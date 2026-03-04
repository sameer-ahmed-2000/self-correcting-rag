import json
import asyncio
import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from app.agents.rag_agent import run_agent_async
from app.core.llm import get_async_groq_client
from app.core.cache import get_cached, set_cached
from app.core.database import (
    create_session, add_message, touch_session, get_session
)

logger = logging.getLogger("rag.query")
router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []
    session_id: int | None = None

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        if len(v) > 1000:
            raise ValueError("Question must be 1000 characters or fewer.")
        return v

    @field_validator("history")
    @classmethod
    def history_max_depth(cls, v: list) -> list:
        return v[-40:] if len(v) > 40 else v


@router.post("/")
async def query(req: QueryRequest, request: Request):
    req_id = getattr(request.state, "request_id", "n/a")
    logger.info(f"[{req_id}] Query: '{req.question[:80]}...' " if len(req.question) > 80 else f"[{req_id}] Query: '{req.question}'")

    get_async_groq_client()  # validate API key + warm up

    # ── Redis cache check ──────────────────────────────────────
    cached = get_cached(req.question, len(req.history))
    if cached:
        logger.info(f"[{req_id}] Serving from cache")
        # Still persist to session if given
        if req.session_id:
            add_message(req.session_id, "user", req.question)
            add_message(
                req.session_id, "assistant", cached["answer"],
                metadata=cached,
            )
            touch_session(req.session_id)

        async def cached_stream():
            yield f"data: {json.dumps({'type': 'cache_hit'})}\n\n"
            # Stream cached answer token-by-token for the typing effect
            for word in cached["answer"].split():
                yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
                await asyncio.sleep(0.01)
            yield f"data: {json.dumps({'type': 'end', 'result': cached})}\n\n"
        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    # ── Session management ─────────────────────────────────────
    session_id = req.session_id
    is_new_session = False
    if session_id is None:
        session = create_session(req.question[:60])
        session_id = session.id
        is_new_session = True
    else:
        if not get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

    add_message(session_id, "user", req.question)

    async def event_generator():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'is_new': is_new_session})}\n\n"

        queue = asyncio.Queue()
        task = asyncio.create_task(
            run_agent_async(req.question, req.history, queue)
        )

        final_result = None
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=90.0)
            except asyncio.TimeoutError:
                logger.error(f"[{req_id}] Pipeline timed out")
                yield f"data: {json.dumps({'type': 'error', 'content': 'Request timed out'})}\n\n"
                task.cancel()
                break

            if event["type"] == "end":
                final_result = event["result"]
                add_message(
                    session_id, "assistant", final_result["answer"],
                    metadata={
                        "sources": final_result.get("sources", []),
                        "hallucination_status": final_result.get("hallucination_status"),
                        "rewritten_query": final_result.get("rewritten_query"),
                        "trace": final_result.get("trace", []),
                    },
                )
                touch_session(session_id)
                # Cache result for future identical queries
                set_cached(req.question, len(req.history), final_result)
                yield f"data: {json.dumps({'type': 'end', 'result': final_result})}\n\n"
                break

            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
