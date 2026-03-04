"""
Agentic RAG workflow using LangGraph.

Graph flow:
  retrieve → evaluate_relevance → generate → check_hallucination
                ↓ (low relevance)                    ↓ (not supported)
            rewrite_query ──────────────────→ retrieve (retry, max 2x)
"""
import asyncio
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END

from app.core.vector_store import similarity_search
from app.core.llm import agenerate, build_rag_prompt, build_query_rewrite_prompt, build_hallucination_check_prompt

class AgentState(TypedDict):
    query: str
    history: list[dict]
    rewritten_query: Optional[str]
    retrieved_docs: list[dict]
    answer: str
    hallucination_status: str
    retry_count: int
    web_search_done: bool
    trace: list[str]
    stream_queue: Optional[asyncio.Queue]


MAX_RETRIES = 2
RELEVANCE_THRESHOLD = 0.35

async def append_trace(state: AgentState, msg: str):
    state["trace"].append(msg)
    if state.get("stream_queue"):
        await state["stream_queue"].put({"type": "trace", "content": msg})

async def retrieve(state: AgentState) -> AgentState:
    q = state.get("rewritten_query") or state["query"]
    docs = await asyncio.to_thread(similarity_search, q, k=5)
    state["retrieved_docs"] = docs
    msg = f"[RETRIEVE] Query: '{q}' → {len(docs)} docs (top score: {docs[0]['score']:.3f})" if docs else "[RETRIEVE] No docs found"
    await append_trace(state, msg)
    return state

async def evaluate_relevance(state: AgentState) -> Literal["generate", "rewrite", "web_search"]:
    docs = state["retrieved_docs"]
    top_score = docs[0]["score"] if docs else 0.0

    if not docs or top_score < RELEVANCE_THRESHOLD:
        if state["retry_count"] < MAX_RETRIES:
            await append_trace(state, f"[EVALUATE] Score {top_score:.3f} < threshold {RELEVANCE_THRESHOLD} → rewriting query")
            return "rewrite"
        elif not state.get("web_search_done"):
            await append_trace(state, f"[EVALUATE] Score {top_score:.3f} < threshold {RELEVANCE_THRESHOLD} after retries → web search fallback")
            return "web_search"

    await append_trace(state, f"[EVALUATE] Relevance OK (score: {top_score:.3f}) → generating answer")
    return "generate"

async def web_search_node(state: AgentState) -> AgentState:
    state["web_search_done"] = True
    q = state.get("rewritten_query") or state["query"]
    await append_trace(state, f"[WEB_SEARCH] Searching DuckDuckGo for: '{q}'")
    try:
        from ddgs import DDGS
        def do_search():
            with DDGS() as ddgs:
                return list(ddgs.text(q, max_results=3))
        results = await asyncio.to_thread(do_search)
        docs = []
        for r in results:
            docs.append({
                "title": r.get("title", ""),
                "text": r.get("body", ""),
                "source": r.get("href", ""),
                "score": 1.0
            })
        state["retrieved_docs"] = docs
        if docs:
            await append_trace(state, f"[WEB_SEARCH] Found {len(docs)} external sources")
        else:
            await append_trace(state, "[WEB_SEARCH] No external sources found")
    except Exception as e:
        await append_trace(state, f"[WEB_SEARCH] Search failed: {str(e)}")
        state["retrieved_docs"] = []
    return state

async def rewrite_query(state: AgentState) -> AgentState:
    state["retry_count"] += 1
    current_answer = state.get("answer", "No answer yet.")
    prompt = build_query_rewrite_prompt(state["query"], current_answer)
    rewritten = await agenerate(prompt, history=state.get("history", []))
    rewritten = rewritten.strip().split("\n")[0]
    state["rewritten_query"] = rewritten
    await append_trace(state, f"[REWRITE] Retry {state['retry_count']}: '{rewritten}'")
    
    if state.get("stream_queue"):
        await state["stream_queue"].put({"type": "rewrite", "content": rewritten})
        
    return state

async def generate_answer(state: AgentState) -> AgentState:
    prompt = build_rag_prompt(state["query"], state["retrieved_docs"])
    
    if state.get("stream_queue"):
        await state["stream_queue"].put({"type": "clear_tokens"})
        
    answer = await agenerate(prompt, history=state.get("history", []), token_queue=state.get("stream_queue"))
    state["answer"] = answer
    await append_trace(state, f"[GENERATE] Answer produced ({len(answer)} chars)")
    return state

async def check_hallucination(state: AgentState) -> AgentState:
    context = " ".join(d["text"] for d in state["retrieved_docs"][:3])
    prompt = build_hallucination_check_prompt(state["query"], state["answer"], context)
    result = (await agenerate(prompt)).strip().upper()
    if "NOT_SUPPORTED" in result:
        status = "NOT_SUPPORTED"
    elif "PARTIALLY" in result:
        status = "PARTIALLY_SUPPORTED"
    else:
        status = "SUPPORTED"
    state["hallucination_status"] = status
    await append_trace(state, f"[HALLUCINATION_CHECK] Status: {status}")
    return state

async def should_retry_after_hallucination(state: AgentState) -> Literal["retrieve", "web_search", "__end__"]:
    if state["hallucination_status"] == "NOT_SUPPORTED" or "don't have enough information" in state["answer"].lower():
        if state["retry_count"] < MAX_RETRIES:
            await append_trace(state, "[DECISION] Answer not supported → retrying with rewritten query")
            return "retrieve"
        elif not state.get("web_search_done"):
            await append_trace(state, "[DECISION] Answer not supported after retries → trying web search")
            return "web_search"
    await append_trace(state, "[DECISION] Finalising answer")
    return "__end__"

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("web_search", web_search_node)
    g.add_node("generate_answer", generate_answer)
    g.add_node("check_hallucination", check_hallucination)
    
    g.set_entry_point("retrieve")
    
    g.add_conditional_edges("retrieve", evaluate_relevance, {"generate": "generate_answer", "rewrite": "rewrite_query", "web_search": "web_search"})
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("web_search", "generate_answer")
    g.add_edge("generate_answer", "check_hallucination")
    g.add_conditional_edges("check_hallucination", should_retry_after_hallucination, {"retrieve": "rewrite_query", "web_search": "web_search", "__end__": END})
    
    return g.compile()

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

async def run_agent_async(query: str, history: list[dict] = None, queue: asyncio.Queue = None) -> dict:
    graph = get_graph()
    initial_state: AgentState = {
        "query": query,
        "history": history or [],
        "rewritten_query": None,
        "retrieved_docs": [],
        "answer": "",
        "hallucination_status": "",
        "retry_count": 0,
        "web_search_done": False,
        "trace": [],
        "stream_queue": queue
    }
    final_state = await graph.ainvoke(initial_state)
    result = {
        "query": query,
        "answer": final_state["answer"],
        "sources": [{"title": d["title"], "source": d["source"], "score": d["score"]} for d in final_state["retrieved_docs"]],
        "hallucination_status": final_state["hallucination_status"],
        "rewritten_query": final_state.get("rewritten_query"),
        "trace": final_state["trace"],
    }
    if queue:
        await queue.put({"type": "end", "result": result})
    return result
