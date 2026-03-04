"""
Sessions REST API — CRUD for chat sessions and message history.
"""
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.database import (
    create_session, get_all_sessions, get_session,
    get_messages, delete_session, update_session_title
)

router = APIRouter()


class CreateSessionRequest(BaseModel):
    title: str = "New Chat"


class RenameSessionRequest(BaseModel):
    title: str


@router.get("/")
def list_sessions():
    sessions = get_all_sessions()
    return [
        {"id": s.id, "title": s.title, "created_at": s.created_at, "updated_at": s.updated_at}
        for s in sessions
    ]


@router.post("/")
def new_session(req: CreateSessionRequest = CreateSessionRequest()):
    session = create_session(req.title)
    return {"id": session.id, "title": session.title, "created_at": session.created_at}


@router.get("/{session_id}/messages")
def list_messages(session_id: int):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = get_messages(session_id)
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "metadata": json.loads(m.metadata_json) if m.metadata_json else None,
            "created_at": m.created_at,
        }
        for m in messages
    ]


@router.patch("/{session_id}")
def rename_session(session_id: int, req: RenameSessionRequest):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    update_session_title(session_id, req.title)
    return {"ok": True}


@router.delete("/{session_id}")
def remove_session(session_id: int):
    ok = delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}
