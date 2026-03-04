"""
SQLite database setup using SQLModel.
Stores chat sessions and messages for persistent conversation history.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from sqlmodel import Field, SQLModel, create_engine, Session, select

DB_PATH = Path("data/chat.db")
ENGINE = None


def get_engine():
    global ENGINE
    if ENGINE is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        ENGINE = create_engine(f"sqlite:///{DB_PATH}", echo=False)
        SQLModel.metadata.create_all(ENGINE)
    return ENGINE


# ── Models ────────────────────────────────────────────────────────────────────

class ChatSession(SQLModel, table=True):
    __tablename__ = "sessions"
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(default="New Chat")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ChatMessage(SQLModel, table=True):
    __tablename__ = "messages"
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="sessions.id", index=True)
    role: str  # "user" | "assistant"
    content: str
    metadata_json: Optional[str] = Field(default=None)  # JSON: sources, trace, status
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def create_session(title: str = "New Chat") -> ChatSession:
    with Session(get_engine()) as db:
        session = ChatSession(title=title)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session


def get_all_sessions() -> List[ChatSession]:
    with Session(get_engine()) as db:
        return db.exec(select(ChatSession).order_by(ChatSession.updated_at.desc())).all()


def get_session(session_id: int) -> Optional[ChatSession]:
    with Session(get_engine()) as db:
        return db.get(ChatSession, session_id)


def update_session_title(session_id: int, title: str) -> None:
    with Session(get_engine()) as db:
        session = db.get(ChatSession, session_id)
        if session:
            session.title = title
            session.updated_at = datetime.now(timezone.utc).isoformat()
            db.add(session)
            db.commit()


def touch_session(session_id: int) -> None:
    with Session(get_engine()) as db:
        session = db.get(ChatSession, session_id)
        if session:
            session.updated_at = datetime.now(timezone.utc).isoformat()
            db.add(session)
            db.commit()


def add_message(session_id: int, role: str, content: str, metadata: dict = None) -> ChatMessage:
    with Session(get_engine()) as db:
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg


def get_messages(session_id: int) -> List[ChatMessage]:
    with Session(get_engine()) as db:
        return db.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id)
        ).all()


def delete_session(session_id: int) -> bool:
    with Session(get_engine()) as db:
        # Delete messages first
        messages = db.exec(select(ChatMessage).where(ChatMessage.session_id == session_id)).all()
        for m in messages:
            db.delete(m)
        session = db.get(ChatSession, session_id)
        if session:
            db.delete(session)
            db.commit()
            return True
        db.commit()
        return False
