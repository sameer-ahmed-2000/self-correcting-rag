import os
import logging
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api import query, ingest, evaluate, sessions
from app.core.database import get_engine  # ensure DB tables are created on startup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag")

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Self-Correcting RAG",
    description="Multi-step agentic RAG pipeline with self-evaluation, streaming, and persistent chat history.",
    version="2.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────────────────────
_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000,http://localhost")
ALLOWED_ORIGINS = [o.strip() for o in _raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(query.router,    prefix="/api/query",    tags=["Query"])
app.include_router(ingest.router,   prefix="/api/ingest",   tags=["Ingest"])
app.include_router(evaluate.router, prefix="/api/evaluate", tags=["Evaluate"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["Sessions"])

# ── Health & Root ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    get_engine()  # initialise SQLite + create tables
    logger.info("✓ Database initialised")
    logger.info(f"✓ CORS allowed origins: {ALLOWED_ORIGINS}")

@app.get("/", tags=["Health"])
def root():
    return {"message": "Self-Correcting RAG API is running", "version": "2.0.0"}

@app.get("/health", tags=["Health"])
def health():
    from pathlib import Path
    index_exists = Path("data/faiss.index").exists()
    return {
        "status": "ok",
        "vector_store": "ready" if index_exists else "empty (ingest documents first)",
    }
