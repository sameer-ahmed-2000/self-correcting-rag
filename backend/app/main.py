import os
import uuid
import logging
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api import query, ingest, evaluate, sessions
from app.core.database import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("rag")

# ── API Key Auth ──────────────────────────────────────────────────────────────
API_KEY        = os.getenv("API_KEY", "")          # empty = auth disabled
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    """Dependency — skip if API_KEY env var is not set (dev mode)."""
    if not API_KEY:
        return  # auth disabled
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Set X-API-Key header.")

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Self-Correcting RAG",
    description="Production-grade agentic RAG pipeline with streaming, self-evaluation, persistent history, and Redis caching.",
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

# ── Request ID middleware ─────────────────────────────────────────────────────
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        f"[{request.method}] {request.url.path} "
        f"| status={response.status_code} "
        f"| req_id={request_id}"
    )
    return response

# ── Routers ───────────────────────────────────────────────────────────────────
# Protected routes use verify_api_key dependency
app.include_router(
    query.router,
    prefix="/api/query",
    tags=["Query"],
    dependencies=[Security(verify_api_key)],
)
app.include_router(
    ingest.router,
    prefix="/api/ingest",
    tags=["Ingest"],
    dependencies=[Security(verify_api_key)],
)
app.include_router(
    sessions.router,
    prefix="/api/sessions",
    tags=["Sessions"],
    dependencies=[Security(verify_api_key)],
)
app.include_router(evaluate.router, prefix="/api/evaluate", tags=["Evaluate"])

# ── Startup / Health ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    get_engine()
    logger.info("✓ SQLite database initialised")
    logger.info(f"✓ CORS origins: {ALLOWED_ORIGINS}")
    logger.info(f"✓ API Key auth: {'ENABLED' if API_KEY else 'DISABLED (set API_KEY to enable)'}")

    # Log Redis status
    from app.core.cache import _get_client
    _get_client()  # trigger connection attempt + log

@app.get("/", tags=["Health"])
def root():
    return {"message": "Self-Correcting RAG API", "version": "2.0.0", "docs": "/docs"}

@app.get("/health", tags=["Health"])
def health():
    from pathlib import Path
    index_exists = Path("data/faiss.index").exists()
    redis_ok = False
    try:
        from app.core.cache import _get_client, _redis_available
        _get_client()
        redis_ok = _redis_available
    except Exception:
        pass
    return {
        "status": "ok",
        "vector_store": "ready" if index_exists else "empty — ingest documents first",
        "cache": "redis connected" if redis_ok else "disabled (no REDIS_URL)",
    }
