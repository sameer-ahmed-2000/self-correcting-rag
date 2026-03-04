from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import query, ingest, evaluate

app = FastAPI(
    title="Self-Correcting RAG",
    description="Multi-step agentic RAG pipeline with self-evaluation using open-source models.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/api/query", tags=["Query"])
app.include_router(ingest.router, prefix="/api/ingest", tags=["Ingest"])
app.include_router(evaluate.router, prefix="/api/evaluate", tags=["Evaluate"])


@app.get("/health")
def health():
    return {"status": "ok"}
