# Self-Correcting RAG

A production-style multi-step agentic RAG pipeline built with **LangGraph**, **FAISS**, and **Llama 3 via Groq**.

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green) ![LangGraph](https://img.shields.io/badge/LangGraph-0.1-purple) ![License](https://img.shields.io/badge/License-MIT-gray)

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Agent                          │
│                                                             │
│  RETRIEVE ──→ EVALUATE ──→ GENERATE ──→ CHECK_HALLUCINATION │
│      ↑           │                            │             │
│      │    (low relevance)              (not supported)      │
│      └──── REWRITE_QUERY ◄─────────────────── ┘            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
  Answer + Sources + Hallucination Status + Agent Trace
```

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Vector Store | FAISS + `all-MiniLM-L6-v2` | Semantic retrieval |
| LLM | Llama 3.3 70B via Groq API | Answer generation |
| Agentic Workflow | LangGraph | Multi-step reasoning |
| Evaluation | RAGAS (faithfulness, relevancy, precision, recall) | Quality metrics |
| Knowledge Base | Wikipedia articles (auto-ingested) | Sample data |

## Quickstart

### 1. Backend

We use `uv` for lightning-fast dependency management.

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

### 2. Ingest Wikipedia data

```bash
curl -X POST http://localhost:8000/api/ingest/wikipedia
```

This fetches and chunks 10 Wikipedia articles on AI/ML topics and indexes them into FAISS.
Takes ~30 seconds. Add your own topics by passing `{"topics": ["Your Topic"]}`.

### 3. Frontend

```bash
cd frontend
npm install && npm run dev
# Open http://localhost:3000
```

## Setup

Before starting, get a free Groq API key from [Console.groq.com](https://console.groq.com).

Edit `backend/.env`:

```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/query/` | POST | Run agentic RAG on a question |
| `POST /api/ingest/wikipedia` | POST | Ingest Wikipedia articles |
| `POST /api/evaluate/` | POST | Evaluate a QA pair with RAGAS |
| `GET /health` | GET | Health check |

### Example Query

```bash
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "How does retrieval-augmented generation work?"}'
```

Response:
```json
{
  "query": "How does retrieval-augmented generation work?",
  "answer": "RAG combines a retrieval system with a generative model...",
  "sources": [{"title": "Retrieval-augmented generation", "source": "https://...", "score": 0.89}],
  "hallucination_status": "SUPPORTED",
  "rewritten_query": null,
  "trace": [
    "[RETRIEVE] Query: 'How does...' → 5 docs (top score: 0.891)",
    "[EVALUATE] Relevance OK (score: 0.891) → generating answer",
    "[GENERATE] Answer produced (312 chars)",
    "[HALLUCINATION_CHECK] Status: SUPPORTED",
    "[DECISION] Finalising answer"
  ]
}
```

## Project Structure

```
agentic-rag/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   └── rag_agent.py       # LangGraph workflow
│   │   ├── api/
│   │   │   ├── query.py           # /api/query endpoint
│   │   │   ├── ingest.py          # /api/ingest endpoint
│   │   │   └── evaluate.py        # /api/evaluate endpoint
│   │   ├── core/
│   │   │   ├── vector_store.py    # FAISS + embeddings
│   │   │   ├── llm.py             # Groq API client
│   │   │   └── evaluator.py       # RAGAS evaluation
│   │   └── main.py
│   ├── data/                      # Auto-created on first ingest
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── components/
    │   │   ├── QueryPanel.jsx
    │   │   ├── TraceViewer.jsx
    │   │   └── IngestPanel.jsx
    │   └── App.css
    └── package.json
```

## Skills Demonstrated

- **Agentic RAG** with LangGraph state machine and conditional edges
- **Hallucination detection** via LLM self-evaluation layer
- **Query rewriting** for iterative retrieval improvement
- **RAGAS evaluation** (faithfulness, answer relevancy, context precision, recall)
- **Semantic chunking** of Wikipedia articles with overlap
- **FastAPI** async microservice architecture
- **High-speed inference** via Groq API and Llama 3 models

## License

MIT
