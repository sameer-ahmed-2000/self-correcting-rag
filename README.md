# ⬡ Self-Correcting RAG — Agentic Research Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![LangGraph](https://img.shields.io/badge/LangGraph-agentic-blueviolet)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3-F54703?logoColor=white)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A production-grade, multi-step **Retrieval-Augmented Generation** system built with LangGraph. The agent retrieves, evaluates its own retrieval quality, generates an answer, self-checks for hallucinations, rewrites queries if needed, and falls back to live web search — all streamed token-by-token to the frontend in real time.

---

## 🎥 Features

| Feature | Details |
|---|---|
| 🔄 **Self-Correcting Agent** | LangGraph pipeline: Retrieve → Evaluate → Generate → Hallucination Check → Retry / Web Search |
| ⚡ **Live Streaming** | Server-Sent Events stream agent trace steps and LLM tokens word-by-word |
| 📂 **Multi-Modal Ingestion** | Ingest Wikipedia articles, arbitrary web URLs, and PDF files |
| 🌐 **Web Search Fallback** | DuckDuckGo search automatically triggered when local context is insufficient |
| 💬 **Persistent Chat History** | SQLite-backed sessions with a collapsible sidebar (ChatGPT-style) |
| 🔒 **Rate Limiting** | 60 requests/minute per IP via `slowapi` |
| 🐳 **Docker Ready** | `docker compose up` starts both services with data persistence |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        React Frontend                        │
│  Sidebar (Sessions) │ Chat (SSE Stream) │ Ingest Panel       │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / SSE
┌──────────────────────────────▼──────────────────────────────┐
│                       FastAPI Backend                        │
│  /api/query  /api/ingest  /api/sessions  /health             │
└────────────┬──────────────────────────────┬─────────────────┘
             │                              │
     ┌───────▼────────┐            ┌────────▼────────┐
     │  LangGraph     │            │  SQLite (chat   │
     │  Agent Graph   │            │  history)       │
     │                │            └─────────────────┘
     │  retrieve ──→ evaluate
     │       ↓            ↓
     │  rewrite      web_search
     │       ↓            ↓
     │  generate  ←───────┘
     │       ↓
     │  check_hallucination
     └───────┬────────┘
             │
     ┌───────▼────────┐      ┌──────────────────┐
     │  FAISS Index   │      │  Groq API        │
     │  (local embed) │      │  LLaMA 3.1 8B    │
     └────────────────┘      └──────────────────┘
```

---

## 🚀 Quick Start

### Option A — Docker (Recommended)

```bash
# 1. Clone the repo
git clone https://github.com/your-username/agentic-rag.git
cd agentic-rag

# 2. Configure environment
cp backend/.env.example backend/.env
# Edit backend/.env and add your GROQ_API_KEY

# 3. Launch everything
docker compose up --build

# App available at: http://localhost
# API docs at:      http://localhost:8000/docs
```

### Option B — Manual Dev Setup

**Backend:**
```bash
cd backend
cp .env.example .env          # add your GROQ_API_KEY
uv sync                       # install dependencies
uv run uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
cp .env.example .env.local    # set VITE_API_URL=http://localhost:8000
npm install
npm run dev                   # http://localhost:5173
```

---

## ⚙️ Environment Variables

### Backend (`backend/.env`)

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | **required** | Get free at [console.groq.com](https://console.groq.com) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | `llama-3.1-8b-instant` or `llama-3.3-70b-versatile` |
| `MAX_TOKENS` | `512` | Max LLM output tokens |
| `ALLOWED_ORIGINS` | `http://localhost:5173,...` | Comma-separated CORS origins |

### Frontend (`frontend/.env.local`)

| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8000` | Backend API base URL |

---

## 📖 How the Agent Works

1. **Retrieve** — FAISS similarity search returns top-5 document chunks
2. **Evaluate** — If cosine similarity < 0.35, rewrite the query and retry (max 2×)
3. **Web Search Fallback** — If local context fails after retries, DuckDuckGo search is used
4. **Generate** — LLaMA 3.1 produces an answer grounded in the retrieved context
5. **Hallucination Check** — A second LLM call verifies the answer is `SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED`
6. **Stream** — Every step and every token is pushed to the browser via SSE

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Agent** | LangGraph, LangChain |
| **LLM** | Groq API (LLaMA 3.1 / 3.3) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local, no API key) |
| **Vector Store** | FAISS (CPU) |
| **Backend** | FastAPI, SQLModel, SlowAPI |
| **Database** | SQLite |
| **Frontend** | React 19, Vite, Vanilla CSS |
| **Infra** | Docker, Docker Compose, nginx |

---

## 📁 Project Structure

```
agentic-rag/
├── backend/
│   ├── app/
│   │   ├── agents/rag_agent.py   # LangGraph pipeline (async)
│   │   ├── api/
│   │   │   ├── query.py          # Streaming SSE endpoint + session persistence
│   │   │   ├── ingest.py         # Wikipedia / URL / PDF ingestion
│   │   │   └── sessions.py       # Chat history CRUD
│   │   ├── core/
│   │   │   ├── llm.py            # Groq async client + prompt builders
│   │   │   ├── vector_store.py   # FAISS add / search
│   │   │   └── database.py       # SQLite session/message models
│   │   └── main.py               # FastAPI app + CORS + rate limiting
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx  # Sidebar + streaming chat
│   │   │   └── IngestPanel.jsx    # Wikipedia / URL / PDF UI
│   │   ├── App.jsx
│   │   └── App.css
│   ├── Dockerfile
│   └── .env.example
└── docker-compose.yml
```

---

## 📄 License

MIT © 2025 — Built as a portfolio project demonstrating production-grade Agentic AI engineering.
