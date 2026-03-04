"""
Agentic RAG workflow using LangGraph.

Graph flow:
  retrieve → evaluate_relevance → generate → check_hallucination
                ↓ (low relevance)                    ↓ (not supported)
            rewrite_query ──────────────────→ retrieve (retry, max 2x)

States:
  - RETRIEVE: fetch top-k docs from FAISS
  - EVALUATE: check if retrieved docs are relevant enough
  - GENERATE: produce answer from context
  - CHECK_HALLUCINATION: verify answer is grounded
  - REWRITE: rephrase the query and retry
  - DONE: return final answer
"""
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END

from app.core.vector_store import similarity_search
from app.core.llm import generate, build_rag_prompt, build_query_rewrite_prompt, build_hallucination_check_prompt


# ── State schema ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    history: list[dict]
    rewritten_query: Optional[str]
    retrieved_docs: list[dict]
    answer: str
    hallucination_status: str   # SUPPORTED | PARTIALLY_SUPPORTED | NOT_SUPPORTED
    retry_count: int
    web_search_done: bool
    trace: list[str]            # step-by-step reasoning log


MAX_RETRIES = 2
RELEVANCE_THRESHOLD = 0.35      # cosine similarity threshold


# ── Node functions ────────────────────────────────────────────────────────────

def retrieve(state: AgentState) -> AgentState:
    q = state.get("rewritten_query") or state["query"]
    docs = similarity_search(q, k=5)
    state["retrieved_docs"] = docs
    state["trace"].append(f"[RETRIEVE] Query: '{q}' → {len(docs)} docs (top score: {docs[0]['score']:.3f})" if docs else "[RETRIEVE] No docs found")
    return state


def evaluate_relevance(state: AgentState) -> Literal["generate", "rewrite", "web_search"]:
    docs = state["retrieved_docs"]
    top_score = docs[0]["score"] if docs else 0.0

    if not docs or top_score < RELEVANCE_THRESHOLD:
        if state["retry_count"] < MAX_RETRIES:
            state["trace"].append(f"[EVALUATE] Score {top_score:.3f} < threshold {RELEVANCE_THRESHOLD} → rewriting query")
            return "rewrite"
        elif not state.get("web_search_done"):
            state["trace"].append(f"[EVALUATE] Score {top_score:.3f} < threshold {RELEVANCE_THRESHOLD} after retries → web search fallback")
            return "web_search"

    state["trace"].append(f"[EVALUATE] Relevance OK (score: {top_score:.3f}) → generating answer")
    return "generate"

def web_search_node(state: AgentState) -> AgentState:
    state["web_search_done"] = True
    q = state.get("rewritten_query") or state["query"]
    state["trace"].append(f"[WEB_SEARCH] Searching DuckDuckGo for: '{q}'")
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(q, max_results=3))
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
            state["trace"].append(f"[WEB_SEARCH] Found {len(docs)} external sources")
        else:
            state["trace"].append("[WEB_SEARCH] No external sources found")
    except Exception as e:
        state["trace"].append(f"[WEB_SEARCH] Search failed: {str(e)}")
        state["retrieved_docs"] = []
    
    return state


def rewrite_query(state: AgentState) -> AgentState:
    state["retry_count"] += 1
    current_answer = state.get("answer", "No answer yet.")
    prompt = build_query_rewrite_prompt(state["query"], current_answer)
    rewritten = generate(prompt, history=state.get("history", []))
    # Clean up — take only the first line
    rewritten = rewritten.strip().split("\n")[0]
    state["rewritten_query"] = rewritten
    state["trace"].append(f"[REWRITE] Retry {state['retry_count']}: '{rewritten}'")
    return state


def generate_answer(state: AgentState) -> AgentState:
    prompt = build_rag_prompt(state["query"], state["retrieved_docs"])
    answer = generate(prompt, history=state.get("history", []))
    state["answer"] = answer
    state["trace"].append(f"[GENERATE] Answer produced ({len(answer)} chars)")
    return state


def check_hallucination(state: AgentState) -> AgentState:
    context = " ".join(d["text"] for d in state["retrieved_docs"][:3])
    prompt = build_hallucination_check_prompt(state["query"], state["answer"], context)
    result = generate(prompt).strip().upper()
    # Normalise output
    if "NOT_SUPPORTED" in result:
        status = "NOT_SUPPORTED"
    elif "PARTIALLY" in result:
        status = "PARTIALLY_SUPPORTED"
    else:
        status = "SUPPORTED"
    state["hallucination_status"] = status
    state["trace"].append(f"[HALLUCINATION_CHECK] Status: {status}")
    return state


def should_retry_after_hallucination(state: AgentState) -> Literal["retrieve", "web_search", "__end__"]:
    # If the LLM said it doesn't have enough info, or the answer is ungrounded
    if state["hallucination_status"] == "NOT_SUPPORTED" or "don't have enough information" in state["answer"].lower():
        if state["retry_count"] < MAX_RETRIES:
            state["trace"].append("[DECISION] Answer not supported → retrying with rewritten query")
            return "retrieve"
        elif not state.get("web_search_done"):
            state["trace"].append("[DECISION] Answer not supported after retries → trying web search")
            return "web_search"
    state["trace"].append("[DECISION] Finalising answer")
    return "__end__"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("retrieve", retrieve)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("web_search", web_search_node)
    g.add_node("generate_answer", generate_answer)
    g.add_node("check_hallucination", check_hallucination)

    g.set_entry_point("retrieve")

    g.add_conditional_edges(
        "retrieve",
        evaluate_relevance,
        {"generate": "generate_answer", "rewrite": "rewrite_query", "web_search": "web_search"},
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("web_search", "generate_answer")
    g.add_edge("generate_answer", "check_hallucination")
    g.add_conditional_edges(
        "check_hallucination",
        should_retry_after_hallucination,
        {"retrieve": "rewrite_query", "web_search": "web_search", "__end__": END},
    )

    return g.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(query: str, history: list[dict] = None) -> dict:
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
    }
    final_state = graph.invoke(initial_state)
    return {
        "query": query,
        "answer": final_state["answer"],
        "sources": [
            {"title": d["title"], "source": d["source"], "score": d["score"]}
            for d in final_state["retrieved_docs"]
        ],
        "hallucination_status": final_state["hallucination_status"],
        "rewritten_query": final_state.get("rewritten_query"),
        "trace": final_state["trace"],
    }
