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
    rewritten_query: Optional[str]
    retrieved_docs: list[dict]
    answer: str
    hallucination_status: str   # SUPPORTED | PARTIALLY_SUPPORTED | NOT_SUPPORTED
    retry_count: int
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


def evaluate_relevance(state: AgentState) -> Literal["generate", "rewrite"]:
    docs = state["retrieved_docs"]
    if not docs:
        state["trace"].append("[EVALUATE] No documents retrieved → rewriting query")
        return "rewrite"
    top_score = docs[0]["score"]
    if top_score < RELEVANCE_THRESHOLD and state["retry_count"] < MAX_RETRIES:
        state["trace"].append(f"[EVALUATE] Top score {top_score:.3f} < threshold {RELEVANCE_THRESHOLD} → rewriting query")
        return "rewrite"
    state["trace"].append(f"[EVALUATE] Relevance OK (score: {top_score:.3f}) → generating answer")
    return "generate"


def rewrite_query(state: AgentState) -> AgentState:
    state["retry_count"] += 1
    current_answer = state.get("answer", "No answer yet.")
    prompt = build_query_rewrite_prompt(state["query"], current_answer)
    rewritten = generate(prompt)
    # Clean up — take only the first line
    rewritten = rewritten.strip().split("\n")[0]
    state["rewritten_query"] = rewritten
    state["trace"].append(f"[REWRITE] Retry {state['retry_count']}: '{rewritten}'")
    return state


def generate_answer(state: AgentState) -> AgentState:
    prompt = build_rag_prompt(state["query"], state["retrieved_docs"])
    answer = generate(prompt)
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


def should_retry_after_hallucination(state: AgentState) -> Literal["retrieve", "__end__"]:
    if state["hallucination_status"] == "NOT_SUPPORTED" and state["retry_count"] < MAX_RETRIES:
        state["trace"].append("[DECISION] Answer not supported → retrying with rewritten query")
        return "retrieve"
    state["trace"].append("[DECISION] Finalising answer")
    return "__end__"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("retrieve", retrieve)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate_answer", generate_answer)
    g.add_node("check_hallucination", check_hallucination)

    g.set_entry_point("retrieve")

    g.add_conditional_edges(
        "retrieve",
        evaluate_relevance,
        {"generate": "generate_answer", "rewrite": "rewrite_query"},
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate_answer", "check_hallucination")
    g.add_conditional_edges(
        "check_hallucination",
        should_retry_after_hallucination,
        {"retrieve": "rewrite_query", "__end__": END},
    )

    return g.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(query: str) -> dict:
    graph = get_graph()
    initial_state: AgentState = {
        "query": query,
        "rewritten_query": None,
        "retrieved_docs": [],
        "answer": "",
        "hallucination_status": "",
        "retry_count": 0,
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
