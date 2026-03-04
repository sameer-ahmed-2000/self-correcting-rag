import { useState } from "react";

const API = "http://localhost:8000";

export default function QueryPanel({ onResult }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const EXAMPLES = [
    "What is retrieval-augmented generation?",
    "How does the transformer architecture work?",
    "What are the key differences between BERT and GPT?",
    "Explain vector databases and their use in AI.",
  ];

  async function handleSubmit(q = question) {
    if (!q.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API}/api/query/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      onResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel query-panel">
      <h2 className="panel-title">Ask a Research Question</h2>
      <p className="panel-sub">The agent will retrieve, evaluate, and self-check its answer before responding.</p>

      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button key={ex} className="example-pill" onClick={() => { setQuestion(ex); handleSubmit(ex); }}>
            {ex}
          </button>
        ))}
      </div>

      <div className="input-row">
        <textarea
          className="question-input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask anything about AI, ML, NLP..."
          rows={3}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); } }}
        />
        <button className="submit-btn" onClick={() => handleSubmit()} disabled={loading || !question.trim()}>
          {loading ? <span className="spinner" /> : "Ask →"}
        </button>
      </div>

      {error && <div className="error-box">⚠ {error}</div>}

      {loading && (
        <div className="loading-panel">
          <div className="loading-steps">
            {["Retrieving documents…", "Evaluating relevance…", "Generating answer…", "Checking for hallucinations…"].map((s, i) => (
              <div key={i} className="loading-step" style={{ animationDelay: `${i * 0.6}s` }}>{s}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
