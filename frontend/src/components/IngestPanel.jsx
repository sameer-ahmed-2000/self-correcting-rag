import { useState } from "react";

const API = "http://localhost:8000";

const DEFAULT_TOPICS = [
  "Artificial intelligence", "Machine learning", "Large language model",
  "Retrieval-augmented generation", "Natural language processing",
  "Vector database", "Transformer (deep learning architecture)",
  "BERT (language model)", "GPT-4", "Information retrieval",
];

export default function IngestPanel() {
  const [topics, setTopics] = useState(DEFAULT_TOPICS.join("\n"));
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function handleIngest() {
    setLoading(true); setError(""); setResult(null);
    const topicList = topics.split("\n").map(t => t.trim()).filter(Boolean);
    try {
      const res = await fetch(`${API}/api/ingest/wikipedia`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topics: topicList }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel">
      <h2 className="panel-title">Ingest Wikipedia Articles</h2>
      <p className="panel-sub">Each article is chunked into 300-word passages and indexed into FAISS.</p>

      <label className="field-label">Topics (one per line)</label>
      <textarea
        className="question-input"
        rows={12}
        value={topics}
        onChange={(e) => setTopics(e.target.value)}
      />

      <button className="submit-btn" onClick={handleIngest} disabled={loading} style={{ marginTop: "1rem" }}>
        {loading ? <><span className="spinner" /> Ingesting…</> : "📥 Ingest into FAISS"}
      </button>

      {error && <div className="error-box">⚠ {error}</div>}

      {result && (
        <div className="result-card" style={{ marginTop: "1.5rem" }}>
          <div className="stat-row">
            <div className="stat"><span>{result.chunks_added}</span>Chunks Added</div>
            <div className="stat"><span>{result.topics_ingested.length}</span>Topics Ingested</div>
            <div className="stat"><span>{result.errors.length}</span>Errors</div>
          </div>
          {result.errors.length > 0 && (
            <div className="error-box" style={{ marginTop: "1rem" }}>
              {result.errors.map((e, i) => <div key={i}>{e}</div>)}
            </div>
          )}
          <div className="success-note">✓ Vector store ready. Switch to the Ask tab to query.</div>
        </div>
      )}
    </div>
  );
}
