import { useState } from "react";

const API = "http://localhost:8000";

const DEFAULT_TOPICS = [
  "Artificial intelligence", "Machine learning", "Large language model",
  "Retrieval-augmented generation", "Natural language processing",
  "Vector database", "Transformer (deep learning architecture)",
  "BERT (language model)", "GPT-4", "Information retrieval",
];

export default function IngestPanel() {
  const [activeTab, setActiveTab] = useState("wikipedia");
  // Wikipedia state
  const [topics, setTopics] = useState(DEFAULT_TOPICS.join("\n"));
  // URL state
  const [urls, setUrls] = useState("");
  // PDF state
  const [file, setFile] = useState(null);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function handleIngestWikipedia() {
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

  async function handleIngestUrl() {
    setLoading(true); setError(""); setResult(null);
    const urlList = urls.split("\n").map(t => t.trim()).filter(Boolean);
    try {
      const res = await fetch(`${API}/api/ingest/url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls: urlList }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleIngestPdf() {
    if (!file) return;
    setLoading(true); setError(""); setResult(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${API}/api/ingest/pdf`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }
      setResult(await res.json());
      setFile(null); // Clear file after successful upload
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel ingest-panel" style={{ maxWidth: '800px', margin: '0 auto', width: '100%' }}>
      <h2 className="panel-title">Add Data to Knowledge Base</h2>
      <p className="panel-sub">Expand your agent's brain by ingesting new documents.</p>

      <div className="tabs" style={{ marginBottom: '1.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.5rem' }}>
        <button className={`tab ${activeTab === 'wikipedia' ? 'active' : ''}`} onClick={() => { setActiveTab('wikipedia'); setResult(null); setError(""); }}>Wikipedia</button>
        <button className={`tab ${activeTab === 'url' ? 'active' : ''}`} onClick={() => { setActiveTab('url'); setResult(null); setError(""); }}>Web URL</button>
        <button className={`tab ${activeTab === 'pdf' ? 'active' : ''}`} onClick={() => { setActiveTab('pdf'); setResult(null); setError(""); }}>PDF File</button>
      </div>

      <div className="ingest-content">
        {activeTab === "wikipedia" && (
          <div className="ingest-form">
            <label className="field-label">Topics (one per line)</label>
            <textarea className="question-input" rows={8} value={topics} onChange={(e) => setTopics(e.target.value)} />
            <button className="submit-btn" onClick={handleIngestWikipedia} disabled={loading || !topics.trim()} style={{ marginTop: "1rem" }}>
              {loading ? <><span className="spinner" /> Ingesting…</> : "📥 Ingest Wikipedia"}
            </button>
          </div>
        )}

        {activeTab === "url" && (
          <div className="ingest-form">
            <label className="field-label">Web URLs (one per line)</label>
            <textarea className="question-input text-input" rows={6} value={urls} onChange={(e) => setUrls(e.target.value)} placeholder="https://example.com/article" />
            <button className="submit-btn" onClick={handleIngestUrl} disabled={loading || !urls.trim()} style={{ marginTop: "1rem" }}>
              {loading ? <><span className="spinner" /> Scanning URLs…</> : "📥 Scrape & Ingest URLs"}
            </button>
          </div>
        )}

        {activeTab === "pdf" && (
          <div className="ingest-form">
            <label className="field-label">Upload a PDF Document</label>
            <div
              className="pdf-dropzone"
              style={{
                border: '2px dashed var(--border)', borderRadius: '12px', padding: '3rem 2rem',
                textAlign: 'center', background: 'var(--surface)', cursor: 'pointer',
                transition: 'border-color 0.2s', borderColor: file ? 'var(--accent)' : 'var(--border)'
              }}
              onClick={() => document.getElementById('pdf-upload').click()}
            >
              <span style={{ fontSize: '2rem', display: 'block', marginBottom: '1rem', color: file ? 'var(--accent)' : 'var(--muted)' }}>
                {file ? '📄' : '📁'}
              </span>
              <p style={{ color: file ? 'var(--text)' : 'var(--muted)', fontWeight: file ? 500 : 400 }}>
                {file ? file.name : 'Click to browse for a PDF file'}
              </p>
              <input
                id="pdf-upload"
                type="file"
                accept="application/pdf"
                style={{ display: 'none' }}
                onChange={e => e.target.files && setFile(e.target.files[0])}
              />
            </div>
            <button className="submit-btn" onClick={handleIngestPdf} disabled={loading || !file} style={{ marginTop: "1.5rem" }}>
              {loading ? <><span className="spinner" /> Processing PDF…</> : "📥 Extract & Ingest PDF"}
            </button>
          </div>
        )}
      </div>

      {error && <div className="error-box" style={{ marginTop: '1.5rem' }}>⚠ {error}</div>}

      {result && (
        <div className="result-card" style={{ marginTop: "1.5rem", borderLeft: '4px solid var(--accent)' }}>
          <div className="stat-row">
            <div className="stat"><span>{result.chunks_added}</span>Chunks Added</div>
            <div className="stat"><span>{result.topics_ingested.length}</span>Sources Ingested</div>
            <div className="stat"><span>{result.errors.length}</span>Errors</div>
          </div>
          {result.errors.length > 0 && (
            <div className="error-box" style={{ marginTop: "1rem", background: 'rgba(248,113,113,0.1)' }}>
              {result.errors.map((e, i) => <div key={i}>{e}</div>)}
            </div>
          )}
          <div className="success-note">✓ Vector store updated. Switch to the Chat tab to query.</div>
        </div>
      )}
    </div>
  );
}
