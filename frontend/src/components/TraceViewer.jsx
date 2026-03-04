export default function TraceViewer({ result, onBack }) {
  if (!result) return (
    <div className="panel">
      <p className="muted">No results yet. Ask a question first.</p>
      <button className="back-btn" onClick={onBack}>← Ask a question</button>
    </div>
  );

  const statusColor = {
    SUPPORTED: "#4ade80",
    PARTIALLY_SUPPORTED: "#facc15",
    NOT_SUPPORTED: "#f87171",
  };

  return (
    <div className="panel trace-panel">
      <button className="back-btn" onClick={onBack}>← Ask another</button>

      <div className="answer-card">
        <div className="answer-header">
          <h3>Answer</h3>
          <span
            className="status-badge"
            style={{ background: statusColor[result.hallucination_status] + "22", color: statusColor[result.hallucination_status], border: `1px solid ${statusColor[result.hallucination_status]}44` }}
          >
            {result.hallucination_status}
          </span>
        </div>
        <p className="question-display">Q: {result.query}</p>
        {result.rewritten_query && (
          <p className="rewrite-display">🔄 Rewritten to: <em>{result.rewritten_query}</em></p>
        )}
        <div className="answer-text">{result.answer}</div>
      </div>

      {result.sources?.length > 0 && (
        <div className="sources-section">
          <h4>Sources Used</h4>
          <div className="sources-list">
            {result.sources.map((s, i) => (
              <a key={i} href={s.source} target="_blank" rel="noreferrer" className="source-item">
                <span className="source-score">{(s.score * 100).toFixed(0)}%</span>
                <span className="source-title">{s.title}</span>
                <span className="source-link">↗</span>
              </a>
            ))}
          </div>
        </div>
      )}

      <div className="trace-section">
        <h4>Agent Trace</h4>
        <div className="trace-steps">
          {result.trace?.map((step, i) => {
            const colors = { RETRIEVE: "#60a5fa", EVALUATE: "#facc15", GENERATE: "#4ade80", HALLUCINATION: "#c084fc", REWRITE: "#fb923c", DECISION: "#94a3b8" };
            const tag = step.match(/\[([A-Z_]+)\]/)?.[1] || "INFO";
            const color = Object.entries(colors).find(([k]) => tag.startsWith(k))?.[1] || "#94a3b8";
            return (
              <div key={i} className="trace-step">
                <span className="trace-num">{String(i + 1).padStart(2, "0")}</span>
                <span className="trace-tag" style={{ color }}>[{tag}]</span>
                <span className="trace-msg">{step.replace(/\[[A-Z_]+\]\s*/, "")}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
