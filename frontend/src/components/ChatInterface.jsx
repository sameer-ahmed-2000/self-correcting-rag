import { useEffect, useRef, useState } from "react";

const API = "http://localhost:8000";

export default function ChatInterface() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const EXAMPLES = [
        "What is retrieval-augmented generation?",
        "How does the transformer architecture work?",
        "What are the key differences between BERT and GPT?"
    ];

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, loading]);

    async function handleSubmit(q = input) {
        if (!q.trim()) return;

        const newUserMsg = { role: "user", content: q };
        setMessages(prev => [...prev, newUserMsg]);
        setInput("");
        setLoading(true);

        // Build history for backend: array of {role, content}
        const historyForBackend = messages.map(m => ({
            role: m.role,
            content: m.role === 'assistant' && m.raw ? m.raw.answer : m.content
        }));

        try {
            const res = await fetch(`${API}/api/query/`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: q, history: historyForBackend }),
            });
            if (!res.ok) throw new Error(`Server error: ${res.status}`);
            const data = await res.json();

            setMessages(prev => [...prev, { role: "assistant", raw: data }]);
        } catch (e) {
            setMessages(prev => [...prev, { role: "assistant", error: e.message }]);
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="chat-container">
            <div className="chat-messages">
                {messages.length === 0 && (
                    <div className="chat-empty">
                        <span className="chat-empty-icon">⬡</span>
                        <h2>Self-Correcting RAG Chat</h2>
                        <p className="muted">Ask any research question. The agent will retrieve, evaluate, and check its facts automatically.</p>
                        <div className="examples chat-examples">
                            {EXAMPLES.map(ex => (
                                <button key={ex} className="example-pill" onClick={() => handleSubmit(ex)}>{ex}</button>
                            ))}
                        </div>
                    </div>
                )}

                {messages.map((m, idx) => (
                    <div key={idx} className={`chat-message ${m.role}`}>
                        <div className="chat-avatar">{m.role === "user" ? "You" : "⬡"}</div>
                        <div className="chat-content">
                            {m.role === "user" ? (
                                <div className="user-text">{m.content}</div>
                            ) : m.error ? (
                                <div className="error-box">⚠ {m.error}</div>
                            ) : (
                                <AssistantMessage data={m.raw} />
                            )}
                        </div>
                    </div>
                ))}
                {loading && (
                    <div className="chat-message assistant">
                        <div className="chat-avatar">⬡</div>
                        <div className="chat-content">
                            <div className="loading-dots"><span className="spinner small-spinner" /> Thinking, retrieving, and verifying...</div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <div className="chat-input-wrapper">
                <div className="chat-input-area">
                    <textarea
                        className="chat-input"
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        placeholder="Message Agentic RAG..."
                        rows={1}
                        onKeyDown={e => {
                            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
                        }}
                    />
                    <button className="chat-send-btn" onClick={() => handleSubmit()} disabled={loading || !input.trim()}>
                        ↑
                    </button>
                </div>
            </div>
        </div>
    );
}

function AssistantMessage({ data }) {
    const [showTrace, setShowTrace] = useState(false);

    if (!data) return null;

    const statusColor = {
        SUPPORTED: "#4ade80",
        PARTIALLY_SUPPORTED: "#facc15",
        NOT_SUPPORTED: "#f87171",
    };

    return (
        <div className="assistant-msg-inner">
            {data.rewritten_query && (
                <div className="chat-rewritten">
                    <span className="muted">Searched for: </span>
                    <em>"{data.rewritten_query}"</em>
                </div>
            )}

            <div className="chat-answer">{data.answer}</div>

            <div className="chat-meta">
                <button className="toggle-trace-btn" onClick={() => setShowTrace(!showTrace)}>
                    {showTrace ? "Hide Agent Trace ↑" : "View Agent Trace ↓"}
                </button>
                <span
                    className="status-badge"
                    style={{ background: statusColor[data.hallucination_status] + "22", color: statusColor[data.hallucination_status], border: `1px solid ${statusColor[data.hallucination_status]}44` }}
                    title="Hallucination Check Result"
                >
                    {data.hallucination_status}
                </span>
            </div>

            {showTrace && (
                <div className="chat-trace-box">
                    {data.sources?.length > 0 && (
                        <div className="sources-list chat-sources">
                            {data.sources.map((s, i) => (
                                <a key={i} href={s.source} target="_blank" rel="noreferrer" className="source-item mini">
                                    <span className="source-score">{(s.score * 100).toFixed(0)}%</span>
                                    <span className="source-title" style={{ fontSize: '0.8rem' }}>{s.title}</span>
                                </a>
                            ))}
                        </div>
                    )}
                    <div className="trace-steps chat-traces" style={{ padding: '0.8rem' }}>
                        {data.trace?.map((step, i) => {
                            const colors = { RETRIEVE: "#60a5fa", EVALUATE: "#facc15", GENERATE: "#4ade80", HALLUCINATION: "#c084fc", REWRITE: "#fb923c", WEB: "#2dd4bf", DECISION: "#94a3b8" };
                            const tag = step.match(/\[([A-Z_]+)\]/)?.[1] || "INFO";
                            const color = Object.entries(colors).find(([k]) => tag.startsWith(k))?.[1] || "#94a3b8";
                            return (
                                <div key={i} className="trace-step mini-trace" style={{ fontSize: '0.75rem' }}>
                                    <span className="trace-tag" style={{ color, minWidth: '70px' }}>[{tag}]</span>
                                    <span className="trace-msg">{step.replace(/\[[A-Z_]+\]\s*/, "")}</span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}
