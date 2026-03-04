import { useEffect, useRef, useState } from "react";

const API = "http://localhost:8000";

const EXAMPLES = [
    "What is retrieval-augmented generation?",
    "How does the transformer architecture work?",
    "What are the key differences between BERT and GPT?",
];

export default function ChatInterface() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    async function handleSubmit(q = input) {
        const question = q.trim();
        if (!question || loading) return;

        const newUserMsg = { role: "user", content: question };

        // Build history for backend from settled (non-streaming) messages
        const historyForBackend = messages
            .filter(m => m.settled)
            .map(m => ({
                role: m.role,
                content: m.role === "assistant" ? (m.raw?.answer || m.streamText || "") : m.content,
            }));

        setMessages(prev => [...prev, newUserMsg]);
        setInput("");
        setLoading(true);

        // Add a placeholder streaming assistant message
        const assistantPlaceholder = {
            role: "assistant",
            streaming: true,
            settled: false,
            streamText: "",
            traceSteps: [],
            rewrittenQuery: null,
            raw: null,
        };
        setMessages(prev => [...prev, assistantPlaceholder]);

        try {
            const res = await fetch(`${API}/api/query/`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question, history: historyForBackend }),
            });

            if (!res.ok) throw new Error(`Server error: ${res.status}`);

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop(); // keep incomplete last line

                for (const line of lines) {
                    if (!line.startsWith("data: ")) continue;
                    try {
                        const event = JSON.parse(line.slice(6));
                        handleStreamEvent(event);
                    } catch { /* skip malformed */ }
                }
            }
        } catch (e) {
            setMessages(prev => {
                const updated = [...prev];
                const lastIdx = updated.length - 1;
                updated[lastIdx] = { ...updated[lastIdx], streaming: false, settled: true, error: e.message };
                return updated;
            });
        } finally {
            setLoading(false);
        }
    }

    function handleStreamEvent(event) {
        setMessages(prev => {
            const updated = [...prev];
            const lastIdx = updated.length - 1;
            const last = { ...updated[lastIdx] };

            if (event.type === "trace") {
                last.traceSteps = [...(last.traceSteps || []), event.content];
            } else if (event.type === "rewrite") {
                last.rewrittenQuery = event.content;
            } else if (event.type === "clear_tokens") {
                last.streamText = "";
            } else if (event.type === "token") {
                last.streamText = (last.streamText || "") + event.content;
            } else if (event.type === "end") {
                last.streaming = false;
                last.settled = true;
                last.raw = event.result;
                last.streamText = event.result.answer;
            }

            updated[lastIdx] = last;
            return updated;
        });
    }

    return (
        <div className="chat-container">
            <div className="chat-messages">
                {messages.length === 0 && (
                    <div className="chat-empty">
                        <span className="chat-empty-icon">⬡</span>
                        <h2>Self-Correcting RAG Chat</h2>
                        <p className="muted">Ask any research question. The agent retrieves, evaluates, and verifies its own facts automatically.</p>
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
                            {m.role === "user"
                                ? <div className="user-text">{m.content}</div>
                                : m.error
                                    ? <div className="error-box">⚠ {m.error}</div>
                                    : <AssistantBubble msg={m} />
                            }
                        </div>
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>

            <div className="chat-input-wrapper">
                <div className="chat-input-area">
                    <textarea
                        className="chat-input"
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        placeholder="Message Agentic RAG…"
                        rows={1}
                        onKeyDown={e => {
                            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
                        }}
                    />
                    <button
                        className="chat-send-btn"
                        onClick={() => handleSubmit()}
                        disabled={loading || !input.trim()}
                    >
                        ↑
                    </button>
                </div>
            </div>
        </div>
    );
}

function AssistantBubble({ msg }) {
    const [showTrace, setShowTrace] = useState(false);

    const statusColor = {
        SUPPORTED: "#4ade80",
        PARTIALLY_SUPPORTED: "#facc15",
        NOT_SUPPORTED: "#f87171",
    };

    const status = msg.raw?.hallucination_status;
    const sources = msg.raw?.sources || [];
    const trace = msg.raw?.trace || msg.traceSteps || [];

    return (
        <div className="assistant-msg-inner">
            {/* Live trace steps while streaming */}
            {msg.streaming && msg.traceSteps?.length > 0 && (
                <div className="stream-trace-live">
                    {msg.traceSteps.map((s, i) => {
                        const colors = { RETRIEVE: "#60a5fa", EVALUATE: "#facc15", GENERATE: "#4ade80", HALLUCINATION: "#c084fc", REWRITE: "#fb923c", WEB: "#2dd4bf", DECISION: "#94a3b8" };
                        const tag = s.match(/\[([A-Z_]+)\]/)?.[1] || "INFO";
                        const color = Object.entries(colors).find(([k]) => tag.startsWith(k))?.[1] || "#94a3b8";
                        return (
                            <div key={i} className="trace-step mini-trace" style={{ opacity: 0.7 }}>
                                <span className="trace-tag" style={{ color, minWidth: '70px' }}>[{tag}]</span>
                                <span className="trace-msg">{s.replace(/\[[A-Z_]+\]\s*/, "")}</span>
                            </div>
                        );
                    })}
                </div>
            )}

            {/* Rewritten query info */}
            {msg.rewrittenQuery && (
                <div className="chat-rewritten">
                    <span className="muted">Searched for: </span>
                    <em>"{msg.rewrittenQuery}"</em>
                </div>
            )}

            {/* The main answer — streams word by word */}
            <div className="chat-answer">
                {msg.streamText || ""}
                {msg.streaming && <span className="cursor-blink">▋</span>}
            </div>

            {/* Meta bar shown after streaming finishes */}
            {!msg.streaming && msg.settled && (
                <div className="chat-meta">
                    <button className="toggle-trace-btn" onClick={() => setShowTrace(!showTrace)}>
                        {showTrace ? "Hide Agent Trace ↑" : "View Agent Trace ↓"}
                    </button>
                    {status && (
                        <span
                            className="status-badge"
                            style={{ background: statusColor[status] + "22", color: statusColor[status], border: `1px solid ${statusColor[status]}44` }}
                        >
                            {status}
                        </span>
                    )}
                </div>
            )}

            {/* Expandable trace panel */}
            {showTrace && (
                <div className="chat-trace-box">
                    {sources.length > 0 && (
                        <div className="sources-list chat-sources">
                            {sources.map((s, i) => (
                                <a key={i} href={s.source} target="_blank" rel="noreferrer" className="source-item mini">
                                    <span className="source-score">{(s.score * 100).toFixed(0)}%</span>
                                    <span className="source-title" style={{ fontSize: '0.8rem' }}>{s.title}</span>
                                </a>
                            ))}
                        </div>
                    )}
                    <div className="trace-steps chat-traces">
                        {trace.map((step, i) => {
                            const colors = { RETRIEVE: "#60a5fa", EVALUATE: "#facc15", GENERATE: "#4ade80", HALLUCINATION: "#c084fc", REWRITE: "#fb923c", WEB: "#2dd4bf", DECISION: "#94a3b8" };
                            const tag = step.match(/\[([A-Z_]+)\]/)?.[1] || "INFO";
                            const color = Object.entries(colors).find(([k]) => tag.startsWith(k))?.[1] || "#94a3b8";
                            return (
                                <div key={i} className="trace-step mini-trace">
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
