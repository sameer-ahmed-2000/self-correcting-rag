import { useState } from "react";
import QueryPanel from "./components/QueryPanel";
import IngestPanel from "./components/IngestPanel";
import TraceViewer from "./components/TraceViewer";
import "./App.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("query");
  const [lastResult, setLastResult] = useState(null);

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">⬡</span>
            <span className="logo-text">Agentic RAG <span>Research Assistant</span></span>
          </div>
          <nav className="tabs">
            {["query", "ingest", "trace"].map((tab) => (
              <button
                key={tab}
                className={`tab ${activeTab === tab ? "active" : ""}`}
                onClick={() => setActiveTab(tab)}
              >
                {tab === "query" && "🔍 Ask"}
                {tab === "ingest" && "📥 Ingest"}
                {tab === "trace" && "🔬 Trace"}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="main">
        {activeTab === "query" && (
          <QueryPanel onResult={(r) => { setLastResult(r); setActiveTab("trace"); }} />
        )}
        {activeTab === "ingest" && <IngestPanel />}
        {activeTab === "trace" && <TraceViewer result={lastResult} onBack={() => setActiveTab("query")} />}
      </main>
    </div>
  );
}
