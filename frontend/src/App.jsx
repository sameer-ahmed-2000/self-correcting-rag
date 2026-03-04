import { useState } from "react";
import "./App.css";
import ChatInterface from "./components/ChatInterface";
import IngestPanel from "./components/IngestPanel";

export default function App() {
  const [activeTab, setActiveTab] = useState("chat");

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">⬡</span>
            <span className="logo-text">Agentic RAG <span>Research Assistant</span></span>
          </div>
          <nav className="tabs">
            {["chat", "ingest"].map((tab) => (
              <button
                key={tab}
                className={`tab ${activeTab === tab ? "active" : ""}`}
                onClick={() => setActiveTab(tab)}
              >
                {tab === "chat" && "💬 Chat"}
                {tab === "ingest" && "📥 Ingest"}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="main" style={{ height: 'calc(100vh - 60px)', padding: '1rem', display: 'flex', flexDirection: 'column' }}>
        {activeTab === "chat" && <ChatInterface />}
        {activeTab === "ingest" && <IngestPanel />}
      </main>
    </div>
  );
}
