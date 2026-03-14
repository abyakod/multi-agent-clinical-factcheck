# 🤖 Multi-Agent Production Stack

**Memory · Fact-Check · LLM-as-Judge · Powered by Ollama (llama3)**

> A production-grade 5-layer agentic architecture demonstrated with real WHO clinical guidelines and drug interaction data, running fully offline.

---

## Why This Is Different: 5-Layer Architecture vs. Standard RAG

Most enterprise AI today is "Naive RAG." This demo shows how a production 5-layer stack fixes the trust gap.

| Feature | Standard RAG (Chatbot) | **Our 5-Layer Stack** |
|---|---|---|
| **Verification** | None. Takes the LLM at its word. | **Adversarial.** A second agent (Fact-Check) is built to prove the first one wrong. |
| **Trust** | "Black Box." It just gives an answer. | **Observable.** Google ADK-style logs show every agent's internal monologue live. |
| **Observability** | Guesswork. Was the answer "good"? | **Scored.** Automatically scored on Faithfulness and Relevance by a Judge agent. |
| **Knowledge** | Static context windows or simple file loads. | **Dynamic.** ChromaDB vector store ensures only relevant context reaches the agent. |
| **Memory** | Resets every message. | **Persistent.** Remembers facts and context across the entire session. |

---

## Demo Scenarios: The Production "Torture Test"

Use these presets to see the pipeline fight to ensure safety and quality in real-time.

| Preset | The Question | What to watch for |
|---|---|---|
| **📋 Metformin Dosage** | Standard clinical lookup. | Watch **Retrieval** search ChromaDB and **Judge** score it for high faithfulness. |
| **🚨 Ibuprofen + Warfarin** | Predetermined safety risk. | **The Fact-Check Layer.** Watch for a `CONTRADICTED` verdict based on clinical guidelines. |
| **❓ Melatonin (Unknown)** | Fact not in knowledge base. | **Hallucination Guard.** Watch the Fact-Checker flag this as `UNVERIFIABLE` rather than guessing. |
| **⚡ Enalapril + Potassium** | Complex interaction. | **Full Pipeline Flow.** Watch all agents (Router → Retrieval → Fact-Check → Judge) fire in sequence. |
| **🧠 Memory Recall** | Follow-up context. | **Memory Layer.** Ask about a drug, then ask "does it have contraindications?" without naming it. |

---

## Quick Start

### 1. Requirements
- **Ollama** installed and running
- **Model**: `llama3` pulled (`ollama pull llama3`)
- **Python**: 3.10+

### 2. Install
```bash
cd multiagent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run
```bash
python app.py
# Open http://localhost:7860
```

---

## Architecture Flow

The system uses a stateful Graph (LangGraph) to coordinate specialized agents:

1. **Router**: Classifies task type (knowledge lookup vs memory recall).
2. **Retrieval**: Searches ChromaDB for clinical context.
3. **Fact-Checker**: Adversarially verifies every claim in the answer.
4. **Judge**: Scores the response on Faithfulness, Relevance, and Completeness.
5. **Memory**: Updates session context with the verified result.

---

## Google ADK-Style UI

The dashboard is built using a **Google Agent Developer Kit (ADK)** pattern:
- **Chat Interface**: Left-hand side for clinical conversation.
- **Event Stream**: Right-hand side showing real-time timestamps and emoji-prefixed agent logs.
- **Agent Cards**: Top row showing execution status (⏳ Waiting → 🔄 Running → ✅ Done).
- **Deep-Dive Tabs**: Fact-check reports, memory state, and quality score history.

---

## Project Structure

```
multiagent/
├── app.py                        ← Gradio UI (Google ADK style)
├── graph/
│   ├── state.py                  ← LangGraph State definition
│   ├── orchestrator.py           ← Logic & conditional routing
│   └── event_formatter.py        ← ADK-style log formatter
├── agents/
│   ├── retrieval_agent.py        ← Ollama retrieval specialist
│   ├── factcheck_agent.py        ← Adversarial fact-checker
│   └── judge_agent.py            ← LLM-as-Judge evaluator
├── memory/
│   └── memory_manager.py         ← Short + long term memory
├── tools/
│   └── knowledge_base_tool.py    ← ChromaDB vector store manager
├── knowledge_base/               ← Text-based clinical documents
├── requirements.txt
└── README.md
```

---

## License

MIT
