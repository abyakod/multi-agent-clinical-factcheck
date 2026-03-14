# Vibe Coding Prompt — Multi-Agent Production Stack Demo
# For: Claude Code / Cursor / Windsurf / Any Agentic IDE
# Purpose: Blog video demonstration — paste this as your planning prompt

---

## ROLE

You are a senior AI systems engineer building a production-grade
multi-agent demonstration application. This is not a toy. Every
component must reflect real enterprise patterns — clean code,
proper separation of concerns, and a UI that communicates
system behaviour clearly to a non-technical observer watching
a screen recording.

---

## THE VIBE

**Feel:** Google ADK-style mission control. Dark theme. Every internal
agent event is visible as it happens — router decisions, agent calls,
tool invocations, MCP results, judge scores. After execution, a live
flow diagram renders the exact path the system took. This is what
makes the video compelling: the audience watches the system think.

**Purpose:** Demonstrate the full 5-layer agentic production stack:
  1. LangGraph Orchestrator — task routing, state, retries
  2. Multi-Agent Crews — retrieval, fact-check, judge specialists
  3. Persistent Memory — short-term + long-term
  4. MCP-style Tool Registry — standardised tool connectivity
  5. LLM-as-a-Judge — continuous quality evaluation

**Model:** Use `claude-sonnet-4-6` via the Anthropic SDK for ALL
agent calls. Each agent is a separate Claude call with a different
system prompt — same model family, different specialist personality.

**Key UI requirement:** Two panels that mirror Google ADK exactly:

  Panel A — LIVE EVENT STREAM (left side, updates in real time)
  Shows every internal event as it fires, formatted like:
  ```
  [00:00.1] 🔀 ROUTER      → Classifying task type...
  [00:00.4] 🔀 ROUTER      → Task: knowledge_lookup
  [00:00.5] 🧠 MEMORY      → Loading session context...
  [00:00.7] 🧠 MEMORY      → Found 2 previous exchanges
  [00:01.1] 🤖 RETRIEVAL   → Agent firing (claude-sonnet-4-6)
  [00:01.2] 🔧 TOOL        → knowledge_base.search called
  [00:01.8] 🔧 TOOL        → Returned 3 document excerpts
  [00:02.1] 🤖 RETRIEVAL   → Confidence: HIGH
  [00:02.3] 🔎 FACTCHECK   → Adversarial agent firing...
  [00:02.4] 🔎 FACTCHECK   → Claim 1: SUPPORTED ✅
  [00:02.6] 🔎 FACTCHECK   → Claim 2: CONTRADICTED 🚨 HIGH
  [00:02.8] 🔎 FACTCHECK   → Overall verdict: FAIL
  [00:03.0] ⚖️  JUDGE       → Scoring faithfulness...
  [00:03.2] ⚖️  JUDGE       → Faithfulness: 0.82
  [00:03.3] ⚖️  JUDGE       → Relevance: 0.91
  [00:03.5] ⚖️  JUDGE       → Completeness: 0.78
  [00:03.6] ⚖️  JUDGE       → Recommendation: APPROVE ✅
  [00:03.7] 💾 MEMORY      → Short-term updated
  [00:03.8] ✅ PIPELINE    → Complete
  ```

  Panel B — FLOW DIAGRAM TAB (appears after execution completes)
  Uses LangGraph's built-in `get_graph().draw_mermaid()` to render
  the actual execution path as a Mermaid diagram in Gradio.
  Nodes that were visited are highlighted. Nodes that were skipped
  are dimmed. Shows the exact route the system took for THIS query.

**Use LangGraph's native `astream_events()` for the event stream.**
This is the built-in LangGraph streaming API — it emits granular
events for every node entry, exit, LLM call and tool call as they
happen. Wire these directly to the Gradio event stream panel.

---

## DEMO SCENARIO

**Scenario: Clinical Drug Safety Assistant**

Why this scenario wins on video:
- Wrong drug interaction answers have obvious, visceral stakes
- Universally relatable — every enterprise audience understands
- Rich enough for complex multi-document retrieval
- Fact-Checker contradiction moment is unmissable on screen
- Memory layer earns its keep with natural clinical follow-ups

Knowledge base has three real public-domain documents:
  - `who_essential_medicines.txt`  — WHO 23rd Essential Medicines List
  - `drug_interactions.txt`        — DrugBank/DDInter open data
  - `clinical_guidelines.txt`      — WHO/CDC/NHS treatment protocols

Four preset questions — each engineered to test one layer dramatically:

  Q1 — Tests Retrieval + Memory:
  First ask: "What is the recommended dosage for metformin
  in Type 2 diabetes and what are its contraindications?"
  Follow-up: "What did you just tell me about contraindications?"
  → Second answer comes from short-term memory — no retrieval.
  Stream shows: [MEMORY] → context found, skipping retrieval
  Flow diagram: retrieval node dimmed on follow-up query

  Q2 — Tests Fact-Check CONTRADICTION (the money shot):
  Inject deliberately wrong clinical claim:
  "Ibuprofen is a safe and recommended analgesic for patients
  who are currently taking warfarin for atrial fibrillation."
  → Fact-Checker fires immediately:
    CLAIM: Ibuprofen is safe with warfarin
    VERDICT: CONTRADICTED
    EVIDENCE: [direct WHO guideline quote — fatal bleeding risk]
    SEVERITY: HIGH
    OVERALL: FAIL 🚨
  Stream: 🔎 FACTCHECK → 🚨 FAIL | ✅0 🚨1 ❓0
  This is the moment that makes every viewer understand
  why this architecture matters.

  Q3 — Tests Fact-Check UNVERIFIABLE (honesty moment):
  "What is the recommended daily dose of melatonin for
  managing jet lag in long-haul flight crew?"
  → Not in any of the three documents
  Fact-Checker: UNVERIFIABLE — not in knowledge base
  System says "I don't know" instead of hallucinating
  This demonstrates: honesty is an architectural property,
  not a model property.

  Q4 — Full Pipeline End to End (closing impressive moment):
  "What are the risks of prescribing enalapril to a patient
  who is already taking potassium supplements daily?"
  → Requires cross-document retrieval:
    who_essential_medicines.txt → enalapril entry
    drug_interactions.txt → ACEi + potassium interaction
    clinical_guidelines.txt → hypertension monitoring protocol
  → All 5 layers fire in sequence
  → Judge scores all above 0.80
  → APPROVE verdict
  → Flow diagram fully lit end to end
  → Best closing shot for the video

---

## TECHNICAL SPECIFICATION

### Project Structure

```
multi_agent_demo/
├── app.py                        ← Gradio UI (main entry)
├── graph/
│   ├── __init__.py
│   ├── state.py                  ← LangGraph state definition
│   ├── orchestrator.py           ← LangGraph graph + nodes
│   └── event_formatter.py        ← Format astream_events for display
├── agents/
│   ├── __init__.py
│   ├── retrieval_agent.py        ← Claude call #1 (retrieval)
│   ├── factcheck_agent.py        ← Claude call #2 (adversarial)
│   └── judge_agent.py            ← Claude call #3 (quality eval)
├── memory/
│   ├── __init__.py
│   └── memory_manager.py         ← Short + long term memory
├── tools/
│   ├── __init__.py
│   └── knowledge_base_tool.py    ← MCP-style tool abstraction
├── knowledge_base/
│   ├── who_essential_medicines.txt   ← WHO 23rd Essential Medicines List
│   ├── drug_interactions.txt         ← DrugBank/DDInter open data
│   └── clinical_guidelines.txt       ← WHO/CDC/NHS treatment protocols
├── requirements.txt
├── .env.example
└── README.md
```

---

### State Definition

```python
# graph/state.py
"""
LangGraph state object — carries all data through the pipeline.
Every node reads from and writes to this state.
This is what makes LangGraph stateful.
"""

from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Input
    question: str

    # Router output
    task_type: str                # knowledge_lookup | memory_response | unknown

    # Memory
    memory_context: str           # Injected by memory node
    has_prior_context: bool

    # Retrieval output
    retrieved_answer: str
    retrieval_confidence: str     # HIGH | MEDIUM | LOW
    source_context: str           # Raw knowledge base content

    # Fact-check output
    factcheck_report: str
    factcheck_verdict: str        # PASS | FAIL | PARTIAL
    supported_count: int
    contradicted_count: int
    unverifiable_count: int

    # Judge output
    faithfulness: float
    relevance: float
    completeness: float
    avg_score: float
    recommendation: str           # APPROVE | FLAG | REJECT

    # Pipeline tracking
    events: List[str]             # Accumulated event log
    final_answer: str
    pipeline_complete: bool
    error: Optional[str]
```

---

### LangGraph Orchestrator with astream_events

```python
# graph/orchestrator.py
"""
Core LangGraph graph definition.

KEY IMPLEMENTATION NOTE:
Use graph.astream_events(input, version="v2") in app.py to get
granular real-time events from every node. This is LangGraph's
native equivalent of Google ADK's event stream.

Events emitted:
  - on_chain_start / on_chain_end  → node entry and exit
  - on_chat_model_start / stream   → LLM call start and token stream
  - on_tool_start / on_tool_end    → tool call entry and result
"""

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from anthropic import Anthropic
import time
from graph.state import AgentState
from agents.retrieval_agent import run_retrieval_agent
from agents.factcheck_agent import run_factcheck_agent
from agents.judge_agent import run_judge_agent
from memory.memory_manager import AgentMemoryManager
from tools.knowledge_base_tool import load_knowledge_base

client = Anthropic()
memory = AgentMemoryManager()

# ── Node: Router ──────────────────────────────────────────────────
def router_node(state: AgentState) -> AgentState:
    """
    Classifies the incoming question.
    Decides: use memory only, or do full retrieval?
    This is where LangGraph's conditional routing begins.
    """
    question = state["question"]
    memory_context = memory.get_context()

    # Check if this is a memory-only request
    memory_keywords = ["summarise", "summarize", "what did you say",
                       "what did i ask", "repeat", "again", "recall"]

    is_memory_request = any(kw in question.lower() for kw in memory_keywords)

    if is_memory_request and memory_context:
        task_type = "memory_response"
    else:
        task_type = "knowledge_lookup"

    return {
        **state,
        "task_type": task_type,
        "memory_context": memory_context,
        "has_prior_context": bool(memory_context),
        "events": state.get("events", []) + [
            f"ROUTER → Task classified: {task_type}"
        ]
    }

def route_decision(state: AgentState) -> str:
    """Conditional edge — determines which node fires next."""
    return state["task_type"]  # "memory_response" or "knowledge_lookup"


# ── Node: Memory Response (skip retrieval) ────────────────────────
def memory_response_node(state: AgentState) -> AgentState:
    """
    Answers from memory only — no retrieval needed.
    Shows memory layer working without touching knowledge base.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=(
            "You are a helpful assistant. Answer the user's question "
            "using ONLY the conversation history provided. "
            "Do not invent any new information."
        ),
        messages=[{
            "role": "user",
            "content": f"Conversation history:\n{state['memory_context']}\n\nQuestion: {state['question']}"
        }]
    )
    answer = response.content[0].text

    return {
        **state,
        "retrieved_answer": answer,
        "retrieval_confidence": "HIGH",
        "source_context": state["memory_context"],
        "events": state["events"] + [
            "MEMORY → Answered from session context (no retrieval needed)"
        ]
    }


# ── Node: Retrieval ───────────────────────────────────────────────
def retrieval_node(state: AgentState) -> AgentState:
    """
    Calls the retrieval specialist agent.
    Reads knowledge base documents, returns grounded excerpts.
    """
    knowledge_base = load_knowledge_base()
    result = run_retrieval_agent(state["question"], state["memory_context"])

    return {
        **state,
        "retrieved_answer": result["response"],
        "retrieval_confidence": result["confidence"],
        "source_context": knowledge_base,
        "events": state["events"] + [
            f"RETRIEVAL → Agent fired (claude-sonnet-4-6)",
            f"RETRIEVAL → Confidence: {result['confidence']}",
            f"RETRIEVAL → Tokens: {result['tokens_used']}"
        ]
    }


# ── Node: Fact-Checker ────────────────────────────────────────────
def factcheck_node(state: AgentState) -> AgentState:
    """
    Adversarial fact-checking agent.
    Every claim in the retrieved answer is verified against source docs.
    CONTRADICTED at HIGH severity → pipeline flags the response.
    """
    result = run_factcheck_agent(
        state["question"],
        state["retrieved_answer"],
        state["source_context"]
    )

    return {
        **state,
        "factcheck_report": result["response"],
        "factcheck_verdict": result["overall_verdict"],
        "supported_count": result["supported_count"],
        "contradicted_count": result["contradicted_count"],
        "unverifiable_count": result["unverifiable_count"],
        "events": state["events"] + [
            f"FACTCHECK → Adversarial agent fired (claude-sonnet-4-6)",
            f"FACTCHECK → Supported: {result['supported_count']} | "
            f"Contradicted: {result['contradicted_count']} | "
            f"Unverifiable: {result['unverifiable_count']}",
            f"FACTCHECK → Overall verdict: {result['overall_verdict']}"
        ]
    }


# ── Node: Judge ───────────────────────────────────────────────────
def judge_node(state: AgentState) -> AgentState:
    """
    LLM-as-a-Judge quality evaluation.
    Scores faithfulness, relevance, completeness.
    Emits structured scores for Gradio dashboard gauges.
    """
    result = run_judge_agent(
        state["question"],
        state["source_context"],
        state["retrieved_answer"]
    )
    scores = result["scores"]

    return {
        **state,
        "faithfulness": scores["faithfulness"],
        "relevance": scores["relevance"],
        "completeness": scores["completeness"],
        "avg_score": scores["average"],
        "recommendation": scores["recommendation"],
        "events": state["events"] + [
            f"JUDGE → Agent fired (claude-sonnet-4-6)",
            f"JUDGE → Faithfulness:  {scores['faithfulness']:.2f}",
            f"JUDGE → Relevance:     {scores['relevance']:.2f}",
            f"JUDGE → Completeness:  {scores['completeness']:.2f}",
            f"JUDGE → Average:       {scores['average']:.2f}",
            f"JUDGE → Recommendation: {scores['recommendation']}"
        ]
    }


# ── Node: Memory Update ───────────────────────────────────────────
def memory_update_node(state: AgentState) -> AgentState:
    """
    Updates short-term and long-term memory after pipeline completes.
    Only stores to long-term if judge score is above threshold.
    """
    store_long_term = state.get("avg_score", 0) > 0.7

    memory.after_response(
        question=state["question"],
        answer=state["retrieved_answer"],
        store_long_term=store_long_term
    )

    return {
        **state,
        "pipeline_complete": True,
        "final_answer": state["retrieved_answer"],
        "events": state["events"] + [
            f"MEMORY → Short-term updated",
            f"MEMORY → Long-term: {'stored' if store_long_term else 'skipped (score below threshold)'}",
            "PIPELINE → Complete ✅"
        ]
    }


# ── Build the Graph ───────────────────────────────────────────────
def build_graph():
    """
    Constructs the LangGraph state machine.

    Graph structure:
      router
        ├── knowledge_lookup → retrieval → factcheck → judge → memory_update
        └── memory_response  → ──────────────────────────────→ memory_update

    After building, call:
      graph.get_graph().draw_mermaid()   → Mermaid diagram string
      graph.get_graph().draw_ascii()     → ASCII fallback

    For streaming events during execution:
      async for event in graph.astream_events(input, version="v2"):
          event["event"]  → "on_chain_start", "on_chain_end", etc.
          event["name"]   → node name
          event["data"]   → event payload
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("memory_response", memory_response_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("factcheck", factcheck_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("memory_update", memory_update_node)

    # Entry point
    workflow.set_entry_point("router")

    # Conditional routing from router
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "knowledge_lookup": "retrieval",
            "memory_response": "memory_response"
        }
    )

    # Linear flow after retrieval
    workflow.add_edge("retrieval", "factcheck")
    workflow.add_edge("factcheck", "judge")
    workflow.add_edge("judge", "memory_update")

    # Memory-only path also ends at memory_update
    workflow.add_edge("memory_response", "memory_update")

    # Terminal
    workflow.add_edge("memory_update", END)

    return workflow.compile()


# Export compiled graph
pipeline = build_graph()
```

---

### Event Formatter (Google ADK Style)

```python
# graph/event_formatter.py
"""
Formats LangGraph astream_events into Google ADK-style
human-readable event log lines.

LangGraph event types we care about:
  on_chain_start   → node started executing
  on_chain_end     → node finished
  on_chat_model_start → LLM call began
  on_chat_model_stream → token streaming (we show a typing indicator)
  on_tool_start    → tool was called
  on_tool_end      → tool returned result
"""

import time
from datetime import datetime

# Map LangGraph node names to display labels + emoji
NODE_DISPLAY = {
    "router":          ("🔀", "ROUTER     "),
    "memory_response": ("🧠", "MEMORY     "),
    "retrieval":       ("🤖", "RETRIEVAL  "),
    "factcheck":       ("🔎", "FACTCHECK  "),
    "judge":           ("⚖️ ", "JUDGE      "),
    "memory_update":   ("💾", "MEMORY     "),
    "LangGraph":       ("🔗", "LANGGRAPH  "),
    "__start__":       ("▶️ ", "PIPELINE   "),
}

# Map event types to human readable descriptions
def format_event(event: dict, start_time: float) -> str | None:
    """
    Takes a raw LangGraph astream_event and returns a
    formatted display string, or None if the event should be skipped.

    Args:
        event: Raw event dict from astream_events()
        start_time: Pipeline start time (for elapsed time display)

    Returns:
        Formatted string like:
        [00:01.3] 🤖 RETRIEVAL   → Agent firing (claude-sonnet-4-6)
    """
    event_type = event.get("event", "")
    node_name = event.get("name", "")
    elapsed = time.time() - start_time
    timestamp = f"[{elapsed:05.2f}s]"

    emoji, label = NODE_DISPLAY.get(node_name, ("⚙️ ", f"{node_name:<11}"))

    # Skip internal LangGraph housekeeping events
    skip_nodes = {"__start__", "LangGraph", ""}
    skip_events = {"on_chain_stream"}

    if node_name in skip_nodes:
        return None
    if event_type in skip_events:
        return None

    # Format based on event type
    if event_type == "on_chain_start":
        return f"{timestamp} {emoji} {label} → Starting..."

    elif event_type == "on_chain_end":
        output = event.get("data", {}).get("output", {})

        # Extract meaningful info from node output
        if node_name == "router":
            task = output.get("task_type", "unknown")
            has_mem = output.get("has_prior_context", False)
            return (
                f"{timestamp} {emoji} {label} → Task: {task} | "
                f"Memory: {'found' if has_mem else 'empty'}"
            )

        elif node_name == "retrieval":
            conf = output.get("retrieval_confidence", "?")
            return f"{timestamp} {emoji} {label} → Complete | Confidence: {conf}"

        elif node_name == "factcheck":
            verdict = output.get("factcheck_verdict", "?")
            s = output.get("supported_count", 0)
            c = output.get("contradicted_count", 0)
            u = output.get("unverifiable_count", 0)
            verdict_icon = {"PASS": "✅", "FAIL": "🚨", "PARTIAL": "⚠️"}.get(verdict, "❓")
            return (
                f"{timestamp} {emoji} {label} → {verdict_icon} {verdict} | "
                f"✅{s} 🚨{c} ❓{u}"
            )

        elif node_name == "judge":
            rec = output.get("recommendation", "?")
            avg = output.get("avg_score", 0)
            f_score = output.get("faithfulness", 0)
            r_score = output.get("relevance", 0)
            c_score = output.get("completeness", 0)
            rec_icon = {"APPROVE": "✅", "FLAG": "⚠️", "REJECT": "🚨"}.get(rec, "❓")
            return (
                f"{timestamp} {emoji} {label} → {rec_icon} {rec} | "
                f"F:{f_score:.2f} R:{r_score:.2f} C:{c_score:.2f} Avg:{avg:.2f}"
            )

        elif node_name == "memory_update":
            return f"{timestamp} {emoji} {label} → Session updated ✅"

        elif node_name == "memory_response":
            return f"{timestamp} {emoji} {label} → Answered from memory (no retrieval)"

        return f"{timestamp} {emoji} {label} → Done"

    elif event_type == "on_chat_model_start":
        model = event.get("data", {}).get("serialized", {}).get("id", [""])[−1]
        return f"{timestamp} {emoji} {label} → Calling claude-sonnet-4-6..."

    elif event_type == "on_tool_start":
        tool = event.get("name", "unknown_tool")
        return f"{timestamp} 🔧 TOOL       → {tool} called"

    elif event_type == "on_tool_end":
        return f"{timestamp} 🔧 TOOL       → Result returned"

    return None


def get_pipeline_complete_event(start_time: float) -> str:
    elapsed = time.time() - start_time
    return f"[{elapsed:05.2f}s] ✅ PIPELINE   → Execution complete"
```

---

### Gradio UI — Full Application with ADK-Style Event Stream

```python
# app.py
"""
Multi-Agent Production Stack — Gradio Demo
Google ADK-style live event stream + Mermaid flow diagram.

UI Layout:
  ┌─────────────────────────────────────────────────┐
  │  Question Input + Preset Selector + Run Button  │
  ├──────────────────────┬──────────────────────────┤
  │  LIVE EVENT STREAM   │   AGENT OUTPUT           │
  │  (updates in real    │   Retrieval answer        │
  │   time as pipeline   │   Fact-check report       │
  │   executes)          │   Judge scores (gauges)   │
  ├──────────────────────┴──────────────────────────┤
  │  TABS:                                          │
  │  [Flow Diagram] [Memory View] [Score History]   │
  └─────────────────────────────────────────────────┘

The Flow Diagram tab renders the Mermaid diagram from
LangGraph's get_graph().draw_mermaid() — showing the exact
execution path for the last query.
"""

import gradio as gr
import asyncio
import time
from graph.orchestrator import pipeline, memory
from graph.event_formatter import format_event, get_pipeline_complete_event

# ── Preset questions ──────────────────────────────────────────────
PRESETS = {
    "🧠 Test Memory — Follow-up Recall": "Summarise what you just told me",
    "📋 Test Retrieval — Policy Lookup": "What is the remote work policy for employees?",
    "🚨 Test Fact-Check — Contradiction": "Employees can work remotely 7 days a week with no office days required",
    "❓ Test Fact-Check — Unknown Info": "What is the monthly gym reimbursement limit?",
    "⚡ Full Pipeline — End to End": "What are the data security requirements for remote employees?",
}

score_history = []

# ── Mermaid diagram renderer ──────────────────────────────────────
def get_mermaid_html(highlight_nodes: list = None) -> str:
    """
    Gets the LangGraph Mermaid diagram and renders it as HTML.
    Uses mermaid.js CDN for rendering in Gradio HTML component.

    LangGraph's draw_mermaid() returns a Mermaid string like:
      graph TD
        router([router]) --> retrieval
        router --> memory_response
        retrieval --> factcheck
        ...

    We inject this into a <div class="mermaid"> block and
    load mermaid.js to render it client-side.
    """
    try:
        mermaid_str = pipeline.get_graph().draw_mermaid()
    except Exception:
        mermaid_str = """graph TD
    router([🔀 Router]) --> retrieval([🤖 Retrieval])
    router --> memory_response([🧠 Memory Response])
    retrieval --> factcheck([🔎 Fact-Checker])
    factcheck --> judge([⚖️ Judge])
    judge --> memory_update([💾 Memory Update])
    memory_response --> memory_update
    memory_update --> END([✅ End])"""

    # Add styling for visited nodes if provided
    style_lines = ""
    if highlight_nodes:
        node_styles = {
            "router": "fill:#1e40af,color:#fff",
            "retrieval": "fill:#065f46,color:#fff",
            "memory_response": "fill:#065f46,color:#fff",
            "factcheck": "fill:#7c2d12,color:#fff",
            "judge": "fill:#4c1d95,color:#fff",
            "memory_update": "fill:#134e4a,color:#fff",
        }
        for node in highlight_nodes:
            if node in node_styles:
                style_lines += f"\nstyle {node} {node_styles[node]}"

    mermaid_with_styles = mermaid_str + style_lines

    return f"""
    <div style="background:#0f172a;padding:20px;border-radius:8px;min-height:300px">
      <div class="mermaid" style="background:transparent">
        {mermaid_with_styles}
      </div>
    </div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{
        startOnLoad: true,
        theme: 'dark',
        themeVariables: {{
          primaryColor: '#1e293b',
          primaryTextColor: '#e2e8f0',
          primaryBorderColor: '#334155',
          lineColor: '#64748b',
          background: '#0f172a'
        }}
      }});
    </script>
    """


# ── Core pipeline runner with streaming ──────────────────────────
def run_pipeline_streaming(question: str, preset_choice: str):
    """
    Runs the LangGraph pipeline and streams events to Gradio.

    Uses pipeline.astream_events() to get real-time events.
    Yields tuples that update all Gradio components simultaneously.

    Yields: (
        event_stream_text,    ← running log of all events
        retrieval_output,     ← retrieved answer text
        factcheck_output,     ← fact-check report text
        faithfulness_val,     ← slider 0-1
        relevance_val,        ← slider 0-1
        completeness_val,     ← slider 0-1
        verdict_text,         ← final recommendation
        mermaid_html,         ← flow diagram HTML (shown after completion)
        memory_display,       ← short-term memory text
    )
    """
    # Use preset if question is empty
    if not question.strip() and preset_choice in PRESETS:
        question = PRESETS[preset_choice]
    if not question.strip():
        yield ("Please enter a question.", "", "", 0, 0, 0, "—", "", "")
        return

    start_time = time.time()
    event_lines = ["─" * 55, f"▶  Pipeline started: {question[:60]}...", "─" * 55]
    visited_nodes = []

    # Initial state
    initial_state = {
        "question": question,
        "task_type": "",
        "memory_context": "",
        "has_prior_context": False,
        "retrieved_answer": "",
        "retrieval_confidence": "",
        "source_context": "",
        "factcheck_report": "",
        "factcheck_verdict": "",
        "supported_count": 0,
        "contradicted_count": 0,
        "unverifiable_count": 0,
        "faithfulness": 0.0,
        "relevance": 0.0,
        "completeness": 0.0,
        "avg_score": 0.0,
        "recommendation": "—",
        "events": [],
        "final_answer": "",
        "pipeline_complete": False,
        "error": None
    }

    # Tracking state as it builds
    current_state = {}

    # Run astream_events — this is LangGraph's native event stream API
    # version="v2" gives the most granular events
    async def collect_events():
        async for event in pipeline.astream_events(initial_state, version="v2"):
            yield event

    def run_async_generator():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def process():
            async for event in pipeline.astream_events(initial_state, version="v2"):
                event_type = event.get("event", "")
                node_name = event.get("name", "")

                # Track visited nodes for flow diagram highlighting
                if event_type == "on_chain_start" and node_name not in (
                    "__start__", "LangGraph", ""
                ):
                    if node_name not in visited_nodes:
                        visited_nodes.append(node_name)

                # Capture final state from memory_update completion
                if event_type == "on_chain_end" and node_name == "memory_update":
                    output = event.get("data", {}).get("output", {})
                    current_state.update(output)

                # Format and add to display
                formatted = format_event(event, start_time)
                if formatted:
                    event_lines.append(formatted)
                    yield event_lines.copy(), current_state.copy(), visited_nodes.copy()

        async def run_and_collect():
            results = []
            async for item in process():
                results.append(item)
            return results

        return loop.run_until_complete(run_and_collect())

    # Stream events to Gradio
    for events_snapshot, state_snapshot, nodes_snapshot in run_async_generator():
        stream_text = "\n".join(events_snapshot)

        yield (
            stream_text,
            state_snapshot.get("retrieved_answer", ""),
            state_snapshot.get("factcheck_report", ""),
            state_snapshot.get("faithfulness", 0.0),
            state_snapshot.get("relevance", 0.0),
            state_snapshot.get("completeness", 0.0),
            state_snapshot.get("recommendation", "—"),
            "",   # Flow diagram not shown yet during execution
            memory.short_term.get_display()
        )

    # Pipeline complete — add completion event and render flow diagram
    event_lines.append(get_pipeline_complete_event(start_time))
    event_lines.append("─" * 55)
    final_stream = "\n".join(event_lines)

    # Update score history
    if current_state.get("avg_score"):
        score_history.append({
            "question": question[:50],
            "faithfulness": current_state.get("faithfulness", 0),
            "relevance": current_state.get("relevance", 0),
            "completeness": current_state.get("completeness", 0),
            "avg": current_state.get("avg_score", 0),
            "verdict": current_state.get("factcheck_verdict", "—"),
            "recommendation": current_state.get("recommendation", "—")
        })

    # Render Mermaid flow diagram with visited nodes highlighted
    mermaid_html = get_mermaid_html(highlight_nodes=visited_nodes)

    yield (
        final_stream,
        current_state.get("retrieved_answer", ""),
        current_state.get("factcheck_report", "No fact-check run"),
        current_state.get("faithfulness", 0.0),
        current_state.get("relevance", 0.0),
        current_state.get("completeness", 0.0),
        current_state.get("recommendation", "—"),
        mermaid_html,
        memory.short_term.get_display()
    )


# ── Gradio UI ─────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("IBM Plex Mono")
        ),
        css="""
        body { background: #0f172a; }
        .event-stream textarea {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 12px !important;
            background: #020617 !important;
            color: #94a3b8 !important;
            border: 1px solid #1e293b !important;
        }
        .verdict-box textarea {
            font-size: 1.2em !important;
            font-weight: bold !important;
            text-align: center !important;
        }
        .section-label {
            font-size: 0.7em;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 4px;
        }
        footer { display: none !important; }
        """,
        title="Multi-Agent Production Stack"
    ) as demo:

        gr.Markdown("""
# 🤖 Multi-Agent Production Stack
**Memory · Fact-Check · LLM-as-a-Judge · Powered by `claude-sonnet-4-6`**

> From the article: *Multi-Agents That Remember, Factcheck + LLM-as-Judge*
— every internal event visible in real time, flow diagram renders after execution
        """)

        # ── INPUT ROW ─────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                preset_dd = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    label="🎯 Load Demo Scenario",
                    value=None,
                    info="Each preset tests a specific layer"
                )
                question_box = gr.Textbox(
                    label="Question",
                    placeholder="Ask anything about company policies...",
                    lines=2
                )
            with gr.Column(scale=1, min_width=160):
                run_btn = gr.Button(
                    "▶  Run Pipeline",
                    variant="primary",
                    size="lg",
                    elem_id="run-btn"
                )
                clear_btn = gr.Button("🗑 Clear Memory", variant="stop", size="sm")

        # ── MAIN PANEL ROW ────────────────────────────────────────
        with gr.Row():

            # LEFT: Live Event Stream (Google ADK style)
            with gr.Column(scale=1):
                gr.HTML("<div class='section-label'>⚡ Live Event Stream</div>")
                event_stream = gr.Textbox(
                    label="",
                    lines=22,
                    max_lines=22,
                    interactive=False,
                    placeholder=(
                        "Events will appear here as the pipeline executes...\n\n"
                        "You will see:\n"
                        "  → Router classifying task\n"
                        "  → Memory context loading\n"
                        "  → Retrieval agent firing\n"
                        "  → Fact-checker verdicts\n"
                        "  → Judge scores\n"
                        "  → Memory update"
                    ),
                    elem_classes=["event-stream"]
                )

            # RIGHT: Agent Outputs + Scores
            with gr.Column(scale=1):
                gr.HTML("<div class='section-label'>🤖 Agent Output</div>")
                retrieval_box = gr.Textbox(
                    label="Retrieval Agent",
                    lines=8,
                    interactive=False
                )

                gr.HTML("<div class='section-label'>⚖️ Judge Scores</div>")
                with gr.Row():
                    faith_slider = gr.Slider(
                        label="Faithfulness",
                        minimum=0, maximum=1, value=0,
                        interactive=False
                    )
                    rel_slider = gr.Slider(
                        label="Relevance",
                        minimum=0, maximum=1, value=0,
                        interactive=False
                    )
                    comp_slider = gr.Slider(
                        label="Completeness",
                        minimum=0, maximum=1, value=0,
                        interactive=False
                    )

                verdict_box = gr.Textbox(
                    label="Final Recommendation",
                    interactive=False,
                    elem_classes=["verdict-box"]
                )

        # ── BOTTOM TABS ───────────────────────────────────────────
        with gr.Tabs():

            # TAB 1: Flow Diagram (the main feature)
            with gr.TabItem("🗺️ Execution Flow Diagram"):
                gr.Markdown("""
**Rendered after each pipeline run.** Shows the exact path the system took.
Highlighted nodes = executed. Generated from LangGraph's native graph structure.
                """)
                flow_diagram = gr.HTML(
                    value="<div style='padding:40px;color:#475569;text-align:center'>"
                          "Run a question above — the execution flow diagram will appear here.</div>"
                )

            # TAB 2: Fact-Check Report
            with gr.TabItem("🔎 Fact-Check Report"):
                factcheck_box = gr.Textbox(
                    label="Adversarial Fact-Checker Output",
                    lines=15,
                    interactive=False
                )

            # TAB 3: Memory View
            with gr.TabItem("🧠 Memory State"):
                gr.Markdown("""
Watch memory populate in real time as you ask questions.
Short-term holds your session. Long-term persists across sessions.
                """)
                memory_display = gr.Textbox(
                    label="Short-Term Memory",
                    lines=12,
                    interactive=False,
                    value="No memory yet."
                )

            # TAB 4: Score History
            with gr.TabItem("📊 Score History"):
                gr.Markdown("Every query scored. Watch for trends.")
                score_display = gr.JSON(label="All Scores", value=[])
                refresh_btn = gr.Button("🔄 Refresh")
                refresh_btn.click(
                    fn=lambda: list(reversed(score_history)),
                    outputs=score_display
                )

        # ── Wire preset to question box ───────────────────────────
        def load_preset(choice):
            return PRESETS.get(choice, "")

        preset_dd.change(
            fn=load_preset,
            inputs=preset_dd,
            outputs=question_box
        )

        # ── Wire run button ───────────────────────────────────────
        run_btn.click(
            fn=run_pipeline_streaming,
            inputs=[question_box, preset_dd],
            outputs=[
                event_stream,       # Live event log
                retrieval_box,      # Retrieved answer
                factcheck_box,      # Fact-check report
                faith_slider,       # Faithfulness gauge
                rel_slider,         # Relevance gauge
                comp_slider,        # Completeness gauge
                verdict_box,        # Final recommendation
                flow_diagram,       # Mermaid flow diagram
                memory_display      # Memory state
            ]
        )

        # ── Wire clear memory ─────────────────────────────────────
        def clear_all():
            memory.short_term.clear()
            memory.long_term.clear()
            return "Memory cleared."

        clear_btn.click(fn=clear_all, outputs=memory_display)

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False
    )
```

---

### Requirements

```
anthropic>=0.40.0
langchain>=0.2.0
langchain-openai>=0.1.0
langgraph>=0.2.0
gradio>=4.40.0
python-dotenv>=1.0.0
```

---

### .env.example

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

---

### How to Run

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your Anthropic API key to .env
python app.py
# Open http://localhost:7860
```

---

## VIDEO RECORDING SCRIPT

```
00:00 — Open app. Orient viewer: left = live events, right = output,
         bottom tabs = flow diagram + memory + scores. (20 sec)

00:20 — Select "📋 Metformin dosage query". Run it.
         Watch event stream populate line by line:
         [ROUTER] → [MEMORY] → [RETRIEVAL] → [FACTCHECK] → [JUDGE]
         Point to each event as it fires. Scores appear right side.
         Click Flow Diagram tab — show highlighted execution path.
         Pause on the Mermaid diagram for 5 seconds.

01:15 — Ask follow-up: "What did you just tell me about contraindications?"
         Watch event stream: ROUTER → memory_response (skips retrieval).
         Flow diagram shows retrieval node DIMMED — not visited.
         Switch to Memory tab — show short-term context populated.
         Say: "The agent answered from memory. No retrieval needed."

02:00 — Select "🚨 Ibuprofen safe with warfarin?". Run it.
         Watch FACTCHECK line fire:
         🔎 FACTCHECK → 🚨 FAIL | ✅0 🚨1 ❓0
         Click Fact-Check Report tab. Show full report:
           CLAIM: Ibuprofen is safe and recommended with warfarin
           VERDICT: CONTRADICTED
           EVIDENCE: [WHO guideline — fatal haemorrhage documented]
           SEVERITY: HIGH
         Pause here — this is the money shot.
         Say: "This is a dangerous clinical hallucination.
         The Fact-Checker blocked it before it reached a user."

03:00 — Select "❓ Melatonin for jet lag?". Run it.
         Watch FACTCHECK: PARTIAL | ✅0 🚨0 ❓1
         Show UNVERIFIABLE verdict.
         Say: "The system says I don't know instead of hallucinating.
         Honesty is an architecture property, not a model property."

03:45 — Select "⚡ ACE inhibitor + potassium supplements". Run it.
         Watch all 5 agents fire in sequence. Full event stream.
         Point to cross-document retrieval in the log —
         agent pulling from who_essential_medicines AND drug_interactions.
         Judge scores all above 0.80. APPROVE verdict.
         Click Flow Diagram — every node highlighted end to end.
         Click Score History — all 4 queries tracked.

04:45 — End on GitHub link.
```

---

## CONSTRAINTS FOR THE IDE

1. Build in this exact order:
   - `graph/state.py` first — all other files depend on it
   - `agents/` layer second — test each agent independently
   - `memory/` third — test with simple in/out calls
   - `graph/orchestrator.py` fourth — wire agents into graph
   - `graph/event_formatter.py` fifth
   - `app.py` last — only after all layers tested

2. Test astream_events in isolation before wiring to Gradio:
   ```python
   async for event in pipeline.astream_events(state, version="v2"):
       print(event["event"], event["name"])
   ```

3. Use `temperature=0` for factcheck_agent and judge_agent
   Use `temperature=0.3` for retrieval_agent

4. The Mermaid diagram must load from mermaid.js CDN — do not
   attempt to install a Python mermaid package

5. Never hardcode ANTHROPIC_API_KEY — always read from environment

6. Gradio must run on port 7860 with `server_name="0.0.0.0"`

7. All Claude calls use model string: `claude-sonnet-4-6`
