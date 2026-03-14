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

Graph structure:
  router
    ├── knowledge_lookup → retrieval → factcheck → judge → memory_update
    └── memory_response  → ──────────────────────────────→ memory_update
"""

from langgraph.graph import StateGraph, END
from openai import OpenAI
from graph.state import AgentState
from agents.retrieval_agent import run_retrieval_agent
from agents.factcheck_agent import run_factcheck_agent
from agents.judge_agent import run_judge_agent
from memory.memory_manager import AgentMemoryManager
from tools.knowledge_base_tool import load_knowledge_base

# Ollama's OpenAI-compatible endpoint
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "llama3"
memory = AgentMemoryManager()


# ── Node: Router ──────────────────────────────────────────────────
def router_node(state: AgentState) -> dict:
    """
    Classifies the incoming question.
    Decides: use memory only, or do full retrieval?
    This is where LangGraph's conditional routing begins.
    """
    question = state["question"]
    memory_context = memory.get_context()

    # Check if this is a memory-only request
    memory_keywords = [
        "summarise", "summarize", "what did you say",
        "what did i ask", "repeat", "again", "recall",
        "what did you just", "tell me about what you said",
        "previous", "earlier", "before"
    ]

    is_memory_request = any(kw in question.lower() for kw in memory_keywords)

    if is_memory_request and memory_context:
        task_type = "memory_response"
    else:
        task_type = "knowledge_lookup"

    return {
        "task_type": task_type,
        "memory_context": memory_context,
        "has_prior_context": bool(memory_context),
        "events": state.get("events", []) + [
            f"ROUTER → Task classified: {task_type}",
            f"ROUTER → Memory: {'found prior context' if memory_context else 'empty'}"
        ]
    }


def route_decision(state: AgentState) -> str:
    """Conditional edge — determines which node fires next."""
    return state["task_type"]  # "memory_response" or "knowledge_lookup"


# ── Node: Memory Response (skip retrieval) ────────────────────────
def memory_response_node(state: AgentState) -> dict:
    """
    Answers from memory only — no retrieval needed.
    Shows memory layer working without touching knowledge base.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.3,
            messages=[
                {"role": "system", "content": (
                    "You are a helpful clinical assistant. Answer the user's question "
                    "using ONLY the conversation history provided. "
                    "Do not invent any new information. If the conversation history "
                    "doesn't contain what's needed, say so."
                )},
                {"role": "user", "content": (
                    f"Conversation history:\n{state['memory_context']}\n\n"
                    f"Question: {state['question']}"
                )}
            ]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Memory response error: {str(e)}"

    return {
        "retrieved_answer": answer,
        "retrieval_confidence": "HIGH",
        "source_context": state["memory_context"],
        "factcheck_verdict": "SKIPPED",
        "recommendation": "APPROVE",
        "events": state["events"] + [
            "MEMORY → Answered from session context (no retrieval needed)"
        ]
    }


# ── Node: Retrieval ───────────────────────────────────────────────
def retrieval_node(state: AgentState) -> dict:
    """
    Calls the retrieval specialist agent.
    Uses ChromaDB semantic search for relevant passages.
    """
    result = run_retrieval_agent(state["question"], state["memory_context"])

    return {
        "retrieved_answer": result["response"],
        "retrieval_confidence": result["confidence"],
        "source_context": result.get("source_context", ""),
        "events": state["events"] + [
            "RETRIEVAL → Agent fired (llama3)",
            f"RETRIEVAL → Confidence: {result['confidence']}",
            f"RETRIEVAL → Tokens used: {result['tokens_used']}"
        ]
    }


# ── Node: Fact-Checker ────────────────────────────────────────────
def factcheck_node(state: AgentState) -> dict:
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
        "factcheck_report": result["response"],
        "factcheck_verdict": result["overall_verdict"],
        "supported_count": result["supported_count"],
        "contradicted_count": result["contradicted_count"],
        "unverifiable_count": result["unverifiable_count"],
        "events": state["events"] + [
            "FACTCHECK → Adversarial agent fired (llama3)",
            f"FACTCHECK → Supported: {result['supported_count']} | "
            f"Contradicted: {result['contradicted_count']} | "
            f"Unverifiable: {result['unverifiable_count']}",
            f"FACTCHECK → Overall verdict: {result['overall_verdict']}"
        ]
    }


# ── Node: Judge ───────────────────────────────────────────────────
def judge_node(state: AgentState) -> dict:
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
        "faithfulness": scores["faithfulness"],
        "relevance": scores["relevance"],
        "completeness": scores["completeness"],
        "avg_score": scores["average"],
        "recommendation": scores["recommendation"],
        "events": state["events"] + [
            "JUDGE → Agent fired (llama3)",
            f"JUDGE → Faithfulness:  {scores['faithfulness']:.2f}",
            f"JUDGE → Relevance:     {scores['relevance']:.2f}",
            f"JUDGE → Completeness:  {scores['completeness']:.2f}",
            f"JUDGE → Average:       {scores['average']:.2f}",
            f"JUDGE → Recommendation: {scores['recommendation']}"
        ]
    }


# ── Node: Memory Update ───────────────────────────────────────────
def memory_update_node(state: AgentState) -> dict:
    """
    Updates short-term and long-term memory after pipeline completes.
    Only stores to long-term if judge score is above threshold.
    """
    store_long_term = state.get("avg_score", 0) > 0.7

    memory.after_response(
        question=state["question"],
        answer=state.get("retrieved_answer", ""),
        store_long_term=store_long_term
    )

    return {
        "pipeline_complete": True,
        "final_answer": state.get("retrieved_answer", ""),
        "events": state["events"] + [
            "MEMORY → Short-term updated",
            f"MEMORY → Long-term: {'stored' if store_long_term else 'skipped (score below threshold)'}",
            "PIPELINE → Complete ✅"
        ]
    }


# ── Build the Graph ───────────────────────────────────────────────
def build_graph():
    """
    Constructs the LangGraph state machine.

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
