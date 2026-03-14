"""
Formats LangGraph astream_events into Google ADK-style
human-readable event log lines.

LangGraph event types we care about:
  on_chain_start   → node started executing
  on_chain_end     → node finished
  on_chat_model_start → LLM call began
  on_chat_model_stream → token streaming (show typing indicator)
  on_tool_start    → tool was called
  on_tool_end      → tool returned result
"""

import time

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


def format_event(event: dict, start_time: float) -> str | None:
    """
    Takes a raw LangGraph astream_event and returns a
    formatted display string, or None if the event should be skipped.

    Args:
        event: Raw event dict from astream_events()
        start_time: Pipeline start time (for elapsed time display)

    Returns:
        Formatted string like:
        [00:01.3] 🤖 RETRIEVAL   → Agent firing (llama3)
    """
    event_type = event.get("event", "")
    node_name = event.get("name", "")
    elapsed = time.time() - start_time
    timestamp = f"[{elapsed:05.1f}s]"

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
            verdict_icon = {
                "PASS": "✅", "FAIL": "🚨", "PARTIAL": "⚠️", "SKIPPED": "⏭️"
            }.get(verdict, "❓")
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
            rec_icon = {
                "APPROVE": "✅", "FLAG": "⚠️", "REJECT": "🚨"
            }.get(rec, "❓")
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
        return f"{timestamp} {emoji} {label} → Calling llama3..."

    elif event_type == "on_tool_start":
        tool = event.get("name", "unknown_tool")
        return f"{timestamp} 🔧 TOOL       → {tool} called"

    elif event_type == "on_tool_end":
        return f"{timestamp} 🔧 TOOL       → Result returned"

    return None


def get_pipeline_complete_event(start_time: float) -> str:
    """Returns the pipeline completion event line."""
    elapsed = time.time() - start_time
    return f"[{elapsed:05.1f}s] ✅ PIPELINE   → Execution complete"
