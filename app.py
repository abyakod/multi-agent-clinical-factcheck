"""
Multi-Agent Production Stack — Google ADK-Style UI

Layout:
┌─────────────────────────────────────────────────────────────┐
│  🤖 Multi-Agent Production Stack       [Agent Status Cards] │
├───────────────────────────┬─────────────────────────────────┤
│                           │  ⚡ Events & Agent Activity     │
│   💬 Chat                 │  [real-time streaming events]   │
│                           │                                 │
│   User: question          │  🔀 Router → classified...      │
│   Agent: response         │  🤖 Retrieval → searching...    │
│                           │  🔎 Factcheck → verifying...    │
│   [presets] [input] [▶]   │  ⚖️  Judge → scoring...         │
├───────────────────────────┴─────────────────────────────────┤
│  Tabs: [Fact-Check] [Memory] [Flow Diagram] [Scores]        │
└─────────────────────────────────────────────────────────────┘
"""

import gradio as gr
import time
from graph.orchestrator import pipeline, memory
from graph.event_formatter import get_pipeline_complete_event
from tools.knowledge_base_tool import get_collection_stats

# Mermaid rendering script
MERMAID_JS = """
<script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({ startOnLoad: true, theme: 'dark' });
</script>
"""

# ── Preset scenarios ──────────────────────────────────────────────
PRESETS = {
    "📋 Metformin Dosage": (
        "What is the recommended dosage for metformin in Type 2 diabetes "
        "and what are its contraindications?"
    ),
    "🚨 Ibuprofen + Warfarin": (
        "Ibuprofen is a safe and recommended analgesic for patients "
        "who are currently taking warfarin for atrial fibrillation."
    ),
    "❓ Melatonin": (
        "What is the recommended daily dose of melatonin for "
        "managing jet lag in long-haul flight crew?"
    ),
    "💊 Vitamin Z": (
        "What is the standard dose of Vitamin Z for hypertension?"
    ),
    "⚡ Enalapril + Potassium": (
        "What are the risks of prescribing enalapril to a patient "
        "who is already taking potassium supplements daily?"
    ),
}

score_history = []


# ── Agent status card HTML ────────────────────────────────────────
def make_agent_cards(active_agents: dict = None) -> str:
    """Generate HTML cards showing which agents are active."""
    agents = {
        "router": ("🔀", "Router"),
        "retrieval": ("🤖", "Retrieval"),
        "factcheck": ("🔎", "Fact-Check"),
        "judge": ("⚖️", "Judge"),
        "memory_update": ("💾", "Memory"),
    }
    if active_agents is None:
        active_agents = {}

    cards_html = ""
    for key, (emoji, label) in agents.items():
        status = active_agents.get(key, "idle")
        if status == "done":
            bg = "#065f46"; border = "#10b981"; badge = "✅"
        elif status == "running":
            bg = "#1e40af"; border = "#3b82f6"; badge = "🔄"
        elif status == "error":
            bg = "#7c2d12"; border = "#f97316"; badge = "❌"
        else:
            bg = "#1e293b"; border = "#334155"; badge = "⏳"

        cards_html += f"""
        <div style="display:inline-block;padding:8px 14px;margin:3px;
                    background:{bg};border:1px solid {border};border-radius:8px;
                    font-size:0.75em;color:#e2e8f0;min-width:80px;text-align:center">
            <div style="font-size:1.2em">{emoji}</div>
            <div style="font-weight:600">{label}</div>
            <div style="font-size:0.85em;opacity:0.8">{badge}</div>
        </div>"""

    return f"""<div style="display:flex;justify-content:center;flex-wrap:wrap;
                gap:4px;padding:8px 0">{cards_html}</div>"""


# ── Mermaid Diagram Generator ─────────────────────────────────────
def generate_mermaid(visited_nodes: list = None) -> str:
    """Generate a Mermaid diagram string, highlighting visited nodes."""
    if visited_nodes is None:
        visited_nodes = []
    
    # Base graph
    mermaid_str = "graph TD\n"
    mermaid_str += "  Start(({{Start}})) --> Router{Router}\n"
    mermaid_str += "  Router -- knowledge_lookup --> Retrieval[[Retrieval]]\n"
    mermaid_str += "  Router -- memory_response --> MemoryResp[[Memory Response]]\n"
    mermaid_str += "  Retrieval --> FactCheck[[Fact-Check]]\n"
    mermaid_str += "  FactCheck --> Judge[[Judge]]\n"
    mermaid_str += "  Judge --> MemUpdate[[Memory Update]]\n"
    mermaid_str += "  MemoryResp --> MemUpdate\n"
    mermaid_str += "  MemUpdate --> End((End))\n"

    # Highlighting
    if visited_nodes:
        mermaid_str += "\n  %% Highlighting visited path\n"
        node_map = {
            "router": "Router",
            "retrieval": "Retrieval",
            "factcheck": "FactCheck",
            "judge": "Judge",
            "memory_response": "MemoryResp",
            "memory_update": "MemUpdate"
        }
        highlighted = [node_map[n] for n in visited_nodes if n in node_map]
        if highlighted:
            mermaid_str += f"  style {' style '.join(highlighted)} fill:#1e40af,stroke:#3b82f6,stroke-width:2px,color:#fff\n"

    return f'<div class="mermaid">{mermaid_str}</div>'

def format_chat_response(state: dict) -> str:
    """Format the pipeline result into a nice chat response."""
    answer = state.get("retrieved_answer", "No answer generated.")
    verdict = state.get("factcheck_verdict", "N/A")
    recommendation = state.get("recommendation", "N/A")
    avg_score = state.get("avg_score", 0)

    verdict_emoji = {"PASS": "✅", "FAIL": "🚨", "PARTIAL": "⚠️",
                     "SKIPPED": "⏭️"}.get(verdict, "❓")
    rec_emoji = {"APPROVE": "✅", "FLAG": "⚠️", "REJECT": "🚨"}.get(
        recommendation, "❓")

    response = f"{answer}\n\n"
    response += "---\n"
    response += f"**Fact-Check:** {verdict_emoji} {verdict}  |  "
    response += f"**Quality:** {rec_emoji} {recommendation} ({avg_score:.2f})"

    return response


# ── Pipeline runner ───────────────────────────────────────────────
def run_pipeline(question: str, chat_history: list):
    """
    Runs the full pipeline and yields incremental updates.
    Gradio generator function — yields (chat, events, cards, factcheck, memory)
    at each stage.
    """
    if not question.strip():
        yield chat_history, "", make_agent_cards(), "", memory.short_term.get_display()
        return

    # Add user message to chat
    chat_history = chat_history + [{"role": "user", "content": question}]

    start_time = time.time()
    event_lines = [
        f"{'═' * 50}",
        f"▶  Pipeline started",
        f"   Q: {question[:65]}{'...' if len(question) > 65 else ''}",
        f"{'═' * 50}",
        ""
    ]

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

    visited = []

    # Show "thinking" in chat + initial event
    chat_with_thinking = chat_history + [
        {"role": "assistant", "content": "🔄 *Processing through pipeline...*"}
    ]
    yield (chat_with_thinking, "\n".join(event_lines),
           make_agent_cards(), "", memory.short_term.get_display(), 
           generate_mermaid())

    # Run pipeline synchronously with node tracking
    try:
        # We use a custom stream to track nodes as they happen
        current_state = initial_state
        visited = []
        
        # Simulating sub-steps for UI 
        # In a real LangGraph setup we'd use graph.stream(..., stream_mode="values")
        # to see state updates per node.
        for result_state in pipeline.stream(initial_state, stream_mode="values"):
            # Identify which node just ran by comparing events
            new_events = result_state.get("events", [])
            if len(new_events) > len(current_state.get("events", [])):
                latest_ev = new_events[-1]
                if "ROUTER" in latest_ev: visited.append("router")
                elif "RETRIEVAL" in latest_ev: visited.append("retrieval")
                elif "FACTCHECK" in latest_ev: visited.append("factcheck")
                elif "JUDGE" in latest_ev: visited.append("judge")
                elif "MEMORY RESPONSE" in latest_ev: visited.append("memory_response")
                elif "MEMORY UPDATE" in latest_ev: visited.append("memory_update")

                # Update UI for this node
                elapsed = time.time() - start_time
                timestamp = f"[{elapsed:05.1f}s]"
                
                # Format event for display
                label_map = {
                    "router": ("🔀", "ROUTER     "),
                    "retrieval": ("🤖", "RETRIEVAL  "),
                    "factcheck": ("🔎", "FACTCHECK  "),
                    "judge": ("⚖️ ", "JUDGE      "),
                    "memory_response": ("🧠", "MEMORY     "),
                    "memory_update": ("💾", "MEMORY     ")
                }
                
                # Check which node was just added to visited
                last_node = visited[-1]
                emoji, label = label_map.get(last_node, ("⚙️ ", "STEP       "))
                event_lines.append(f"{timestamp} {emoji} {label} → Complete")
                
                # Update status dict for cards
                status_dict = {v: "done" for v in visited}
                status_dict[visited[-1]] = "running" # Mark latest as running for effect
                
                yield (
                    chat_with_thinking,
                    "\n".join(event_lines),
                    make_agent_cards(status_dict),
                    result_state.get("factcheck_report", ""),
                    memory.short_term.get_display(),
                    generate_mermaid(visited)
                )
            
            current_state = result_state

        # Final yield with complete formatting
        result = current_state
        event_lines.append("")
        event_lines.append(get_pipeline_complete_event(start_time))
        event_lines.append(f"{'═' * 50}")

        response = format_chat_response(result)
        chat_final = chat_history + [{"role": "assistant", "content": response}]

        # Update scores etc
        if result.get("avg_score"):
            score_history.append({
                "question": question[:50],
                "faithfulness": result.get("faithfulness", 0),
                "relevance": result.get("relevance", 0),
                "completeness": result.get("completeness", 0),
                "avg": result.get("avg_score", 0),
                "verdict": result.get("factcheck_verdict", "—"),
                "recommendation": result.get("recommendation", "—")
            })

        status_dict = {v: "done" for v in visited}
        yield (
            chat_final,
            "\n".join(event_lines),
            make_agent_cards(status_dict),
            result.get("factcheck_report", "No fact-check performed."),
            memory.short_term.get_display(),
            generate_mermaid(visited)
        )

    except Exception as e:
        event_lines.append(f"\n❌ ERROR: {str(e)}")
        error_response = chat_history + [
            {"role": "assistant", "content": f"❌ Pipeline error: {str(e)}"}
        ]
        yield (
            error_response,
            "\n".join(event_lines),
            make_agent_cards({"router": "error"}),
            "",
            memory.short_term.get_display()
        )


# ── Theme + CSS ───────────────────────────────────────────────────
CUSTOM_THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter")
)

CUSTOM_CSS = """
body, .gradio-container { background: #0a0f1e !important; }

/* Chat styling */
.chatbot-container .message { font-size: 0.9em !important; }

/* Event stream */
.event-stream textarea {
    font-family: 'IBM Plex Mono', 'Fira Code', monospace !important;
    font-size: 11px !important;
    background: #020617 !important;
    color: #94a3b8 !important;
    border: 1px solid #1e293b !important;
    line-height: 1.7 !important;
}

/* Section labels */
.section-label {
    font-size: 0.7em; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 4px; font-weight: 600;
}

/* Header */
.header-block {
    background: linear-gradient(135deg, #0a0f1e 0%, #1e293b 100%);
    border: 1px solid #1e293b; border-radius: 12px;
    padding: 16px 20px; margin-bottom: 8px;
}

/* Preset buttons */
.preset-btn { font-size: 0.8em !important; }

footer { display: none !important; }
"""


# ── Build UI ──────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="Multi-Agent Production Stack") as demo:
        gr.HTML(MERMAID_JS) # Load Mermaid.js

        # ── HEADER ────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-block">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <h1 style="margin:0;color:#f1f5f9;font-size:1.4em;font-weight:700">
                        🤖 Multi-Agent Production Stack
                    </h1>
                    <p style="margin:4px 0 0;color:#64748b;font-size:0.8em">
                        Memory · Fact-Check · LLM-as-a-Judge · 5-Layer Architecture ·
                        Powered by <code style="color:#60a5fa">llama3</code> via Ollama
                    </p>
                </div>
                <div style="color:#334155;font-size:0.7em;text-align:right">
                    ChromaDB Vector Store<br>LangGraph Orchestration
                </div>
            </div>
        </div>
        """)

        # ── AGENT STATUS CARDS ────────────────────────────────────
        agent_cards = gr.HTML(value=make_agent_cards())

        # ── MAIN LAYOUT: Chat (left) + Events (right) ────────────
        with gr.Row():

            # LEFT: Chat Interface
            with gr.Column(scale=1):
                gr.HTML("<div class='section-label'>💬 Agent Chat</div>")
                chatbot = gr.Chatbot(
                    label="",
                    height=420,
                )

                # Preset buttons
                gr.HTML("<div class='section-label'>🎯 Demo Scenarios</div>")
                with gr.Row():
                    preset_btns = {}
                    for label, q in PRESETS.items():
                        preset_btns[label] = gr.Button(
                            label, size="sm", elem_classes=["preset-btn"]
                        )

                # Input row
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type a clinical question...",
                        show_label=False,
                        scale=4,
                        container=False
                    )
                    send_btn = gr.Button("▶ Run", variant="primary", scale=1)

            # RIGHT: Event Stream
            with gr.Column(scale=1):
                gr.HTML("<div class='section-label'>⚡ Pipeline Events</div>")
                event_stream = gr.Textbox(
                    label="",
                    lines=25,
                    max_lines=35,
                    interactive=False,
                    placeholder=(
                        "Events will stream here as agents process your query...\n\n"
                        "  🔀 Router classifying task\n"
                        "  🤖 Retrieval searching vector DB\n"
                        "  🔎 Fact-checker verifying claims\n"
                        "  ⚖️  Judge scoring quality\n"
                        "  💾 Memory storing exchange"
                    ),
                    elem_classes=["event-stream"]
                )

        # ── BOTTOM TABS ───────────────────────────────────────────
        with gr.Tabs():
            with gr.TabItem("📊 Execution Flow"):
                gr.HTML("<div class='section-label' style='margin-top:10px'>LangGraph Execution Path</div>")
                mermaid_output = gr.HTML(value=generate_mermaid(), label="Execution Flow")

            with gr.TabItem("🔎 Fact-Check Report"):
                factcheck_box = gr.Textbox(
                    label="Adversarial Fact-Checker Output",
                    lines=15, interactive=False,
                    placeholder="SUPPORTED ✅ / CONTRADICTED 🚨 / UNVERIFIABLE ❓"
                )

            with gr.TabItem("🧠 Memory State"):
                memory_display = gr.Textbox(
                    label="Short-Term Memory",
                    lines=12, interactive=False,
                    value="No memory yet — ask a question to build context."
                )

            with gr.TabItem("📊 Score History"):
                score_display = gr.JSON(label="All Scores", value=[])
                refresh_btn = gr.Button("🔄 Refresh")
                refresh_btn.click(
                    fn=lambda: list(reversed(score_history)),
                    outputs=score_display
                )

            with gr.TabItem("ℹ️ Knowledge Base"):
                try:
                    stats = get_collection_stats()
                    kb_info = (
                        f"**Vector Store:** ChromaDB\n\n"
                        f"**Total Chunks:** {stats['total_chunks']}\n\n"
                        f"**Documents:** {', '.join(stats['documents'])}"
                    )
                except Exception:
                    kb_info = "Knowledge base loading..."
                gr.Markdown(kb_info)

        # ── FOOTER ────────────────────────────────────────────────
        with gr.Row():
            clear_btn = gr.Button("🗑 Clear Memory & Chat", variant="stop", size="sm")

        # ── WIRING ────────────────────────────────────────────────
        outputs = [chatbot, event_stream, agent_cards, factcheck_box, memory_display, mermaid_output]

        def send_message(msg, history):
            """Handle send button / enter key."""
            return run_pipeline(msg, history or [])

        def send_preset(preset_text, history):
            """Handle preset button click."""
            return run_pipeline(preset_text, history or [])

        # Wire send button + enter
        send_btn.click(
            fn=send_message,
            inputs=[msg_input, chatbot],
            outputs=outputs
        ).then(fn=lambda: "", outputs=msg_input)  # Clear input after send

        msg_input.submit(
            fn=send_message,
            inputs=[msg_input, chatbot],
            outputs=outputs
        ).then(fn=lambda: "", outputs=msg_input)

        # Wire preset buttons
        for label_text, q in PRESETS.items():
            def make_preset_fn(text):
                def fn(history):
                    yield from run_pipeline(text, history or [])
                return fn

            preset_btns[label_text].click(
                fn=make_preset_fn(q),
                inputs=[chatbot],
                outputs=outputs
            )

        # Wire clear
        def clear_all():
            memory.short_term.clear()
            memory.long_term.clear()
            return [], "", make_agent_cards(), "", "Memory cleared. ✅"

        clear_btn.click(fn=clear_all, outputs=outputs)

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=CUSTOM_THEME,
        css=CUSTOM_CSS
    )
