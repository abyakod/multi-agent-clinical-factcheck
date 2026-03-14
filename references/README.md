# Multi-Agents That Remember, Factcheck + LLM-as-Judge

**The complete reference architecture for shipping AI agents that actually work in production.**

> Read the article: [Multi-Agents That Remember, Factcheck + LLM-as-Judge](https://medium.com/your-article-link)

---

## Why This Exists

Between 70–95% of enterprise AI pilots fail to reach production. Not because the models are bad — because the architecture wasn't built for reality.

This repository implements a complete 5-layer production agentic architecture, demonstrated using real WHO clinical guidelines and drug interaction data. Why clinical data? Because when an agent confidently states *"ibuprofen is safe alongside warfarin"* and the Fact-Checker fires `CONTRADICTED — HIGH severity` with a direct guideline quote showing fatal bleeding risk — the stakes of getting architecture right become immediately obvious to any audience.

The same architecture applies to any enterprise domain: finance, legal, operations, compliance. The clinical demo simply makes every layer's value undeniable.

| Layer | What It Solves |
|---|---|
| [MCP — Tool Registry](#layer-1-mcp--tool-registry) | NxM integration problem |
| [Memory Manager](#layer-2-memory-manager) | Stateless agents that forget everything |
| [Multi-Agent Orchestration](#layer-3-multi-agent-orchestration) | Single agents hitting a capability ceiling |
| [CrewAI Fact-Checker](#layer-4-crewai-fact-checker) | Hallucinations that look convincing |
| [LLM-as-a-Judge](#layer-5-llm-as-a-judge) | Zero production observability |

---

## Quick Start

```bash
git clone https://github.com/your-repo/agentic-production-stack
cd agentic-production-stack
pip install -r requirements.txt
cp .env.example .env
```

Add your keys to `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
```

Run the full stack:

```bash
python app.py
# Open http://localhost:7860
```

**Four demo scenarios are preloaded — each tests a specific layer:**

| Preset | What It Tests | Expected Result |
|---|---|---|
| 📋 Metformin dosage query | Retrieval + Memory | Accurate WHO guideline answer |
| 🚨 Ibuprofen safe with warfarin? | Fact-Checker contradiction | `CONTRADICTED — HIGH severity` 🚨 |
| ❓ Gym reimbursement limit? | Fact-Checker unverifiable | `UNVERIFIABLE` — not in knowledge base |
| ⚡ ACE inhibitor + potassium supplements | Full pipeline end to end | All 5 layers fire, APPROVE verdict |

---

## Architecture

```
User Request
     ↓
[MCP Layer]          — One registry. Every agent, every tool, one protocol.
     ↓
[Memory Layer]       — Short-term session context + long-term persistent memory
     ↓
[LangGraph Orchestrator]
     ├── [RAG Crew]        Internal knowledge retrieval
     ├── [Research Crew]   Live external data
     └── [Action Crew]     Tool execution via MCP
     ↓
[Fact-Checker]       — Adversarial agent: SUPPORTED / CONTRADICTED / UNVERIFIABLE
     ↓
[LLM-as-a-Judge]     — Faithfulness · Relevance · Completeness scored continuously
     ↓
[Router]             — PASS → User | FAIL → Human review queue via Slack
```

---

## Layer 1: MCP — Tool Registry

**Problem it solves:** 5 agents × 8 tools = 40 custom integrations. One API change breaks three agents. Maintenance eats your sprints.

**How it works:** Define tools once in a central registry. Every agent in your system connects to any tool through one standardized interface — today and when you add the fifteenth tool next quarter.

```python
# layers/mcp/registry.py
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dataclasses import dataclass
from typing import Dict, Any
import asyncio

@dataclass
class MCPTool:
    name: str
    command: str
    args: list
    description: str

# Define tools once. Every agent uses this registry.
TOOL_REGISTRY: Dict[str, MCPTool] = {
    "postgres": MCPTool(
        name="postgres",
        command="uvx",
        args=["mcp-server-postgres", "postgresql://user:pass@localhost/db"],
        description="Query internal databases — customer records, product data, logs"
    ),
    "slack": MCPTool(
        name="slack",
        command="uvx",
        args=["mcp-server-slack"],
        description="Read and send Slack messages — human-in-the-loop routing"
    ),
    "github": MCPTool(
        name="github",
        command="uvx",
        args=["mcp-server-github"],
        description="Read repos, issues, PRs — for code-aware agents"
    ),
    "filesystem": MCPTool(
        name="filesystem",
        command="uvx",
        args=["mcp-server-filesystem", "/knowledge_base"],
        description="Read internal documents — powers the RAG crew"
    ),
}

class AgentToolbox:
    """
    Single interface for all agent tool connections.
    Add a tool to TOOL_REGISTRY — every agent gets it automatically.
    """
    def __init__(self):
        self.active_sessions: Dict[str, ClientSession] = {}

    async def connect(self, tool_name: str) -> ClientSession:
        if tool_name in self.active_sessions:
            return self.active_sessions[tool_name]

        tool = TOOL_REGISTRY.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not in registry. Add it to TOOL_REGISTRY.")

        server_params = StdioServerParameters(command=tool.command, args=tool.args)

        async with stdio_client(server_params) as (read, write):
            session = ClientSession(read, write)
            await session.initialize()
            self.active_sessions[tool_name] = session
            tools = await session.list_tools()
            print(f"✅ Connected: {tool_name} ({len(tools.tools)} tools available)")
            return session

    async def execute(self, tool_name: str, action: str, params: Dict[str, Any]) -> Any:
        session = await self.connect(tool_name)
        result = await session.call_tool(action, params)
        return result

toolbox = AgentToolbox()

# Usage from any agent:
# result = await toolbox.execute("postgres", "query", {"sql": "SELECT * FROM customers LIMIT 10"})
# result = await toolbox.execute("slack", "send_message", {"channel": "#ai-review", "text": "..."})
```

---

## Layer 2: Memory Manager

**Problem it solves:** Agents that start every session from zero. Users repeat context. Preferences set last week are forgotten today.

**How it works:** Two memory types working together — short-term conversation buffer with auto-summarization (prevents token overflow), and long-term semantic vector store (persists across sessions, retrieved automatically by relevance).

```python
# layers/memory/manager.py
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
from typing import Optional

class ShortTermMemory:
    """
    Keeps recent messages verbatim, auto-summarizes older context.
    Prevents token overflow without losing conversational continuity.
    """
    def __init__(self, llm: ChatOpenAI, max_token_limit: int = 2000):
        self.memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=max_token_limit,
            return_messages=True,
            memory_key="chat_history"
        )

    def add_exchange(self, human_input: str, ai_response: str):
        self.memory.save_context(
            {"input": human_input},
            {"output": ai_response}
        )

    def get_context(self) -> list:
        return self.memory.load_memory_variables({})["chat_history"]

    def get_summary(self) -> str:
        return self.memory.moving_summary_buffer or "No summary yet."


class LongTermMemory:
    """
    Stores facts, preferences, and observations persistently.
    Retrieved semantically — agent finds relevant past context automatically.
    """
    def __init__(self, collection_name: str = "agent_memory"):
        self.embeddings = OpenAIEmbeddings()
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=f"./memory_store/{collection_name}"
        )

    def remember(self, content: str, metadata: Optional[dict] = None):
        doc_metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "observation",
            **(metadata or {})
        }
        self.store.add_texts(texts=[content], metadatas=[doc_metadata])

    def recall(self, query: str, k: int = 3) -> list:
        results = self.store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

    def remember_user_preference(self, user_id: str, preference: str):
        self.remember(
            content=preference,
            metadata={"type": "user_preference", "user_id": user_id}
        )


class AgentMemoryManager:
    """
    Single interface for all agent memory operations.
    Automatically injects relevant context into every agent call.
    """
    def __init__(self, agent_id: str, llm: ChatOpenAI):
        self.agent_id = agent_id
        self.short_term = ShortTermMemory(llm=llm)
        self.long_term = LongTermMemory(collection_name=f"agent_{agent_id}")

    def build_context_prompt(self, current_input: str) -> str:
        recent = self.short_term.get_context()
        relevant_memories = self.long_term.recall(query=current_input, k=3)
        context_parts = []

        if relevant_memories:
            context_parts.append(
                "RELEVANT CONTEXT FROM MEMORY:\n" +
                "\n".join(f"- {m}" for m in relevant_memories)
            )

        if recent:
            recent_str = "\n".join([
                f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
                for m in recent[-4:]
            ])
            context_parts.append(f"RECENT CONVERSATION:\n{recent_str}")

        return "\n\n".join(context_parts)

    def after_response(self, user_input: str, agent_response: str, important: bool = False):
        self.short_term.add_exchange(user_input, agent_response)
        if important:
            self.long_term.remember(
                content=f"User asked: '{user_input}' → Agent: '{agent_response[:200]}'",
                metadata={"agent_id": self.agent_id}
            )
```

> **Anti-pattern to avoid:** Don't store everything in long-term memory. Be deliberate — store decisions, preferences, constraints, and outcomes. Let short-term handle conversational flow. Polluted long-term memory degrades retrieval quality fast.

---

## Layer 3: Multi-Agent Orchestration

**Problem it solves:** One agent trying to do everything — retrieval, research, synthesis, execution — produces mediocre results across all dimensions.

**How it works:** LangGraph as the conductor (state management, routing, retries). CrewAI crews as specialist teams (each one doing one thing exceptionally well).

### LangGraph Orchestrator

```python
# layers/orchestrator/langgraph_orchestrator.py
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from enum import Enum

llm = ChatOpenAI(model="gpt-4o", temperature=0)

class TaskType(str, Enum):
    KNOWLEDGE_LOOKUP = "knowledge_lookup"
    LIVE_RESEARCH = "live_research"
    ACTION_EXECUTION = "action_execution"
    COMPLEX = "complex"

class OrchestratorState(TypedDict):
    user_request: str
    task_type: TaskType
    memory_context: str
    crew_results: List[dict]
    final_answer: str
    confidence: float
    requires_human_review: bool
    iteration: int

def classify_task(state: OrchestratorState) -> OrchestratorState:
    prompt = f"""Classify this request into exactly one category:
- knowledge_lookup: needs internal documents or stored data
- live_research: needs current external information
- action_execution: needs to perform actions (send, update, query)
- complex: needs multiple of the above

Request: {state['user_request']}
Respond with ONLY the category name."""

    task_type = llm.invoke(prompt).content.strip().lower()
    if task_type not in [t.value for t in TaskType]:
        task_type = TaskType.KNOWLEDGE_LOOKUP

    return {**state, "task_type": task_type}

def route_to_crew(state: OrchestratorState) -> str:
    return {
        TaskType.KNOWLEDGE_LOOKUP: "rag_crew",
        TaskType.LIVE_RESEARCH: "research_crew",
        TaskType.ACTION_EXECUTION: "action_crew",
        TaskType.COMPLEX: "rag_crew",
    }.get(state["task_type"], "rag_crew")

def synthesize_results(state: OrchestratorState) -> OrchestratorState:
    results_text = "\n\n".join([
        f"[{r['crew'].upper()} CREW]: {r['result']}"
        for r in state["crew_results"]
    ])
    prompt = f"""Synthesize these crew outputs into a single coherent answer.
{results_text}
User's request: {state['user_request']}"""
    return {**state, "final_answer": llm.invoke(prompt).content}

def check_confidence(state: OrchestratorState) -> str:
    if state["confidence"] < 0.7 and state["iteration"] < 2:
        return "retry"
    if state["confidence"] < 0.5:
        return "human_review"
    return "complete"

workflow = StateGraph(OrchestratorState)
workflow.add_node("classify", classify_task)
workflow.add_node("synthesize", synthesize_results)
workflow.set_entry_point("classify")
workflow.add_conditional_edges("classify", route_to_crew, {
    "rag_crew": "rag_crew",
    "research_crew": "research_crew",
    "action_crew": "action_crew",
})
workflow.add_edge("rag_crew", "synthesize")
workflow.add_edge("research_crew", "synthesize")
workflow.add_edge("action_crew", "synthesize")
workflow.add_conditional_edges("synthesize", check_confidence, {
    "retry": "classify",
    "human_review": END,
    "complete": END,
})
orchestrator = workflow.compile()
```

### CrewAI Specialist Crews

```python
# layers/crews/specialist_crews.py
from crewai import Agent, Task, Crew, Process
from crewai_tools import FileReadTool, WebsiteSearchTool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

def run_rag_crew(question: str, memory_context: str = "") -> str:
    """RAG is a capability here — not the entire system."""

    retrieval_agent = Agent(
        role="Knowledge Retrieval Specialist",
        goal="Find the most accurate information from internal documents",
        backstory=(
            "You find precise information in large document collections. "
            "You flag low-confidence results instead of guessing. "
            f"Context from memory: {memory_context}"
        ),
        tools=[FileReadTool(file_path="./knowledge_base/")],
        llm=llm, max_iter=3
    )

    synthesis_agent = Agent(
        role="Knowledge Synthesizer",
        goal="Produce a clear, grounded answer with source attribution",
        backstory=(
            "You combine retrieved information into accurate answers. "
            "You always cite sources. You state uncertainty explicitly."
        ),
        llm=llm
    )

    retrieval_task = Task(
        description=(
            f"Answer: '{question}'\n"
            f"Return top 3 excerpts with source filename, quote, confidence (HIGH/MEDIUM/LOW)."
        ),
        expected_output="3 excerpts with source, quote, and confidence",
        agent=retrieval_agent
    )

    synthesis_task = Task(
        description="Synthesize into: Answer | Evidence | Sources | Confidence | Gaps",
        expected_output="Structured answer with all 5 components",
        agent=synthesis_agent,
        context=[retrieval_task]
    )

    return str(Crew(
        agents=[retrieval_agent, synthesis_agent],
        tasks=[retrieval_task, synthesis_task],
        process=Process.sequential
    ).kickoff())


def run_research_crew(question: str, memory_context: str = "") -> str:
    """Live external research with built-in cross-verification."""

    research_agent = Agent(
        role="External Research Specialist",
        goal="Find current, accurate information from verified external sources",
        backstory="You find authoritative external information and never present speculation as fact.",
        tools=[WebsiteSearchTool()],
        llm=llm
    )

    verifier_agent = Agent(
        role="Research Verifier",
        goal="Cross-reference findings across multiple sources",
        backstory="You flag single-source claims as unverified. No unverified claim passes through.",
        tools=[WebsiteSearchTool()],
        llm=llm
    )

    research_task = Task(
        description=f"Research: '{question}'. Find 2-3 authoritative sources.",
        expected_output="Findings with source URLs and dates",
        agent=research_agent
    )

    verify_task = Task(
        description="Verify findings. Flag single-source or unverified claims.",
        expected_output="Verified findings with confidence per claim",
        agent=verifier_agent,
        context=[research_task]
    )

    return str(Crew(
        agents=[research_agent, verifier_agent],
        tasks=[research_task, verify_task],
        process=Process.sequential
    ).kickoff())
```

**LangGraph vs CrewAI — the mental model:**

| | LangGraph | CrewAI |
|---|---|---|
| **Role** | The conductor | The orchestra section |
| **Best for** | State, routing, retries, memory | Specialist roles, parallel work |
| **Use when** | Managing the top-level agent flow | Executing a specific capability |
| **Combined** | Orchestrator graph | Nodes inside the graph |

---

## Layer 4: CrewAI Fact-Checker

**Problem it solves:** Fluent-sounding answers that are factually wrong. The generator is optimized for fluency — fluency and accuracy are not the same thing.

**How it works:** A dedicated adversarial agent whose only job is finding contradictions. It classifies every factual claim as `SUPPORTED`, `CONTRADICTED`, or `UNVERIFIABLE` — with severity ratings. FAIL at HIGH severity blocks the response.

> **Why "adversarial agents" is a gap:** Most CrewAI tutorials show collaborative agents. Nobody writes about agents designed to challenge other agents. This pattern is one of the most effective production reliability techniques available — and barely documented.

```python
# layers/fact_checker/crewai_fact_checker.py
from crewai import Agent, Task, Crew, Process
from crewai_tools import FileReadTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

llm = ChatOpenAI(model="gpt-4o", temperature=0)

class ClaimVerdict(BaseModel):
    claim: str
    verdict: str        # SUPPORTED | CONTRADICTED | UNVERIFIABLE
    evidence: str       # Direct quote or NOT FOUND
    severity: str       # HIGH | MEDIUM | LOW

class FactCheckReport(BaseModel):
    verdicts: List[ClaimVerdict]
    overall_verdict: str   # PASS | FAIL | PARTIAL
    confidence: float
    summary: str

fact_checker = Agent(
    role="Adversarial Fact-Checker",
    goal=(
        "Find every factual error, unsupported claim, and contradiction. "
        "Assume the answer is wrong until proven otherwise by direct source evidence."
    ),
    backstory=(
        "You have zero tolerance for unsupported claims. "
        "You never give the benefit of the doubt. "
        "A plausible claim is not a verified claim."
    ),
    tools=[FileReadTool(file_path="./knowledge_base/")],
    llm=llm, max_iter=2
)

verdict_reporter = Agent(
    role="Verification Report Compiler",
    goal="Compile verdicts into a clear, prioritized, actionable report",
    backstory="HIGH severity contradictions are always first. PASS/FAIL must be unambiguous.",
    llm=llm
)

def run_fact_check(question: str, generated_answer: str) -> dict:
    extraction_task = Task(
        description=(
            f"QUESTION: {question}\n"
            f"ANSWER TO VERIFY: {generated_answer}\n\n"
            f"For each factual claim:\n"
            f"  SUPPORTED    — direct quote found, claim accurate\n"
            f"  CONTRADICTED — source says something different\n"
            f"  UNVERIFIABLE — not found in any source\n"
            f"Rate severity HIGH/MEDIUM/LOW per claim."
        ),
        expected_output="Claim list with verdict, evidence, severity",
        agent=fact_checker
    )

    report_task = Task(
        description=(
            "FAIL if any CONTRADICTED at HIGH severity.\n"
            "PARTIAL if UNVERIFIABLE but none CONTRADICTED.\n"
            "PASS only if all SUPPORTED.\n"
            "Sort by severity. Include confidence (0.0-1.0)."
        ),
        expected_output="Report: overall_verdict, confidence, verdicts sorted by severity, summary",
        agent=verdict_reporter,
        context=[extraction_task]
    )

    result = str(Crew(
        agents=[fact_checker, verdict_reporter],
        tasks=[extraction_task, report_task],
        process=Process.sequential
    ).kickoff())

    if "FAIL" in result:
        return {"status": "blocked", "answer": None, "report": result}
    elif "PARTIAL" in result:
        return {"status": "partial", "answer": generated_answer, "report": result}
    return {"status": "approved", "answer": generated_answer, "report": result}
```

> **Cost note:** Don't fact-check every response. Run 100% on compliance/financial content, 100% on anything the Judge flags, and 20% random sample on everything else.

**What Fact-Checker catches vs Judge:**

| Failure | Judge | Fact-Checker |
|---|---|---|
| Answer ignores the question | ✅ | ❌ |
| Fluent but factually wrong | ⚠️ Sometimes | ✅ |
| Invented plausible detail | ❌ | ✅ |
| Correct facts used incorrectly | ❌ | ✅ |

Run both. They're complementary.

---

## Layer 5: LLM-as-a-Judge

**Problem it solves:** No continuous quality signal. You find out about degradation from users, not metrics.

**How it works:** Every agent output scored on faithfulness, relevance, and completeness. Scores emit to your observability stack. Anything below 0.6 routes to human review via Slack MCP automatically.

```python
# layers/judge/llm_judge.py
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import json

llm = ChatOpenAI(model="gpt-4o", temperature=0)

class AgentEvaluation(BaseModel):
    faithfulness_score: float = Field(..., ge=0.0, le=1.0)
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    flag_for_review: bool

JUDGE_PROMPT = ChatPromptTemplate.from_template("""
You are a strict quality evaluator for an enterprise AI agent system.

USER REQUEST: {question}
CONTEXT PROVIDED TO AGENT: {context}
AGENT RESPONSE: {answer}

Score on three dimensions (0.0–1.0):
1. FAITHFULNESS: Every claim traceable to context? Penalize anything not grounded.
2. RELEVANCE: Directly addresses the request? Penalize tangents.
3. COMPLETENESS: Fully resolves the request? Penalize partial answers.

Return ONLY valid JSON:
{{
  "faithfulness_score": <float>,
  "relevance_score": <float>,
  "completeness_score": <float>,
  "reasoning": "<one sentence>",
  "flag_for_review": <true if any score below 0.6>
}}
""")

def run_judge(question: str, context: str, answer: str) -> AgentEvaluation:
    response = llm.invoke(JUDGE_PROMPT.format(
        question=question, context=context, answer=answer
    ))
    evaluation = AgentEvaluation(**json.loads(response.content))

    print(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "scores": {
            "faithfulness": evaluation.faithfulness_score,
            "relevance": evaluation.relevance_score,
            "completeness": evaluation.completeness_score,
            "average": round((
                evaluation.faithfulness_score +
                evaluation.relevance_score +
                evaluation.completeness_score
            ) / 3, 3)
        },
        "flag": evaluation.flag_for_review,
        "reason": evaluation.reasoning
    }, indent=2))

    return evaluation

def evaluate_and_route(question: str, context: str, answer: str) -> dict:
    evaluation = run_judge(question, context, answer)

    if evaluation.flag_for_review:
        # Push to Slack via MCP:
        # await toolbox.execute("slack", "send_message", {
        #     "channel": "#ai-review",
        #     "text": f"🚨 Flagged: {question[:100]}\nReason: {evaluation.reasoning}"
        # })
        return {"status": "flagged", "answer": answer, "scores": evaluation.dict()}

    return {"status": "approved", "answer": answer, "scores": evaluation.dict()}
```

**What low scores tell you:**

| Score | Below Threshold | Root Cause | Fix |
|---|---|---|---|
| Faithfulness | < 0.7 | Agent generating beyond context | Tighten retrieval, add Fact-Checker |
| Relevance | < 0.7 | Wrong crew being invoked | Improve task classification |
| Completeness | < 0.7 | Context too small or docs too sparse | Increase retrieval k, improve chunking |

---

## Production Readiness Checklist

Before you ship, verify these targets:

| Checkpoint | Target | Consequence if missed |
|---|---|---|
| P95 end-to-end latency | < 2 seconds | Users abandon at 3s |
| Memory retrieval relevance | ≥ 80% | Irrelevant memories degrade responses |
| Task classification accuracy | ≥ 90% | Misrouted tasks waste compute |
| Faithfulness score (avg) | ≥ 0.80 | Hallucination risk |
| Relevance score (avg) | ≥ 0.75 | Poor user experience |
| Fact-Checker FAIL rate | < 5% | Systemic generation problem |
| Fact-Checker UNVERIFIABLE rate | < 15% | Knowledge base gaps |
| LangGraph retry rate | < 15% | Poor classification or retrieval |
| Judge flag rate | < 10% | Systemic quality issue |
| MCP tool error rate | < 1% | Integration instability |
| Judge-human agreement | ≥ 80% | Validate monthly on real samples |

---

## Project Structure

```
agentic-production-stack/
├── layers/
│   ├── mcp/
│   │   └── registry.py          # Tool registry + AgentToolbox
│   ├── memory/
│   │   └── manager.py           # ShortTermMemory + LongTermMemory + AgentMemoryManager
│   ├── orchestrator/
│   │   └── langgraph_orchestrator.py   # Top-level state graph
│   ├── crews/
│   │   └── specialist_crews.py  # RAG, Research, Action crews
│   ├── fact_checker/
│   │   └── crewai_fact_checker.py      # Adversarial verification agent
│   └── judge/
│       └── llm_judge.py         # Evaluation + observability emitter
├── knowledge_base/              # Drop your documents here
├── memory_store/                # Auto-created by LongTermMemory
├── main.py                      # Full pipeline runner
├── requirements.txt
└── .env.example
```

---

## Requirements

```
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-community>=0.2.0
langgraph>=0.1.0
crewai>=0.28.0
crewai-tools>=0.4.0
chromadb>=0.5.0
mcp>=1.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

---

## What to Wire Next

1. **Grafana / Datadog** — emit Judge scores as metrics. Score trends reveal silent degradation you'd never catch in logs.
2. **Adversarial testing** — feed deliberately wrong answers to the Fact-Checker during QA. Verify it catches them every time before it matters in production.
3. **Slack auto-routing via MCP** — the `evaluate_and_route` function has the Slack call commented in. Uncomment, add your channel, and your human review loop closes automatically.

---

## Related Articles

- [Your AI Agent Demos. It Doesn't Ship.](https://medium.com/your-article-link) — the story behind this architecture
- Coming next: Mastering Google ADK for Multi-Modal Enterprise Agents

---

## License

MIT
