"""
LangGraph state object — carries all data through the pipeline.
Every node reads from and writes to this state.
This is what makes LangGraph stateful.
"""

from typing import TypedDict, List, Optional


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
