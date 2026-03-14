"""
Memory Manager — Short-Term + Long-Term Agent Memory

Short-term: Conversation buffer — keeps recent exchanges for session continuity.
Long-term: Important facts and preferences stored persistently.

The memory manager provides a unified interface for all memory operations.
"""

from datetime import datetime
from typing import List, Optional


class ShortTermMemory:
    """
    Keeps recent conversation exchanges verbatim.
    Provides display-formatted output for the Gradio memory tab.
    """

    def __init__(self, max_exchanges: int = 10):
        self.exchanges: List[dict] = []
        self.max_exchanges = max_exchanges

    def add_exchange(self, question: str, answer: str):
        """Store a Q&A exchange."""
        self.exchanges.append({
            "question": question,
            "answer": answer[:500],  # Truncate long answers for memory
            "timestamp": datetime.utcnow().isoformat()
        })
        # Keep only recent exchanges
        if len(self.exchanges) > self.max_exchanges:
            self.exchanges = self.exchanges[-self.max_exchanges:]

    def get_context(self) -> str:
        """Return formatted conversation history."""
        if not self.exchanges:
            return ""
        lines = []
        for i, ex in enumerate(self.exchanges, 1):
            lines.append(f"[Exchange {i}]")
            lines.append(f"  User: {ex['question']}")
            lines.append(f"  Agent: {ex['answer'][:300]}")
            lines.append("")
        return "\n".join(lines)

    def get_display(self) -> str:
        """Return formatted display for Gradio memory tab."""
        if not self.exchanges:
            return "No memory yet — ask a question to start building context."
        lines = ["═" * 40, "SHORT-TERM MEMORY", "═" * 40, ""]
        for i, ex in enumerate(self.exchanges, 1):
            lines.append(f"── Exchange {i} ({ex['timestamp'][:16]}) ──")
            lines.append(f"  Q: {ex['question'][:80]}")
            lines.append(f"  A: {ex['answer'][:200]}...")
            lines.append("")
        lines.append(f"Total exchanges: {len(self.exchanges)}")
        return "\n".join(lines)

    def clear(self):
        """Clear all short-term memory."""
        self.exchanges = []


class LongTermMemory:
    """
    Stores important facts and observations persistently.
    Simple list-based implementation — production would use a vector store.
    """

    def __init__(self):
        self.memories: List[dict] = []

    def store(self, content: str, metadata: Optional[dict] = None):
        """Store a fact in long-term memory."""
        self.memories.append({
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        })

    def recall(self, query: str, k: int = 3) -> List[str]:
        """
        Simple keyword-based recall.
        Production would use semantic vector search.
        """
        if not self.memories:
            return []

        # Simple relevance scoring by keyword overlap
        query_words = set(query.lower().split())
        scored = []
        for mem in self.memories:
            mem_words = set(mem["content"].lower().split())
            overlap = len(query_words & mem_words)
            scored.append((overlap, mem["content"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [content for _, content in scored[:k] if _ > 0]

    def clear(self):
        """Clear all long-term memory."""
        self.memories = []


class AgentMemoryManager:
    """
    Single interface for all agent memory operations.
    Automatically manages both short-term and long-term memory.
    """

    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    def get_context(self) -> str:
        """Get conversation context for injecting into agent prompts."""
        return self.short_term.get_context()

    def after_response(self, question: str, answer: str, store_long_term: bool = False):
        """
        Update memory after a pipeline response.

        Args:
            question: User's question
            answer: Agent's response
            store_long_term: Whether to persist to long-term memory
                           (only for high-quality responses)
        """
        self.short_term.add_exchange(question, answer)

        if store_long_term:
            self.long_term.store(
                content=f"Q: {question[:100]} → A: {answer[:200]}",
                metadata={"type": "high_quality_exchange"}
            )
