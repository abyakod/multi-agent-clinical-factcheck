"""
Retrieval Agent — Knowledge Base Specialist

Uses Ollama (llama3) to find relevant information from clinical
knowledge base documents via ChromaDB semantic search.

Temperature: 0.3 (slightly creative for synthesis)
"""

from openai import OpenAI
from tools.knowledge_base_tool import search_knowledge_base

# Ollama's OpenAI-compatible endpoint
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "llama3"

RETRIEVAL_SYSTEM_PROMPT = """You are a Clinical Knowledge Retrieval Specialist.

Your job is to find the most accurate, relevant information from the provided
clinical knowledge base (connected via Model Context Protocol - MCP) to 
answer the user's question.

RULES:
1. ONLY use information explicitly stated in the knowledge base passages.
2. NEVER invent, fabricate, or assume information not present in the source.
3. Quote directly from source documents when possible.
4. If the information is not in the knowledge base, say so explicitly.
5. Rate your confidence as HIGH, MEDIUM, or LOW:
   - HIGH: Direct, unambiguous answer found in sources
   - MEDIUM: Answer found but requires some interpretation
   - LOW: Partial answer or information may be incomplete

RESPONSE FORMAT:
Answer the question directly and thoroughly, citing which document
the information comes from. End with a confidence line:

CONFIDENCE: [HIGH|MEDIUM|LOW]
"""


def run_retrieval_agent(question: str, memory_context: str = "") -> dict:
    """
    Runs the retrieval specialist agent with semantic search.

    Args:
        question: User's clinical question
        memory_context: Previous conversation context (if any)

    Returns:
        dict with keys: response, confidence, tokens_used, source_context
    """
    # Semantic search via ChromaDB
    source_context = search_knowledge_base(question, n_results=5)

    user_message = f"""RELEVANT KNOWLEDGE BASE PASSAGES (Retrieved via MCP `search_kb` tool):
{source_context}

---

"""
    if memory_context:
        user_message += f"""PREVIOUS CONVERSATION CONTEXT:
{memory_context}

---

"""

    user_message += f"""QUESTION: {question}

Answer using ONLY the knowledge base passages above. Cite sources. State confidence."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.3,
            messages=[
                {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )

        response_text = response.choices[0].message.content
        tokens_used = (response.usage.prompt_tokens + response.usage.completion_tokens
                       if response.usage else 0)

        # Extract confidence from response
        confidence = "MEDIUM"
        for line in response_text.split("\n"):
            line_upper = line.strip().upper()
            if "CONFIDENCE:" in line_upper:
                if "HIGH" in line_upper:
                    confidence = "HIGH"
                elif "LOW" in line_upper:
                    confidence = "LOW"
                else:
                    confidence = "MEDIUM"
                break

        return {
            "response": response_text,
            "confidence": confidence,
            "tokens_used": tokens_used,
            "source_context": source_context
        }

    except Exception as e:
        return {
            "response": f"Retrieval agent error: {str(e)}",
            "confidence": "LOW",
            "tokens_used": 0,
            "source_context": source_context
        }
