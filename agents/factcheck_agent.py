"""
Fact-Check Agent — Adversarial Verification Specialist

Uses Ollama (llama3) to adversarially verify every factual claim
in a generated answer against source documents. Classifies each
claim as SUPPORTED, CONTRADICTED, or UNVERIFIABLE with severity ratings.

Temperature: 0 (deterministic — no creativity in verification)

This is the pattern that catches hallucinations that look most
convincing — where the model fills a gap with a plausible-sounding
detail that happens to be wrong.
"""

from openai import OpenAI
import re

# Ollama's OpenAI-compatible endpoint
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "llama3"

FACTCHECK_SYSTEM_PROMPT = """You are an Adversarial Clinical Fact-Checker.

Your SOLE PURPOSE is to find every factual error, unsupported claim, and
contradiction in the answer you are given. You assume the answer is WRONG
until proven otherwise by DIRECT source evidence.

You have ZERO tolerance for unsupported claims. A plausible claim is NOT
a verified claim. You NEVER give the benefit of the doubt.

FOR EACH FACTUAL CLAIM in the answer, classify it as:
  SUPPORTED    — Direct quote found in source documents, claim is accurate
  CONTRADICTED — Source documents say something DIFFERENT (include the quote)
  UNVERIFIABLE — Claim not found in ANY source document

Rate severity for each claim:
  HIGH   — Could cause patient harm or is a critical clinical error
  MEDIUM — Clinically relevant but not immediately dangerous
  LOW    — Minor detail, low clinical impact

RESPONSE FORMAT (use EXACTLY this format):

CLAIM 1: [exact claim from the answer]
VERDICT: [SUPPORTED|CONTRADICTED|UNVERIFIABLE]
EVIDENCE: [Direct quote from source OR "NOT FOUND IN KNOWLEDGE BASE"]
SEVERITY: [HIGH|MEDIUM|LOW]

CLAIM 2: ...
(continue for all claims)

---
SUMMARY:
SUPPORTED: [count]
CONTRADICTED: [count]
UNVERIFIABLE: [count]
OVERALL VERDICT: [PASS|FAIL|PARTIAL]
- FAIL if ANY claim is CONTRADICTED at HIGH severity
- PARTIAL if any UNVERIFIABLE but none CONTRADICTED at HIGH
- PASS only if ALL claims SUPPORTED
"""


def run_factcheck_agent(question: str, answer: str, source_context: str) -> dict:
    """
    Runs the adversarial fact-checking agent.

    Args:
        question: Original user question
        answer: Generated answer to verify
        source_context: Source documents to verify against

    Returns:
        dict with keys: response, overall_verdict, supported_count,
                        contradicted_count, unverifiable_count
    """
    user_message = f"""SOURCE DOCUMENTS (ground truth):
{source_context}

---

QUESTION THAT WAS ASKED: {question}

ANSWER TO VERIFY:
{answer}

---

Verify EVERY factual claim in the answer above against the source documents.
Use the exact format specified. Be ruthlessly thorough."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": FACTCHECK_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )

        response_text = response.choices[0].message.content

        # Parse counts from response
        supported = _count_verdicts(response_text, "SUPPORTED")
        contradicted = _count_verdicts(response_text, "CONTRADICTED")
        unverifiable = _count_verdicts(response_text, "UNVERIFIABLE")

        # Determine overall verdict
        overall = _determine_verdict(response_text, contradicted, unverifiable)

        return {
            "response": response_text,
            "overall_verdict": overall,
            "supported_count": supported,
            "contradicted_count": contradicted,
            "unverifiable_count": unverifiable
        }

    except Exception as e:
        return {
            "response": f"Fact-check agent error: {str(e)}",
            "overall_verdict": "ERROR",
            "supported_count": 0,
            "contradicted_count": 0,
            "unverifiable_count": 0
        }


def _count_verdicts(text: str, verdict_type: str) -> int:
    """Count occurrences of a verdict type in the response."""
    pattern = rf"VERDICT:\s*{verdict_type}"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        return len(matches)

    summary_pattern = rf"{verdict_type}:\s*(\d+)"
    summary_match = re.search(summary_pattern, text, re.IGNORECASE)
    if summary_match:
        return int(summary_match.group(1))

    return 0


def _determine_verdict(text: str, contradicted: int, unverifiable: int) -> str:
    """Determine overall verdict from response text and counts."""
    verdict_match = re.search(
        r"OVERALL\s*VERDICT:\s*(PASS|FAIL|PARTIAL)",
        text, re.IGNORECASE
    )
    if verdict_match:
        return verdict_match.group(1).upper()

    if contradicted > 0:
        return "FAIL"
    elif unverifiable > 0:
        return "PARTIAL"
    return "PASS"
