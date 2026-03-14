"""
Judge Agent — LLM-as-a-Judge Quality Evaluator

Uses Ollama (llama3) to score every agent response on three dimensions:
  - Faithfulness: Is every claim grounded in context?
  - Relevance: Does the answer directly address the question?
  - Completeness: Does it fully resolve the request?

Temperature: 0 (deterministic scoring)

Scores emit to the Gradio dashboard as gauge values.
Below 0.6 on any dimension → FLAG for human review.
"""

from openai import OpenAI
import json
import re

# Ollama's OpenAI-compatible endpoint
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "llama3"

JUDGE_SYSTEM_PROMPT = """You are a strict Quality Evaluator for an enterprise AI agent system.

You score agent responses on three dimensions, each from 0.0 to 1.0:

1. FAITHFULNESS (0.0–1.0):
   Is every single claim in the response directly traceable to the provided context?
   Penalize ANY claim not grounded in the source material.
   1.0 = Every claim has a direct source. 0.0 = Entirely fabricated.

2. RELEVANCE (0.0–1.0):
   Does the response directly address what was asked?
   Penalize tangents, off-topic information, and padding.
   1.0 = Perfectly targeted answer. 0.0 = Completely off-topic.

3. COMPLETENESS (0.0–1.0):
   Does it fully resolve the user's request?
   Penalize partial answers, missing important aspects.
   1.0 = Fully resolved. 0.0 = Nothing useful provided.

RESPONSE FORMAT — Return ONLY valid JSON, nothing else:
{"faithfulness": <float>, "relevance": <float>, "completeness": <float>, "reasoning": "<one sentence>", "recommendation": "<APPROVE if all >= 0.6, FLAG if any < 0.6, REJECT if any < 0.3>"}
"""


def run_judge_agent(question: str, context: str, answer: str) -> dict:
    """
    Runs the LLM-as-Judge evaluation agent.

    Args:
        question: Original user question
        context: Source context provided to the retrieval agent
        answer: Agent's generated answer to evaluate

    Returns:
        dict with key 'scores' containing: faithfulness, relevance,
        completeness, average, recommendation, reasoning
    """
    user_message = f"""USER REQUEST: {question}

CONTEXT PROVIDED TO AGENT:
{context[:3000]}

AGENT RESPONSE:
{answer}

Score this response. Return ONLY valid JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )

        response_text = response.choices[0].message.content

        # Parse JSON from response (handle potential markdown wrapping)
        json_text = response_text
        if "```" in json_text:
            json_match = re.search(r"```(?:json)?\s*(.*?)```", json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)

        # Try to find JSON object in the response
        json_match = re.search(r"\{[^{}]*\}", json_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)

        scores = json.loads(json_text.strip())

        faithfulness = float(scores.get("faithfulness", 0.5))
        relevance = float(scores.get("relevance", 0.5))
        completeness = float(scores.get("completeness", 0.5))
        average = round((faithfulness + relevance + completeness) / 3, 3)

        recommendation = scores.get("recommendation", "FLAG")
        if recommendation not in ("APPROVE", "FLAG", "REJECT"):
            if min(faithfulness, relevance, completeness) < 0.3:
                recommendation = "REJECT"
            elif min(faithfulness, relevance, completeness) < 0.6:
                recommendation = "FLAG"
            else:
                recommendation = "APPROVE"

        return {
            "scores": {
                "faithfulness": faithfulness,
                "relevance": relevance,
                "completeness": completeness,
                "average": average,
                "recommendation": recommendation,
                "reasoning": scores.get("reasoning", "No reasoning provided")
            }
        }

    except Exception as e:
        return {
            "scores": {
                "faithfulness": 0.0,
                "relevance": 0.0,
                "completeness": 0.0,
                "average": 0.0,
                "recommendation": "REJECT",
                "reasoning": f"Judge agent error: {str(e)}"
            }
        }
