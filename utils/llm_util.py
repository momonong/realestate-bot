"""
utils/llm_util.py

Handles all interaction with the local Ollama instance.
The ollama Python SDK is synchronous; callers MUST use asyncio.to_thread().
"""

from __future__ import annotations

import logging
import os

import ollama

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read once at import time; overridable via environment vars)
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a professional real estate analyst AI. Your task is to analyze the
provided web search context and produce a concise property valuation report.

STRICT RULES:
1. Rely ONLY on the numbers and properties found in the provided Web Search
   Context below. Do NOT hallucinate or invent comparable properties.
2. From the comparable sales data, calculate the Estimated Home Value as the
   median (or best estimate) of the comparable sold prices.
3. From the rental market data, calculate the Estimated Monthly Rent.
4. Calculate Cap Rate using this exact formula:
     Cap Rate = (Annual Rent * 0.65 / Home Value) * 100
   The 0.65 factor accounts for 35% combined vacancy + operating expenses.
5. Your entire response MUST be under 1500 characters so it fits in Discord.
6. Do NOT use LaTeX or complex math notation. Write math as plain text
   (e.g., Cap Rate = 5.4%).
7. If the search context is insufficient to make a reliable estimate, clearly
   state the limitation and provide a best-effort range instead.

OUTPUT FORMAT (Markdown, Discord-compatible):

## 🏠 Property Valuation Report
**Address:** <address>
**Specs:** <specs>
| Metric | Estimate |
|--------|----------|
| Estimated Market Value | $X,XXX,XXX |
| Estimated Monthly Rent | $X,XXX |
| Cap Rate | X.X% |

## 📊 Comparable Properties
<2-4 rows of a compact markdown table with address, sold price, sqft>

## 💡 Market Context & Logic
<2-3 sentences explaining your reasoning and any caveats>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_valuation(
    user_input: str,
    search_context: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Send the search context to a local Ollama model and return the valuation
    report as a string.

    This function is **blocking** (synchronous Ollama SDK call). Wrap with
    ``asyncio.to_thread(run_valuation, user_input, search_context)`` when
    calling from an async Discord handler.

    Args:
        user_input:     The original user query (address + specs).
        search_context: Web search results assembled by search_util.
        model:          Name of the locally available Ollama model.

    Returns:
        The model's markdown report string, or an error message string.
    """
    logger.info("run_valuation | model=%s | context_len=%d", model, len(search_context))

    user_message = (
        f"USER REQUEST:\n{user_input}\n\n"
        f"WEB SEARCH CONTEXT:\n{search_context}"
    )

    try:
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            options={
                "temperature": 0.2,   # low temperature → more factual
                "num_predict": 800,   # cap tokens to stay within Discord limit
            },
        )
        report: str = response["message"]["content"].strip()
        logger.info("run_valuation | success | report_len=%d", len(report))
        return report

    except ollama.ResponseError as exc:
        logger.error("Ollama ResponseError: %s", exc)
        return (
            f"**Ollama Error:** The model `{model}` returned an error.\n"
            f"> {exc}\n\n"
            f"Make sure the model is pulled: `ollama pull {model}`"
        )
    except ConnectionRefusedError:
        logger.error("Ollama connection refused at %s", OLLAMA_HOST)
        return (
            f"**Ollama Offline:** Cannot connect to Ollama at `{OLLAMA_HOST}`.\n"
            "Please start Ollama with `ollama serve` and try again."
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in run_valuation")
        return (
            f"**Unexpected LLM Error:** `{type(exc).__name__}: {exc}`\n"
            "Please check the bot logs for details."
        )
