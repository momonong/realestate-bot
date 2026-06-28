"""
utils/search_util.py

Responsible for fetching real estate comparable sales and rental market data
from Zillow, Redfin, and Rent.com via DuckDuckGo Search (no API key required).

All network I/O is synchronous (duckduckgo_search is blocking), so callers
MUST wrap calls with asyncio.to_thread() to avoid blocking the Discord event loop.
"""

from __future__ import annotations

import logging
from typing import Optional

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SALES_QUERY_TEMPLATE = (
    'site:zillow.com OR site:redfin.com "{address}" sold comps price square feet'
)
_RENT_QUERY_TEMPLATE = (
    'site:zillow.com OR site:rent.com "{address}" monthly rent price market'
)
_MAX_SALES_RESULTS = 5
_MAX_RENT_RESULTS = 4
_DDGS_TIMEOUT = 20  # seconds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_results(results: list[dict], label: str) -> str:
    """
    Convert a list of DDGS result dicts into a readable text block.

    Each result dict typically contains: 'title', 'href', 'body'.
    """
    if not results:
        return f"[{label}]: No results found.\n"

    lines: list[str] = [f"=== {label} ==="]
    for i, r in enumerate(results, start=1):
        title = r.get("title", "(no title)").strip()
        body = r.get("body", "(no snippet)").strip()
        lines.append(f"[{i}] {title}")
        lines.append(f"    {body}")
    lines.append("")  # blank separator
    return "\n".join(lines)


def _safe_ddgs_text(
    query: str,
    max_results: int,
    label: str,
) -> str:
    """
    Execute a DuckDuckGo text search synchronously and return a formatted
    text block.  All exceptions are caught; a descriptive error string is
    returned instead of propagating the exception.

    Args:
        query:       The search query string.
        max_results: Maximum number of results to fetch.
        label:       Section label used in the formatted output.

    Returns:
        A formatted string block with the search results or an error notice.
    """
    logger.info("DDG search | label=%s | query=%r", label, query)
    try:
        with DDGS(timeout=_DDGS_TIMEOUT) as ddgs:
            results: list[dict] = list(ddgs.text(query, max_results=max_results))
        logger.info("DDG search | label=%s | got %d results", label, len(results))
        return _format_results(results, label)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DDG search failed | label=%s | error=%s", label, exc)
        return f"[{label}]: Search failed ({type(exc).__name__}: {exc})\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_real_estate_context(address: str, specs: str) -> str:
    """
    Perform two DuckDuckGo searches (comparable sales + rental market) for the
    given property and return a single concatenated context string.

    This function is **blocking** (synchronous I/O).  Wrap with
    ``asyncio.to_thread(fetch_real_estate_context, address, specs)`` when
    calling from an async Discord command handler.

    Args:
        address: The property street address (e.g. "123 Main St, Austin, TX").
        specs:   Additional specs such as sqft / beds / baths, included in
                 the output header for the LLM but NOT added to queries
                 (keeps queries concise for better DDG results).

    Returns:
        A multi-section text string suitable for inclusion in an LLM prompt.
    """
    logger.info("fetch_real_estate_context | address=%r specs=%r", address, specs)

    sales_query = _SALES_QUERY_TEMPLATE.format(address=address)
    rent_query = _RENT_QUERY_TEMPLATE.format(address=address)

    sales_context = _safe_ddgs_text(sales_query, _MAX_SALES_RESULTS, "COMPARABLE SALES")
    rent_context = _safe_ddgs_text(rent_query, _MAX_RENT_RESULTS, "RENTAL MARKET")

    header = (
        f"PROPERTY UNDER ANALYSIS\n"
        f"  Address : {address}\n"
        f"  Specs   : {specs}\n"
        f"\n"
    )

    context = header + sales_context + "\n" + rent_context
    logger.debug("Context assembled | length=%d chars", len(context))
    return context
