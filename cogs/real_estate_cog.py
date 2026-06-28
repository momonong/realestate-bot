"""
cogs/real_estate_cog.py

Discord Cog that exposes the `!estimate` command.

Usage:
    !estimate 123 Main St, Austin, TX | 1800 sqft | 3 beds | 2 baths
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime

import discord
from discord.ext import commands

from utils.search_util import fetch_real_estate_context
from utils.llm_util import run_valuation

logger = logging.getLogger(__name__)

_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")
_MAX_DISCORD_CHARS = 2000
_SAFE_CHAR_LIMIT = 1900


class RealEstateCog(commands.Cog, name="RealEstate"):
    """
    Cog providing the !estimate command for AI-powered property valuation.
    """

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # ------------------------------------------------------------------
    # Helper: build the "thinking" embed
    # ------------------------------------------------------------------
    @staticmethod
    def _thinking_embed(address: str, specs: str) -> discord.Embed:
        embed = discord.Embed(
            title="\U0001f3e0 Real Estate Analysis in Progress",
            description=(
                "Searching comparable sales & rental data, "
                "then running local AI valuation...\n\u200b"
            ),
            color=discord.Color.from_rgb(52, 152, 219),
            timestamp=datetime.utcnow(),
        )
        embed.add_field(name="\U0001f4cd Address", value=f"`{address}`", inline=False)
        embed.add_field(name="\U0001f4d0 Specs",   value=f"`{specs}`",   inline=False)
        embed.add_field(
            name="\u23f3 Steps",
            value=(
                "1\ufe0f\u20e3 DuckDuckGo → Zillow / Redfin comp search\n"
                "2\ufe0f\u20e3 DuckDuckGo → Zillow / Rent.com rental search\n"
                f"3\ufe0f\u20e3 Local Ollama (`{_MODEL}`) valuation"
            ),
            inline=False,
        )
        embed.set_footer(text="This may take 30-90 seconds. Please wait.")
        return embed

    # ------------------------------------------------------------------
    # Helper: truncate report if needed
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_report(report: str) -> str:
        if len(report) <= _SAFE_CHAR_LIMIT:
            return report
        logger.warning(
            "Report truncated from %d to %d chars", len(report), _SAFE_CHAR_LIMIT
        )
        truncated = report[:_SAFE_CHAR_LIMIT]
        # Try to cut at the last newline to avoid breaking markdown
        last_nl = truncated.rfind("\n")
        if last_nl > _SAFE_CHAR_LIMIT - 200:
            truncated = truncated[:last_nl]
        return truncated + "\n\n> \u26a0\ufe0f *Report truncated to fit Discord's 2000-char limit.*"

    # ------------------------------------------------------------------
    # Command: !estimate
    # ------------------------------------------------------------------
    @commands.command(
        name="estimate",
        aliases=["comp", "valuation", "val"],
        help=(
            "Estimate market value, rent, and cap rate for a US property.\n\n"
            "Usage:\n"
            "  !estimate <address> | <sqft> | <beds> | <baths>\n\n"
            "Example:\n"
            "  !estimate 123 Main St, Austin, TX | 1800 sqft | 3 beds | 2 baths"
        ),
    )
    async def estimate(self, ctx: commands.Context, *, raw_input: str) -> None:
        """
        Main command handler for property valuation.

        Parses the '|'-delimited input, kicks off async search + LLM calls,
        and replies with a markdown valuation report.
        """
        # ---- Parse input ---------------------------------------------------
        parts = [p.strip() for p in raw_input.split("|")]
        if not parts or not parts[0]:
            await ctx.send(
                "\u26a0\ufe0f **Invalid input.**\n"
                "Usage: `!estimate <address> | <sqft> | <beds> | <baths>`\n"
                "Example: `!estimate 123 Main St, Austin, TX | 1800 sqft | 3 beds | 2 baths`"
            )
            return

        address = parts[0]
        specs = " | ".join(parts[1:]) if len(parts) > 1 else "(specs not provided)"
        user_input = raw_input

        logger.info(
            "estimate command | user=%s | address=%r | specs=%r",
            ctx.author,
            address,
            specs,
        )

        # ---- Send thinking embed -------------------------------------------
        thinking_msg = await ctx.send(embed=self._thinking_embed(address, specs))

        # ---- Step 1: Web search (blocking → thread) ------------------------
        try:
            search_context: str = await asyncio.to_thread(
                fetch_real_estate_context, address, specs
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Search step failed unexpectedly")
            await thinking_msg.delete()
            await ctx.send(
                f"\u274c **Search Error:** `{type(exc).__name__}: {exc}`\n"
                "The web search step failed. Please try again."
            )
            return

        # ---- Step 2: LLM valuation (blocking → thread) --------------------
        try:
            report: str = await asyncio.to_thread(
                run_valuation, user_input, search_context, _MODEL
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM step failed unexpectedly")
            await thinking_msg.delete()
            await ctx.send(
                f"\u274c **LLM Error:** `{type(exc).__name__}: {exc}`\n"
                "The AI valuation step failed. Please check that Ollama is running."
            )
            return

        # ---- Step 3: Delete thinking embed, send report -------------------
        try:
            await thinking_msg.delete()
        except discord.HTTPException:
            pass  # Not critical if delete fails

        safe_report = self._safe_report(report)
        await ctx.send(safe_report)

        logger.info(
            "estimate command done | user=%s | report_len=%d",
            ctx.author,
            len(report),
        )

    # ------------------------------------------------------------------
    # Error handler scoped to this Cog
    # ------------------------------------------------------------------
    @estimate.error
    async def estimate_error(self, ctx: commands.Context, error: commands.CommandError) -> None:
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(
                "\u26a0\ufe0f **Missing input.**\n"
                "Usage: `!estimate <address> | <sqft> | <beds> | <baths>`"
            )
        else:
            logger.error("Unhandled error in !estimate: %s", error)
            await ctx.send(f"\u274c An unexpected error occurred: `{error}`")


# ---------------------------------------------------------------------------
# Required setup function for discord.py Cog loader
# ---------------------------------------------------------------------------
async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(RealEstateCog(bot))
