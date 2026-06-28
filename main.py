"""
main.py

Entry point for the RealEstate-Comp-Agent Discord Bot.

Startup:
    uv run python main.py
    # or
    python main.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import discord
from discord.ext import commands
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
# Force UTF-8 on Windows terminals (cp950 cannot encode some Unicode chars)
_stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(_stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("realestate-bot")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

DISCORD_TOKEN: str | None = os.getenv("DISCORD_TOKEN")
COMMAND_PREFIX: str       = os.getenv("COMMAND_PREFIX", "!")
OLLAMA_MODEL: str         = os.getenv("OLLAMA_MODEL", "qwen2.5")
OLLAMA_HOST: str          = os.getenv("OLLAMA_HOST", "http://localhost:11434")

if not DISCORD_TOKEN or DISCORD_TOKEN == "your_discord_bot_token_here":
    logger.error("DISCORD_TOKEN is not set in .env — aborting.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True  # Required for prefix commands in discord.py v2


class RealEstateBot(commands.Bot):
    """Custom Bot subclass that loads Cogs in setup_hook."""

    async def setup_hook(self) -> None:
        """Called once before the bot connects. Load all Cogs here."""
        cogs = [
            "cogs.real_estate_cog",
        ]
        for cog in cogs:
            try:
                await self.load_extension(cog)
                logger.info("Cog loaded: %s", cog)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load cog %s: %s", cog, exc)

    async def on_ready(self) -> None:
        """Fired when the bot successfully connects to Discord."""
        logger.info("=" * 60)
        logger.info("Bot online  : %s (ID: %s)", self.user, self.user.id)
        logger.info("Prefix      : %s", COMMAND_PREFIX)
        logger.info("Ollama model: %s @ %s", OLLAMA_MODEL, OLLAMA_HOST)
        logger.info("Guilds      : %d", len(self.guilds))
        logger.info("=" * 60)

        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name=f"{COMMAND_PREFIX}estimate <address> | sqft | beds | baths",
            )
        )

    async def on_command_error(
        self, ctx: commands.Context, error: commands.CommandError
    ) -> None:
        """Global error handler — only handles errors not caught by Cog handlers."""
        if isinstance(error, commands.CommandNotFound):
            return  # Silently ignore unknown commands
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(f"Slow down! Try again in {error.retry_after:.1f}s.")
            return
        logger.error("Global command error: %s", error)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
async def main() -> None:
    bot = RealEstateBot(
        command_prefix=COMMAND_PREFIX,
        intents=intents,
        help_command=commands.DefaultHelpCommand(),
    )

    try:
        async with bot:
            await bot.start(DISCORD_TOKEN)
    except discord.LoginFailure:
        logger.error("Invalid DISCORD_TOKEN. Please reset your token in the Developer Portal.")
        sys.exit(1)
    except discord.PrivilegedIntentsRequired:
        logger.error(
            "Message Content Intent is not enabled. "
            "Go to: Discord Developer Portal -> Your App -> Bot -> "
            "Enable 'MESSAGE CONTENT INTENT'."
        )
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C).")


if __name__ == "__main__":
    asyncio.run(main())
