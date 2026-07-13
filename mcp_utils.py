import asyncio
from typing import Any

from agents import UserError
from agents.mcp import MCPServerStreamableHttp
from agents.mcp.server import _IsolatedSessionRetryFailed
from aiolimiter import AsyncLimiter
import httpx
from mcp.types import CallToolResult

from logs import get_logger

logger = get_logger(__name__, level="DEBUG")

_RETRY_BACKOFF_BASE_SECONDS = 10
_MAX_RETRY_BACKOFF_SECONDS = 120


class CustomMCPServerStreamableHttp(MCPServerStreamableHttp):
    def __init__(self, max_calls_per_minute: int = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_calls_per_minute = max_calls_per_minute
        self.rate_limiter = AsyncLimiter(
            max_rate=max_calls_per_minute / 60
            if max_calls_per_minute
            else float("inf"),
            time_period=1,
        )

    async def list_tools(self, *args, **kwargs):
        # cache_tools_list=True means this never hits the network after the first
        # call, so skip the rate limiter to avoid wasting tokens on cached lookups.
        logger.debug("Listing MCP tools")
        return await super().list_tools(*args, **kwargs)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        attempt = 0
        while True:
            try:
                async with self.rate_limiter:
                    logger.debug(
                        f"Calling tool: {tool_name}. "
                        f"Rate limiter has capacity: {self.rate_limiter.has_capacity()}"
                    )
                    return await super().call_tool(tool_name, arguments, meta=meta)

            except UserError as exc:
                # The SDK converts HTTP 429 responses to UserError. Retry with backoff
                # until the server's rate limit window clears.
                if "HTTP error 429" not in str(exc):
                    raise
                logger.warning(
                    f"Tool '{tool_name}' on '{self.name}' received HTTP 429 "
                    f"(attempt {attempt + 1}). Will retry after backoff."
                )

            except asyncio.CancelledError as exc:
                # CancelledError with "cancel scope" in the message is from anyio's
                # cancel scope firing when the MCP background streaming task fails
                # (the server returning 429 kills the background POST/SSE handler,
                # anyio then cancels the host task via task.cancel()). We uncancel
                # the current task and retry so inference can complete.
                #
                # CancelledError WITHOUT "cancel scope" is a genuine external
                # cancellation (Ctrl+C, explicit task.cancel()) and must propagate.
                if "cancel scope" not in str(exc):
                    raise
                current_task = asyncio.current_task()
                if current_task is not None:
                    current_task.uncancel()
                logger.warning(
                    f"Tool '{tool_name}' on '{self.name}' cancelled by MCP "
                    f"connection failure (attempt {attempt + 1}, likely rate "
                    "limiting). Will retry after backoff."
                )

            attempt += 1
            backoff = min(
                _RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
                _MAX_RETRY_BACKOFF_SECONDS,
            )
            logger.debug(
                f"Tool '{tool_name}': waiting {backoff:.0f}s before "
                f"attempt {attempt + 1}."
            )
            await asyncio.sleep(backoff)
