from __future__ import annotations

import os
import random
import threading
import time
from typing import Callable, List, TypeVar, Optional

import dotenv
import parallel
from agents import RunContextWrapper, function_tool
from parallel import Parallel
from logs import get_logger

dotenv.load_dotenv()

logger = get_logger(__name__, level="DEBUG")

client = Parallel()

T = TypeVar("T")


class GlobalRateLimiter:
    """
    Thread-safe process-global leaky-bucket rate limiter.

    This avoids bursts by enforcing a minimum interval between calls.
    For example:
        max_calls_per_minute = 60  -> at most 1 call/second
        max_calls_per_minute = 600 -> at most 10 calls/second

    Important:
        This is global within one Python process. If you run multiple
        worker processes, each process will have its own limiter unless
        you move the limiter state to Redis, a database, etc.
    """

    def __init__(self, max_calls_per_minute: int) -> None:
        if max_calls_per_minute <= 0:
            raise ValueError("max_calls_per_minute must be > 0")

        self.max_calls_per_minute = max_calls_per_minute
        self.min_interval_seconds = 60.0 / max_calls_per_minute

        self._lock = threading.Lock()
        self._next_allowed_time = 0.0

    def acquire(self) -> None:
        """
        Blocks until this caller is allowed to perform the next API call.
        """
        with self._lock:
            now = time.monotonic()

            if now < self._next_allowed_time:
                sleep_seconds = self._next_allowed_time - now
                time.sleep(sleep_seconds)
                now = time.monotonic()

            self._next_allowed_time = now + self.min_interval_seconds


WEB_SEARCH_MAX_CALLS_PER_MINUTE = int(
    os.getenv("WEB_SEARCH_MAX_CALLS_PER_MINUTE", "600")
)

WEB_SEARCH_MAX_RETRIES = int(os.getenv("WEB_SEARCH_MAX_RETRIES", "5"))

WEB_SEARCH_BACKOFF_INITIAL_SECONDS = float(
    os.getenv("WEB_SEARCH_BACKOFF_INITIAL_SECONDS", "1.0")
)

WEB_SEARCH_BACKOFF_MAX_SECONDS = float(
    os.getenv("WEB_SEARCH_BACKOFF_MAX_SECONDS", "60.0")
)

WEB_SEARCH_BACKOFF_JITTER_SECONDS = float(
    os.getenv("WEB_SEARCH_BACKOFF_JITTER_SECONDS", "0.25")
)


# Process-global limiter. Every tool in this module should use this same object.
_web_search_rate_limiter = GlobalRateLimiter(
    max_calls_per_minute=WEB_SEARCH_MAX_CALLS_PER_MINUTE
)


def _get_retry_after_seconds(exc: BaseException) -> float | None:
    """
    Try to read Retry-After from the underlying HTTP response, if available.

    The Parallel SDK exposes `response` on APIStatusError. Depending on the
    HTTP backend, headers may or may not be present.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)

    if not headers:
        return None

    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        return None

    try:
        return max(0.0, float(retry_after))
    except ValueError:
        return None


def _is_429_error(exc: BaseException) -> bool:
    """
    Detect Parallel 429 errors robustly.
    """
    if isinstance(exc, parallel.RateLimitError):
        return True

    status_code = getattr(exc, "status_code", None)
    return status_code == 429


def _call_with_rate_limit_and_retries(
    call: Callable[[], T],
    *,
    max_retries: int = WEB_SEARCH_MAX_RETRIES,
    initial_backoff_seconds: float = WEB_SEARCH_BACKOFF_INITIAL_SECONDS,
    max_backoff_seconds: float = WEB_SEARCH_BACKOFF_MAX_SECONDS,
    jitter_seconds: float = WEB_SEARCH_BACKOFF_JITTER_SECONDS,
) -> T:
    """
    Run an API call under the global rate limiter and retry on 429.

    The limiter is applied before every attempt, including retries. This is
    important because retries themselves can otherwise create a second burst.
    """
    attempt = 0

    while True:
        _web_search_rate_limiter.acquire()

        try:
            return call()

        except Exception as exc:
            if not _is_429_error(exc):
                raise

            if attempt >= max_retries:
                logger.exception(
                    "Parallel API call failed after %s retries due to repeated 429 errors.",
                    max_retries,
                )
                raise

            retry_after_seconds = _get_retry_after_seconds(exc)

            if retry_after_seconds is not None:
                sleep_seconds = retry_after_seconds
            else:
                exponential_backoff = initial_backoff_seconds * (2**attempt)
                sleep_seconds = min(exponential_backoff, max_backoff_seconds)

                if jitter_seconds > 0:
                    sleep_seconds += random.uniform(0, jitter_seconds)

            attempt += 1

            logger.warning(
                "Received 429 from Parallel API. Retrying attempt %s/%s after %.2f seconds.",
                attempt,
                max_retries,
                sleep_seconds,
            )

            time.sleep(sleep_seconds)


###############################################################################
# TOOLS
###############################################################################


@function_tool
async def web_search(
    context: RunContextWrapper, objective: str, search_queries: List[str]
) -> dict:
    """
    Searches the web.

    Args:
        search_queries: Concise keyword search queries, 3-6 words each. At least one query is required,
            provide 2-3 for best results. Used together with objective to focus results on
            the most relevant content.

        objective: Natural-language description of the underlying question or goal driving the
            search. Used together with search_queries to focus results on the most relevant
            content. Should be self-contained with enough context to understand the intent
            of the search.
    """
    logger.debug("Tool called: web_search. Searching the web...")
    response = _call_with_rate_limit_and_retries(
        lambda: client.search(
            objective=objective,
            search_queries=search_queries,
        )
    )

    return response.model_dump()


@function_tool
async def extract_web_page(
    context: RunContextWrapper,
    objective: str,
    urls: List[str],
    search_queries: Optional[List[str]] = None,
) -> dict:
    """
    Searches the web.

    Args:
        urls: URLs to extract content from. Up to 20 URLs.

        search_queries: Concise keyword search queries, 3-6 words each. At least one query is required,
            provide 2-3 for best results. Used together with objective to focus results on
            the most relevant content.

        objective: Natural-language description of the underlying question or goal driving the
            search. Used together with search_queries to focus results on the most relevant
            content. Should be self-contained with enough context to understand the intent
            of the search.
    """
    logger.debug("Tool called: extract_web_page. Searching the web...")

    response = _call_with_rate_limit_and_retries(
        lambda: client.extract(
            objective=objective, search_queries=search_queries, urls=urls
        )
    )

    return response.model_dump()
