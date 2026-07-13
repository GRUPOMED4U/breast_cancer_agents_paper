"""
PubMed MCP Server
Provides article search and download functionality through Model Context Protocol.

Sources:

- Adapted from https://github.com/aeghnnsw/pubmed-mcp
"""

import os
import threading
import time
from typing import Any, Callable, List, TypeVar

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pubmed_client import PubMedClient
import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent logs from propagating to the root logger
    logger.propagate = False

    # Avoid duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


logger = get_logger("pubmed_search", level="DEBUG")

# Load environment variables
load_dotenv()

T = TypeVar("T")


def _env_float(name: str, default: float) -> float:
    """Read a positive float from the environment, falling back to a default."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = float(raw_value)
    except ValueError:
        return default

    return value if value > 0 else default


class RateLimiter:
    """
    Thread-safe fixed-interval rate limiter.

    This limiter serializes outbound PubMed client calls inside this Python
    process. It intentionally avoids bursts by spacing request start times by at
    least 1 / requests_per_second seconds.
    """

    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be greater than zero")

        self._min_interval_seconds = 1.0 / requests_per_second
        self._next_allowed_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until another outbound request is allowed."""
        with self._lock:
            now = time.monotonic()
            sleep_seconds = self._next_allowed_at - now

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
                now = time.monotonic()

            self._next_allowed_at = (
                max(now, self._next_allowed_at) + self._min_interval_seconds
            )


# NCBI E-utilities currently allow 3 rps without an API key and 10 rps with one.
# Use slightly conservative defaults and allow explicit override for deployments
# that have negotiated higher limits.
DEFAULT_NCBI_RPS = 9.0 if os.getenv("NCBI_API_KEY") else 2.8
NCBI_RATE_LIMIT_RPS = _env_float("NCBI_RATE_LIMIT_RPS", DEFAULT_NCBI_RPS)
NCBI_MAX_RETRIES = int(_env_float("NCBI_MAX_RETRIES", 3))
NCBI_RETRY_BASE_SECONDS = _env_float("NCBI_RETRY_BASE_SECONDS", 1.0)

pubmed_rate_limiter = RateLimiter(requests_per_second=NCBI_RATE_LIMIT_RPS)


# Initialize MCP server
mcp = FastMCP(
    "PubMed",
    instructions="Use PubMed search whenever you need to gather more information on the medical literature and clinical guidelines to support your recommendations. Use `search_articles` to retrieve a list of PubMed IDs based on a query. The, use `get_full_text_of_pmc_article` to retrieve the full text of articles that have a PMCID associated with them.",
)

# Initialize PubMed client
pubmed_client = PubMedClient(
    api_key=os.getenv("NCBI_API_KEY"), email=os.getenv("NCBI_EMAIL")
)


def _is_rate_limit_error(error: Exception) -> bool:
    """Return True when an exception looks like an HTTP 429/rate-limit error."""
    message = str(error).lower()
    return (
        "429" in message
        or "too many requests" in message
        or "rate limit" in message
        or "rate-limit" in message
    )


def _call_pubmed(operation: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Execute one PubMed client operation with rate limiting and 429 retries.

    Rate limiting is applied immediately before every outbound client call.
    If NCBI still returns a rate-limit error, the call is retried with
    exponential backoff.
    """
    for attempt in range(NCBI_MAX_RETRIES + 1):
        pubmed_rate_limiter.wait()

        try:
            return operation(*args, **kwargs)
        except Exception as error:
            if not _is_rate_limit_error(error) or attempt >= NCBI_MAX_RETRIES:
                raise

            sleep_seconds = NCBI_RETRY_BASE_SECONDS * (2**attempt)
            time.sleep(sleep_seconds)

    # This line should never be reached because the loop either returns or raises.
    raise RuntimeError("PubMed request failed unexpectedly")


@mcp.tool(
    description="Retrieve a list of articles in the form of PubMed IDs based on a query. By default, it returns the top 5 results together with their abstracts."
)
def search_articles(
    query: str,
    max_results: int = 5,
    sort: str = "relevance",
    include_abstracts: bool = True,
) -> dict:
    """
    Search PubMed for articles matching the query.

    Args:
        query: Search query string (e.g., "COVID-19 vaccines", "machine learning AND healthcare")
        max_results: Maximum number of results to return (default: 20, max: 200)
        sort: Sort order - "relevance", "pub_date", or "first_author" (default: "relevance")
        include_abstracts: Whether to include abstracts in the results

    Returns:
        Dictionary containing:
        - pmids: List of PubMed IDs
        - total_count: Total number of matching articles
        - query_used: The search query that was executed
    """
    logger.debug(
        f"Calling tool: `search_articles`. Called with query: {query}, max_results: {max_results}, sort: {sort}, include_abstracts: {include_abstracts}"
    )
    try:
        # Validate inputs
        if not query.strip():
            return {"error": "Query cannot be empty"}

        if max_results < 1 or max_results > 200:
            max_results = min(max(max_results, 1), 200)

        if sort not in ["relevance", "pub_date", "first_author"]:
            sort = "relevance"

        # Perform search
        results = _call_pubmed(
            pubmed_client.search_articles,
            query=query,
            max_results=max_results,
            sort=sort,
        )

        response = {
            "pmids": results["pmids"],
            "total_count": results["total_count"],
            "query_used": query,
            "results_returned": len(results["pmids"]),
            "sort_order": sort,
        }

        if include_abstracts:
            # Download abstracts for each article. This call is also rate limited.
            abstracts = download_articles_batch(
                pmids=results["pmids"], return_mode="json"
            )

            if "error" in abstracts:
                response["abstracts_error"] = abstracts["error"]
            else:
                response["abstracts"] = abstracts["content"]

        return response

    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


@mcp.tool(description="Retrieve the full text of an article with PMCID.")
def get_full_text_of_pmc_article(pmcid: str):
    """
    Get full text of a PMC article in markdown format if available.

    Args:
        pmcid: PubMed Central ID

    Returns:
        Full text as string
    """
    logger.debug(
        f"Calling tool: `get_full_text_of_pmc_article`. Called with PMCID: {pmcid}."
    )
    return pubmed_client.get_full_text_of_pmc_article(pmcid)


@mcp.tool(description="Retrieve a summary of one article.")
def download_article(
    pmid: str, format_type: str = "abstract", return_mode: str = "json"
) -> dict:
    """
    Download article details by PubMed ID.

    Args:
        pmid: PubMed ID (e.g., "33073741")
        format_type: Content format - "abstract" or "medline" (default: "abstract")
        return_mode: Return format - "xml", "text", or "json" (default: "json")

    Returns:
        Dictionary containing:
        - pmid: The PubMed ID
        - content: Article content in requested format
        - format_type: Format type used
        - return_mode: Return mode used
    """
    try:
        # Validate inputs
        if not pmid.strip():
            return {"error": "PMID cannot be empty"}

        # Clean PMID (remove any non-numeric characters)
        pmid_clean = "".join(filter(str.isdigit, pmid))
        if not pmid_clean:
            return {"error": "PMID must contain numeric characters"}

        if format_type not in ["abstract", "medline", "full"]:
            format_type = "abstract"

        if return_mode not in ["xml", "text", "json"]:
            return_mode = "xml"

        # Fetch article
        content = _call_pubmed(
            pubmed_client.fetch_article,
            pmid=pmid_clean,
            rettype=format_type,
            retmode=return_mode,
        )

        return {
            "pmid": pmid_clean,
            "content": content,
            "format_type": format_type,
            "return_mode": return_mode,
            "content_length": len(content),
        }

    except Exception as e:
        return {"error": f"Download failed: {str(e)}"}


@mcp.tool(
    description="Retrieve a summary of a list of articles PubMed IDs to understand their content."
)
def download_articles_batch(
    pmids: List[str], format_type: str = "abstract", return_mode: str = "json"
) -> dict:
    """
    Download multiple articles by PubMed IDs in a single request.

    Args:
        pmids: List of PubMed IDs (e.g., ["33073741", "33073726"])
        format_type: Content format - "abstract", "medline", or "full" (default: "abstract")
        return_mode: Return format - "xml", "text", or "json" (default: "json")

    Returns:
        Dictionary containing:
        - pmids: List of requested PMIDs
        - content: Combined article content
        - format_type: Format type used
        - return_mode: Return mode used
        - article_count: Number of articles requested
    """
    try:
        # Validate inputs
        if not pmids or len(pmids) == 0:
            return {"error": "PMIDs list cannot be empty"}

        # Clean PMIDs
        pmids_clean = []
        for pmid in pmids:
            pmid_clean = "".join(filter(str.isdigit, str(pmid)))
            if pmid_clean:
                pmids_clean.append(pmid_clean)

        if not pmids_clean:
            return {"error": "No valid PMIDs provided"}

        # Limit batch size to prevent timeout
        if len(pmids_clean) > 50:
            pmids_clean = pmids_clean[:50]

        if format_type not in ["abstract", "medline", "full"]:
            format_type = "abstract"

        if return_mode not in ["xml", "text", "json"]:
            return_mode = "json"

        # Fetch articles
        content = _call_pubmed(
            pubmed_client.fetch_articles_batch,
            pmids=pmids_clean,
            rettype=format_type,
            retmode=return_mode,
        )

        return {
            "pmids": pmids_clean,
            "content": content,
            "format_type": format_type,
            "return_mode": return_mode,
            "article_count": len(pmids_clean),
            "content_length": len(content),
        }

    except Exception as e:
        return {"error": f"Batch download failed: {str(e)}"}


@mcp.tool(
    description="Get a list of articles metadata in XML format without abstract content."
)
def get_article_xml_metadata(pmids: List[str]) -> dict:
    """
    Get document summaries for articles (metadata without full content).

    Args:
        pmids: List of PubMed IDs (e.g., ["33073741", "33073726"])

    Returns:
        Dictionary containing:
        - pmids: List of requested PMIDs
        - summaries: XML summary data
        - article_count: Number of articles requested
    """
    try:
        # Validate inputs
        if not pmids or len(pmids) == 0:
            return {"error": "PMIDs list cannot be empty"}

        # Clean PMIDs
        pmids_clean = []
        for pmid in pmids:
            pmid_clean = "".join(filter(str.isdigit, str(pmid)))
            if pmid_clean:
                pmids_clean.append(pmid_clean)

        if not pmids_clean:
            return {"error": "No valid PMIDs provided"}

        # Limit batch size
        if len(pmids_clean) > 50:
            pmids_clean = pmids_clean[:50]

        # Get summaries
        summaries = _call_pubmed(pubmed_client.get_article_summary, pmids_clean)

        return {
            "pmids": pmids_clean,
            "summaries": summaries,
            "article_count": len(pmids_clean),
            "content_length": len(summaries),
        }

    except Exception as e:
        return {"error": f"Summary retrieval failed: {str(e)}"}


if __name__ == "__main__":
    mcp.run()
