"""
PubMed API client for E-utilities integration.
Provides methods for searching and fetching articles from PubMed database.

Sources:

- Adapted from https://github.com/aeghnnsw/pubmed-mcp
"""

from pathlib import Path
import subprocess
import requests
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union

import re


def clean_string(text: str) -> str:
    # If your string contains literal "\n" instead of real newlines
    text = text.replace("\\n", "\n")

    # Normalize Windows/Mac newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse spaces/tabs, but not newlines
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    # Collapse multiple consecutive newlines into one
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


class PubMedClient:
    """Client for interacting with PubMed E-utilities API."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None):
        """
        Initialize PubMed client.

        Args:
            api_key: NCBI API key for increased rate limits
            email: Email address for API requests (recommended)
        """
        self.api_key = api_key
        self.email = email
        self.last_request_time = 0
        self.rate_limit_delay = 0.34 if api_key else 0.34  # ~3 requests per second

    def _make_request(self, endpoint: str, params: Dict[str, str]) -> requests.Response:
        """
        Make a rate-limited request to PubMed API.

        Args:
            endpoint: API endpoint (e.g., 'esearch.fcgi', 'efetch.fcgi')
            params: Query parameters

        Returns:
            Response object
        """
        # Add common parameters
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email

        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)

        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, params=params)
        self.last_request_time = time.time()

        response.raise_for_status()
        return response

    def search_articles(
        self, query: str, max_results: int = 20, sort: str = "relevance"
    ) -> Dict[str, Union[List[str], int]]:
        """
        Search PubMed for articles matching the query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sort: Sort order ('relevance', 'pub_date', 'first_author')

        Returns:
            Dictionary with 'pmids' list and 'total_count'
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "xml",
            "sort": sort,
        }

        response = self._make_request("esearch.fcgi", params)

        # Parse XML response
        root = ET.fromstring(response.text)

        # Extract PMIDs
        pmids = []
        id_list = root.find("IdList")
        if id_list is not None:
            pmids = [id_elem.text for id_elem in id_list.findall("Id")]

        # Extract total count
        count_elem = root.find("Count")
        total_count = int(count_elem.text) if count_elem is not None else 0

        return {"pmids": pmids, "total_count": total_count}

    def fetch_article(
        self, pmid: str, rettype: str = "abstract", retmode: str = "xml"
    ) -> str:
        """
        Fetch article details by PMID.

        Args:
            pmid: PubMed ID
            rettype: Return type ('abstract', 'medline', 'full')
            retmode: Return format ('xml', 'text', 'json')

        Returns:
            Article data as string
        """
        params = {"db": "pubmed", "id": pmid, "rettype": rettype, "retmode": retmode}

        response = self._make_request("efetch.fcgi", params)
        return response.text

    def fetch_articles_batch(
        self, pmids: List[str], rettype: str = "abstract", retmode: str = "xml"
    ) -> str:
        """
        Fetch multiple articles by PMIDs in a single request.

        Args:
            pmids: List of PubMed IDs
            rettype: Return type ('abstract', 'medline', 'full')
            retmode: Return format ('xml', 'text', 'json')

        Returns:
            Combined article data as string
        """
        if not pmids:
            return ""

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": rettype,
            "retmode": retmode,
        }

        response = self._make_request("efetch.fcgi", params)
        return response.text

    def get_article_summary(self, pmids: List[str]) -> str:
        """
        Get document summaries for articles.

        Args:
            pmids: List of PubMed IDs

        Returns:
            XML summary data
        """
        if not pmids:
            return ""

        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}

        response = self._make_request("esummary.fcgi", params)
        return response.text

    def get_full_text_of_pmc_article(self, pmcid: str) -> str:
        """
        Get full text of a PMC article.

        Args:
            pmcid: PubMed Central ID

        Returns:
            Full text as string
        """
        temp_path = Path("temp")
        temp_path.mkdir(exist_ok=True)

        new_file = temp_path / f"{pmcid}.md"
        result = subprocess.run(
            [
                "./obscura",
                "fetch",
                f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                "--dump",
                "markdown",
                "--output",
                new_file,
                "--wait-until",
                "domcontentloaded",
            ]
        )

        if result.returncode != 0:
            return "Error: Failed to fetch PMC article."

        with open(new_file, "r", encoding="utf-8") as f:
            text = f.read()

        if "Page Not Found" in text:
            return "The PMC article could not be found or the PMCID is invalid."

        return clean_string(text)
