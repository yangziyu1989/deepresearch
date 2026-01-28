"""Literature search across multiple sources."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import arxiv
import httpx

from deepresearch.core.exceptions import LiteratureSearchError
from deepresearch.core.state import Paper


@dataclass
class SearchQuery:
    """Query for literature search."""

    query: str
    max_results: int = 50
    year_from: int | None = None
    year_to: int | None = None
    categories: list[str] | None = None


class ArxivClient:
    """Client for arXiv API."""

    def __init__(self) -> None:
        self.client = arxiv.Client()

    async def search(self, query: SearchQuery) -> list[Paper]:
        """Search arXiv for papers."""
        try:
            # Build search query
            search_query = query.query

            # Add category filter if specified
            if query.categories:
                cat_filter = " OR ".join(f"cat:{cat}" for cat in query.categories)
                search_query = f"({search_query}) AND ({cat_filter})"

            search = arxiv.Search(
                query=search_query,
                max_results=query.max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            # Run in thread pool since arxiv library is synchronous
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, lambda: list(self.client.results(search))
            )

            papers = []
            for result in results:
                # Filter by year if specified
                year = result.published.year
                if query.year_from and year < query.year_from:
                    continue
                if query.year_to and year > query.year_to:
                    continue

                paper = Paper(
                    paper_id=result.entry_id.split("/")[-1],
                    title=result.title,
                    abstract=result.summary,
                    authors=[a.name for a in result.authors],
                    year=year,
                    source="arxiv",
                    url=result.entry_id,
                )
                papers.append(paper)

            return papers

        except Exception as e:
            raise LiteratureSearchError(
                f"arXiv search failed: {e}",
                source="arxiv",
                query=query.query,
            )


class SemanticScholarClient:
    """Client for Semantic Scholar API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key

    async def search(self, query: SearchQuery) -> list[Paper]:
        """Search Semantic Scholar for papers."""
        try:
            async with httpx.AsyncClient() as client:
                params = {
                    "query": query.query,
                    "limit": min(query.max_results, 100),  # API limit
                    "fields": "paperId,title,abstract,authors,year,url,citationCount",
                }

                if query.year_from:
                    params["year"] = f"{query.year_from}-"
                if query.year_to:
                    if "year" in params:
                        params["year"] = f"{query.year_from}-{query.year_to}"
                    else:
                        params["year"] = f"-{query.year_to}"

                response = await client.get(
                    f"{self.BASE_URL}/paper/search",
                    params=params,
                    headers=self.headers,
                    timeout=30.0,
                )

                if response.status_code == 429:
                    raise LiteratureSearchError(
                        "Semantic Scholar rate limit exceeded",
                        source="semantic_scholar",
                        query=query.query,
                    )

                response.raise_for_status()
                data = response.json()

                papers = []
                for item in data.get("data", []):
                    if not item.get("abstract"):
                        continue  # Skip papers without abstracts

                    paper = Paper(
                        paper_id=item["paperId"],
                        title=item["title"],
                        abstract=item["abstract"],
                        authors=[a.get("name", "") for a in item.get("authors", [])],
                        year=item.get("year", 0),
                        source="semantic_scholar",
                        url=item.get("url", ""),
                        citations=item.get("citationCount", 0),
                    )
                    papers.append(paper)

                return papers

        except httpx.HTTPError as e:
            raise LiteratureSearchError(
                f"Semantic Scholar search failed: {e}",
                source="semantic_scholar",
                query=query.query,
            )


class LiteratureSearcher:
    """Aggregated literature search across multiple sources."""

    def __init__(
        self,
        arxiv_client: ArxivClient | None = None,
        semantic_scholar_client: SemanticScholarClient | None = None,
    ) -> None:
        self.arxiv_client = arxiv_client or ArxivClient()
        self.semantic_scholar_client = semantic_scholar_client or SemanticScholarClient()

    async def search(
        self,
        query: str,
        max_results: int = 50,
        sources: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[Paper]:
        """Search for papers across all configured sources."""
        sources = sources or ["arxiv", "semantic_scholar"]
        search_query = SearchQuery(
            query=query,
            max_results=max_results,
            year_from=year_from,
            year_to=year_to,
        )

        all_papers: list[Paper] = []
        tasks = []

        if "arxiv" in sources:
            tasks.append(self._search_arxiv(search_query))
        if "semantic_scholar" in sources:
            tasks.append(self._search_semantic_scholar(search_query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                # Log but continue with other sources
                continue
            all_papers.extend(result)

        # Deduplicate by title similarity
        deduplicated = self._deduplicate_papers(all_papers)

        # Sort by relevance (citations as proxy)
        deduplicated.sort(key=lambda p: p.citations, reverse=True)

        return deduplicated[:max_results]

    async def _search_arxiv(self, query: SearchQuery) -> list[Paper]:
        """Search arXiv."""
        return await self.arxiv_client.search(query)

    async def _search_semantic_scholar(self, query: SearchQuery) -> list[Paper]:
        """Search Semantic Scholar."""
        return await self.semantic_scholar_client.search(query)

    def _deduplicate_papers(self, papers: list[Paper]) -> list[Paper]:
        """Remove duplicate papers based on title similarity."""
        seen_titles: set[str] = set()
        unique_papers: list[Paper] = []

        for paper in papers:
            # Normalize title for comparison
            normalized = paper.title.lower().strip()
            # Remove common variations
            normalized = normalized.replace("-", " ").replace(":", " ")

            # Simple check - could use fuzzy matching for better results
            if normalized not in seen_titles:
                seen_titles.add(normalized)
                unique_papers.append(paper)

        return unique_papers

    async def get_paper_details(
        self, paper_id: str, source: str
    ) -> Paper | None:
        """Get detailed information for a specific paper."""
        if source == "semantic_scholar":
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{SemanticScholarClient.BASE_URL}/paper/{paper_id}",
                        params={
                            "fields": "paperId,title,abstract,authors,year,url,citationCount,references"
                        },
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    data = response.json()

                    return Paper(
                        paper_id=data["paperId"],
                        title=data["title"],
                        abstract=data.get("abstract", ""),
                        authors=[a.get("name", "") for a in data.get("authors", [])],
                        year=data.get("year", 0),
                        source="semantic_scholar",
                        url=data.get("url", ""),
                        citations=data.get("citationCount", 0),
                    )
            except Exception:
                return None

        return None
