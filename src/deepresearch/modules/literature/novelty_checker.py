"""Novelty checking for research ideas."""

import json
from dataclasses import dataclass

import numpy as np

from deepresearch.core.config import ResearchIdea
from deepresearch.core.exceptions import NoveltyCheckError
from deepresearch.core.state import Paper
from deepresearch.modules.experiment.api_manager import APIManager
from deepresearch.providers.base import GenerationRequest, Message


@dataclass
class NoveltyScore:
    """Novelty assessment result."""

    overall_score: float  # 0-1, higher = more novel
    methodology_score: float
    contribution_score: float
    similar_papers: list[tuple[str, float]]  # (paper_id, similarity)
    explanation: str
    suggestions: list[str]


NOVELTY_PROMPT = """You are a research novelty assessor. Analyze the proposed research idea against existing literature.

Proposed Research:
Title: {title}
Description: {description}
Methodology: {methodology}
Key Contributions: {contributions}
Hypothesis: {hypothesis}

Related Papers (from literature search):
{papers}

Assess the novelty of this research idea. Consider:
1. Has this exact approach been tried before?
2. How different is the methodology from existing work?
3. Are the claimed contributions truly novel?
4. What gaps does this fill in the literature?

Output a JSON object with:
{{
    "overall_novelty_score": 0.0-1.0 (1.0 = completely novel),
    "methodology_novelty_score": 0.0-1.0,
    "contribution_novelty_score": 0.0-1.0,
    "most_similar_papers": [
        {{"paper_title": "...", "similarity_reason": "...", "similarity_score": 0.0-1.0}}
    ],
    "novel_aspects": ["list of truly novel aspects"],
    "overlap_concerns": ["list of potential overlap with existing work"],
    "suggestions_for_differentiation": ["how to make it more novel"],
    "explanation": "overall assessment explanation"
}}
"""


class NoveltyChecker:
    """Checks novelty of research ideas against literature."""

    def __init__(
        self,
        api_manager: APIManager,
        similarity_threshold: float = 0.85,
    ) -> None:
        self.api_manager = api_manager
        self.similarity_threshold = similarity_threshold
        self._embeddings_cache: dict[str, list[float]] = {}

    async def check_novelty(
        self,
        idea: ResearchIdea,
        papers: list[Paper],
    ) -> NoveltyScore:
        """Check the novelty of a research idea against related papers."""
        if not papers:
            return NoveltyScore(
                overall_score=1.0,
                methodology_score=1.0,
                contribution_score=1.0,
                similar_papers=[],
                explanation="No related papers found for comparison.",
                suggestions=[],
            )

        # First, compute embedding-based similarity
        embedding_similarities = await self._compute_embedding_similarities(
            idea, papers
        )

        # Then, use LLM for detailed novelty analysis
        llm_assessment = await self._llm_novelty_assessment(idea, papers)

        # Combine scores
        similar_papers = []
        for paper_id, sim in embedding_similarities[:5]:
            similar_papers.append((paper_id, sim))

        # Weight embedding and LLM assessments
        overall_score = (
            0.3 * (1 - max(s for _, s in embedding_similarities[:3]) if embedding_similarities else 1.0)
            + 0.7 * llm_assessment["overall_novelty_score"]
        )

        return NoveltyScore(
            overall_score=overall_score,
            methodology_score=llm_assessment["methodology_novelty_score"],
            contribution_score=llm_assessment["contribution_novelty_score"],
            similar_papers=similar_papers,
            explanation=llm_assessment["explanation"],
            suggestions=llm_assessment["suggestions_for_differentiation"],
        )

    async def _compute_embedding_similarities(
        self,
        idea: ResearchIdea,
        papers: list[Paper],
    ) -> list[tuple[str, float]]:
        """Compute embedding-based similarities."""
        try:
            from deepresearch.providers.openai_provider import OpenAIProvider
            from deepresearch.core.config import ProviderConfig, ProviderType

            # Create embedding provider
            config = ProviderConfig(
                provider_type=ProviderType.OPENAI,
                model="text-embedding-3-small",
            )
            provider = OpenAIProvider(config)

            # Get idea embedding
            idea_text = f"{idea.title}\n{idea.description}\n{idea.methodology}"
            idea_embedding = await provider.embed([idea_text])
            idea_vec = np.array(idea_embedding.embeddings[0])

            # Get paper embeddings (batch for efficiency)
            paper_texts = [f"{p.title}\n{p.abstract}" for p in papers]
            paper_embeddings = await provider.embed(paper_texts)

            # Compute cosine similarities
            similarities = []
            for paper, embedding in zip(papers, paper_embeddings.embeddings):
                paper_vec = np.array(embedding)
                sim = np.dot(idea_vec, paper_vec) / (
                    np.linalg.norm(idea_vec) * np.linalg.norm(paper_vec)
                )
                similarities.append((paper.paper_id, float(sim)))

            await provider.close()

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities

        except Exception as e:
            # Fall back to no embedding comparison
            return []

    async def _llm_novelty_assessment(
        self,
        idea: ResearchIdea,
        papers: list[Paper],
    ) -> dict:
        """Use LLM for detailed novelty assessment."""
        # Format papers for prompt
        papers_text = "\n\n".join(
            f"Paper {i+1}: {p.title}\nAbstract: {p.abstract[:500]}..."
            for i, p in enumerate(papers[:10])  # Limit to top 10
        )

        prompt = NOVELTY_PROMPT.format(
            title=idea.title,
            description=idea.description,
            methodology=idea.methodology,
            contributions=", ".join(idea.key_contributions),
            hypothesis=idea.hypothesis,
            papers=papers_text,
        )

        request = GenerationRequest(
            messages=[
                Message(role="system", content="You are a research novelty assessor. Output only valid JSON."),
                Message(role="user", content=prompt),
            ],
            temperature=0.3,
            max_tokens=2048,
            json_mode=True,
        )

        try:
            response = await self.api_manager.generate(request)
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Return default scores if parsing fails
            return {
                "overall_novelty_score": 0.5,
                "methodology_novelty_score": 0.5,
                "contribution_novelty_score": 0.5,
                "most_similar_papers": [],
                "novel_aspects": [],
                "overlap_concerns": [],
                "suggestions_for_differentiation": [],
                "explanation": "Unable to assess novelty due to parsing error.",
            }

    def is_sufficiently_novel(self, score: NoveltyScore) -> bool:
        """Check if a novelty score meets the threshold."""
        return score.overall_score >= (1 - self.similarity_threshold)
