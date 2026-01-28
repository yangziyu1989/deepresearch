"""LLM-based research idea generation."""

import json
from dataclasses import dataclass

from deepresearch.core.config import ResearchIdea
from deepresearch.core.state import Paper
from deepresearch.modules.experiment.api_manager import APIManager
from deepresearch.providers.base import GenerationRequest, Message


IDEA_GENERATION_PROMPT = """You are an AI research scientist. Generate novel research ideas based on the given topic and literature context.

Research Topic: {topic}

Recent Related Papers:
{papers}

Generate {num_ideas} novel research ideas. For each idea, consider:
1. What gap in the literature does it address?
2. What is the core hypothesis?
3. How could it be tested experimentally?
4. What would be the key contributions?

Output a JSON object with:
{{
    "ideas": [
        {{
            "title": "concise, descriptive title",
            "description": "2-3 sentence description of the idea",
            "methodology": "proposed approach/method",
            "key_contributions": ["contribution 1", "contribution 2"],
            "hypothesis": "the main hypothesis to test",
            "feasibility_score": 0.0-1.0 (how easy to implement),
            "novelty_score": 0.0-1.0 (how novel compared to existing work),
            "impact_score": 0.0-1.0 (potential impact if successful),
            "related_papers": ["paper titles that inspired this"]
        }}
    ]
}}

Focus on ideas that:
- Are technically feasible with current AI capabilities
- Can be tested with existing benchmarks or datasets
- Address meaningful research questions
- Build on but clearly differentiate from existing work
"""


REFINEMENT_PROMPT = """You are an AI research scientist. Refine and improve the following research idea based on feedback.

Original Idea:
Title: {title}
Description: {description}
Methodology: {methodology}
Hypothesis: {hypothesis}

Feedback:
{feedback}

Novelty Concerns:
{novelty_concerns}

Suggestions for Differentiation:
{suggestions}

Provide an improved version of this idea that addresses the concerns. Output JSON:
{{
    "title": "refined title",
    "description": "refined description",
    "methodology": "refined methodology",
    "key_contributions": ["refined contributions"],
    "hypothesis": "refined hypothesis",
    "changes_made": ["list of changes made to address concerns"]
}}
"""


class IdeaGenerator:
    """Generates research ideas using LLMs."""

    def __init__(self, api_manager: APIManager) -> None:
        self.api_manager = api_manager

    async def generate_ideas(
        self,
        topic: str,
        papers: list[Paper],
        num_ideas: int = 5,
    ) -> list[ResearchIdea]:
        """Generate research ideas based on topic and literature."""
        # Format papers for context
        papers_text = "\n\n".join(
            f"{i+1}. {p.title} ({p.year})\n   {p.abstract[:300]}..."
            for i, p in enumerate(papers[:15])  # Limit context
        )

        prompt = IDEA_GENERATION_PROMPT.format(
            topic=topic,
            papers=papers_text,
            num_ideas=num_ideas,
        )

        request = GenerationRequest(
            messages=[
                Message(
                    role="system",
                    content="You are an AI research scientist. Generate creative but feasible research ideas. Output only valid JSON.",
                ),
                Message(role="user", content=prompt),
            ],
            temperature=0.8,  # Higher temperature for creativity
            max_tokens=4096,
            json_mode=True,
        )

        response = await self.api_manager.generate(request)

        try:
            data = json.loads(response.content)
            ideas = []
            for item in data.get("ideas", []):
                idea = ResearchIdea(
                    title=item["title"],
                    description=item["description"],
                    methodology=item["methodology"],
                    key_contributions=item.get("key_contributions", []),
                    hypothesis=item["hypothesis"],
                    novelty_score=item.get("novelty_score", 0.5),
                    feasibility_score=item.get("feasibility_score", 0.5),
                    impact_score=item.get("impact_score", 0.5),
                    related_papers=item.get("related_papers", []),
                )
                ideas.append(idea)
            return ideas
        except json.JSONDecodeError:
            return []

    async def refine_idea(
        self,
        idea: ResearchIdea,
        feedback: str,
        novelty_concerns: list[str],
        suggestions: list[str],
    ) -> ResearchIdea:
        """Refine an idea based on novelty feedback."""
        prompt = REFINEMENT_PROMPT.format(
            title=idea.title,
            description=idea.description,
            methodology=idea.methodology,
            hypothesis=idea.hypothesis,
            feedback=feedback,
            novelty_concerns="\n".join(f"- {c}" for c in novelty_concerns),
            suggestions="\n".join(f"- {s}" for s in suggestions),
        )

        request = GenerationRequest(
            messages=[
                Message(
                    role="system",
                    content="You are an AI research scientist. Refine research ideas to improve novelty. Output only valid JSON.",
                ),
                Message(role="user", content=prompt),
            ],
            temperature=0.7,
            max_tokens=2048,
            json_mode=True,
        )

        response = await self.api_manager.generate(request)

        try:
            data = json.loads(response.content)
            return ResearchIdea(
                title=data.get("title", idea.title),
                description=data.get("description", idea.description),
                methodology=data.get("methodology", idea.methodology),
                key_contributions=data.get("key_contributions", idea.key_contributions),
                hypothesis=data.get("hypothesis", idea.hypothesis),
                novelty_score=idea.novelty_score,  # Will be re-evaluated
                feasibility_score=idea.feasibility_score,
                impact_score=idea.impact_score,
                related_papers=idea.related_papers,
            )
        except json.JSONDecodeError:
            return idea

    def rank_ideas(self, ideas: list[ResearchIdea]) -> list[ResearchIdea]:
        """Rank ideas by combined score."""
        def combined_score(idea: ResearchIdea) -> float:
            # Weighted combination of scores
            return (
                0.4 * idea.novelty_score
                + 0.3 * idea.feasibility_score
                + 0.3 * idea.impact_score
            )

        return sorted(ideas, key=combined_score, reverse=True)

    async def select_best_idea(
        self,
        ideas: list[ResearchIdea],
        criteria: str | None = None,
    ) -> ResearchIdea:
        """Select the best idea, optionally using LLM for complex criteria."""
        if not ideas:
            raise ValueError("No ideas to select from")

        if criteria:
            # Use LLM to select based on custom criteria
            ideas_text = "\n\n".join(
                f"Idea {i+1}: {idea.title}\n"
                f"Description: {idea.description}\n"
                f"Methodology: {idea.methodology}\n"
                f"Scores: novelty={idea.novelty_score:.2f}, feasibility={idea.feasibility_score:.2f}, impact={idea.impact_score:.2f}"
                for i, idea in enumerate(ideas)
            )

            request = GenerationRequest(
                messages=[
                    Message(
                        role="system",
                        content="You are a research advisor. Select the best research idea based on the given criteria.",
                    ),
                    Message(
                        role="user",
                        content=f"Ideas:\n{ideas_text}\n\nCriteria: {criteria}\n\nWhich idea number is best? Reply with just the number.",
                    ),
                ],
                temperature=0.3,
                max_tokens=10,
            )

            response = await self.api_manager.generate(request)
            try:
                idx = int(response.content.strip()) - 1
                if 0 <= idx < len(ideas):
                    return ideas[idx]
            except ValueError:
                pass

        # Fall back to ranking
        ranked = self.rank_ideas(ideas)
        return ranked[0]
