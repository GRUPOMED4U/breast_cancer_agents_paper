from abc import ABC, abstractmethod
import asyncio
from typing import List, Optional
import numpy as np
from tqdm import tqdm
from google import genai
from google.genai import types

from data_models import (
    EvaluationEntry,
    EvaluationMetrics,
    GraderResponse,
    Message,
    RubricTag,
)
from logs import get_logger

logger = get_logger(__name__)

GRADER_TEMPLATE = """
Your job is to look at a query, a system response and a rubric, and score the system response on how well it follows the rubric.

# Query

{query}

# System response

{agent_response}

# Rubric

{rubric}

# Instructions

Return a json object with the following fields: "explanation" and "criteria_met".

# Dealing with missing information on the case summary

If the system correctly claims that an important information is missing from the case summary in order to meet the criteria of the rubric item and presents correct options for different scenarios, you should return true for the corresponding "criteria_met" and justify accordingly.

# Constraints

The "explanation" field should be a string explaining why the system response does or does not meet the criteria of the rubric item.

The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.

One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.

# Final instruction

Return just the json object. Do not include any other text in the response.
""".strip()


class AsyncBaseEvaluator(ABC):
    """
    Abstract base class for asynchronous evaluators.
    """

    @abstractmethod
    async def evaluate(self, inputs: List[EvaluationEntry]) -> List[EvaluationEntry]:
        pass

    @abstractmethod
    def compute_metrics(
        self,
        inputs: List[EvaluationEntry],
    ) -> EvaluationMetrics:
        """Compute aggregate evaluation metrics from asynchronous grader outputs.

        Args:
            inputs: List of evaluated system responses.
            targets: List of target evaluation entries.
            grader_responses: Nested list of grader responses for each evaluation
                entry and rubric item.

        Returns:
            An `EvaluationMetrics` instance containing the computed metrics.
        """
        pass

    async def __call__(
        self,
        inputs: List[EvaluationEntry],
    ) -> EvaluationMetrics:

        graded_inputs = await self.evaluate(inputs)
        metrics: EvaluationMetrics = self.compute_metrics(graded_inputs)

        return metrics


class HealthBenchMetricsMixin:
    """Mixin that computes HealthBench-style aggregate evaluation metrics.

    This mixin provides a reusable implementation for transforming rubric-level
    grader responses into normalized entry scores, per-tag aggregates, bootstrap
    score variability estimates, and per-example evaluation details.
    """

    def compute_metrics(
        self,
        inputs: List[EvaluationEntry],
    ) -> EvaluationMetrics:
        """Compute aggregate metrics from rubric-level grader responses.

        For each evaluation entry, this method computes a normalized score based on
        the points of rubric items whose criteria were met. It also aggregates scores
        by example tag and rubric tag, estimates score variability with bootstrap
        sampling, and attaches system responses and grader responses back to the
        evaluation entries.

        Args:
            inputs: List[EvaluationEntry].

        Returns:
            An `EvaluationMetrics` object containing global, per-tag, and per-example
                evaluation results.
        """
        entry_scores = []
        rubric_tag_scores = {}
        example_tag_scores = {}
        for evaluation_entry in tqdm(
            inputs,
            desc="Computing metrics",
            total=len(inputs),
        ):
            if not isinstance(evaluation_entry, EvaluationEntry):
                try:
                    evaluation_entry = EvaluationEntry(**evaluation_entry)
                except Exception as e:
                    logger.error(f"Failed to parse evaluation entry: {e}")
            entry_grader_responses = evaluation_entry.grader_responses
            # Global score
            ## Compute entry score
            entry_score = sum(
                r.points
                for r, g in zip(evaluation_entry.rubrics, entry_grader_responses)
                if g.criteria_met
            )
            ## Compute entry max possible score
            entry_max_possible_score = sum(
                r.points for r in evaluation_entry.rubrics if r.points > 0
            )

            entry_normalized_score = entry_score / entry_max_possible_score
            entry_scores.append(entry_normalized_score)

            # Aggregate per example tag
            for example_tag in evaluation_entry.example_tags:
                example_tag_scores.setdefault(example_tag, []).append(
                    entry_normalized_score
                )

            # Aggregate per rubric tag score
            for rubric_tag in RubricTag._value2member_map_.keys():
                ## Compute entry score
                entry_score = sum(
                    r.points
                    for r, g in zip(evaluation_entry.rubrics, entry_grader_responses)
                    if g.criteria_met and rubric_tag in r.tags
                )

                entry_normalized_score = entry_score / entry_max_possible_score
                rubric_tag_scores.setdefault(rubric_tag, []).append(
                    entry_normalized_score
                )

        # Compute global scores
        global_score = sum(entry_scores) / len(entry_scores)
        global_score = np.clip(global_score, 0, 1).item()

        ## Compute global scores std
        bootstrap_samples = [
            np.random.choice(entry_scores, len(entry_scores)) for _ in range(10000)
        ]
        bootstrap_means = [
            np.clip(np.mean(sample), 0, 1) for sample in bootstrap_samples
        ]
        global_score_std = np.std(bootstrap_means).item()

        ## Global score per example tag
        per_example_tag_score = {}
        for example_tag, scores in example_tag_scores.items():
            current_example_tag_score = sum(scores) / len(scores)
            per_example_tag_score[example_tag] = np.clip(
                current_example_tag_score, 0, 1
            ).item()

        ## Global score per rubric tag
        per_rubric_tag_score = {}
        for rubric_tag, scores in rubric_tag_scores.items():
            current_rubric_tag_score = sum(scores) / len(scores)
            per_rubric_tag_score[rubric_tag] = current_rubric_tag_score

        return EvaluationMetrics(
            global_score=global_score,
            global_score_std=global_score_std,
            per_example_tag_score=per_example_tag_score,
            per_rubric_tag_score=per_rubric_tag_score,
            per_example_details=inputs,
        )


class AsyncGeminiEvaluator(HealthBenchMetricsMixin, AsyncBaseEvaluator):
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        prompt_template: Optional[List[Message]] = None,
    ):
        """Initialize the asynchronous Gemini evaluator.

        Args:
            model_id: Identifier of the Google GenAI model used for grading.
            api_key: Optional API key for the Google GenAI client.
            prompt_template: Optional prompt template represented as a list of
                messages. If not provided, a default grading template is used.
        """
        self.model_id = model_id
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)

        # Define prompt template
        self.prompt_template = prompt_template
        if self.prompt_template is None:
            self.prompt_template = [
                {
                    "role": "user",
                    "content": GRADER_TEMPLATE,
                },
            ]

    async def evaluate(self, inputs: List[EvaluationEntry]) -> List[EvaluationEntry]:
        processed_entries: List[EvaluationEntry] = []
        token_usage = {}
        for evaluation_entry in tqdm(
            inputs, desc="Grading responses", total=len(inputs)
        ):
            entry_level_grader_responses: List[GraderResponse] = []
            agent_response = evaluation_entry.agent_response

            async with asyncio.TaskGroup() as tg:
                tasks = []
                for rubric in evaluation_entry.rubrics:
                    tasks.append(
                        tg.create_task(
                            self.client.aio.models.generate_content(
                                model=self.model_id,
                                contents=GRADER_TEMPLATE.format(
                                    query=evaluation_entry.prompt,
                                    agent_response=agent_response,
                                    rubric=rubric,
                                ),
                                config={
                                    "response_mime_type": "application/json",
                                    "response_json_schema": GraderResponse.model_json_schema(),
                                },
                            )
                        )
                    )

            responses: List[types.GenerateContentResponse] = [t.result() for t in tasks]
            for response in responses:
                try:
                    result = GraderResponse.model_validate_json(response.text)
                except Exception:
                    logger.error(
                        f"Failed to parse GraderResponse from response: {response.text}"
                    )
                    result = GraderResponse()
                entry_level_grader_responses.append(result)

                # Aggregate token usage
                relevant_token_usage_keys = [
                    "prompt_token_count",
                    "thoughts_token_count",
                    "total_token_count",
                    "tool_use_prompt_token_count",
                ]
                for k in relevant_token_usage_keys:
                    if getattr(response.usage_metadata, k, None) is not None:
                        token_usage[k] = token_usage.get(k, 0) + getattr(
                            response.usage_metadata, k
                        )
            evaluation_entry.grader_responses = entry_level_grader_responses
            evaluation_entry.evaluation_token_usage = token_usage
            processed_entries.append(evaluation_entry)
        return processed_entries
