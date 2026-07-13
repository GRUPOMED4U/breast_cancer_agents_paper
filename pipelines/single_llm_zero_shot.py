from dataclasses import asdict
from typing import List
from agents import Agent, ModelSettings, Runner, Usage, RunResult
from agents.extensions.models.litellm_model import LitellmModel
import asyncio

from data_models import EvaluationEntry, Prompt
from logs import get_logger
from pipelines.base_class import Pipeline
from prompt_helpers import format_prompt

logger = get_logger(__name__)


class SingleLLMZeroShotPipeline(Pipeline):
    def __init__(self, prompt_template: Prompt, agent_definition: dict[str, str]):
        self.model_id = agent_definition["model"].replace("/", "-")

        # Use LitellmModel if model is from a provider different from openai
        if "/" in agent_definition["model"]:
            agent_definition["model"] = LitellmModel(model=agent_definition["model"])

        self.prompt_template = prompt_template
        self.agent_definition = agent_definition
        self.agent_definition["model_settings"] = ModelSettings(
            **self.agent_definition["model_settings"]
        )

    def _make_agent(self) -> Agent:
        return Agent(
            **self.agent_definition,
        )

    async def _run_single_task(
        self,
        entry: EvaluationEntry,
    ) -> RunResult:
        if getattr(entry, "agent_response", None) is not None:
            logger.debug(f"Entry {entry.prompt_id} already completed. Skipping.")
            return None
        try:
            logger.debug(f"Running inference for entry {entry.prompt_id}")
            prompt = format_prompt(
                self.prompt_template,
                {"case_summary": entry.case_summary},
            )

            agent = self._make_agent()

            return await Runner.run(
                agent,
                prompt,
            )

        except asyncio.CancelledError:
            logger.exception(
                f"Pipeline cancelled at entry {entry.prompt_id}. Aborting."
            )
            raise

        except Exception as exc:
            logger.exception(f"Inference failed for entry {entry.prompt_id}")
            raise exc

    async def run_pipeline(
        self, dataset: List[EvaluationEntry]
    ) -> List[EvaluationEntry]:

        async with asyncio.TaskGroup() as tg:
            tasks = []
            for entry in dataset:
                tasks.append(tg.create_task(self._run_single_task(entry=entry)))

        errors: list[tuple[int, BaseException]] = []

        for idx, task in enumerate(tasks):
            result: RunResult = task.result()

            if isinstance(result, BaseException):
                errors.append((idx, result))
                continue

            if result is None:
                continue

            dataset[idx].agent_response = result.final_output

            # get token usage and convert to JSON serializable obj
            inference_token_usage: Usage = result.context_wrapper.usage
            inference_token_usage.input_tokens_details = (
                inference_token_usage.input_tokens_details.model_dump()
            )
            inference_token_usage.output_tokens_details = (
                inference_token_usage.output_tokens_details.model_dump()
            )
            inference_token_usage = asdict(inference_token_usage)

            dataset[idx].inference_token_usage = inference_token_usage

        if errors:
            logger.error(f"{len(errors)} inference tasks failed.")
            for idx, error in errors:
                logger.error(f"[entry={idx}] {type(error).__name__}: {error}")

        return dataset
