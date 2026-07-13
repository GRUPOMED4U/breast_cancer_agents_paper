from dataclasses import asdict
from typing import List
import asyncio
import copy
from agents import (
    Agent,
    ModelSettings,
    Runner,
    ToolExecutionConfig,
    Usage,
    RunResult,
    RunConfig,
)
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp import (
    MCPServerStdio,
    MCPServerStreamableHttp,
    MCPServerManager,
    ToolFilterStatic,
)

from data_models import EvaluationEntry, Prompt
from pipelines.base_class import Pipeline
from prompt_helpers import format_prompt
from logs import get_logger

logger = get_logger(__name__, level="DEBUG")


class SingleLLMWithPubMedSearchAndFullPMCArticlesPipeline(Pipeline):
    def __init__(
        self,
        prompt_template: Prompt,
        agent_definition: dict,
        max_calls_per_minute: int = 600,
        max_concurrency: int = 5,
    ):
        # set_tracing_disabled(True)

        self.max_concurrency = max_concurrency
        self.prompt_template = prompt_template
        self.max_calls_per_minute = max_calls_per_minute

        self.agent_definition = copy.copy(agent_definition)
        self.model_id = self.agent_definition["model"].replace("/", "-")

        if "/" in self.agent_definition["model"]:
            self.agent_definition["model"] = LitellmModel(
                model=self.agent_definition["model"]
            )

        self.agent_definition["model_settings"] = ModelSettings(
            **self.agent_definition["model_settings"]
        )

    def _make_mcp_servers(self) -> MCPServerStreamableHttp:
        servers = []
        servers.append(
            MCPServerStdio(
                name="PubMed search",
                params={
                    "command": ".venv/Scripts/python.exe",
                    "args": ["custom_mcps/pubmed_search/server.py"],
                },
                cache_tools_list=True,
                max_retry_attempts=3,
                retry_backoff_seconds_base=1.0,
                tool_filter=ToolFilterStatic(
                    allowed_tool_names=[
                        "search_articles",
                        "get_full_text_of_pmc_article",
                    ]
                ),
                client_session_timeout_seconds=60,
            )
        )
        return servers

    def _make_agent(self, mcp_servers) -> Agent:
        return Agent(
            **self.agent_definition,
            mcp_servers=mcp_servers,
        )

    @staticmethod
    def _usage_to_dict(usage: Usage) -> dict:
        usage_dict = asdict(usage)

        input_details = getattr(usage, "input_tokens_details", None)
        if hasattr(input_details, "model_dump"):
            usage_dict["input_tokens_details"] = input_details.model_dump()

        output_details = getattr(usage, "output_tokens_details", None)
        if hasattr(output_details, "model_dump"):
            usage_dict["output_tokens_details"] = output_details.model_dump()

        return usage_dict

    async def _run_single_task(
        self,
        entry: EvaluationEntry,
        active_mcp_servers,
        concurrency_limiter: asyncio.Semaphore,
    ) -> RunResult:

        async with concurrency_limiter:
            if getattr(entry, "agent_response", None) is not None:
                logger.debug(f"Entry {entry.prompt_id} already completed. Skipping.")
                return None
            try:
                logger.debug(f"Running inference for entry {entry.prompt_id}")
                prompt = format_prompt(
                    self.prompt_template,
                    {"case_summary": entry.case_summary},
                )

                agent = self._make_agent(active_mcp_servers)

                return await Runner.run(
                    agent,
                    prompt,
                    run_config=RunConfig(
                        tool_execution=ToolExecutionConfig(
                            max_function_tool_concurrency=1,
                        ),
                    ),
                    max_turns=None,
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
        self,
        dataset: List[EvaluationEntry],
    ) -> List[EvaluationEntry]:
        concurrency_limiter = asyncio.Semaphore(self.max_concurrency)
        mcp_servers = self._make_mcp_servers()

        results = []

        async with MCPServerManager(mcp_servers) as manager:
            tasks = []
            for idx, entry in enumerate(dataset):
                task = self._run_single_task(
                    entry=entry,
                    active_mcp_servers=manager.active_servers,
                    concurrency_limiter=concurrency_limiter,
                )
                tasks.append(task)

            results = await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

        errors: list[tuple[int, BaseException]] = []

        for idx, result in enumerate(results):
            if isinstance(result, BaseException):
                errors.append((idx, result))
                continue

            if result is None:
                continue

            dataset[idx].agent_response = result.final_output

            inference_token_usage: Usage = result.context_wrapper.usage
            dataset[idx].inference_token_usage = self._usage_to_dict(
                inference_token_usage
            )

        if errors:
            logger.error(f"{len(errors)} inference tasks failed.")
            for idx, error in errors:
                logger.error(f"[entry={idx}] {type(error).__name__}: {error}")

        return dataset
