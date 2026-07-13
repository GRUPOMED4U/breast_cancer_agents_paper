import asyncio
from dataclasses import asdict
import os
from typing import List
from agents.mcp import (
    MCPServer,
    MCPServerStdio,
    MCPServerManager,
    ToolFilterStatic,
)
from agents import (
    Agent,
    ItemHelpers,
    ModelSettings,
    RunConfig,
    RunContextWrapper,
    Runner,
    Tool,
    ToolExecutionConfig,
    Usage,
    RunResult,
    function_tool,
)
from agents.extensions.models.litellm_model import LitellmModel
from pydantic import BaseModel, Field, RootModel

from custom_error_handlers import max_turns_error_handler
from logs import get_logger
from data_models import EvaluationEntry, Prompt
from mcp_utils import CustomMCPServerStreamableHttp
from pipelines.base_class import Pipeline
from prompt_helpers import format_prompt

logger = get_logger(__name__, level="DEBUG")


# Constants

MODELS_WITHOUT_STRUCTURED_OUTPUT_SUPPORT = set(["gemini-gemini-2.5-flash"])


# Pipeline-specific data models
class Questions(BaseModel):
    questions: List[str] = Field(
        default_factory=list,
        description="List of questions to the specialist subagent.",
    )

    def format_questions_for_agent(
        self,
        current_entry: EvaluationEntry,
    ) -> str:
        questions = "\n".join(
            f"{i + 1}. {question.strip()}"
            for i, question in enumerate(self.questions)
            if question and question.strip()
        )

        return f"Answer the following questions:\n\n{questions}\n\nCase summary:\n\n{current_entry.case_summary}"


class QuestionAnswerPair(BaseModel):
    """
    QuestionAnswerPair pattern.
    """

    question: str = Field(description="The question made to the specialist subagent.")
    answer: str = Field(description="The answer provided by the specialist subagent.")


class QuestionAnswerPairsList(BaseModel):
    question_answer_pairs: List[QuestionAnswerPair] = Field(
        default_factory=list,
        description="List of question-answer pairs to the specialist subagent.",
    )


class Report(BaseModel):
    """
    Report pattern.

    The report text is expected to be in markdown format. The recommendations are expected to be organized in sections and include at least the following:
    - Systemic therapy
    - Surgical treatment
    - Radiotherapy
    - Complementary exams

    Other sections can be included if deemed necessary.

    The recommendations inside each section are numbered in the format "1. ", "2. ", etc, and always include the recommendation itself and a justification.
    """

    text: str = Field(description="The report text in markdown format.")


# Helpers


def agent_as_tool(
    agent: Agent,
    tool_name: str | None,
    tool_description: str | None,
    max_turns: int = 10,
    current_entry: EvaluationEntry = None,
) -> Tool:
    """
    Convert an Agent into a Tool that runs via Runner.run with full control.
    """

    @function_tool(
        name_override=tool_name,
        description_override=tool_description,
    )
    async def run_agent(
        context: RunContextWrapper,
        input: Questions,
    ) -> str:
        """
        Specialist subagent tool.

        Args:
            context (RunContextWrapper): The context of the tool.
            input (Questions): The input to the tool.

        Returns:
            str: The output of the tool.
        """

        agent_input = input.format_questions_for_agent(current_entry)

        result = await Runner.run(
            starting_agent=agent,
            input=agent_input,
            context=context.context,
            max_turns=max_turns,
            error_handlers={
                "max_turns": max_turns_error_handler,
            },
        )

        return ItemHelpers.text_message_outputs(result.new_items)

    return run_agent


# Main class
class DivideAndConquerPipeline(Pipeline):
    def __init__(
        self,
        prompt_template: Prompt,
        agent_definition: dict[str, str],
        subagents_definitions: List[dict[str, str]],
        max_concurrency: int = 1,
        max_calls_per_minute: float = 600.0,
        max_turns_for_main_agent: int = 2,
        max_turns_for_subagents: int = 10,
    ):
        self.model_id = agent_definition["model"].replace("/", "-")
        self.max_concurrency = max_concurrency
        self.max_calls_per_minute = max_calls_per_minute
        self.max_turns_for_main_agent = max_turns_for_main_agent
        self.max_turns_for_subagents = max_turns_for_subagents

        # Use LitellmModel if model is from a provider different from openai
        if "/" in agent_definition["model"]:
            agent_definition["model"] = LitellmModel(model=agent_definition["model"])

        self.prompt_template = prompt_template
        self.agent_definition = agent_definition
        self.agent_definition["model_settings"] = ModelSettings(
            **self.agent_definition["model_settings"]
        )

        self.subagents_definitions = subagents_definitions
        for idx, _ in enumerate(self.subagents_definitions):
            if "/" in self.subagents_definitions[idx]["model"]:
                self.subagents_definitions[idx]["model"] = LitellmModel(
                    model=self.subagents_definitions[idx]["model"]
                )
            self.subagents_definitions[idx]["model_settings"] = ModelSettings(
                **self.subagents_definitions[idx]["model_settings"]
            )

    def _make_mcp_servers(self) -> List[MCPServer]:
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
        servers.append(
            CustomMCPServerStreamableHttp(
                name="web search",
                params={
                    "url": "https://search.parallel.ai/mcp",
                    "timeout": 30,
                    "sse_read_timeout": 300,
                    "headers": {
                        "Authorization": f"Bearer {os.environ['PARALLEL_API_KEY']}"
                    },
                },
                cache_tools_list=True,
                max_retry_attempts=5,
                retry_backoff_seconds_base=1.0,
                max_calls_per_minute=self.max_calls_per_minute,
            )
        )
        return servers

    def _make_agent(
        self, tools: List[Tool] = None, mcp_servers: List[MCPServer] = None
    ) -> Agent:

        output_type = (
            Report
            if self.model_id not in MODELS_WITHOUT_STRUCTURED_OUTPUT_SUPPORT
            else None
        )

        return Agent(
            **self.agent_definition,
            tools=tools if tools is not None else [],
            mcp_servers=mcp_servers if mcp_servers is not None else [],
            output_type=output_type,
        )

    def _make_subagents(
        self, tools: List[Tool] = None, mcp_servers: List[MCPServer] = None
    ) -> List[Agent]:

        output_type = (
            QuestionAnswerPairsList
            if self.model_id not in MODELS_WITHOUT_STRUCTURED_OUTPUT_SUPPORT
            else None
        )

        return [
            Agent(
                **self.subagents_definitions[idx],
                tools=tools if tools is not None else [],
                mcp_servers=mcp_servers if mcp_servers is not None else [],
                output_type=output_type,
            )
            for idx, _ in enumerate(self.subagents_definitions)
        ]

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
        concurrency_limiter: asyncio.Semaphore,
        tools: List[Tool] = None,
        active_mcp_servers: List[MCPServer] = None,
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

                subagents = self._make_subagents(
                    tools=tools, mcp_servers=active_mcp_servers
                )
                subagents_as_tools = [
                    agent_as_tool(
                        agent=subagent,
                        tool_name=subagent.name.lower().replace(" ", "_"),
                        tool_description=f"{subagent.name}: a specialist subagent with access to web search tools and scientific databases.",
                        max_turns=self.max_turns_for_subagents,
                        current_entry=entry,
                    )
                    for subagent in subagents
                ]

                agent = self._make_agent(tools=subagents_as_tools)

                return await Runner.run(
                    agent,
                    prompt,
                    run_config=RunConfig(
                        tool_execution=ToolExecutionConfig(
                            max_function_tool_concurrency=1,
                        ),
                    ),
                    max_turns=self.max_turns_for_main_agent,
                    error_handlers={
                        "max_turns": max_turns_error_handler,
                    },
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
