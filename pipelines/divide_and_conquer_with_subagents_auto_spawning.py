from dataclasses import dataclass, field
import asyncio
from dataclasses import asdict
from typing import List, Literal
from agents.mcp import (
    MCPServer,
    MCPServerStdio,
    MCPServerManager,
    ToolFilterStatic,
)
from agents import (
    Agent,
    FunctionTool,
    Handoff,
    HandoffInputData,
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
    handoff,
)
from agents.extensions.models.litellm_model import LitellmModel
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from pydantic import BaseModel, Field

from custom_error_handlers import max_turns_error_handler
from custom_tools.parallel_ai import extract_web_page, web_search
from logs import get_logger
from data_models import EvaluationEntry, Prompt
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


class AgentDefinition(BaseModel):
    name: str = Field(description="The name of the agent.")
    tool_description: str = Field(description="Tool description.")
    instructions: str = Field(description="Specific instructions for the agent.")
    tools: List[
        Literal[
            "web_search", "web_fetch", "search_articles", "get_full_text_of_pmc_article"
        ]
    ] = Field(default_factory=list, description="A list of tools the agent can use.")


class ListOfAgentDefinitions(BaseModel):
    agents: List[AgentDefinition] = Field(
        default_factory=list, description="A list of agent definitions."
    )


@dataclass
class PipelineRunContext:
    current_entry: EvaluationEntry
    report_to_fact_check: Report | None = None
    main_agent: Agent | None = None
    available_tools: list[Tool] = field(default_factory=list)
    active_mcp_servers: list[MCPServer] = field(default_factory=list)


# Helpers and tools


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
            run_config=RunConfig(
                tool_execution=ToolExecutionConfig(
                    max_function_tool_concurrency=1,
                ),
            ),
        )

        return ItemHelpers.text_message_outputs(result.new_items)

    return run_agent


def spawned_subagent_as_tool(
    subagent: Agent, tool_name: str, tool_description: str, max_turns: int = 10
) -> Tool:
    @function_tool(
        name_override=tool_name,
        description_override=tool_description,
    )
    async def run_agent(
        context: RunContextWrapper,
        input: str,
    ) -> str:
        """
        Specialist subagent tool.

        Args:
            context (RunContextWrapper): The context of the tool.
            input (str): The input to the tool.

        Returns:
            str: The output of the tool.
        """

        result = await Runner.run(
            starting_agent=subagent,
            input=input,
            context=context.context,
            max_turns=max_turns,
            error_handlers={
                "max_turns": max_turns_error_handler,
            },
            run_config=RunConfig(
                tool_execution=ToolExecutionConfig(
                    max_function_tool_concurrency=1,
                ),
            ),
        )

        return ItemHelpers.text_message_outputs(result.new_items)

    return run_agent


@function_tool(name_override="spawn_subagents")
async def spawn_subagents(
    ctx: RunContextWrapper[PipelineRunContext],
    agent_definitions: ListOfAgentDefinitions,
) -> str:
    """
    Spawn subagents based on agent definitions. Spawned subagents will be available as tools after creation.

    A set of tools will be available to each subagent. It includes:
    - web_search: run a web search and returns a list of top-k results with relevant excerpts from each page.
    - web_fetch: returns a markdown representation of the specified url.
    - search_articles: returns top_k PubMed results sorted by relevance with their complete citation and abstract.
    - get_full_text_of_pmc_article: returns the full text in markdown format for the PMCID provided.

    Args:
        ctx (RunContextWrapper): The context of the tool.
        agent_definitions (ListOfAgentDefinitions): A list of agent definitions.

    Returns:
        str: A message indicating the subagents have been spawned.
    """
    logger.debug("Tool called: spawn_subagents. Spawning subagents...")
    default_model = LitellmModel(model="gemini/gemini-3.5-flash")
    default_model_settings = ModelSettings(reasoning={"effort": "medium"})
    spawned_subagents_as_tools: List[FunctionTool] = []

    for agent_definition in agent_definitions.agents:
        subagent = Agent(
            model=default_model,
            model_settings=default_model_settings,
            instructions=agent_definition.instructions,
            name=agent_definition.name,
            mcp_servers=ctx.context.active_mcp_servers,
            tools=[web_search, extract_web_page],
        )
        spawned_subagents_as_tools.append(
            spawned_subagent_as_tool(
                subagent,
                tool_name=agent_definition.name.lower().replace(" ", "_"),
                tool_description=agent_definition.tool_description,
            )
        )

    ctx.context.main_agent.tools.extend(spawned_subagents_as_tools)

    output_msg = f"Subagents spawned: {', '.join([subagent.name for subagent in spawned_subagents_as_tools])}. You are now able to use them as tools."

    logger.debug(output_msg)

    return output_msg


# Main class
class DivideAndConquerPipelineWithSubAgentsAutoSpawning(Pipeline):
    def __init__(
        self,
        prompt_template: Prompt,
        agent_definition: dict[str, str],
        fact_checker_definition: dict[str, str],
        max_concurrency: int = 1,
        max_calls_per_minute: float = 600.0,
        max_turns_for_main_agent: int = 10,
        max_turns_for_subagents: int = 10,
        max_function_tool_concurrency: int | None = 1,
    ):
        self.model_id = agent_definition["model"].replace("/", "-")
        self.max_concurrency = max_concurrency
        self.max_calls_per_minute = max_calls_per_minute
        self.max_turns_for_main_agent = max_turns_for_main_agent
        self.max_turns_for_subagents = max_turns_for_subagents
        self.max_function_tool_concurrency = max_function_tool_concurrency

        # Use LitellmModel if model is from a provider different from openai
        if "/" in agent_definition["model"]:
            agent_definition["model"] = LitellmModel(model=agent_definition["model"])

        self.prompt_template = prompt_template
        self.agent_definition = agent_definition
        self.agent_definition["model_settings"] = ModelSettings(
            **self.agent_definition["model_settings"]
        )

        self.fact_checker_definition = fact_checker_definition
        if "/" in fact_checker_definition["model"]:
            fact_checker_definition["model"] = LitellmModel(
                model=fact_checker_definition["model"]
            )
        self.fact_checker_definition["model_settings"] = ModelSettings(
            **self.fact_checker_definition["model_settings"]
        )
        self.fact_checker_definition["instructions"] = (
            f"{RECOMMENDED_PROMPT_PREFIX}\n\n{self.fact_checker_definition['instructions']}"
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

        return servers

    def _make_agent(
        self,
        tools: List[Tool] = None,
        mcp_servers: List[MCPServer] = None,
        handoffs: List[Handoff] = None,
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
            handoffs=handoffs if handoffs is not None else [],
        )

    def _make_fact_checker_handoff(
        self,
        tools: List[Tool] = None,
        mcp_servers: List[MCPServer] = None,
    ) -> Handoff:

        async def on_handoff(
            ctx: RunContextWrapper[PipelineRunContext],
            input_data: Report,
        ):
            ctx.context.report_to_fact_check = input_data

        def fact_checker_input_filter(
            handoff_input_data: HandoffInputData,
        ) -> HandoffInputData:
            ctx = handoff_input_data.run_context

            if ctx is None or ctx.context is None:
                case_summary = "No case summary was available."
                report_text = "No report was available in the run context."
            else:
                case_summary = ctx.context.current_entry.case_summary
                report = ctx.context.report_to_fact_check
                report_text = (
                    report.text
                    if report is not None
                    else "No report was submitted through the handoff."
                )

            new_input = (
                {
                    "role": "user",
                    "content": (
                        "Please fact-check the following clinical report.\n\n"
                        "Check whether each recommendation is supported by the case summary "
                        "and by available evidence. Correct any inaccurate, unsupported, "
                        "incomplete, or unsafe statements. Return the corrected final report "
                        "using the Report schema.\n\n"
                        f"# Case summary\n\n{case_summary}\n\n"
                        f"# Report\n\n{report_text}\n"
                    ),
                },
            )

            return handoff_input_data.clone(
                input_history=new_input,
                input_items=(),
                pre_handoff_items=(),
            )

        output_type = (
            Report
            if self.model_id not in MODELS_WITHOUT_STRUCTURED_OUTPUT_SUPPORT
            else None
        )

        fact_checker_agent = Agent(
            **self.fact_checker_definition,
            tools=tools if tools is not None else [],
            mcp_servers=mcp_servers if mcp_servers is not None else [],
            output_type=output_type,
        )

        return handoff(
            fact_checker_agent,
            tool_name_override="transfer_to_fact_checker_agent",
            tool_description_override=(
                "Submit the complete draft clinical report for fact checking. "
                "The input must be a Report object containing the full markdown report "
                "in the `text` field."
            ),
            input_type=Report,
            on_handoff=on_handoff,
            input_filter=fact_checker_input_filter,
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
        concurrency_limiter: asyncio.Semaphore,
        tools: List[Tool] = None,
        active_mcp_servers: List[MCPServer] = None,
    ) -> RunResult | BaseException | None:

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

                fact_checker_handoff = self._make_fact_checker_handoff(
                    tools=[web_search, extract_web_page],
                    mcp_servers=active_mcp_servers,
                )

                agent = self._make_agent(
                    tools=tools,
                    handoffs=[fact_checker_handoff],
                )

                run_context = PipelineRunContext(
                    current_entry=entry,
                    main_agent=agent,
                    active_mcp_servers=active_mcp_servers,
                )

                result = await Runner.run(
                    agent,
                    prompt,
                    context=run_context,
                    run_config=RunConfig(
                        tool_execution=ToolExecutionConfig(
                            max_function_tool_concurrency=(
                                self.max_function_tool_concurrency
                                if self.max_function_tool_concurrency is not None
                                else 1
                            ),
                        ),
                    ),
                    max_turns=self.max_turns_for_main_agent,
                    error_handlers={
                        "max_turns": max_turns_error_handler,
                    },
                )

                return result

            except asyncio.CancelledError as exc:
                # anyio cancel scope fires with "cancel scope" in the message.
                # Uncancel and return the exception so the worker can continue to
                # the next entry. For genuine external cancellation (no "cancel scope")
                # re-raise so the pipeline stops cleanly.
                if "cancel scope" not in str(exc):
                    raise
                logger.exception(
                    f"Inference cancelled for entry {entry.prompt_id}. "
                    "Keeping the batch alive and recording this entry as failed."
                )
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
                return exc

            except BaseException as exc:
                logger.exception(f"Inference failed for entry {entry.prompt_id}")
                return exc

    async def run_pipeline(
        self,
        dataset: List[EvaluationEntry],
    ) -> List[EvaluationEntry]:
        """
        Run inference with bounded concurrency while keeping MCP servers alive
        until all worker tasks are finished, failed, or explicitly cancelled.

        Important:
        - Do not create one asyncio task per dataset entry.
        - Do not leave MCPServerManager while workers are still running.
        - Save each completed result immediately into `dataset`.
        """

        if not dataset:
            return dataset

        concurrency_limiter = asyncio.Semaphore(self.max_concurrency)
        mcp_servers = self._make_mcp_servers()

        worker_count = max(1, min(self.max_concurrency, len(dataset)))
        work_queue: asyncio.Queue[int | None] = asyncio.Queue()

        errors: list[tuple[int, BaseException]] = []

        for idx, entry in enumerate(dataset):
            if getattr(entry, "agent_response", None) is not None:
                logger.debug(f"Entry {entry.prompt_id} already completed. Skipping.")
                continue
            work_queue.put_nowait(idx)

        for _ in range(worker_count):
            work_queue.put_nowait(None)

        async with MCPServerManager(mcp_servers) as manager:

            async def worker(worker_id: int) -> None:
                while True:
                    idx = await work_queue.get()

                    try:
                        if idx is None:
                            return

                        entry = dataset[idx]

                        result = await self._run_single_task(
                            entry=entry,
                            active_mcp_servers=manager.active_servers,
                            concurrency_limiter=concurrency_limiter,
                            tools=[spawn_subagents],
                        )

                        if isinstance(result, BaseException):
                            errors.append((idx, result))
                            logger.error(
                                f"[worker={worker_id}, idx={idx}, "
                                f"prompt_id={entry.prompt_id}] "
                                f"{type(result).__name__}: {result}"
                            )
                            continue

                        if result is None:
                            continue

                        dataset[idx].agent_response = result.final_output

                        inference_token_usage: Usage = result.context_wrapper.usage
                        dataset[idx].inference_token_usage = self._usage_to_dict(
                            inference_token_usage
                        )

                        logger.debug(
                            f"[worker={worker_id}] Completed entry {entry.prompt_id}"
                        )

                    except asyncio.CancelledError as exc:
                        if idx is not None:
                            errors.append((idx, exc))
                            logger.exception(
                                f"[worker={worker_id}, idx={idx}] "
                                "Worker cancelled while processing entry."
                            )
                        raise

                    except BaseException as exc:
                        if idx is not None:
                            errors.append((idx, exc))
                            logger.exception(
                                f"[worker={worker_id}, idx={idx}] "
                                "Unexpected worker failure."
                            )

                    finally:
                        work_queue.task_done()

            workers = [
                asyncio.create_task(worker(worker_id))
                for worker_id in range(worker_count)
            ]

            try:
                # asyncio.shield prevents anyio's cancel scope (fired when the
                # MCP background streaming task gets a 429) from propagating
                # task.cancel() to the worker tasks. The main task still gets
                # CancelledError from the shield, but workers keep running.
                await asyncio.shield(asyncio.gather(*workers))

            except asyncio.CancelledError as exc:
                if "cancel scope" not in str(exc):
                    # Genuine external cancellation (Ctrl+C, explicit task.cancel()).
                    # Cancel workers and let the pipeline stop cleanly.
                    logger.warning(
                        "Pipeline was externally cancelled. Cancelling workers."
                    )
                    for task in workers:
                        task.cancel()
                    await asyncio.gather(*workers, return_exceptions=True)
                    return dataset

                logger.warning(
                    "Pipeline main task was cancelled by an MCP connection failure "
                    "(likely a rate-limit response). Workers were not cancelled — "
                    "waiting for them to finish."
                )

                # Uncancel this task so the next awaits (gather below, then
                # MCPServerManager cleanup) don't keep raising CancelledError.
                # anyio's task.cancel() increments _must_cancel; without uncancel()
                # the next await would immediately re-raise CancelledError.
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()

                # Workers were shielded and are still running — just wait.
                worker_results = await asyncio.gather(
                    *workers,
                    return_exceptions=True,
                )

                for result in worker_results:
                    if isinstance(result, BaseException) and not isinstance(
                        result, asyncio.CancelledError
                    ):
                        logger.error(
                            f"Worker shutdown result: {type(result).__name__}: {result}"
                        )

            except BaseException as exc:
                logger.exception(
                    "Unexpected pipeline-level failure. Cancelling all workers before "
                    "leaving MCPServerManager."
                )

                for task in workers:
                    task.cancel()

                await asyncio.gather(
                    *workers,
                    return_exceptions=True,
                )

                errors.append((-1, exc))

        if errors:
            logger.error(f"{len(errors)} inference tasks failed or were cancelled.")
            for idx, error in errors:
                if idx >= 0:
                    logger.error(
                        f"[idx={idx}, prompt_id={dataset[idx].prompt_id}] "
                        f"{type(error).__name__}: {error}"
                    )
                else:
                    logger.error(f"[idx=unknown] {type(error).__name__}: {error}")

        return dataset
