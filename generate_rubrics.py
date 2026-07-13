import asyncio
import os
from typing import List

from agents.mcp import MCPServer, MCPServerManager, MCPServerStdio, ToolFilterStatic
import jsonlines
from pathlib import Path
import yaml
from agents import (
    Agent,
    ModelSettings,
    RunConfig,
    RunResult,
    Runner,
    Tool,
    ToolExecutionConfig,
)
from agents.extensions.models.litellm_model import LitellmModel

import dotenv

from custom_error_handlers import max_turns_error_handler
from logs import get_logger
import data_models
from mcp_utils import CustomMCPServerStreamableHttp

logger = get_logger(__name__)
dotenv.load_dotenv()

# Define output path
output_path = Path("data/eval_dataset.jsonl")

# Load data
case_summaries_path = Path("data/case_summaries.jsonl")
case_summaries_data = []
with jsonlines.open(case_summaries_path, "r") as reader:
    for obj in reader:
        case_summaries_data.append(obj)
logger.info(f"Loaded {len(case_summaries_data)} case summaries")

# Load agent definitions
agent_key = "rubric_generator"
agent_definitions_path = Path("agent_definitions.yaml")
with open(agent_definitions_path, "r", encoding="utf-8") as f:
    agent_definitions = yaml.safe_load(f)

agent_definitions[agent_key]["output_type"] = getattr(
    data_models, agent_definitions[agent_key]["output_type"]
)
agent_definitions[agent_key]["model_settings"] = ModelSettings(
    **agent_definitions[agent_key]["model_settings"]
)

# Use LitellmModel if model is from a provider different from openai
if "/" in agent_definitions[agent_key]["model"]:
    agent_definitions[agent_key]["model"] = LitellmModel(
        model=agent_definitions[agent_key]["model"]
    )

# Define concurrency limiter
concurrency_limiter = asyncio.Semaphore(1)
max_turns_for_main_agent = 10


def make_mcp_servers(max_calls_per_minute: int = 600) -> List[MCPServer]:
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
            max_calls_per_minute=max_calls_per_minute,
        )
    )
    return servers


def make_agent(
    agent_definition: dict,
    tools: List[Tool] = None,
    mcp_servers: List[MCPServer] = None,
) -> Agent:
    return Agent(
        **agent_definition,
        tools=tools if tools is not None else [],
        mcp_servers=mcp_servers if mcp_servers is not None else [],
    )


async def run_single_task(agent, prompt) -> RunResult:
    async with concurrency_limiter:
        try:
            return await Runner.run(
                agent,
                prompt,
                run_config=RunConfig(
                    tool_execution=ToolExecutionConfig(
                        max_function_tool_concurrency=1,
                    ),
                ),
                max_turns=max_turns_for_main_agent,
                error_handlers={
                    "max_turns": max_turns_error_handler,
                },
            )

        except asyncio.CancelledError:
            logger.exception("Pipeline cancelled. Aborting.")
            raise

        except Exception as exc:
            logger.exception("Inference failed")
            raise exc


# Define main function
async def main() -> None:
    mcp_servers = make_mcp_servers()
    async with MCPServerManager(mcp_servers) as manager:
        tasks = []
        for case_data in case_summaries_data:
            # Define prompt
            prompt = f"# Case summary\n{case_data['case_summary']}\n\n# Official recommmendations\n{case_data['official_recommendations']}"

            # Instantiate agent
            agent = make_agent(
                agent_definitions[agent_key], mcp_servers=manager.active_servers
            )
            # Queue task
            tasks.append(run_single_task(agent, prompt))

        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        with jsonlines.open(output_path, "w") as writer:
            for result, case_data in zip(results, case_summaries_data):
                prompt = [
                    {
                        "role": "user",
                        "content": f"# Case summary\n{case_data['case_summary']}",
                    }
                ]

                writer.write(
                    {
                        "prompt_id": case_data["id"],
                        "prompt": prompt,
                        "case_summary": case_data["case_summary"],
                        "official_recommendations": case_data[
                            "official_recommendations"
                        ],
                        "breast_cancer_stage": case_data["breast_cancer_stage"],
                        "filename": case_data["filename"],
                        **result.final_output.model_dump(),
                    }
                )

        logger.info(f"Saved {len(tasks)} prompts to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
