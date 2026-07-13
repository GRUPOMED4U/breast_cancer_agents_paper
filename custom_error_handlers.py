import asyncio
import copy

from agents import (
    ModelBehaviorError,
    RunErrorHandlerInput,
    RunErrorHandlerResult,
    Runner,
)
from logs import get_logger

logger = get_logger(__name__, level="DEBUG")


async def max_turns_error_handler(
    handler_input: RunErrorHandlerInput,
) -> RunErrorHandlerResult:
    logger.debug("MaxTurnsExceeded: max_turns_error_handler called")

    run_data = handler_input.run_data
    agent = copy.copy(run_data.last_agent)
    agent.tools = []
    agent.mcp_servers = []
    input = run_data.history

    max_retries = 3
    n_retries = 0

    while n_retries <= max_retries:
        try:
            input.append(
                {
                    "role": "user",
                    "content": "You have reached the maximum number of turns allowed. Stop making tool calls and return a final answer. You won't be able to make any more tool calls before generating your final response.",
                }
            )
            result = await Runner.run(
                agent,
                input,
                context=handler_input.context.context,
                max_turns=1,
                error_handlers=None,
            )
            return RunErrorHandlerResult(
                final_output=result.final_output, include_in_history=True
            )

        except ModelBehaviorError as exc:
            n_retries += 1
            logger.exception(
                f"ModelBehaviorError during max_turns_error_handler execution. Retry number {n_retries}/{max_retries}. See: {exc}"
            )
            continue

        except asyncio.CancelledError:
            logger.exception(
                "Pipeline cancelled during max_turns_error_handler execution. Aborting."
            )
            raise

        except Exception as exc:
            logger.exception(
                "Pipeline cancelled during max_turns_error_handler execution."
            )
            raise exc

    raise Exception(
        "Maximum number of retries exceeded for max_turns_error_handler execution."
    )
