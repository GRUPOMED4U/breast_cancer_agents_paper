from typing import List

from data_models import Message, Prompt


def format_prompt(
    prompt_template: Prompt | list[dict[str, str]], kwargs: dict[str, str]
) -> Prompt:
    if isinstance(prompt_template, List):
        prompt_template = [Message(**message) for message in prompt_template]

    formatted_prompt: Prompt = []

    used_kwargs = set()

    for message in prompt_template:
        current_message_kwargs = {
            k: v for k, v in kwargs.items() if f"{{{k}}}" in message.content
        }
        formatted_prompt.append(
            Message(
                role=message.role,
                content=message.content.format(**current_message_kwargs),
            )
        )
        used_kwargs.update(current_message_kwargs.keys())

    if len(used_kwargs) != len(kwargs):
        raise ValueError(f"Missing required kwargs: {set(kwargs.keys()) - used_kwargs}")

    formatted_prompt = [msg.model_dump() for msg in formatted_prompt]

    return formatted_prompt
