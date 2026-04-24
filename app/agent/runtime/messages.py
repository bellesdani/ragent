from __future__ import annotations

from app.api.schemas.openai import ChatMessage
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart


def split_messages(messages: list[ChatMessage], instructions: str) -> tuple[str, list[ModelMessage]]:
    latest_user_index = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user" and messages[index].content:
            latest_user_index = index
            break
    if latest_user_index is None:
        raise ValueError("user message content is required")

    latest_user_prompt = messages[latest_user_index].content or ""
    history = messages[:latest_user_index]
    extra_instructions = "\n\n".join(
        message.content
        for message in history
        if message.role == "system" and message.content
    )
    merged_instructions = merge_instructions(instructions, extra_instructions)
    model_messages: list[ModelMessage] = []
    first_request = True

    for message in history:
        if not message.content:
            continue
        if message.role == "system":
            continue
        if message.role == "assistant":
            model_messages.append(ModelResponse(parts=[TextPart(content=message.content)]))
            continue
        if message.role == "user":
            request_instructions = None
            if first_request:
                request_instructions = merged_instructions
            model_messages.append(
                ModelRequest.user_text_prompt(
                    message.content,
                    instructions=request_instructions,
                )
            )
            first_request = False

    if not model_messages:
        if merged_instructions != instructions:
            return latest_user_prompt, [ModelRequest(parts=[], instructions=merged_instructions)]
        return latest_user_prompt, []

    first_message = model_messages[0]
    if isinstance(first_message, ModelRequest):
        if not first_message.instructions:
            first_message.instructions = merged_instructions
    else:
        model_messages.insert(0, ModelRequest(parts=[], instructions=merged_instructions))
    return latest_user_prompt, model_messages


def merge_instructions(base_instructions: str, extra_instruction: str) -> str:
    extra_instruction = extra_instruction.strip()
    if not extra_instruction:
        return base_instructions
    return f"{base_instructions}\n\n{extra_instruction}"
