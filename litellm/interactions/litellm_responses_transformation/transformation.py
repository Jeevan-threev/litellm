"""
Transformation utilities for bridging Interactions API to Responses API.

This module handles transforming between:
- Interactions API format (Google's format with Turn[], system_instruction, etc.)
- Responses API format (OpenAI's format with input[], instructions, etc.)
"""

import base64
from typing import Any, Final, cast

from litellm.types.interactions import (
    InteractionInput,
    InteractionsAPIOptionalRequestParams,
    InteractionsAPIResponse,
    Turn,
)
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIResponse,
)

_IMAGE_MAGIC_BYTES: Final[tuple[tuple[bytes, str], ...]] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


class LiteLLMResponsesInteractionsConfig:
    """Configuration class for transforming between Interactions API and Responses API."""

    @staticmethod
    def transform_interactions_request_to_responses_request(
        model: str,
        input: InteractionInput | None,
        optional_params: InteractionsAPIOptionalRequestParams,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Transform an Interactions API request to a Responses API request.

        Key transformations:
        - system_instruction -> instructions
        - input (string | Turn[]) -> input (ResponseInputParam)
        - tools -> tools (similar format)
        - generation_config -> temperature, top_p, etc.
        """
        responses_request: Final[dict[str, Any]] = {
            "model": model,
        }

        # Transform input
        if input is not None:
            responses_request["input"] = (
                LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(input)
            )

        # Transform system_instruction -> instructions
        if optional_params.get("system_instruction"):
            responses_request["instructions"] = optional_params["system_instruction"]

        # Transform tools (similar format, pass through for now)
        if optional_params.get("tools"):
            responses_request["tools"] = optional_params["tools"]

        # Transform generation_config to temperature, top_p, etc.
        generation_config: Final = optional_params.get("generation_config")
        if generation_config:
            if isinstance(generation_config, dict):
                if "temperature" in generation_config:
                    responses_request["temperature"] = generation_config["temperature"]
                if "top_p" in generation_config:
                    responses_request["top_p"] = generation_config["top_p"]
                if "top_k" in generation_config:
                    # Responses API doesn't have top_k, skip it
                    pass
                if "max_output_tokens" in generation_config:
                    responses_request["max_output_tokens"] = generation_config["max_output_tokens"]

        # Pass through other optional params that match
        passthrough_params: Final = ["stream", "store", "metadata", "user"]
        for param in passthrough_params:
            if param in optional_params and optional_params[param] is not None:
                responses_request[param] = optional_params[param]

        # Add any extra kwargs
        responses_request.update(kwargs)

        return responses_request

    @staticmethod
    def _transform_interactions_input_to_responses_input(
        input: InteractionInput,
    ) -> ResponseInputParam:
        """
        Transform Interactions API input to Responses API input format.

        Interactions API input can be:
        - string: "Hello"
        - Turn[]: [{"role": "user", "content": [...]}]
        - Content object

        Responses API input is:
        - string: "Hello"
        - Message[]: [{"role": "user", "content": [...]}]
        """
        if isinstance(input, str):
            # ResponseInputParam accepts str
            return cast(ResponseInputParam, input)

        if isinstance(input, list):
            # Turn[] format - convert to Responses API Message[] format
            messages: Final = []
            for turn in input:
                if isinstance(turn, dict):
                    role = turn.get("role", "user")
                    content = turn.get("content", [])

                    # Transform content array
                    transformed_content = LiteLLMResponsesInteractionsConfig._transform_content_array(content)

                    messages.append(
                        {
                            "role": role,
                            "content": transformed_content,
                        }
                    )
                elif isinstance(turn, Turn):
                    # Pydantic model
                    role = turn.role if hasattr(turn, "role") else "user"
                    content = turn.content if hasattr(turn, "content") else []

                    # Ensure content is a list for _transform_content_array
                    # Cast to List[Any] to handle various content types
                    if isinstance(content, list):
                        content_list: list[Any] = list(content)
                    elif content is not None:
                        content_list = [content]
                    else:
                        content_list = []

                    transformed_content = LiteLLMResponsesInteractionsConfig._transform_content_array(content_list)

                    messages.append(
                        {
                            "role": role,
                            "content": transformed_content,
                        }
                    )

            return cast(ResponseInputParam, messages)

        # Single content object - wrap in message
        if isinstance(input, dict):
            return cast(
                ResponseInputParam,
                [
                    {
                        "role": "user",
                        "content": LiteLLMResponsesInteractionsConfig._transform_content_array(
                            input.get("content", []) if isinstance(input.get("content"), list) else [input]
                        ),
                    }
                ],
            )

        # Fallback: convert to string
        return cast(ResponseInputParam, str(input))

    @staticmethod
    def _transform_content_array(content: list[Any]) -> list[dict[str, Any]]:
        """Transform Interactions API content array to Responses API format."""
        if not isinstance(content, list):
            # Single content item - wrap in array
            content = [content]

        transformed: Final[list[dict[str, Any]]] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "image":
                    transformed.append(LiteLLMResponsesInteractionsConfig._transform_image_content_item(item))
                    continue
                # Already in dict format, pass through
                transformed.append(item)
            elif isinstance(item, str):
                # Plain string - wrap in text format
                transformed.append({"type": "text", "text": item})
            else:
                # Pydantic model or other - convert to dict
                if hasattr(item, "model_dump"):
                    dumped = item.model_dump()
                    if isinstance(dumped, dict):
                        transformed.append(dumped)
                    else:
                        # Fallback: wrap in text format
                        transformed.append({"type": "text", "text": str(dumped)})
                elif hasattr(item, "dict"):
                    dumped = item.dict()
                    if isinstance(dumped, dict):
                        transformed.append(dumped)
                    else:
                        # Fallback: wrap in text format
                        transformed.append({"type": "text", "text": str(dumped)})
                else:
                    # Fallback: wrap in text format
                    transformed.append({"type": "text", "text": str(item)})

        return transformed

    @staticmethod
    def _transform_image_content_item(item: dict[str, Any]) -> dict[str, Any]:
        """
        Gemini's Interactions API image part ({"type": "image", "data": <base64>,
        "mime_type": ...} or {"uri": ...}) is not a Responses API content type
        (input_text/input_image/input_file) and is silently dropped downstream if
        passed through unchanged. Map it to a Responses API `input_image` part.
        """
        uri = item.get("uri")
        if isinstance(uri, str) and uri:
            return {"type": "input_image", "image_url": uri}

        data = item.get("data")
        if isinstance(data, str) and data:
            mime_type = (
                item.get("mime_type")
                or LiteLLMResponsesInteractionsConfig._sniff_image_mime_type(data)
                or "application/octet-stream"
            )
            return {"type": "input_image", "image_url": f"data:{mime_type};base64,{data}"}

        return item

    @staticmethod
    def _sniff_image_mime_type(data: str) -> str | None:
        prefix_len = (min(len(data), 24) // 4) * 4
        if prefix_len == 0:
            return None
        try:
            prefix = base64.b64decode(data[:prefix_len])
        except ValueError:
            return None
        for magic, mime in _IMAGE_MAGIC_BYTES:
            if prefix.startswith(magic):
                return mime
        if prefix[:4] == b"RIFF" and prefix[8:12] == b"WEBP":
            return "image/webp"
        return None

    @staticmethod
    def transform_responses_response_to_interactions_response(
        responses_response: ResponsesAPIResponse,
        model: str | None = None,
    ) -> InteractionsAPIResponse:
        """
        Transform a Responses API response to an Interactions API response.

        Key transformations:
        - Extract text from output[].content[].text
        - Convert created_at (int) to created (ISO string)
        - Map status
        - Extract usage
        """
        # Extract text from outputs and build both `outputs` (legacy) and `steps` (new schema).
        outputs: Final[list[dict[str, Any]]] = []
        steps: Final[list[dict[str, Any]]] = []
        if hasattr(responses_response, "output") and responses_response.output:
            for output_item in responses_response.output:
                # Use getattr with None default to safely access content
                content = getattr(output_item, "content", None)
                if content is not None:
                    content_items = content if isinstance(content, list) else [content]
                    model_output_contents: list[dict[str, Any]] = []
                    for content_item in content_items:
                        # Check if content_item has text attribute
                        text = getattr(content_item, "text", None)
                        if text is not None:
                            # Use independent dict instances so mutations to one
                            # of `outputs` / `steps` don't leak into the other.
                            outputs.append({"type": "text", "text": text})
                            model_output_contents.append({"type": "text", "text": text})
                        elif isinstance(content_item, dict) and content_item.get("type") == "text":
                            outputs.append({**content_item})
                            model_output_contents.append({**content_item})
                    if model_output_contents:
                        steps.append(
                            {
                                "type": "model_output",
                                "content": model_output_contents,
                            }
                        )

        # Convert created_at to ISO string
        created_at: Final = getattr(responses_response, "created_at", None)
        if isinstance(created_at, int):
            from datetime import datetime

            created = datetime.fromtimestamp(created_at).isoformat()
        elif created_at is not None and hasattr(created_at, "isoformat"):
            created = created_at.isoformat()
        else:
            created = None

        # Map status
        status: Final = getattr(responses_response, "status", "completed")
        if status == "completed":
            interactions_status = "completed"
        elif status == "in_progress":
            interactions_status = "in_progress"
        else:
            interactions_status = status

        # Build interactions response — populate both `outputs` (legacy schema) and
        # `steps` (new schema) so callers work regardless of which schema they expect.
        interactions_response_dict: Final[dict[str, Any]] = {
            "id": getattr(responses_response, "id", ""),
            "object": "interaction",
            "status": interactions_status,
            "outputs": outputs,
            "steps": steps,
            "model": model or getattr(responses_response, "model", ""),
            "created": created,
        }

        # Add usage if available
        # Map Responses API usage (input_tokens, output_tokens) to Interactions API spec format
        # (total_input_tokens, total_output_tokens)
        usage: Final = getattr(responses_response, "usage", None)
        if usage:
            interactions_response_dict["usage"] = {
                "total_input_tokens": getattr(usage, "input_tokens", 0),
                "total_output_tokens": getattr(usage, "output_tokens", 0),
            }

        # Add updated (same as created for now)
        interactions_response_dict["updated"] = created

        return InteractionsAPIResponse(**interactions_response_dict)
