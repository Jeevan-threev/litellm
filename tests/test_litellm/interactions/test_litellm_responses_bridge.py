"""
Tests for LiteLLM Responses bridge provider.

Inherits from BaseInteractionsTest to run the same test suite against
the litellm_responses bridge provider, which calls litellm.responses() internally.
"""

import base64
import os

from litellm.interactions.litellm_responses_transformation.transformation import (
    LiteLLMResponsesInteractionsConfig,
)
from tests.test_litellm.interactions.base_interactions_test import (
    BaseInteractionsTest,
)


class TestLiteLLMResponsesBridge(BaseInteractionsTest):
    """Test LiteLLM Responses bridge using the base test suite."""

    def get_model(self) -> str:
        """Return the model string for the bridge provider.

        The bridge provider uses litellm.responses() internally, so we can
        use any model that litellm.responses() supports (e.g., gpt-4o).
        """
        return "gpt-4o"

    def get_api_key(self) -> str:
        """Return the OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY", "")


class TestBridgeImageContentTransformation:
    """Regression tests for GH issue #41427: Gemini image content parts were
    silently dropped by the litellm_proxy/Responses bridge because they kept
    their native "image" type tag, which the Responses API doesn't recognize."""

    def test_image_content_with_mime_type_is_transformed_to_input_image(self):
        image_part = {"type": "image", "data": "base64data", "mime_type": "image/png"}
        transformed = LiteLLMResponsesInteractionsConfig._transform_content_array([image_part])
        assert transformed == [{"type": "input_image", "image_url": "data:image/png;base64,base64data"}]

    def test_image_content_with_uri_is_transformed_to_input_image(self):
        image_part = {"type": "image", "uri": "https://example.com/cat.jpg"}
        transformed = LiteLLMResponsesInteractionsConfig._transform_content_array([image_part])
        assert transformed == [{"type": "input_image", "image_url": "https://example.com/cat.jpg"}]

    def test_image_content_missing_mime_type_is_sniffed_from_data(self):
        png_signature_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"rest-of-file").decode()
        image_part = {"type": "image", "data": png_signature_b64}
        transformed = LiteLLMResponsesInteractionsConfig._transform_content_array([image_part])
        assert transformed == [{"type": "input_image", "image_url": f"data:image/png;base64,{png_signature_b64}"}]

    def test_image_content_webp_signature_is_sniffed_from_data(self):
        webp_signature_b64 = base64.b64encode(b"RIFF\x00\x00\x00\x00WEBPrest-of-file").decode()
        image_part = {"type": "image", "data": webp_signature_b64}
        transformed = LiteLLMResponsesInteractionsConfig._transform_content_array([image_part])
        assert transformed == [{"type": "input_image", "image_url": f"data:image/webp;base64,{webp_signature_b64}"}]

    def test_image_content_missing_mime_type_and_unrecognized_data_defaults_to_octet_stream(self):
        image_part = {"type": "image", "data": "not-a-real-image-signature"}
        transformed = LiteLLMResponsesInteractionsConfig._transform_content_array([image_part])
        assert transformed[0]["image_url"].startswith("data:application/octet-stream;base64,")

    def test_image_content_with_undecodable_data_defaults_to_octet_stream(self):
        image_part = {"type": "image", "data": "a" + "!" * 23}
        transformed = LiteLLMResponsesInteractionsConfig._transform_content_array([image_part])
        assert transformed[0]["image_url"].startswith("data:application/octet-stream;base64,")

    def test_image_content_with_too_short_data_defaults_to_octet_stream(self):
        image_part = {"type": "image", "data": "ab"}
        transformed = LiteLLMResponsesInteractionsConfig._transform_content_array([image_part])
        assert transformed == [{"type": "input_image", "image_url": "data:application/octet-stream;base64,ab"}]

    def test_image_content_without_data_or_uri_passes_through_unchanged(self):
        image_part = {"type": "image", "mime_type": "image/png"}
        transformed = LiteLLMResponsesInteractionsConfig._transform_content_array([image_part])
        assert transformed == [image_part]

    def test_text_content_passes_through_unchanged(self):
        text_part = {"type": "text", "text": "hello"}
        transformed = LiteLLMResponsesInteractionsConfig._transform_content_array([text_part])
        assert transformed == [text_part]
