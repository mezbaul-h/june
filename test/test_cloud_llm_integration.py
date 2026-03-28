"""Integration tests for the CloudLLM provider.

These tests verify the integration with the MiniMax API.
They require a valid MINIMAX_API_KEY environment variable to be set.

Run with: pytest test/test_cloud_llm_integration.py -v
"""

import os

import pytest

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
skip_no_key = pytest.mark.skipif(not MINIMAX_API_KEY, reason="MINIMAX_API_KEY not set")


@skip_no_key
class TestCloudLLMIntegration:
    """Integration tests that call the real MiniMax API."""

    def test_minimax_streaming_response(self):
        from june_va.models.cloud_llm import CloudLLM

        llm = CloudLLM(
            model="MiniMax-M2.7-highspeed",
            api_key=MINIMAX_API_KEY,
            base_url="https://api.minimax.io/v1",
            temperature=0.7,
        )

        assert llm.exists() is True

        tokens = list(llm.forward("Reply with exactly one word: hello"))
        full_response = "".join(tokens)

        assert len(full_response) > 0
        assert len(llm.messages) == 2
        assert llm.messages[0]["role"] == "user"
        assert llm.messages[1]["role"] == "assistant"

    def test_minimax_with_system_prompt(self):
        from june_va.models.cloud_llm import CloudLLM

        llm = CloudLLM(
            model="MiniMax-M2.7-highspeed",
            api_key=MINIMAX_API_KEY,
            system_prompt="You are a helpful assistant. Reply concisely.",
            temperature=0.5,
        )

        tokens = list(llm.forward("What is 2+2? Reply with just the number."))
        full_response = "".join(tokens)

        assert len(full_response) > 0
        assert len(llm.messages) == 3  # system + user + assistant

    def test_minimax_disabled_history(self):
        from june_va.models.cloud_llm import CloudLLM

        llm = CloudLLM(
            model="MiniMax-M2.7-highspeed",
            api_key=MINIMAX_API_KEY,
            disable_chat_history=True,
            temperature=0.5,
        )

        tokens = list(llm.forward("Say hi"))
        assert len("".join(tokens)) > 0
        assert len(llm.messages) == 0  # history cleared
