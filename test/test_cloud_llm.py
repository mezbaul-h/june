"""Unit tests for the CloudLLM provider."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestCloudLLMInit:
    """Tests for CloudLLM initialization."""

    @patch("openai.OpenAI")
    def test_init_with_api_key_kwarg(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        llm = CloudLLM(model="MiniMax-M2.7", api_key="test-key-123")
        assert llm.api_key == "test-key-123"
        assert llm.model_id == "MiniMax-M2.7"
        assert llm.base_url == "https://api.minimax.io/v1"
        mock_openai_cls.assert_called_once_with(api_key="test-key-123", base_url="https://api.minimax.io/v1")

    @patch("openai.OpenAI")
    def test_init_with_env_minimax_key(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-minimax-key"}, clear=False):
            llm = CloudLLM(model="MiniMax-M2.7")
            assert llm.api_key == "env-minimax-key"

    @patch("openai.OpenAI")
    def test_init_with_custom_base_url(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        llm = CloudLLM(model="gpt-4", api_key="key", base_url="https://api.openai.com/v1")
        assert llm.base_url == "https://api.openai.com/v1"
        mock_openai_cls.assert_called_once_with(api_key="key", base_url="https://api.openai.com/v1")

    @patch("openai.OpenAI")
    def test_init_with_system_prompt(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        llm = CloudLLM(model="MiniMax-M2.7", api_key="key", system_prompt="You are a helpful assistant.")
        assert llm.system_prompt == "You are a helpful assistant."
        assert len(llm.messages) == 1
        assert llm.messages[0] == {"role": "system", "content": "You are a helpful assistant."}

    @patch("openai.OpenAI")
    def test_init_without_system_prompt(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        llm = CloudLLM(model="MiniMax-M2.7", api_key="key")
        assert llm.system_prompt is None
        assert len(llm.messages) == 0

    @patch("openai.OpenAI")
    def test_init_default_temperature(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        llm = CloudLLM(model="MiniMax-M2.7", api_key="key")
        assert llm.temperature == 0.7

    @patch("openai.OpenAI")
    def test_init_custom_temperature(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        llm = CloudLLM(model="MiniMax-M2.7", api_key="key", temperature=0.3)
        assert llm.temperature == 0.3

    @patch("openai.OpenAI")
    def test_init_disable_chat_history(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        llm = CloudLLM(model="MiniMax-M2.7", api_key="key", disable_chat_history=True)
        assert llm.is_chat_history_disabled is True


class TestCloudLLMExists:
    """Tests for CloudLLM.exists() method."""

    @patch("openai.OpenAI")
    def test_exists_with_api_key(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        llm = CloudLLM(model="MiniMax-M2.7", api_key="valid-key")
        assert llm.exists() is True

    @patch("openai.OpenAI")
    def test_exists_without_api_key(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        with patch.dict(os.environ, {}, clear=True):
            llm = CloudLLM(model="MiniMax-M2.7")
            assert llm.exists() is False


class TestCloudLLMForward:
    """Tests for CloudLLM.forward() method."""

    def _make_chunk(self, content=None):
        """Create a mock streaming chunk."""
        chunk = MagicMock()
        if content is not None:
            choice = MagicMock()
            choice.delta.content = content
            chunk.choices = [choice]
        else:
            chunk.choices = []
        return chunk

    @patch("openai.OpenAI")
    def test_forward_basic(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = [
            self._make_chunk("Hello"),
            self._make_chunk(" world"),
            self._make_chunk("!"),
        ]

        llm = CloudLLM(model="MiniMax-M2.7", api_key="key")
        result = list(llm.forward("Hi"))

        assert result == ["Hello", " world", "!"]
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "MiniMax-M2.7"
        assert call_kwargs["stream"] is True

    @patch("openai.OpenAI")
    def test_forward_stores_history(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = [self._make_chunk("Response")]

        llm = CloudLLM(model="MiniMax-M2.7", api_key="key")
        list(llm.forward("Hello"))

        assert len(llm.messages) == 2
        assert llm.messages[0] == {"role": "user", "content": "Hello"}
        assert llm.messages[1] == {"role": "assistant", "content": "Response"}

    @patch("openai.OpenAI")
    def test_forward_disabled_history(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = [self._make_chunk("Response")]

        llm = CloudLLM(model="MiniMax-M2.7", api_key="key", disable_chat_history=True)
        list(llm.forward("Hello"))

        assert len(llm.messages) == 0

    @patch("openai.OpenAI")
    def test_forward_temperature_clamping_low(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = [self._make_chunk("ok")]

        llm = CloudLLM(model="MiniMax-M2.7", api_key="key", temperature=0.0)
        list(llm.forward("test"))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.01

    @patch("openai.OpenAI")
    def test_forward_temperature_clamping_high(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = [self._make_chunk("ok")]

        llm = CloudLLM(model="MiniMax-M2.7", api_key="key", temperature=2.0)
        list(llm.forward("test"))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 1.0

    @patch("openai.OpenAI")
    def test_forward_skips_empty_chunks(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = [
            self._make_chunk("Hello"),
            self._make_chunk(None),
            self._make_chunk(),  # empty choices
            self._make_chunk(" there"),
        ]

        llm = CloudLLM(model="MiniMax-M2.7", api_key="key")
        result = list(llm.forward("Hi"))

        assert result == ["Hello", " there"]

    @patch("openai.OpenAI")
    def test_forward_with_system_prompt(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = [self._make_chunk("Hi")]

        llm = CloudLLM(model="MiniMax-M2.7", api_key="key", system_prompt="Be helpful")
        list(llm.forward("Hello"))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "Be helpful"}
        assert messages[1] == {"role": "user", "content": "Hello"}

    @patch("openai.OpenAI")
    def test_forward_multi_turn(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_client.chat.completions.create.return_value = [self._make_chunk("First reply")]
        llm = CloudLLM(model="MiniMax-M2.7", api_key="key")
        list(llm.forward("Turn 1"))

        mock_client.chat.completions.create.return_value = [self._make_chunk("Second reply")]
        list(llm.forward("Turn 2"))

        assert len(llm.messages) == 4
        assert llm.messages[0]["content"] == "Turn 1"
        assert llm.messages[1]["content"] == "First reply"
        assert llm.messages[2]["content"] == "Turn 2"
        assert llm.messages[3]["content"] == "Second reply"


class TestCloudLLMModels:
    """Tests for MiniMax model configurations."""

    @patch("openai.OpenAI")
    def test_minimax_m27(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        llm = CloudLLM(model="MiniMax-M2.7", api_key="key")
        assert llm.model_id == "MiniMax-M2.7"

    @patch("openai.OpenAI")
    def test_minimax_m27_highspeed(self, mock_openai_cls):
        from june_va.models.cloud_llm import CloudLLM

        llm = CloudLLM(model="MiniMax-M2.7-highspeed", api_key="key")
        assert llm.model_id == "MiniMax-M2.7-highspeed"
