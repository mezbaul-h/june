"""
This module provides a class for interacting with cloud LLM providers using the OpenAI-compatible API.

Supports MiniMax, OpenAI, and other providers that implement the OpenAI chat completions API.
"""

import os
from typing import Dict, Iterator, List, Optional

from .common import BaseModel


class CloudLLM(BaseModel):
    """
    A class for interacting with cloud LLM providers via OpenAI-compatible API.

    This class inherits from the BaseModel class and provides methods for generating text
    using cloud-hosted language models such as MiniMax and OpenAI.

    Args:
        **kwargs: Keyword arguments for initializing the CloudLLM, including optional arguments
            like 'system_prompt', 'disable_chat_history', 'api_key', 'base_url', and 'temperature'.

    Attributes:
        messages: A list of dictionaries representing the conversation history.
        system_prompt: An optional system prompt to provide context for the conversation.
        is_chat_history_disabled: A flag indicating whether the chat history should be disabled.
        client: An instance of the OpenAI client for interacting with the cloud LLM.
        temperature: The temperature parameter for generation (clamped to (0, 1] for MiniMax).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.messages: List[Dict[str, str]] = []

        self.system_prompt: Optional[str] = kwargs.get("system_prompt")

        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

        self.is_chat_history_disabled: Optional[bool] = kwargs.get("disable_chat_history")

        self.api_key: str = kwargs.get("api_key") or os.environ.get("MINIMAX_API_KEY", "") or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        self.base_url: str = kwargs.get("base_url", "https://api.minimax.io/v1")
        self.temperature: float = kwargs.get("temperature", 0.7)

        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def exists(self) -> bool:
        """
        Check if the cloud LLM provider is configured with a valid API key.

        Returns:
            True if an API key is available, False otherwise.
        """
        return bool(self.api_key)

    def forward(self, message: str) -> Iterator[str]:
        """
        Generate text from user input using the cloud LLM.

        Args:
            message: The user input message.

        Returns:
            An iterator that yields the generated text in chunks.
        """
        self.messages.append({"role": "user", "content": message})

        # Clamp temperature to (0, 1] for MiniMax compatibility
        temperature = max(0.01, min(1.0, self.temperature))

        stream = self.client.chat.completions.create(
            model=self.model_id,
            messages=self.messages,
            stream=True,
            temperature=temperature,
        )

        generated_content = ""

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                generated_content += token
                yield token

        if self.is_chat_history_disabled:
            self.messages.pop()
        else:
            self.messages.append({"role": "assistant", "content": generated_content})
