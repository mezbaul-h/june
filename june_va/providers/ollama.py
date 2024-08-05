"""
This module provides a class for interacting with a Language Model (LLM) using the ollama library.
"""

import abc
import json
from typing import Dict, Iterator, List, Optional

import requests
from pydantic import BaseModel, ConfigDict

from .common import BaseLLMModel, LLMMessage


class OllamaLLMSettings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    server_url: Optional[str] = None


class OllamaLLM(BaseLLMModel):
    def __init__(self, user_settings: OllamaLLMSettings) -> None:
        self.user_settings = user_settings
        self.chat_url = (user_settings.server_url or "http://localhost:11434").rstrip("/") + "/api/chat"

    def forward(self, messages: List[LLMMessage]) -> Iterator[LLMMessage]:
        res = requests.post(
            self.chat_url,
            json={
                "messages": [item.model_dump() for item in messages],
                "model": self.user_settings.model,
                "stream": True,
            },
            stream=True,
        )

        res.raise_for_status()

        for line in res.iter_lines():
            yield LLMMessage(**json.loads(line)["message"])
