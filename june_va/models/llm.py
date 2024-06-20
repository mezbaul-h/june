"""
This module provides a class for interacting with a Language Model (LLM) using the ollama library.
"""

from typing import Dict, Iterator, List, Optional

from ollama import Client, ResponseError

from .common import BaseModel


class LLM(BaseModel):
    """
    A class for interacting with a Language Model (LLM) using the ollama library.

    This class inherits from the BaseModel class and provides methods for checking if a model exists,
    and generating text from user input using the specified LLM.

    Args:
        **kwargs: Keyword arguments for initializing the LLM, including optional arguments
            like 'system_prompt' and 'disable_chat_history'.

    Attributes:
        messages: A list of dictionaries representing the conversation history,
            with each dictionary containing a 'role' (e.g., 'system', 'user', 'assistant') and 'content' keys.
        system_prompt: An optional system prompt to provide context for the conversation.
        is_chat_history_disabled: A flag indicating whether the chat history should be disabled.
        model: An instance of the ollama.Client for interacting with the LLM.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.messages: List[Dict[str, str]] = []

        self.system_prompt: Optional[str] = kwargs.get("system_prompt")

        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

        self.is_chat_history_disabled: Optional[bool] = kwargs.get("disable_chat_history")

        self.model = Client()

    def exists(self) -> bool:
        """
        Check if the specified LLM model exists.

        Returns:
            True if the model exists, False otherwise.
        """
        try:
            # Assert ollama model validity
            _ = self.model.show(self.model_id)

            return True
        except ResponseError:
            return False

    def forward(self, message: str) -> Iterator[str]:
        """
        Generate text from user input using the specified LLM.

        Args:
            message: The user input message.

        Returns:
            An iterator that yields the generated text in chunks.
        """
        self.messages.append({"role": "user", "content": message})

        assistant_role = None
        generated_content = ""

        stream = self.model.chat(
            model=self.model_id,
            messages=self.messages,
            stream=True,
        )

        for chunk in stream:
            # NOTE: `chunk["done"] == True` when ends
            token = chunk["message"]["content"]

            if assistant_role is None:
                assistant_role = chunk["message"]["role"]

            generated_content += token

            yield token

        if self.is_chat_history_disabled:
            self.messages.pop()
        else:
            self.messages.append({"role": assistant_role, "content": generated_content})
