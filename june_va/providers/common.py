import abc
import re
from typing import Iterator, Literal

from pydantic import BaseModel


class BaseLLMModel(abc.ABC):
    @abc.abstractmethod
    def forward(self, text: str) -> Iterator[str]: ...


class BaseSTTModel(abc.ABC):
    @abc.abstractmethod
    def forward(self, speech): ...


class BaseTTSModel(abc.ABC):
    @abc.abstractmethod
    def forward(self, text: str) -> str: ...

    @staticmethod
    def normalize_text(text: str) -> str:
        n1 = re.sub(r'[^A-Za-z0-9-_?!.,;:\'"\s]', " ", text)
        n2 = re.sub(r"\s+", " ", n1)

        return n2


class LLMMessage(BaseModel):
    content: str
    role: Literal["assistant", "system", "user"]
