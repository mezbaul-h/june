"""
This module defines a base class for models used in various tasks such as Text-to-Speech, Speech-to-Text, and Language
Models.
"""

import abc
from typing import Any, Dict

from ..settings import settings


class BaseModel(abc.ABC):
    """
    Base class for models used in various tasks.

    This class provides a common interface for initializing models and defining the
    forward method, which should be implemented by subclasses.

    Args:
        **kwargs: Keyword arguments for initializing the model, including optional
            arguments like 'device', 'generation_args', and 'model'.

    Attributes:
        device: The device on which the model should be loaded (e.g., 'cpu', 'cuda').
        generation_args: A dictionary of arguments to be used during generation or inference.
        model_id: The identifier or name of the model to be loaded.
    """

    def __init__(self, **kwargs) -> None:
        self.device: str = kwargs.get("device") or settings.TORCH_DEVICE
        self.generation_args: Dict[str, Any] = kwargs.get("generation_args") or {}
        self.model_id: str = kwargs["model"]

    @abc.abstractmethod
    def forward(self, model_input: Any) -> Any:
        """
        Abstract method to be implemented by subclasses for running the model on input data.

        Args:
            model_input: The input data for the model.

        Returns:
            The output of the model for the given input.
        """
        ...
