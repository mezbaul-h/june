"""
This module defines a base class for models used in various tasks such as Text-to-Speech, Speech-to-Text, and Language
Models.
"""

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict

from ..settings import settings
from ..utils import print_system_message


class BaseMeta(ABCMeta):
    """
    Metaclass for BaseModel that adds automatic initialization logging.

    This metaclass overrides the __call__ method to print a system message
    when a new instance of a model is created. It logs the model's class name,
    model ID, and the device it's initialized on.
    """

    def __call__(cls, *args, **kwargs):
        # Create the instance using the standard creation process
        instance = super().__call__(*args, **kwargs)

        # Print a system message with information about the initialized model
        print_system_message(
            f"{instance.__class__.__name__} model initialized (model_id={instance.model_id}; device={instance.device})",
        )

        # Return the created instance
        return instance


class BaseModel(ABC, metaclass=BaseMeta):
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

    @abstractmethod
    def forward(self, model_input: Any) -> Any:
        """
        Abstract method to be implemented by subclasses for running the model on input data.

        Args:
            model_input: The input data for the model.

        Returns:
            The output of the model for the given input.
        """
        ...
