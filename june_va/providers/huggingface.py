"""
This module provides a Speech-to-Text (STT) class for transcribing audio data into text using the Transformers library.
"""

import warnings
from typing import Any, Dict, Optional, Union

from numpy import ndarray
from pydantic import BaseModel, ConfigDict, Json

from .common import BaseSTTModel
from ..settings import TORCH_DEVICE


class HuggingfaceSTTSettings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    api_key: Optional[str] = None
    device: str = TORCH_DEVICE
    generation_args: Optional[Dict[Any, Any]] = None
    model: str


class HuggingfaceSTT(BaseSTTModel):
    """
    A class for transcribing audio data into text using the Transformers library.

    This class inherits from the BaseModel class and provides a method for running
    the Speech-to-Text model on audio data.

    Args:
        user_settings: Keyword arguments for initializing the STT model, including optional arguments like 'device', 'generation_args', and 'model'.

    Attributes:
        model: An instance of the Transformers pipeline for automatic speech recognition.
    """

    def __init__(self, user_settings: HuggingfaceSTTSettings) -> None:
        self.user_settings = user_settings

        with warnings.catch_warnings():
            # Ignore the `resume_download` warning raise by Hugging Face's underlying library
            warnings.simplefilter("ignore", lineno=1132)

            from transformers import pipeline

            self.model = pipeline(
                "automatic-speech-recognition",
                chunk_length_s=10,
                device=self.user_settings.device,
                model=self.user_settings.model,
                token=self.user_settings.api_key,
                torch_dtype="auto",
                trust_remote_code=True,
            )

    def forward(self, audio: Dict[str, Union[int, ndarray]]) -> str:
        """
        Transcribe audio data into text using the Speech-to-Text model.

        Args:
            audio: A dictionary containing the audio data,
                with a 'sampling_rate' key for the sample rate (int) and an 'raw' key for the audio array (np.ndarray).

        Returns:
            The transcribed text from the audio data.
        """
        generation_args = self.user_settings.generation_args or {}
        transcription = self.model(audio, **generation_args)

        return transcription["text"].strip()
