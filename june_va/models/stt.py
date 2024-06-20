"""
This module provides a Speech-to-Text (STT) class for transcribing audio data into text using the Transformers library.
"""

import warnings
from typing import Dict, Union

from numpy import ndarray

from ..settings import settings
from .common import BaseModel


class STT(BaseModel):
    """
    A class for transcribing audio data into text using the Transformers library.

    This class inherits from the BaseModel class and provides a method for running
    the Speech-to-Text model on audio data.

    Args:
        **kwargs: Keyword arguments for initializing the STT model, including optional
            arguments like 'device', 'generation_args', and 'model'.

    Attributes:
        model: An instance of the Transformers pipeline for automatic speech recognition.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        with warnings.catch_warnings():
            # Ignore the `resume_download` warning raise by Hugging Face's underlying library
            warnings.simplefilter("ignore", lineno=1132)

            from transformers import pipeline

            self.model = pipeline(
                "automatic-speech-recognition",
                chunk_length_s=10,
                device=self.device,
                model=self.model_id,
                token=settings.HF_TOKEN,
                torch_dtype="auto",
                trust_remote_code=True,
            )

    def forward(self, audio: Dict[str, Union[int, ndarray]]) -> str:
        """
        Transcribe audio data into text using the Speech-to-Text model.

        Args:
            audio: A dictionary containing the audio data,
                with a 'sample_rate' key for the sample rate (int) and an 'array' key for the audio array (np.ndarray).

        Returns:
            The transcribed text from the audio data.
        """
        transcription = self.model(audio, **self.generation_args)

        return transcription["text"].strip()
