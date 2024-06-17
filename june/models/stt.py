"""
This module provides a Text-to-Speech (TTS) class for generating speech from text using the TTS library.
"""

import warnings

from transformers import pipeline

from ..settings import settings
from .common import ModelBase


class STT(ModelBase):
    def __init__(self, kwargs):
        super().__init__(kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", lineno=1132)
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                chunk_length_s=10,
                device=self.device,
                model=self.model_id,
                token=settings.HF_TOKEN,
                torch_dtype="auto",
                trust_remote_code=True,
            )

    def transcribe(self, audio) -> str:
        transcription = self.pipeline(audio, **self.generation_args)

        return transcription["text"].strip()
