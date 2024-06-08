"""
This module provides a Text-to-Speech (TTS) class for generating speech from text using the TTS library.
"""

from transformers import pipeline

from ..settings import settings
from .common import ModelBase


class STT(ModelBase):
    def __init__(self, **kwargs):
        model_id = kwargs["model"]

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            chunk_length_s=30,
            device_map=settings.HF_DEVICE_MAP,
            model=model_id,
            token=settings.HF_TOKEN,
            torch_dtype="auto",
            trust_remote_code=True,
        )

    def transcribe(self, audio, **kwargs) -> str:
        generation_args = kwargs.get("generation_args") or {}

        transcription = self.pipeline(audio, **generation_args)

        return transcription["text"].strip()
