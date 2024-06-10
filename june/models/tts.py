"""
This module provides a Text-to-Speech (TTS) class for generating speech from text using the TTS library.
"""

from transformers import pipeline

from ..settings import settings
from .common import ModelBase


class TTS(ModelBase):
    def __init__(self, **kwargs):
        model_id = kwargs["model"]

        self.pipeline = pipeline(
            "text-to-speech",
            model=model_id,
            token=settings.HF_TOKEN,
            torch_dtype="auto",
        )

    def synthesise(self, text, **kwargs):
        generation_args = kwargs.get("generation_args") or {}

        synthesis = self.pipeline(text, **generation_args)

        return synthesis
