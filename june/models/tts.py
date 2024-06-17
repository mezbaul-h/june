"""
This module provides a Text-to-Speech (TTS) class for generating speech from text using the TTS library.
"""

from TTS.api import TTS as XTTS

from .common import ModelBase


class TTS(ModelBase):
    def __init__(self, kwargs):
        super().__init__(kwargs)

        self.tts = XTTS(self.model_id).to(self.device)
        self.file_path = self.generation_args.pop("file_path", "out.wav")

    def synthesise(self, text):
        return self.tts.tts(text, **self.generation_args)
