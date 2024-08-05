"""
This module provides a Text-to-Speech (TTS) class for generating speech from text using the TTS library.
"""

import abc
import time
from pathlib import Path
from typing import List

import requests

from june_va.settings.models import CoquiTTSSettings, ElevenlabsTTSSettings

from .common import BaseModel


class BaseTTSModel(abc.ABC):
    @abc.abstractmethod
    def forward(self, text: str) -> str: ...


class ElevenlabsTTS(BaseTTSModel):
    _URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    def __init__(self, **kwargs) -> None:
        self.user_settings: ElevenlabsTTSSettings = kwargs["user_settings"]
        self.file_path_template = "tts_out_at_{n}.mp3"

    def forward(self, text: str):
        payload = {
            "text": text,
            "model_id": self.user_settings.model,
            "seed": 123,
        }
        headers = {
            "Content-Type": "application/json",
            "xi-api-key": self.user_settings.api_key,
        }

        response = requests.post(self._URL.format(voice_id=self.user_settings.voice_id), json=payload, headers=headers)

        response.raise_for_status()

        file_path = self.file_path_template.format(n=time.time())

        with open(file_path, "wb") as f:
            f.write(response.content)

        return file_path


class CoquiTTS(BaseTTSModel):
    """
    A class for generating speech from text using the TTS library.

    This class inherits from the BaseModel class and provides a method for running
    the Text-to-Speech model on text input.

    Args:
        **kwargs: Keyword arguments for initializing the TTS model, including optional
            arguments like 'device', 'generation_args', and 'model'.

    Attributes:
        model: An instance of the TTS model from the TTS library.
        file_path: The file path where the generated audio should be saved.
    """

    def __init__(self, **kwargs) -> None:
        from TTS.api import TTS as _CoquiTTS

        self.user_settings: CoquiTTSSettings = kwargs["user_settings"]
        self.file_path: str = kwargs.get("file_path") or "out.wav"

        self.model = _CoquiTTS(self.user_settings.model).to(self.user_settings.device)
        self.generation_args = kwargs.get("generation_args")

    def forward(self, text: str):
        """
        Generate speech from text using the Text-to-Speech model.

        Args:
            text: The input text for which speech should be generated.

        Returns:
            A list of integers representing the generated audio data.
        """
        synthesis: List[int] = self.model.tts(text, **self.generation_args)

        self.model.synthesizer.save_wav(wav=synthesis, path=self.file_path)

        return self.file_path
