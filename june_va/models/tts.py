"""
This module provides a Text-to-Speech (TTS) class for generating speech from text using the TTS library.
"""

from typing import List

from .common import BaseModel


class TTS(BaseModel):
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
        super().__init__(**kwargs)

        # Disable additional splits, as they increase the likelihood of generation errors.
        self.generation_args["split_sentences"] = False

        self.file_path: str = self.generation_args.get("file_path") or "out.wav"

        from TTS.api import TTS as CoquiTTS

        self.model = CoquiTTS(self.model_id).to(self.device)

    def forward(self, text: str) -> List[int]:
        """
        Generate speech from text using the Text-to-Speech model.

        Args:
            text: The input text for which speech should be generated.

        Returns:
            A list of integers representing the generated audio data.
        """

        return self.model.tts(text, **self.generation_args)
