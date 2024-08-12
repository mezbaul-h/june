import io
from typing import Optional, Any, Dict

from pydantic import BaseModel, ConfigDict, Json

from ..settings import TORCH_DEVICE
from ..utils import logger, suppress_stdout_stderr
from .common import BaseTTSModel


class CoquiTTSSettings(BaseModel):
    """
    Attributes:
        api_key: Your API key.
        model: Identifier of the model that will be used, you can query them using GET /v1/models. The model needs to have support for text to speech, you can check this using the can_do_text_to_speech property. (default: eleven_monolingual_v1)
        voice_id: Voice ID to be used, you can use https://api.elevenlabs.io/v1/voices to list all the available voices.
    """

    model_config = ConfigDict(protected_namespaces=())

    device: Optional[str] = TORCH_DEVICE
    generation_args: Optional[Dict[Any, Any]] = None
    model: str


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

    def __init__(self, user_settings: CoquiTTSSettings) -> None:
        from TTS.api import TTS as _CoquiTTS

        self.user_settings = user_settings
        # self.file_path: str = kwargs.get("file_path") or "out.wav"

        self.model = _CoquiTTS(self.user_settings.model).to(self.user_settings.device)
        # self.generation_args = kwargs.get("generation_args")

    def forward(self, text: str):
        """
        Generate speech from text using the Text-to-Speech model.

        Args:
            text: The input text for which speech should be generated.

        Returns:
            A list of integers representing the generated audio data.
        """
        generation_args = self.user_settings.generation_args or {}

        # Create a byte stream
        byte_io = io.BytesIO()

        try:
            self.model.tts_to_file(text, **generation_args, file_path=byte_io)
        except RuntimeError as exc:
            # logger.warning("Failed to synthesize input: %s (Reason: %s)", repr(text), str(exc))
            ...

        return byte_io.getvalue()
