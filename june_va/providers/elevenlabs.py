import io
import time

import requests
from pydantic import BaseModel, ConfigDict
from pydub import AudioSegment

from .common import BaseTTSModel


class ElevenlabsTTSSettings(BaseModel):
    """
    Attributes:
        api_key: Your API key.
        model: Identifier of the model that will be used, you can query them using GET /v1/models. The model needs to have support for text to speech, you can check this using the can_do_text_to_speech property. (default: eleven_monolingual_v1)
        voice_id: Voice ID to be used, you can use https://api.elevenlabs.io/v1/voices to list all the available voices.
    """

    model_config = ConfigDict(protected_namespaces=())

    api_key: str
    model: str
    voice_id: str


class ElevenlabsTTS(BaseTTSModel):
    _URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    def __init__(self, user_settings: ElevenlabsTTSSettings) -> None:
        self.user_settings = user_settings
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
        params = {
            "output_format": "mp3_44100_128",
        }

        response = requests.post(
            self._URL.format(voice_id=self.user_settings.voice_id), headers=headers, json=payload, params=params
        )

        response.raise_for_status()

        file_path = self.file_path_template.format(n=time.time())

        # with open(file_path, "wb") as f:
        #     f.write(response.content)

        audio_segment = AudioSegment.from_mp3(io.BytesIO(response.content))

        buffer = io.BytesIO()

        audio_segment.export(buffer, format="wav")

        return buffer.getvalue()
