from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Json


class CoquiTTSSettings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    device: str
    model: str


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


class TTSSettings(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        # protected_namespaces=(),
    )

    provider: Literal[
        "coqui",
        "elevenlabs",
    ]


class OllamaLLMSettings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str


class LLMSettings(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        # protected_namespaces=(),
    )

    provider: Literal["ollama",]


class UserSettings(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        # protected_namespaces=(),
    )

    llm: LLMSettings
    tts: Optional[TTSSettings] = None
