import pkgutil
from importlib import import_module
from typing import Any, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from . import providers


class _ModuleWrapper:
    def __init__(self, module):
        self.module = module

    def __getitem__(self, name):
        return getattr(self.module, name)


provider_modules = {
    name: _ModuleWrapper(import_module(f".{name}", package=providers.__name__))
    for _, name, _ in pkgutil.iter_modules(providers.__path__)
    if name != "common"
}


class LLMSettings(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        # protected_namespaces=(),
    )

    provider: str


class STTSettings(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        # protected_namespaces=(),
    )

    provider: str


class TTSSettings(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        # protected_namespaces=(),
    )

    # provider: Literal[
    #     "coqui",
    #     "elevenlabs",
    # ]
    provider: str


class UserSettings(BaseModel):
    llm: LLMSettings
    stt: Optional[STTSettings] = None
    tts: Optional[TTSSettings] = None


def process_user_settings(user_settings_dict: dict) -> Tuple[Any, Any, Any]:
    """
    parse user settings and load models
    """
    stt_model = None
    tts_model = None
    user_settings = UserSettings(**user_settings_dict)

    extras = user_settings.llm.model_extra
    provider = provider_modules[user_settings.llm.provider]
    model_class_name = f"{user_settings.llm.provider.capitalize()}LLM"
    llm_model = provider[model_class_name](provider[f"{model_class_name}Settings"](**extras))

    if user_settings.stt:
        extras = user_settings.stt.model_extra
        provider = provider_modules[user_settings.stt.provider]
        model_class_name = f"{user_settings.stt.provider.capitalize()}STT"
        stt_model = provider[model_class_name](provider[f"{model_class_name}Settings"](**extras))

    if user_settings.tts:
        extras = user_settings.tts.model_extra
        provider = provider_modules[user_settings.tts.provider]
        model_class_name = f"{user_settings.tts.provider.capitalize()}TTS"
        tts_model = provider[model_class_name](provider[f"{model_class_name}Settings"](**extras))

    return llm_model, stt_model, tts_model
