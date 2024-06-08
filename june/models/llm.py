import re
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline

from ..settings import settings
from .common import ModelBase


class TokenStreamer(TextStreamer):
    system_token_pattern = re.compile(r"<\|?([a-z\-_]+)\|?>", re.IGNORECASE)

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, skip_prompt=True)

        self.bos_token = kwargs["bos_token"]
        self.eos_token = kwargs["eos_token"]

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if text.startswith(self.bos_token):
            return

        if self.eos_token in text:
            text = text.removesuffix(self.eos_token)

        super().on_finalized_text(text, stream_end)


class LLM(ModelBase):
    def __init__(self, **kwargs):
        model_id = kwargs["model"]

        model_args = {
            "token": settings.HF_TOKEN,
            "torch_dtype": "auto",
            "trust_remote_code": True,
        }
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=settings.HF_DEVICE_MAP,
            **model_args,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        chat_template = kwargs.get("chat_template")
        if chat_template:
            tokenizer.chat_template = chat_template

        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **model_args,
        )

        streamer = TokenStreamer(tokenizer, bos_token=tokenizer.bos_token, eos_token=tokenizer.eos_token)

        self.chat_history = []

        system_prompt = kwargs.get("system_prompt")

        if system_prompt:
            self.chat_history.append({"role": "system", "content": system_prompt})

        self.is_chat_history_disabled = kwargs.get("disable_chat_history")
        self.system_prompt = kwargs.get("system_prompt")
        self.generation_args = kwargs.get("generation_args") or {}
        self.generation_args.update(
            {
                "pad_token_id": self.pipeline.tokenizer.eos_token_id,
                "streamer": streamer,
            }
        )

    def generate(self, message: str):
        self.chat_history.append({"role": "user", "content": message})

        try:
            completion = self.pipeline(self.chat_history, **self.generation_args)[0]["generated_text"]
        except RuntimeError as e:
            if "cutlassF" in str(e) and settings.TORCH_DEVICE == "cuda":
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_flash_sdp(False)

            # Try again
            completion = self.pipeline(self.chat_history, **self.generation_args)[0]["generated_text"]

        if isinstance(completion, str):
            completion = {"role": "assistant", "content": completion}
        else:
            completion = completion[-1]

        if self.is_chat_history_disabled:
            self.chat_history.pop()
        else:
            self.chat_history.append(completion)

        return completion
