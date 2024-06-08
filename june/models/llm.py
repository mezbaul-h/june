import re
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline

from ..settings import settings
from ..utils import DeferredInitProxy
from .common import ModelBase


class TokenStreamer(TextStreamer):
    system_token_pattern = re.compile(r"<\|?[a-z\-_]+\|?>", re.IGNORECASE)

    def __init__(self, tokenizer):
        super().__init__(tokenizer)

        self.stream_started = False

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if not self.stream_started:
            self.stream_started = True
            return  # Avoid printing initial tokens

        matches = self.system_token_pattern.findall(text)

        if len(matches) == 1 and stream_end:
            # If exactly one match is found, replace it with an empty string
            text = self.system_token_pattern.sub("", text, count=1)

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

        self.streamer = TokenStreamer(tokenizer)

        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **model_args,
        )

        self.contexts = {}

    def generate(self, message: str, **kwargs):
        attach_context_id = False
        init_context = False
        context_id = kwargs.get("context_id")
        system_prompt = kwargs.get("system_prompt")
        generation_args = kwargs.get("generation_args") or {}
        generation_args.update(
            {
                "pad_token_id": self.pipeline.tokenizer.eos_token_id,
                "streamer": self.streamer,
            }
        )

        if not context_id:
            attach_context_id = True
            init_context = True
            context_id = str(uuid.uuid4())
        elif context_id not in self.contexts:
            init_context = True

        if init_context:
            self.contexts[context_id] = []

            if system_prompt:
                self.contexts[context_id].append({"role": "system", "content": system_prompt})

        self.contexts[context_id].append({"role": "user", "content": message})

        try:
            completion = self.pipeline(self.contexts[context_id], **generation_args)[0]["generated_text"]
        except RuntimeError as e:
            if "cutlassF" in str(e) and settings.TORCH_DEVICE == "cuda":
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_flash_sdp(False)

            # Try again
            completion = self.pipeline(self.contexts[context_id], **generation_args)[0]["generated_text"]

        if isinstance(completion, str):
            completion = {"role": "assistant", "content": completion}
        else:
            completion = completion[-1]

        self.contexts[context_id].append({**completion})

        if attach_context_id:
            completion.update({"context_id": context_id})

        return completion


zephyr_chat_template = (
    "{{ bos_token }}{% for message in messages %}\n"
    "{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + '<|end|>' }}\n"
    "{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + '<|end|>' }}\n"
    "{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + '<|end|>' }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n"
    "{% endif %}\n{% endfor %}"
)

all_models = dict(
    google_gemma_11_2b=DeferredInitProxy(LLM, model="google/gemma-1.1-2b-it"),
    google_gemma_11_7b=DeferredInitProxy(LLM, model="google/gemma-1.1-7b-it"),
    kunoichi_dpo_v2_7b=DeferredInitProxy(LLM, model="SanjiWatsuki/Kunoichi-DPO-v2-7B"),
    meta_llama_3_8b=DeferredInitProxy(LLM, model="meta-llama/Meta-Llama-3-8B-Instruct"),
    mistral_7b_v03=DeferredInitProxy(LLM, model="mistralai/Mistral-7B-Instruct-v0.3"),
    nous_capybara_3b_v19=DeferredInitProxy(LLM, model="NousResearch/Nous-Capybara-3B-V1.9"),
    nous_hermes_2_mistral_7b_dpo=DeferredInitProxy(LLM, model="NousResearch/Nous-Hermes-2-Mistral-7B-DPO"),
    openchat_35_1210=DeferredInitProxy(LLM, model="openchat/openchat-3.5-1210"),  # ~7B params
    openhermes_2_5_mistral_7b=DeferredInitProxy(LLM, model="teknium/OpenHermes-2.5-Mistral-7B"),  # xoxo
    phi_3_mini_128k=DeferredInitProxy(
        LLM,
        model="microsoft/Phi-3-mini-128k-instruct",
        chat_template=zephyr_chat_template,
    ),  # ~4B params
    qwen2_0_5b=DeferredInitProxy(LLM, model="Qwen/Qwen2-0.5B-Instruct"),  # ~1B params
    qwen2_1_5b=DeferredInitProxy(LLM, model="Qwen/Qwen2-1.5B-Instruct"),  # ~2B params
    qwen2_7b=DeferredInitProxy(LLM, model="Qwen/Qwen2-7B-Instruct"),
    stablelm_2_zephyr_1_6b=DeferredInitProxy(LLM, model="stabilityai/stablelm-2-zephyr-1_6b"),  # ~2B params
    stablelm_zephyr_3b=DeferredInitProxy(LLM, model="stabilityai/stablelm-zephyr-3b"),
    starling_lm_7b_alpha=DeferredInitProxy(LLM, model="berkeley-nest/Starling-LM-7B-alpha"),
)
