import uuid

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..settings import settings
from ..utils import DeferredInitProxy
from .common import ModelBase


class LLM(ModelBase):
    def __init__(self, **kwargs):
        model_id = kwargs["model"]

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=settings.HF_TOKEN,
            torch_dtype="auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        chat_template = kwargs.get("chat_template")
        if chat_template:
            tokenizer.chat_template = chat_template

        self.pipeline = pipeline(
            "text-generation",
            device_map="auto",
            model=model,
            tokenizer=tokenizer,
            token=settings.HF_TOKEN,
            torch_dtype="auto",
            trust_remote_code=True,
        )

        self.contexts = {}

    def generate(self, message: str, **kwargs):
        attach_context_id = False
        init_context = False
        context_id = kwargs.get("context_id")
        system_prompt = kwargs.get("system_prompt")
        generation_args = kwargs.get("generation_args") or {}

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
    phi_3_mini_128k=DeferredInitProxy(
        LLM,
        model="microsoft/Phi-3-mini-128k-instruct",
        chat_template=zephyr_chat_template,
    ),  # ~4B params
    stablelm_2_zephyr_1_6b=DeferredInitProxy(LLM, model="stabilityai/stablelm-2-zephyr-1_6b"),  # ~2B params
    stablelm_zephyr_3b=DeferredInitProxy(LLM, model="stabilityai/stablelm-zephyr-3b"),
    starling_lm_7b_alpha=DeferredInitProxy(LLM, model="berkeley-nest/Starling-LM-7B-alpha"),
)