"""
This module provides a TextGenerator class for generating text using a Large Language Model (LLM).
"""

import uuid

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from june.settings import HF_TOKEN


class TextGenerator:
    """
    Class for generating text using a Large Language Model (LLM).
    """

    @property
    def model(self):
        """
        Loads and returns the LLM model.
        """
        return AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct",
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )

    @property
    def tokenizer(self):
        """
        Returns the LLM tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
        tokenizer.use_default_system_prompt = True
        tokenizer.chat_template = (
            "{{ bos_token }}"
            "{% for message in messages %}\n"
            "{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + '<|end|>' }}\n"
            "{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + '<|end|>' }}\n"
            "{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + '<|end|>' }}\n"
            "{% endif %}\n"
            "{% if loop.last and add_generation_prompt %}\n"
            "{{ '<|assistant|>' }}\n"
            "{% endif %}\n"
            "{% endfor %}"
        )

        return tokenizer

    def __init__(self, **kwargs):
        """
        Initializes the TextGenerator object.
        """
        self.pipeline = pipeline(
            "text-generation",
            device_map="auto",
            model=self.model,
            token=HF_TOKEN,
            tokenizer=self.tokenizer,
        )

        self.contexts = {}
        self.pipeline_kwargs = kwargs.get("pipeline_kwargs") or {}
        self.system_initial_context = kwargs.get("system_initial_context")

    def generate(self, message: str, context_id: str = None):
        """
        Generates text based on the input message and optional context ID.

        Parameters
        ----------
        message : str
            Input message for text generation.
        context_id : str, optional
            Optional context ID for tracking conversation context.

        Returns
        -------
        dict
            Generated text with role and content.
        """
        attach_context_id = False
        init_context = False

        if not context_id:
            attach_context_id = True
            init_context = True
            context_id = str(uuid.uuid4())
        elif context_id not in self.contexts:
            init_context = True

        if init_context:
            self.contexts[context_id] = []

            if self.system_initial_context:
                self.contexts[context_id].append(
                    {
                        "role": "system",
                        "content": self.system_initial_context,
                    }
                )

        self.contexts[context_id].append(
            {
                "role": "user",
                "content": message,
            }
        )

        completion = self.pipeline(
            self.contexts[context_id],
            **self.pipeline_kwargs,
        )[
            0
        ]["generated_text"]

        if isinstance(completion, str):
            completion = {
                "role": "assistant",
                "content": completion,
            }
        else:
            completion = completion[-1]

        self.contexts[context_id].append({**completion})

        if attach_context_id:
            completion.update({"context_id": context_id})

        return completion
