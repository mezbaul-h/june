import ollama

from .common import ModelBase


class LLM(ModelBase):
    def __init__(self, kwargs):
        super().__init__(kwargs)

        self.messages = []

        self.system_prompt = kwargs.pop("system_prompt", None)

        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

        self.is_chat_history_disabled = kwargs.pop("disable_chat_history", None)

        self.client = ollama.Client()

    def get_info(self):
        return ollama.show(self.model_id)

    def generate(self, message: str):
        self.messages.append({"role": "user", "content": message})

        assistant_role = None
        generated_content = ""

        stream = self.client.chat(
            model=self.model_id,
            messages=self.messages,
            stream=True,
        )

        for chunk in stream:
            # NOTE: `chunk["done"] == True` when ends
            token = chunk["message"]["content"]

            if assistant_role is None:
                assistant_role = chunk["message"]["role"]

            generated_content += token

            yield token

        if self.is_chat_history_disabled:
            self.messages.pop()
        else:
            self.messages.append({"role": assistant_role, "content": generated_content})
