import re

import torch
from colorama import Fore, Style
from jinja2 import TemplateError
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline

from ..settings import settings
from ..utils import print_system_message
from .common import ModelBase


class TokenStreamer(TextStreamer):
    """
    Streamer class for handling token streams with special token suppression.

    Attributes
    ----------
    system_token_pattern : re.Pattern
        Compiled regular expression pattern to match system tokens.
    bos_token : str
        Beginning of stream token.
    eos_token : str
        End of stream token.
    stream_started : bool
        Flag to indicate if the streaming has started.
    """

    system_token_pattern = re.compile(r"(<\|?[a-z\-_]+\|?>)", re.IGNORECASE)

    def __init__(self, tokenizer, **kwargs):
        """
        Initialize the TokenStreamer.

        Parameters
        ----------
        tokenizer : AutoTokenizer
            The tokenizer to be used.
        """
        super().__init__(tokenizer, skip_prompt=True)

        self.bos_token = kwargs["bos_token"]
        self.eos_token = kwargs["eos_token"]

        self.stream_started = False

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """
        Process and handle the finalized text stream, printing and removing special tokens as needed.

        Parameters
        ----------
        text : str
            The text to process.
        stream_end : bool, optional
            Flag indicating if the stream has ended, by default False.
        """
        if not self.stream_started:
            self.stream_started = True

            # Print the assistant prompt before processing the text.
            print(f"{Fore.GREEN}[assistant]>{Style.RESET_ALL} ", end="", flush=True)

        # Suppress the beginning of stream token
        if text.startswith(self.bos_token):
            return

        # Remove the end of stream token if it is at the end of the text
        if text.endswith(self.eos_token):
            text = text.removesuffix(self.eos_token)
        else:
            # Handle the case where Phi-3 tokenizer has incorrect `eos_token`
            # and it appears as a normal token in the stream.
            mobj = self.system_token_pattern.search(text)

            if mobj and text.endswith(mobj.group(0)):
                text = text.removesuffix(mobj.group(0))

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
            attn_implementation="eager",
            device_map=settings.HF_DEVICE_MAP,
            **model_args,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        print_system_message(f"LLM context length: {tokenizer.model_max_length}")

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

        def _complete_prompt():
            return self.pipeline(self.chat_history, **self.generation_args)[0]["generated_text"]

        try:
            completion = _complete_prompt()
        except RuntimeError as e:
            if "cutlassF" in str(e) and settings.TORCH_DEVICE == "cuda":
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_flash_sdp(False)

                # Try again
                completion = _complete_prompt()

            raise e
        except TemplateError as e:
            if "System role not supported" in str(e):
                print_system_message(str(e), color=Fore.YELLOW)

                # Remove system prompt
                self.chat_history.pop(0)

                # Try again
                completion = _complete_prompt()

            raise e

        if isinstance(completion, str):
            completion = {"role": "assistant", "content": completion}
        else:
            completion = completion[-1]

        if self.is_chat_history_disabled:
            self.chat_history.pop()
        else:
            self.chat_history.append(completion)

        return completion
