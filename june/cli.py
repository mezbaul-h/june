"""
This module provides a CLI program for interacting with the TextGenerator, Transcriber, and TextToSpeech classes.
"""

import json
import re
import time

import click
import torch
from datasets import load_dataset

from .audio import AudioIO
from .models import LLM, STT, TTS

# from .utils import get_default_microphone_info


@click.command()
@click.option(
    "-c",
    "--config",
    help="Configuration file.",
    nargs=1,
    type=click.File("r", encoding="utf-8"),
)
def main(**kwargs):
    """
    Main function to run the CLI program.
    """
    # system_initial_context = input("[system]> ")
    config = json.loads(kwargs["config"].read())

    llm_model = LLM(
        chat_template=config["llm"].get("chat_template"),
        disable_chat_history=config["llm"].get("disable_chat_history"),
        generation_args=config["llm"].get("generation_args"),
        model=config["llm"]["model"],
        system_prompt=config["llm"].get("system_prompt"),
    )
    speech_recognition = config.get("stt") and config.get("tts")

    if not speech_recognition:
        # get_default_microphone_info()
        audio_io = None
        speaker_embedding = None
        tts_model = None
    else:
        audio_io = AudioIO()
        stt_model = STT(model=config["stt"]["model"])
        tts_model = TTS(model=config["tts"]["model"])
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    def get_user_input():
        if not speech_recognition:
            return input("[user]> ")

        audio_data = audio_io.record_audio()

        if audio_data is not None:
            print("[transcribing]")

            transcription = stt_model.transcribe(
                audio_data,
                generation_args={
                    "batch_size": 8,
                },
            )
            return transcription

    # Regular expression pattern to match 'quit', 'stop', or 'exit', ignoring case
    exit_pattern = re.compile(r"\b(exit|quit|stop)\b", re.IGNORECASE)

    while True:
        user_input = get_user_input()

        if speech_recognition:
            print(f"[user]> {user_input}")

        if user_input:
            if exit_pattern.search(user_input):
                break

            print(f"[assistant]> ", end="", flush=True)  # Print it before so to account for streaming.

            reply = llm_model.generate(user_input)
            # print(f"[assistant]> {reply['content']}")

            if speech_recognition:
                synthesis = tts_model.synthesise(
                    reply["content"], generation_args={"forward_params": {"speaker_embeddings": speaker_embedding}}
                )
                audio_io.play_audio(synthesis)

        time.sleep(1)  # Pause briefly before next listening
