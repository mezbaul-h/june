"""
This module provides a CLI program for interacting with the TextGenerator, Transcriber, and TextToSpeech classes.
"""

import json
import re
import time

import click
import torch
from colorama import Fore, Style, init
from datasets import load_dataset

from .audio import AudioIO
from .models import LLM, STT, TTS
from .utils import print_system_message


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
    init()

    config = json.loads(kwargs["config"].read())
    llm_config = config["llm"]
    stt_config = config.get("stt") or {}
    tts_config = config.get("tts") or {}

    print_system_message(
        f"Models being used: LLM={llm_config['model']}, STT={stt_config.get('model') or 'n/a'}, "
        f"TTS={tts_config.get('model') or 'n/a'}"
    )

    if llm_config.get("disable_chat_history"):
        print_system_message(
            "Chat history is currently disabled. The conversation may not be fully interactive, as the "
            "assistant will not retain previous context. Each interaction will be treated independently.",
            color=Fore.YELLOW,
        )

    if not llm_config.get("system_prompt"):
        print_system_message("No system prompt provided.")

    llm_model = LLM(
        chat_template=llm_config.get("chat_template"),
        disable_chat_history=llm_config.get("disable_chat_history"),
        generation_args=llm_config.get("generation_args"),
        model=llm_config["model"],
        system_prompt=llm_config.get("system_prompt"),
    )
    speech_recognition = stt_config and tts_config

    if speech_recognition:
        try:
            import pyaudio
        except ImportError:
            print_system_message(
                "PyAudio not installed. Please install PyAudio for speech recognition and audio synthesis to " "work.",
                color=Fore.RED,
            )
            return 1

    if not speech_recognition:
        # get_default_microphone_info()
        audio_io = None
        speaker_embedding = None
        tts_model = None
    else:
        audio_io = AudioIO()
        stt_model = STT(model=stt_config["model"])
        tts_model = TTS(model=tts_config["model"])
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    def get_user_input():
        if not speech_recognition:
            return input(f"{Fore.CYAN}[user]>{Style.RESET_ALL} ")

        audio_data = audio_io.record_audio()

        if audio_data is not None:
            print_system_message("Transcribing audio...")

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
            print(f"{Fore.CYAN}[user]>{Style.RESET_ALL} {user_input}")

        if user_input:
            if exit_pattern.search(user_input):
                print_system_message("Exiting...")
                break

            # Print it before so to account for streaming.
            print(f"{Fore.GREEN}[assistant]>{Style.RESET_ALL} ", end="", flush=True)

            reply = llm_model.generate(user_input)
            # print(f"[assistant]> {reply['content']}")

            if speech_recognition:
                synthesis = tts_model.synthesise(
                    reply["content"], generation_args={"forward_params": {"speaker_embeddings": speaker_embedding}}
                )
                print_system_message("Playing synthesised audio...")
                audio_io.play_audio(synthesis)

        time.sleep(1)  # Pause briefly before next listening
