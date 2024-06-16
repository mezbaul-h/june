"""
This module provides a CLI program for interacting with the TextGenerator, Transcriber, and TextToSpeech classes.
"""

import json
import re
import time

import click
from colorama import Fore, Style, init

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
    # init()

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

    llm_model = LLM(**llm_config)
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
        audio_io = None
        tts_model = None
    else:
        audio_io = AudioIO()
        stt_model = STT(**stt_config)
        tts_model = TTS(**tts_config)

    def get_user_input():
        if not speech_recognition:
            return input(f"{Fore.CYAN}[user]>{Style.RESET_ALL} ")

        audio_data = audio_io.record_audio()

        if audio_data is not None:
            print_system_message("Transcribing audio...")

            transcription = stt_model.transcribe(audio_data)
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

            reply = llm_model.generate(user_input)

            if speech_recognition:
                synthesis = tts_model.synthesise(reply["content"])
                print_system_message("Playing synthesised audio...")
                audio_io.play_audio(synthesis)

        time.sleep(1)  # Pause briefly before next listening
