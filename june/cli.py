"""
This module provides a CLI program for interacting with the TextGenerator, Transcriber, and TextToSpeech classes.
"""

import asyncio
import json
import logging
import os.path
import re
import time
from threading import Thread

import click
import pygame
from colorama import Fore, Style, init

from .audio import AudioIO
from .models import LLM, STT, TTS
from .utils import ThreadSafeState, print_system_message

logging.getLogger("TTS").setLevel(logging.ERROR)
pygame.mixer.init()

shutdown_event = asyncio.Event()


class AppState:
    READY_FOR_INPUT = 0  # ready to take user input
    LLM_RESPONSE_GENERATED = 1


current_app_state = ThreadSafeState(AppState.READY_FOR_INPUT)


def producer(text_queue: asyncio.Queue, llm_model: LLM, stt_model: STT, tts_model: TTS):
    """Producer function to put items into the queue."""
    audio_io = AudioIO()
    speech_recognition = stt_model and tts_model
    min_chunk_size = 10
    splitters = [".", ",", "?", ":", ";"]

    def get_user_input():
        if not speech_recognition:
            return input(f"{Style.BRIGHT}{Fore.CYAN}[user]>{Style.RESET_ALL} ")

        audio_data = audio_io.record_audio()

        if audio_data is not None:
            print_system_message("Transcribing audio...")

            transcription = stt_model.forward(audio_data)
            return transcription

    # Regular expression pattern to match 'quit', 'stop', or 'exit', ignoring case
    exit_pattern = re.compile(r"\b(exit|quit|stop)\b", re.IGNORECASE)

    while True:
        if pygame.mixer.music.get_busy() or current_app_state.get_value() != AppState.READY_FOR_INPUT:
            time.sleep(0.1)
            continue

        buffer = []
        user_input = get_user_input()

        if speech_recognition:
            print(f"{Style.BRIGHT}{Fore.CYAN}[user]>{Style.RESET_ALL} {user_input}")

        if user_input:
            if exit_pattern.search(user_input):
                print_system_message("Exiting...")
                break

            print(f"{Style.BRIGHT}{Fore.GREEN}[assistant]> {Style.NORMAL}", end="", flush=True)

            for token in llm_model.forward(user_input):
                print(token, end="", flush=True)

                buffer.append(token)

                # Check if buffer is ready to be chunked
                if token == "\n" or (len(buffer) >= min_chunk_size and token in splitters):
                    chunk = "".join(buffer).strip()

                    buffer.clear()

                    if chunk:
                        # Queue this chunk for async TTS processing
                        text_queue.put_nowait(chunk)

            # Process any remaining text in buffer
            if buffer:
                chunk = "".join(buffer).strip()
                if chunk:
                    text_queue.put_nowait(chunk)

            current_app_state.set_value(AppState.LLM_RESPONSE_GENERATED)

            print(Style.RESET_ALL)

        time.sleep(0.5)  # Pause briefly before next listening

    audio_io.close()


async def consumer(text_queue: asyncio.Queue, tts_model: TTS):
    """Consumer function to process items from the queue."""
    with AudioIO() as audio_io:
        while not shutdown_event.is_set():
            try:
                text_buffer = text_queue.get_nowait()

                if tts_model and not shutdown_event.is_set():
                    synthesis = tts_model.forward(text_buffer)

                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.1)

                    tts_model.model.synthesizer.save_wav(wav=synthesis, path=tts_model.file_path)

                    audio_io.play_wav(tts_model.file_path)

                text_queue.task_done()
            except asyncio.QueueEmpty:
                if current_app_state.get_value() != AppState.READY_FOR_INPUT:
                    current_app_state.set_value(AppState.READY_FOR_INPUT)

                await asyncio.sleep(0.5)


async def start_async_tasks(text_queue, tts_model):
    """Starts asynchronous tasks without directly calling loop.run_forever()."""
    consumer_task = asyncio.create_task(consumer(text_queue, tts_model))
    try:
        # Wait until consumer finishes
        await consumer_task
    except asyncio.CancelledError:
        ...


def run_async_tasks(text_queue, tts_model):
    # Set up the event loop and run the async tasks

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(start_async_tasks(text_queue, tts_model))
    except Exception:
        loop.close()


async def _real_main(**kwargs):
    config = json.loads(kwargs["config"].read())
    llm_config = config["llm"]
    stt_config = config.get("stt") or {}
    tts_config = config.get("tts") or {}
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

    llm_model = LLM(**llm_config)

    if not llm_model.exists():
        print_system_message(f"Invalid ollama model: {llm_model.model_id}", color=Fore.RED)
        return 2

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

    if not speech_recognition:
        stt_model = None
        tts_model = None
    else:
        stt_model = STT(**stt_config)
        tts_model = TTS(**tts_config)

    text_queue = asyncio.Queue()

    thread = Thread(target=run_async_tasks, args=(text_queue, tts_model))
    thread.start()

    try:
        producer(text_queue, llm_model, stt_model, tts_model)
    except KeyboardInterrupt:
        ...
    finally:
        shutdown_event.set()
        thread.join()

        # Wait for the queue to be fully processed
        await text_queue.join()

        if tts_model and os.path.exists(tts_model.file_path):
            os.remove(tts_model.file_path)


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

    asyncio.run(_real_main(**kwargs))
