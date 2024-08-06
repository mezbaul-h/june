import asyncio
import io
import logging
import pathlib
import random
import re
import time
import wave
from threading import Thread
from typing import Optional

import pygame
from colorama import Fore, Style

from june_va.audio import AudioIO
from june_va.engine import Engine
from june_va.providers.common import LLMMessage
from june_va.utils import ThreadSafeState, print_system_message

logging.getLogger("TTS").setLevel(logging.ERROR)
pygame.mixer.init()

engine: Optional[Engine] = None


class AppState:
    """Enumeration for application states."""

    READY_FOR_INPUT = 0  # Ready to take user input
    LLM_RESPONSE_GENERATED = 1


current_app_state = ThreadSafeState(AppState.READY_FOR_INPUT)
shutdown_event = asyncio.Event()


async def _clear_queue(queue: asyncio.Queue[str]):
    """
    Clear all items from the asyncio queue.

    Args:
        queue: The queue to be cleared.
    """
    while not queue.empty():
        _ = await queue.get()
        queue.task_done()


async def consumer(text_queue: asyncio.Queue[str], tts_model):
    """
    Consumer task to process text from the queue and generate TTS output.

    Args:
        text_queue: Queue containing text to process.
        tts_model: Text-to-Speech model for generating audio.
    """

    with AudioIO() as audio_io:
        while not shutdown_event.is_set():
            try:
                text_buffer = text_queue.get_nowait()

                if tts_model:
                    wav_data = tts_model.forward(text_buffer)

                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.25)

                    file_path = f"june-va-audio-{time.time()}.wav"

                    with open(file_path, "wb") as wf:
                        wf.write(wav_data)

                    audio_io.play_wav(file_path)

                text_queue.task_done()
            except asyncio.QueueEmpty:
                if current_app_state.get_value() != AppState.READY_FOR_INPUT:
                    # Wait for the last chunk of speech to be played fully
                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.25)

                    current_app_state.set_value(AppState.READY_FOR_INPUT)

                await asyncio.sleep(0.25)
            except Exception:
                text_queue.task_done()

                if current_app_state.get_value() != AppState.READY_FOR_INPUT:
                    # Wait for the last chunk of speech to be played fully
                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.25)

                    current_app_state.set_value(AppState.READY_FOR_INPUT)


async def start_async_tasks(text_queue: asyncio.Queue[str], tts_model: Optional):
    """
    Start consumer task for processing text queue.

    Args:
        text_queue: Queue containing text to process.
        tts_model: Text-to-Speech model for generating audio.
    """
    consumer_task = asyncio.create_task(consumer(text_queue, tts_model))

    try:
        # Wait until consumer finishes
        await consumer_task
    except asyncio.CancelledError:
        ...


def run_async_tasks(text_queue: asyncio.Queue[str], tts_model: Optional):
    """
    Run async tasks in a new event loop for thread safety.

    Args:
        text_queue: Queue to put processed text chunks.
        tts_model: Text-to-Speech model for generating audio.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(start_async_tasks(text_queue, tts_model))
    except Exception:
        loop.close()


async def _real_main(stt_model, tts_model):
    text_queue = asyncio.Queue()

    # Run consumer task in separate thread
    thread = Thread(target=run_async_tasks, args=(text_queue, tts_model))
    thread.start()

    try:
        await producer(text_queue, stt_model)
    except KeyboardInterrupt:
        ...
    finally:
        shutdown_event.set()
        thread.join()
        await _clear_queue(text_queue)
        await text_queue.join()


async def producer(text_queue, stt_model):
    with AudioIO() as audio_io:

        def get_user_input() -> str:
            if stt_model:
                audio_data = audio_io.record_audio()

                if audio_data is not None:
                    print_system_message("Transcribing audio...")

                    transcription = stt_model.forward(audio_data)

                    return transcription

            return input(f"{Style.BRIGHT}{Fore.CYAN}[user]>{Style.RESET_ALL} ")

        # Regular expression pattern to match 'quit', 'stop', or 'exit', ignoring case
        exit_pattern = re.compile(r"\b(exit|quit|stop)\b", re.IGNORECASE)

        while True:
            if current_app_state.get_value() != AppState.READY_FOR_INPUT:
                await asyncio.sleep(0.25)
                continue

            user_input = get_user_input()

            if stt_model:
                print(f"{Style.BRIGHT}{Fore.CYAN}[user]>{Style.RESET_ALL} {user_input}")

            if user_input:
                if exit_pattern.search(user_input):
                    print_system_message("Exiting...")
                    break

                print(f"{Style.BRIGHT}{Fore.GREEN}[assistant]> {Style.NORMAL}", end="", flush=True)

                async for chunk in engine.process_text(
                    [
                        LLMMessage(role="user", content=user_input),
                    ],
                    cli=True,
                    queue=text_queue,
                ):
                    ...

                current_app_state.set_value(AppState.LLM_RESPONSE_GENERATED)

                print(Style.RESET_ALL)


def main(llm_model, stt_model, tts_model):
    global engine
    engine = Engine(llm_model, stt_model, tts_model)
    asyncio.run(_real_main(stt_model, tts_model))

    for item in pathlib.Path().glob("june-va-audio-*.wav"):
        if item.is_file():
            item.unlink()
