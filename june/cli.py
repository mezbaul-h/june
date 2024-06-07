"""
This module provides a CLI program for interacting with the TextGenerator, Transcriber, and TextToSpeech classes.
"""

import time

import click

# from .gui import main as main3
from .models import llm, stt, tts


@click.command()
@click.option(
    "-lm",
    "--llm-model",
    help="LLM model to use.",
    required=True,
    type=click.Choice(list(llm.all_models.keys()), case_sensitive=True),
)
@click.option(
    "-sm",
    "--stt-model",
    default=list(stt.all_models.keys())[0],
    help="STT model to use.",
    required=True,
    type=click.Choice(list(stt.all_models.keys()), case_sensitive=True),
)
@click.option(
    "-tm",
    "--tts-model",
    default=list(tts.all_models.keys())[0],
    help="TTS model to use.",
    required=True,
    type=click.Choice(list(tts.all_models.keys()), case_sensitive=True),
)
def main(llm_model, stt_model, tts_model):
    """
    Main function to run the CLI program.
    """
    # system_initial_context = input("[system]> ")
    generation_args = {
        "max_new_tokens": 200,
        "num_beams": 1,
    }

    llm_model = llm.all_models[llm_model]
    llm_model.noop()
    context_id = "cli-chat"

    stt_model = stt.all_models[stt_model]
    stt_model.noop()

    tts_model = tts.all_models[tts_model]
    tts_model.noop()

    while True:
        audio_data = stt_model.record_audio()

        if audio_data is not None:
            print("Transcribing...")
            transcription = stt_model.transcribe(audio_data)

            user_input = transcription.strip()

            if user_input:
                print(f"[user]> {user_input}")

                if user_input.lower() in ["exit", "halt", "stop", "quit"]:
                    break

                reply = llm_model.generate(user_input, context_id=context_id, generation_args=generation_args)
                print(f"[assistant]> {reply['content']}")
                tts_model.speak(reply["content"])
        else:
            print("No sound detected.")

        time.sleep(1)  # Pause briefly before next listening
