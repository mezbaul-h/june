"""
This module provides a CLI program for interacting with the TextGenerator, Transcriber, and TextToSpeech classes.
"""

import time

import click
import torch
from datasets import load_dataset

from .audio import AudioIO
from .models import llm, stt, tts
from .utils import get_default_microphone_info


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

    # get_default_microphone_info()
    audio_io = AudioIO()

    llm_model = llm.all_models[llm_model]
    llm_model.noop()
    context_id = "cli-chat"

    stt_model = stt.all_models[stt_model]
    stt_model.noop()

    tts_model = tts.all_models[tts_model]
    tts_model.noop()

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    while True:
        audio_data = audio_io.record_audio()

        if audio_data is not None:
            print("Transcribing...")

            transcription = stt_model.transcribe(
                audio_data,
                generation_args={
                    "batch_size": 8,
                },
            )
            user_input = transcription

            if user_input:
                print(f"[user]> {user_input}")

                if user_input.lower() in ["exit", "halt", "stop", "quit"]:
                    break

                reply = llm_model.generate(user_input, context_id=context_id, generation_args=generation_args)
                print(f"[assistant]> {reply['content']}")

                synthesis = tts_model.synthesise(
                    reply["content"], generation_args={"forward_params": {"speaker_embeddings": speaker_embedding}}
                )
                audio_io.play_audio(synthesis)
        else:
            print("No sound detected.")

        time.sleep(1)  # Pause briefly before next listening
