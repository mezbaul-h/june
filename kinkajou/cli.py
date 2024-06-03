"""
This module provides a CLI program for interacting with the TextGenerator, Transcriber, and TextToSpeech classes.
"""

import click

from .llm import TextGenerator
from .transcriber import Transcriber
from .tts import TextToSpeech


@click.command()
def main():
    """
    Main function to run the CLI program.
    """
    generation_kwargs = {
        "max_new_tokens": 200,
        "num_beams": 1,
    }

    system_initial_context = input("[system]> ")

    generator = TextGenerator(
        pipeline_kwargs=generation_kwargs,
        system_initial_context=system_initial_context,
    )
    context_id = "cli-chat"

    tts = TextToSpeech()
    transcriber = Transcriber(model="base.en")

    for transcription in transcriber.run_forever():
        user_input = transcription.strip()

        if user_input:
            print(f"[user]> {user_input}")

            if user_input.lower() in ["exit", "halt", "stop", "quit"]:
                break

            reply = generator.generate(user_input, context_id)
            print(f"[assistant]> {reply['content']}")
            tts.say(reply["content"])
