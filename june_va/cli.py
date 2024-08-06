import asyncio
import json

import click

from .interfaces.api import main as api_main
from .interfaces.cli import main as cli_main

__version__ = "0.1.0"

from .settings import process_user_settings


@click.group()
@click.option(
    "-v",
    "--verbose",
    help="Verbose mode.",
    is_flag=True,
)
@click.version_option(__version__)
def command_group(**kwargs):
    """Main command group."""
    pass


@command_group.command()
@click.option(
    "-c",
    "--config",
    help="Configuration file.",
    nargs=1,
    required=False,
    type=click.File("r", encoding="utf-8"),
)
def api(**kwargs):
    """Start the service."""
    user_config = json.loads(kwargs["config"].read()) if kwargs["config"] else {}
    llm_model, stt_model, tts_model = process_user_settings(user_config)
    api_main(llm_model, stt_model, tts_model)


@command_group.command()
@click.option(
    "-c",
    "--config",
    help="Configuration file.",
    nargs=1,
    required=False,
    type=click.File("r", encoding="utf-8"),
)
def cli(**kwargs):
    """Stop the service."""
    user_config = json.loads(kwargs["config"].read()) if kwargs["config"] else {}
    llm_model, stt_model, tts_model = process_user_settings(user_config)
    cli_main(llm_model, stt_model, tts_model)
