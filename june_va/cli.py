import json
import logging

import click

from .interfaces.rest_api import main as api_main
from .interfaces.cli import main as cli_main

from june_va import __version__

from .settings import process_user_settings
from .utils import logger


@click.command()
@click.option(
    "-c",
    "--config",
    help="Configuration file.",
    nargs=1,
    required=False,
    type=click.File("r", encoding="utf-8"),
)
@click.option(
    "-H",
    "--host",
    default="127.0.0.1",
    help="REST API host.",
    nargs=1,
    type=str,
)
@click.option(
    "-p",
    "--port",
    default=8000,
    help="REST API server port.",
    nargs=1,
    type=int,
)
@click.option(
    "-s",
    "--serve",
    help="Start REST API server.",
    is_flag=True,
)
@click.option(
    "-v",
    "--verbose",
    help="Verbose mode.",
    is_flag=True,
)
@click.version_option(__version__)
def main(**kwargs):
    """
    Local voice assistant tool.
    """
    if not kwargs["verbose"]:
        logger.setLevel(logging.INFO)

    user_config = json.loads(kwargs["config"].read()) if kwargs["config"] else {}
    llm_model, stt_model, tts_model = process_user_settings(user_config)

    if kwargs["serve"]:
        api_main(llm_model, stt_model, tts_model, kwargs['host'], kwargs['port'])
    else:
        cli_main(llm_model, stt_model, tts_model)
