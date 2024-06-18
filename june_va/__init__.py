from .utils import suppress_stdout_stderr

# Suppress pygame's support prompt without the need to set PYGAME_HIDE_SUPPORT_PROMPT environment variable
with suppress_stdout_stderr():
    import pygame

    _ = pygame


__version__ = "0.0.1"
