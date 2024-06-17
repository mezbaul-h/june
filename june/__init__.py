from .utils import suppress_stdout_stderr

# suppress pygame's support prompt without the need to PYGAME_HIDE_SUPPORT_PROMPT to be set
with suppress_stdout_stderr():
    import pygame

    _ = pygame
