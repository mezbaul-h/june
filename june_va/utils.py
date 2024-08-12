"""
This module provides utility classes and functions.
"""

import logging
import os
import sys
import threading
from typing import Any, Iterator, List

from colorama import Fore, Style

from june_va.providers.common import LLMMessage

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("%(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)
logger.setLevel(logging.DEBUG)


class TokenChunker:
    MIN_CHUNK_SIZE = 15
    SPLITTERS = [".", ",", "?", ":", ";"]

    def combine_buffer(self, clear: bool = False) -> LLMMessage:
        combined_content = "".join([item.content for item in self.buffer])
        role = self.buffer[0].role

        if clear:
            self.buffer.clear()

        return LLMMessage(content=combined_content, role=role)

    def __iter__(self):
        for message in self.source:
            token = message.content

            # Skip empty ("") tokens.
            if token:
                if self.print_tokens:
                    print(token, end="", flush=True)

                self.buffer.append(message)

                # Check if buffer is ready to be chunked
                if token == "\n" or (len(self.buffer) >= self.MIN_CHUNK_SIZE and token in self.SPLITTERS):
                    yield self.combine_buffer(clear=True)

        # Process any remaining text in buffer
        if self.buffer:
            yield self.combine_buffer()

    def __init__(self, source: Iterator[LLMMessage], print_tokens: bool = False) -> None:
        self.buffer: List[LLMMessage] = []
        self.print_tokens = print_tokens
        self.source = source


class ThreadSafeState:
    """
    A thread-safe class for managing a shared state value.

    This class provides methods for setting and getting the value in a thread-safe manner
    using a lock.

    Args:
        value: The initial value of the shared state.

    Attributes:
        _value: The shared state value.
        _lock: A lock object used for thread-safe access to the shared state.
    """

    def __init__(self, value: Any) -> None:
        self._value = value
        self._lock = threading.Lock()

    def set_value(self, value: Any) -> None:
        """
        Set the shared state value in a thread-safe manner.

        Args:
            value: The new value to set for the shared state.
        """
        with self._lock:
            self._value = value

    def get_value(self) -> Any:
        """
        Get the shared state value in a thread-safe manner.

        Returns:
            The current value of the shared state.
        """
        with self._lock:
            return self._value


class suppress_stdout_stderr:
    """
    A context manager for temporarily suppressing stdout and stderr.

    This context manager redirects stdout and stderr to null files
    within the context, and restores them to their original values
    when the context is exited.
    """

    def __enter__(self) -> "suppress_stdout_stderr":
        """
        Suppresses stdout and stderr by redirecting them to null files.

        Returns:
            The instance of the context manager.
        """
        # Open null files for writing
        self.out_null_file = open(os.devnull, "w")
        self.err_null_file = open(os.devnull, "w")

        # Save original file descriptors
        self.old_stdout_file_no_undup = sys.stdout.fileno()
        self.old_stderr_file_no_undup = sys.stderr.fileno()

        # Duplicate file descriptors
        self.old_stdout_file_no = os.dup(sys.stdout.fileno())
        self.old_stderr_file_no = os.dup(sys.stderr.fileno())

        # Redirect stdout and stderr to null files
        os.dup2(self.out_null_file.fileno(), self.old_stdout_file_no_undup)
        os.dup2(self.err_null_file.fileno(), self.old_stderr_file_no_undup)

        # Save original stdout and stderr
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        # Set stdout and stderr to null files
        sys.stdout = self.out_null_file
        sys.stderr = self.err_null_file

        return self

    def __exit__(self, *_) -> None:
        """
        Restores stdout and stderr to their original values.
        """
        # Restore stdout and stderr
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        # Restore original file descriptors
        os.dup2(self.old_stdout_file_no, self.old_stdout_file_no_undup)
        os.dup2(self.old_stderr_file_no, self.old_stderr_file_no_undup)

        # Close duplicate file descriptors
        os.close(self.old_stdout_file_no)
        os.close(self.old_stderr_file_no)

        # Close null files
        self.out_null_file.close()
        self.err_null_file.close()


def print_system_message(message: str, color: str = Fore.BLUE, log_level: int = logging.DEBUG) -> None:
    """
    Print a message with a colored system prompt.

    Args:
        message: The message to be printed.
        color: The color code for the message text (e.g., Fore.BLUE).
            Defaults to Fore.BLUE.
        log_level: The logging level for the message (e.g., logging.DEBUG).
            Defaults to logging.DEBUG.
    """
    logger.log(log_level, f"{Style.BRIGHT}{Fore.YELLOW}[system]> {Style.NORMAL}{color}{message}{Style.RESET_ALL}")
