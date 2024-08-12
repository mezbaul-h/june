"""
This module provides utility classes and functions.
"""

import logging
import os
import sys
import threading
from typing import Any

from colorama import Fore, Style

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("%(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


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


def deep_merge_dicts(old: dict, new: dict) -> dict:
    """
    Merge two dictionaries recursively.

    Args:
        old: The original dictionary.
        new: The new dictionary to merge into the original.

    Returns:
        The merged dictionary.
    """
    merged = old.copy()  # Start with a shallow copy of the old dictionary

    for key, value in new.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value

    return merged


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
