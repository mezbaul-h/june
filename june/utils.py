"""
This module provides various utility methods and classes.
"""

import os
import sys


class suppress_stdout_stderr:
    """
    Context manager to suppress stdout and stderr.
    """

    def __enter__(self):
        """
        Suppresses stdout and stderr.
        """
        # Open null files for writing
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        # Save original file descriptors
        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        # Duplicate file descriptors
        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        # Redirect stdout and stderr to null files
        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        # Save original stdout and stderr
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        # Set stdout and stderr to null files
        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        """
        Restores stdout and stderr.
        """
        # Restore stdout and stderr
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        # Restore original file descriptors
        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        # Close duplicate file descriptors
        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        # Close null files
        self.outnull_file.close()
        self.errnull_file.close()
