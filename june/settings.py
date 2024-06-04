"""
This module provides settings related to the application.
"""

import os

import torch

HF_TOKEN = os.getenv("HF_TOKEN")
"""Hugging Face token for authentication."""

TORCH_DEVICE = "cpu"
"""Torch device to be used for computation."""

if torch.cuda.is_available():
    TORCH_DEVICE = "cuda"
    """Torch device to be used for computation."""
elif torch.backends.mps.is_available():
    TORCH_DEVICE = "mps"
    """Torch device to be used for computation."""
