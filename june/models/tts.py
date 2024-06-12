"""
This module provides a Text-to-Speech (TTS) class for generating speech from text using the TTS library.
"""

import torch
from datasets import load_dataset
from transformers import pipeline

from ..settings import settings
from .common import ModelBase


class TTS(ModelBase):
    def __init__(self, kwargs):
        super().__init__(kwargs)

        self.pipeline = pipeline(
            "text-to-speech",
            device=self.device,
            model=self.model_id,
            token=settings.HF_TOKEN,
            torch_dtype="auto",
        )

        if "speaker_embeddings" in self.generation_args:
            embeddings_dataset = load_dataset(
                "Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True
            )
            speaker_embedding = torch.tensor(
                embeddings_dataset[self.generation_args["speaker_embeddings"]]["xvector"]
            ).unsqueeze(0)
            self.generation_args.update({"speaker_embeddings": speaker_embedding})

    def synthesise(self, text):
        synthesis = self.pipeline(text, forward_params=self.generation_args)

        return synthesis
