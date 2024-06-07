"""
This module provides a Text-to-Speech (TTS) class for generating speech from text using the TTS library.
"""

import tempfile
import wave

import pyaudio
from TTS.api import TTS

from ..settings import settings
from ..utils import DeferredInitProxy, suppress_stdout_stderr
from .common import ModelBase


class TTSWrapper(ModelBase):
    """
    Class for Text-to-Speech functionality.

    Attributes
    ----------
    model : TTS
        TTS instance for generating speech.
    """

    def __init__(self, **kwargs):
        """
        Initializes the TextToSpeech object.
        """
        with suppress_stdout_stderr():
            self.model = TTS(kwargs["model"], gpu=False if settings.TORCH_DEVICE == "cpu" else True)

    def speak(self, text):
        """
        Generates speech from text and plays it.

        Parameters
        ----------
        text : str
            The text to convert to speech.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as audio_file:
            with suppress_stdout_stderr():
                # Generate speech by cloning a voice using default settings
                self.model.tts_to_file(
                    text=text,
                    file_path=audio_file.name,
                )
                self.play_wav(audio_file)

    @staticmethod
    def play_wav(filename):
        """
        Plays a WAV file.

        Parameters
        ----------
        filename : any
            The path to the WAV file to play.
        """
        # Open the WAV file
        wf = wave.open(filename, "rb")

        # Create an interface to PortAudio
        p = pyaudio.PyAudio()

        # Open a .Stream object to write the WAV file to
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )

        # Read data in chunks
        chunk = 1024
        data = wf.readframes(chunk)

        # Play the sound by writing the audio data to the stream
        while data:
            stream.write(data)
            data = wf.readframes(chunk)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

        # Close PyAudio
        p.terminate()

        # Close the WAV file
        wf.close()


all_models = dict(
    ljspeech_glow_tts=DeferredInitProxy(TTSWrapper, model="tts_models/en/ljspeech/glow-tts"),
)
