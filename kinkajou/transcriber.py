"""
This module provides a Transcriber class for recording audio, transcribing it using Whisper, and saving the
transcription.
"""

import tempfile
import time

import numpy as np
import pyaudio
import whisper
from scipy.io import wavfile

from kinkajou.settings import TORCH_DEVICE
from kinkajou.utils import suppress_stdout_stderr


class Transcriber:
    """
    Class for recording audio, transcribing it using Whisper, and saving the transcription.

    Attributes
    ----------
    RATE : int
        Sample rate for audio recording.
    CHUNK : int
        Buffer size for audio recording.
    THRESHOLD : int
        Silence threshold for audio recording.
    SILENCE_LIMIT : int
        Seconds of silence before stopping recording.
    model : whisper.Whisper
        whisper.Whisper model for audio transcription.
    """

    RATE = 24000  # Sample rate
    CHUNK = 2048  # Buffer size
    THRESHOLD = 2000  # Silence threshold
    SILENCE_LIMIT = 5  # Seconds of silence before stopping recording

    def __init__(self, model):
        """
        Initializes the Transcriber object.

        Parameters
        ----------
        model : str
            Name of the Whisper model to be used for audio transcription.
        """
        # Initialize the Whisper model
        self.model = whisper.load_model(model, device=TORCH_DEVICE)

    @staticmethod
    def is_silent(data):
        """
        Checks if the audio data is silent.

        Parameters
        ----------
        data : numpy.ndarray
            Array of audio data.

        Returns
        -------
        bool
            True if the audio data is silent, False otherwise.
        """
        return np.max(data) < Transcriber.THRESHOLD

    def record_audio(self):
        """
        Records audio from the microphone and returns the data.

        Returns
        -------
        numpy.ndarray or None
            Recorded audio data, or None if no sound is detected.
        """
        with suppress_stdout_stderr():
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16, channels=1, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK
            )

        audio_data = []
        current_silence = 0
        recording = False

        print("Listening for sound...")

        while True:
            data = np.frombuffer(stream.read(self.CHUNK), dtype=np.int16)

            if not recording and not self.is_silent(data):
                print("Sound detected. Starting recording...")
                recording = True

            if recording:
                audio_data.append(data)
                if self.is_silent(data):
                    current_silence += 1
                else:
                    current_silence = 0

                if current_silence > (self.SILENCE_LIMIT * self.RATE / self.CHUNK):
                    print("Silence detected. Stopping recording...")
                    break

        stream.stop_stream()
        stream.close()
        p.terminate()

        if recording:
            audio_data = np.concatenate(audio_data)
            return audio_data
        else:
            return None

    @staticmethod
    def save_wav(filename, data):
        """
        Saves audio data to a WAV file.

        Parameters
        ----------
        filename : str
            Name of the WAV file to save.
        data : numpy.ndarray
            Audio data to save.
        """
        wavfile.write(filename, Transcriber.RATE, data)

    def transcribe_audio(self, audio_data):
        """
        Transcribes audio data using Whisper.

        Parameters
        ----------
        audio_data : numpy.ndarray
            Audio data to transcribe.

        Returns
        -------
        str
            Transcribed text.
        """
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            Transcriber.save_wav(temp_file.name, audio_data)

            result = self.model.transcribe(temp_file.name)
        return result["text"]

    def run_forever(self):
        """
        Continuously records audio, transcribes it, and yields the transcription.

        Yields
        ------
        str
            The transcription.
        """
        while True:
            audio_data = self.record_audio()

            if audio_data is not None:
                print("Transcribing...")
                transcription = self.transcribe_audio(audio_data)
                yield transcription
            else:
                print("No sound detected.")

            time.sleep(1)  # Pause briefly before next listening
