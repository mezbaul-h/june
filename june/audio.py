import numpy as np

try:
    import pyaudio
except ImportError:
    pyaudio = None

from .utils import suppress_stdout_stderr


class AudioIO:
    RATE = 24000  # Sample rate
    CHUNK = 2048  # Buffer size
    THRESHOLD = 1000  # Silence threshold
    SILENCE_LIMIT = 3  # Seconds of silence before stopping recording

    def play_audio(self, audio_data):
        if audio_data:
            with suppress_stdout_stderr():
                p = pyaudio.PyAudio()

            stream = p.open(format=pyaudio.paInt16, channels=1, rate=audio_data["sampling_rate"], output=True)

            # Convert normalized float32 data back to int16 for playback
            int_data = (audio_data["audio"] * np.iinfo(np.int16).max).astype(np.int16)

            # Write the audio data to the stream in chunks
            for i in range(0, len(int_data), self.CHUNK):
                stream.write(int_data[i : i + self.CHUNK].tobytes())

            stream.stop_stream()
            stream.close()
            p.terminate()
        else:
            print("No audio data to play.")

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

        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

        frames = []
        current_silence = 0
        recording = False

        print("Listening for sound...")

        while True:
            data = np.frombuffer(stream.read(self.CHUNK), dtype=np.int16)

            if not recording and not self.is_silent(data):
                print("Sound detected. Starting recording...")
                recording = True

            if recording:
                frames.append(data)
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
            raw_data = np.hstack(frames)

            # Convert to float32 and normalize only when returning
            normalized_data = raw_data.astype(np.float32) / np.iinfo(np.int16).max
            return {
                "raw": normalized_data,
                "sampling_rate": self.RATE,
            }
        else:
            return None

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
        return np.max(data) < AudioIO.THRESHOLD
