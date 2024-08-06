import asyncio
import io
import json
import math
import wave
from typing import List, Optional

from june_va.providers.common import LLMMessage
from june_va.utils import TokenChunker


class Engine:
    def __init__(self, llm_model, stt_model, tts_model):
        self.llm_model = llm_model
        self.stt_model = stt_model
        self.tts_model = tts_model

    def create_wav_header(self, params):
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wav_file:
                wav_file.setparams(params)

            header = wav_io.getvalue()

            # Set the file size to 0 (bytes 4 to 8)
            header = header[:4] + bytes(4) + header[8:]

            # Set the data size to 0 (bytes 40 to 44)
            header = header[:40] + bytes(4) + header[44:]

            return header

    async def process_text(self, messages: List[LLMMessage], cli: bool = False, queue: Optional[asyncio.Queue] = None):
        voice_output = self.tts_model is not None

        audio_buffer = io.BytesIO()
        first_chunk = True

        def audio_buffer_generator(_input):
            nonlocal first_chunk
            wav_bytes = self.tts_model.forward(_input)

            if not wav_bytes:
                return

            # Use wave module to read WAV file properly
            with io.BytesIO(wav_bytes) as wav_file:
                with wave.open(wav_file, "rb") as wav:
                    params = wav.getparams()
                    pcm_data = wav.readframes(wav.getnframes())

            if first_chunk:
                yield self.create_wav_header(params)
                first_chunk = False

            audio_buffer.write(pcm_data)
            audio_buffer.seek(0)

            while True:
                data = audio_buffer.read(1024)

                if not data:
                    break

                yield data

            audio_buffer.seek(0)
            audio_buffer.truncate(0)

        for chunk in TokenChunker(self.llm_model.forward(messages), print_tokens=cli):
            if voice_output and chunk.content.strip():
                if queue:
                    queue.put_nowait(chunk.content)
                else:
                    for buffer in audio_buffer_generator(chunk.content):
                        yield buffer
            else:
                yield json.dumps(chunk.model_dump(mode="json")) + "\n"

        remaining_audio_buffer = audio_buffer.getvalue()

        if remaining_audio_buffer:
            yield remaining_audio_buffer
