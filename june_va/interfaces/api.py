import io
import json
import wave
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pydub import AudioSegment

from june_va.providers.common import LLMMessage
from june_va.utils import TokenChunker

app = FastAPI()


class JsonInput(BaseModel):
    text: str


def _manipulate_wav_header(filename):
    with open(filename, "r+b") as f:
        # Read the first 44 bytes of the file
        header = f.read(44)

        # Set the file size to the highest possible value (bytes 4 to 8)
        header = header[:4] + bytes.fromhex("ffffffff") + header[8:]

        # Set the data size to the highest possible value (bytes 40 to 44)
        header = header[:40] + bytes.fromhex("ffffffff") + header[44:]

        # Go back to the beginning of the file
        f.seek(0)

        # Write the modified header back to the file
        f.write(header)


def create_wav_header(params):
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setparams(params)

        header = wav_io.getvalue()

        # Set the file size to 0 (bytes 4 to 8)
        header = header[:4] + bytes(4) + header[8:]

        # Set the data size to 0 (bytes 40 to 44)
        header = header[:40] + bytes(4) + header[44:]

        return header


async def _process_text(messages: List[LLMMessage], voice_output: bool):
    llm_model = getattr(app, "llm_model")
    tts_model = getattr(app, "tts_model")
    tts_input_buffer = ""
    tts_input_queue = []
    audio_buffer = io.BytesIO()
    first_chunk = True

    def audio_buffer_generator(_input):
        nonlocal first_chunk
        wav_bytes = tts_model.forward(_input)

        # Use wave module to read WAV file properly
        with io.BytesIO(wav_bytes) as wav_file:
            with wave.open(wav_file, "rb") as wav:
                params = wav.getparams()
                pcm_data = wav.readframes(wav.getnframes())

        if first_chunk:
            yield create_wav_header(params)
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

    for chunk in TokenChunker(llm_model.forward(messages)):
        if voice_output:
            tts_input_buffer += tts_model.normalize_text(chunk.content)

            if len(tts_input_buffer) >= (TokenChunker.MIN_CHUNK_SIZE * 2):
                tts_input_queue.append(tts_input_buffer)
                tts_input_buffer = ""

                if len(tts_input_queue) == 2:
                    last_input = tts_input_queue[-1]
                    second_last_input = tts_input_queue[-2]

                    if len(last_input) < (len(second_last_input) * 0.5):
                        final_input = second_last_input + last_input
                        tts_input_queue.clear()
                    else:
                        final_input = second_last_input
                        tts_input_queue.pop(0)

                    for buffer in audio_buffer_generator(final_input):
                        yield buffer
        else:
            yield json.dumps(chunk.model_dump(mode="json")) + "\n"

    if tts_input_buffer:
        tts_input_queue.append(tts_input_buffer)

    if tts_input_queue:
        for buffer in audio_buffer_generator("".join(tts_input_queue)):
            yield buffer

    remaining_audio_buffer = audio_buffer.getvalue()

    if remaining_audio_buffer:
        yield remaining_audio_buffer


async def _speech_to_text(file: UploadFile) -> str:
    if "audio/wav" not in file.content_type:
        raise HTTPException(status_code=400, detail="Invalid file type. Only WAV files are accepted.")

        # Read the file
    audio_data = await file.read()

    # Load audio data using pydub
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio file: {str(e)}")

    stt_model = getattr(app, "stt_model")

    raw_data = np.array(audio_segment.get_array_of_samples())
    normalized_data = raw_data.astype(np.float32) / np.iinfo(np.int16).max

    return stt_model.forward(
        {
            "raw": normalized_data,
            "sampling_rate": audio_segment.frame_rate,
        }
    )


async def _text_to_speech(text: str) -> bytes:
    tts_model = getattr(app, "tts_model")

    return tts_model.forward(text)


@app.post("/api/chat")
async def process_input(
    request: Request,
    voice_output: bool = False,
):
    body = await request.json()

    if not isinstance(body, list):
        raise HTTPException(400)

    messages = [LLMMessage(**item) for item in body]

    return StreamingResponse(
        _process_text(messages, voice_output),
        media_type="application/x-ndjson" if not voice_output else "audio/wav",
    )


@app.post("/api/stt")
async def process_input(
    request: Request,
):
    body = dict((await request.form()).items())

    text = await _speech_to_text(body["file"])

    return JSONResponse(
        {
            "text": text,
        }
    )


def main(llm_model, stt_model, tts_model):
    setattr(app, "llm_model", llm_model)
    setattr(app, "stt_model", stt_model)
    setattr(app, "tts_model", tts_model)

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
