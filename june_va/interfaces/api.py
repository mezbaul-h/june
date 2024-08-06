import io
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pydub import AudioSegment

from june_va.engine import Engine
from june_va.providers.common import LLMMessage

app = FastAPI()
engine: Optional[Engine] = None


class JsonInput(BaseModel):
    text: str


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

    raw_data = np.array(audio_segment.get_array_of_samples())
    normalized_data = raw_data.astype(np.float32) / np.iinfo(np.int16).max

    return engine.stt_model.forward(
        {
            "raw": normalized_data,
            "sampling_rate": audio_segment.frame_rate,
        }
    )


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
        engine.process_text(messages),
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
    global engine

    engine = Engine(llm_model, stt_model, tts_model)

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
