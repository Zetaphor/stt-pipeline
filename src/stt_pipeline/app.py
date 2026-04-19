from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from . import config
from .diarize import diarize, is_loaded, load_pipeline
from .merge import merge_transcript
from .transcribe import transcribe

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info("Loading pyannote diarization model (this may take a moment on first run) ...")
    await asyncio.to_thread(load_pipeline)
    log.info("stt-pipeline ready on %s:%s", config.HOST, config.PORT)
    yield


app = FastAPI(title="stt-pipeline", lifespan=lifespan)


@app.get("/health")
async def health():
    whisper_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{config.WHISPER_URL}/v1/models")
            whisper_ok = r.is_success
    except httpx.HTTPError:
        pass

    diarize_ok = is_loaded()
    ok = whisper_ok and diarize_ok
    body = {
        "status": "ok" if ok else "degraded",
        "whisper": "reachable" if whisper_ok else "unreachable",
        "diarization": "loaded" if diarize_ok else "not loaded",
    }
    return JSONResponse(body, status_code=200 if ok else 503)


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default="Whisper-Diarized"),
    response_format: str = Form(default="verbose_json"),
    language: str | None = Form(default=None),
):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    # Whisper (Lemonade) handles MP3 natively; pyannote needs 16kHz mono WAV.
    wav_path: Path | None = None
    if suffix.lower() not in (".wav", ".wave"):
        wav_path = tmp_path.with_suffix(".wav")
        await asyncio.to_thread(
            _convert_to_wav, tmp_path, wav_path,
        )
    diarize_input = wav_path or tmp_path

    try:
        whisper_result, speaker_segments = await asyncio.gather(
            transcribe(tmp_path, language=language),
            asyncio.to_thread(diarize, diarize_input),
        )
        merged = merge_transcript(whisper_result, speaker_segments)

        if response_format == "text":
            return PlainTextResponse(merged["text"])

        return JSONResponse(merged)
    finally:
        tmp_path.unlink(missing_ok=True)
        if wav_path:
            wav_path.unlink(missing_ok=True)


def _convert_to_wav(src: Path, dst: Path) -> None:
    """Convert any audio format to 16kHz mono WAV via ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )


def main():
    uvicorn.run(
        "stt_pipeline.app:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
    )
