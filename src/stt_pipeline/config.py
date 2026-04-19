from __future__ import annotations

import os
from pathlib import Path


def _resolve_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    for candidate in (
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ):
        if candidate.is_file():
            return candidate.read_text().strip()
    return None


WHISPER_URL: str = os.environ.get("WHISPER_URL", "http://127.0.0.1:8002")
WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "Whisper-Large-v3-Turbo")
DIARIZE_MODEL: str = os.environ.get("DIARIZE_MODEL", "pyannote/speaker-diarization-3.1")
DIARIZE_DEVICE: str = os.environ.get("DIARIZE_DEVICE", "cpu")
HOST: str = os.environ.get("STT_HOST", "0.0.0.0")
PORT: int = int(os.environ.get("STT_PORT", "8003"))
HF_TOKEN: str | None = _resolve_hf_token()
