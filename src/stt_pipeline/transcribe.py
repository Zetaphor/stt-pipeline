from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from . import config


async def transcribe(
    audio_path: Path,
    *,
    language: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Send an audio file to the Whisper endpoint and return verbose_json."""
    url = f"{config.WHISPER_URL}/v1/audio/transcriptions"
    fields: dict[str, Any] = {
        "model": (None, config.WHISPER_MODEL),
        "response_format": (None, "verbose_json"),
    }
    if language:
        fields["language"] = (None, language)

    mime = "audio/wav" if audio_path.suffix in (".wav", ".wave") else "application/octet-stream"
    fields["file"] = (audio_path.name, audio_path.read_bytes(), mime)

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=120.0)
    try:
        resp = await client.post(url, files=fields)
        resp.raise_for_status()
        return resp.json()
    finally:
        if owns_client:
            await client.aclose()
