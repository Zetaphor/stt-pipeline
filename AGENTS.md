# Agent Guidelines for stt-pipeline

## Project overview

This is a FastAPI service that combines Whisper speech-to-text (via an external endpoint) with pyannote speaker diarization. It exposes an OpenAI-compatible `/v1/audio/transcriptions` API that returns speaker-labeled transcripts.

## Architecture

- `src/stt_pipeline/app.py` -- FastAPI application, endpoints, lifespan model loading, ffmpeg audio conversion
- `src/stt_pipeline/config.py` -- Environment variable settings with defaults
- `src/stt_pipeline/transcribe.py` -- Async HTTP client that forwards audio to the Whisper endpoint
- `src/stt_pipeline/diarize.py` -- pyannote pipeline loading (singleton) and inference
- `src/stt_pipeline/merge.py` -- Aligns Whisper timestamped segments with pyannote speaker labels

## Key design decisions

- **Whisper runs externally.** This service does NOT run Whisper itself. It forwards audio to a separate Whisper endpoint (Lemonade Server on an AMD NPU by default). Do not add Whisper model loading to this codebase.
- **Diarization runs on CPU by default.** The `DIARIZE_DEVICE` env var controls this. The intent is zero GPU contention with LLM inference running on the same machine.
- **pyannote v4 API.** The pipeline uses `token=` (not `use_auth_token=`) for authentication and returns `DiarizeOutput` where the annotation is at `.speaker_diarization`.
- **Non-WAV conversion.** MP3 and other compressed formats cause sample count mismatches in pyannote's torchaudio backend. All non-WAV uploads are converted to 16 kHz mono WAV via ffmpeg before diarization. The original file is sent to Whisper unchanged.
- **Concurrent execution.** Whisper transcription (async HTTP) and diarization (threaded CPU) run in parallel via `asyncio.gather`.

## Development

```bash
uv sync                    # Install dependencies
uv run stt-pipeline        # Start the server
uv run uvicorn stt_pipeline.app:app --reload  # Dev mode with auto-reload
```

## Testing changes

After modifying source files, restart the systemd service if running in production:

```bash
systemctl --user restart stt-pipeline
```

Verify with the health endpoint:

```bash
curl http://127.0.0.1:8003/health
```

## Dependencies

Managed via `pyproject.toml` with uv. PyTorch is pinned to CPU-only builds via the `[[tool.uv.index]]` pytorch-cpu index. Do not add GPU torch indexes unless the user explicitly requests it.
