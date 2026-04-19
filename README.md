# stt-pipeline

Diarized speech-to-text service that combines [Whisper](https://github.com/openai/whisper) transcription (via [Lemonade Server](https://github.com/lemonade-sdk/lemonade) on an AMD NPU) with [pyannote](https://github.com/pyannote/pyannote-audio) speaker diarization on CPU.

Exposes an OpenAI-compatible `/v1/audio/transcriptions` endpoint that returns speaker-labeled transcripts.

## How it works

1. Client uploads audio to `POST /v1/audio/transcriptions`
2. Non-WAV files are converted to 16 kHz mono WAV via ffmpeg (pyannote requirement)
3. Whisper transcription and speaker diarization run **concurrently**:
   - Original audio is forwarded to a Whisper endpoint for timestamped transcription
   - WAV audio is processed by pyannote's `speaker-diarization-3.1` pipeline on CPU
4. Whisper segments are merged with speaker labels by largest time overlap
5. Response includes `[SPEAKER_XX]:` prefixed text and a `speaker` field on each segment

## Requirements

- Python >= 3.11, < 3.14
- [uv](https://docs.astral.sh/uv/) package manager
- ffmpeg (for non-WAV audio conversion)
- A running Whisper endpoint (default: Lemonade Server on `http://127.0.0.1:8002`)
- HuggingFace account with accepted licenses for:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

## Installation

```bash
git clone git@github.com:Zetaphor/stt-pipeline.git
cd stt-pipeline
uv sync
```

PyTorch is pinned to CPU-only builds to avoid pulling CUDA dependencies.

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_URL` | `http://127.0.0.1:8002` | Whisper endpoint base URL |
| `WHISPER_MODEL` | `Whisper-Large-v3-Turbo` | Model name sent to Whisper |
| `DIARIZE_MODEL` | `pyannote/speaker-diarization-3.1` | HuggingFace diarization model |
| `DIARIZE_DEVICE` | `cpu` | PyTorch device (`cpu` or `cuda`) |
| `STT_HOST` | `0.0.0.0` | Listen address |
| `STT_PORT` | `8003` | Listen port |
| `HF_TOKEN` | *(auto-detected)* | HuggingFace token; falls back to `~/.cache/huggingface/token` |

## Usage

### Start the server

```bash
uv run stt-pipeline
```

Or directly:

```bash
uv run uvicorn stt_pipeline.app:app --host 0.0.0.0 --port 8003
```

### Transcribe audio

```bash
curl http://127.0.0.1:8003/v1/audio/transcriptions \
  -F "file=@recording.mp3" \
  -F "model=Whisper-Diarized"
```

### Plain text output

```bash
curl http://127.0.0.1:8003/v1/audio/transcriptions \
  -F "file=@recording.mp3" \
  -F "response_format=text"
```

### Health check

```bash
curl http://127.0.0.1:8003/health
```

## API

### `POST /v1/audio/transcriptions`

OpenAI-compatible multipart form upload.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | *(required)* | Audio file (WAV, MP3, FLAC, etc.) |
| `model` | string | `Whisper-Diarized` | Model name (for client compatibility) |
| `response_format` | string | `verbose_json` | `verbose_json` or `text` |
| `language` | string | *(auto)* | ISO language code hint |

The `verbose_json` response extends the standard Whisper format with a `speaker` field on each segment.

### `GET /health`

Returns JSON with Whisper reachability and diarization model status.

## Systemd

A systemd user service can be installed for auto-start:

```bash
# Copy or symlink the service file
cp stt-pipeline.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now stt-pipeline
```

## Performance

| Component | Speed |
|-----------|-------|
| Whisper (NPU) | ~17x realtime |
| Diarization (CPU) | ~1-3x realtime |

Whisper and diarization run concurrently, so wall time equals the slower of the two. For audio longer than ~30 seconds, CPU diarization dominates. Set `DIARIZE_DEVICE=cuda` for significantly faster diarization at the cost of GPU memory.

## License

MIT
