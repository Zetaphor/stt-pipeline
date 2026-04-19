from __future__ import annotations

import logging
from pathlib import Path

import torch
from pyannote.audio import Pipeline as DiarizationPipeline

from . import config

log = logging.getLogger(__name__)

_pipeline: DiarizationPipeline | None = None


def load_pipeline() -> DiarizationPipeline:
    """Load the pyannote diarization pipeline (call once at startup)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    log.info("Loading diarization model %s on %s ...", config.DIARIZE_MODEL, config.DIARIZE_DEVICE)
    _pipeline = DiarizationPipeline.from_pretrained(
        config.DIARIZE_MODEL,
        token=config.HF_TOKEN,
    )
    _pipeline.to(torch.device(config.DIARIZE_DEVICE))
    log.info("Diarization model loaded.")
    return _pipeline


def is_loaded() -> bool:
    return _pipeline is not None


def diarize(audio_path: Path) -> list[tuple[float, float, str]]:
    """Run diarization and return a list of (start_sec, end_sec, speaker_label)."""
    pipeline = load_pipeline()
    result = pipeline(str(audio_path))
    # pyannote v4 returns DiarizeOutput; the Annotation is in .speaker_diarization
    annotation = getattr(result, "speaker_diarization", result)
    segments: list[tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))
    return segments
