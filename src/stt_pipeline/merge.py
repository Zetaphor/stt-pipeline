from __future__ import annotations

from typing import Any


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return the overlap duration between two time intervals."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _dominant_speaker(
    seg_start: float,
    seg_end: float,
    speaker_segments: list[tuple[float, float, str]],
) -> str:
    """Find the speaker with the largest overlap for a given time range."""
    best_speaker = "UNKNOWN"
    best_overlap = 0.0
    for sp_start, sp_end, label in speaker_segments:
        if sp_start > seg_end:
            break
        ov = _overlap(seg_start, seg_end, sp_start, sp_end)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = label
    return best_speaker


def merge_transcript(
    whisper_result: dict[str, Any],
    speaker_segments: list[tuple[float, float, str]],
) -> dict[str, Any]:
    """Merge Whisper verbose_json output with pyannote speaker segments.

    Returns an OpenAI-compatible response with speaker labels injected into
    segment text and as a ``speaker`` field on each segment.
    """
    speaker_segments = sorted(speaker_segments, key=lambda s: s[0])

    raw_segments: list[dict[str, Any]] = whisper_result.get("segments", [])
    merged_segments: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for seg in raw_segments:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        text = seg.get("text", "").strip()
        speaker = _dominant_speaker(start, end, speaker_segments)

        labeled_text = f"[{speaker}]: {text}"
        text_parts.append(labeled_text)

        merged_seg = {**seg, "speaker": speaker, "text": text}
        merged_segments.append(merged_seg)

    merged_text = "\n".join(text_parts)

    result: dict[str, Any] = {
        **whisper_result,
        "text": merged_text,
        "segments": merged_segments,
    }
    return result
