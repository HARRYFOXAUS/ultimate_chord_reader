"""Lyric transcription utilities for Ultimate Chord Reader."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple

import whisper


def transcribe(vocal_path: str) -> List[Tuple[float, float, str, float]]:
    """Transcribe the given vocal stem using Whisper.
    Returns a list of tuples of (start_time, end_time, text, confidence).
    """
    model = whisper.load_model("base")
    result = model.transcribe(vocal_path)

    lines = []
    for seg in result.get("segments", []):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = seg.get("text", "").strip()
        conf = seg.get("avg_logprob", 0.0)
        lines.append((start, end, text, conf))
    return lines
