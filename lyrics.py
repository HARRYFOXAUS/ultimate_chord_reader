"""Lyric transcription utilities for Ultimate Chord Reader."""

from __future__ import annotations

from typing import List, Tuple

import whisper  # provided by the *openai-whisper* package (pip install openai-whisper)

import os, shutil
import imageio_ffmpeg  # provides self-contained ffmpeg & ffprobe binaries

# Ensure the bundled ffmpeg and ffprobe executables are visible to Whisper/Demucs
for _exe in (imageio_ffmpeg.get_ffmpeg_exe(), imageio_ffmpeg.get_ffprobe_exe()):
    _dir = os.path.dirname(_exe)
    if not shutil.which(os.path.basename(_exe)):
        os.environ["PATH"] = _dir + os.pathsep + os.environ.get("PATH", "")


def transcribe(vocal_path: str) -> List[Tuple[float, float, str, float]]:
    """Transcribe the given vocal stem using Whisper.
    Returns a list of tuples of (start_time, end_time, text, confidence).
    """
    model = whisper.load_model("base")
    result = model.transcribe(vocal_path)

    lines = []
    for seg in result.get("segments", []):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = seg.get("text", "").strip()
        conf = float(seg.get("avg_logprob", 0.0))
        lines.append((start, end, text, conf))
    return lines
