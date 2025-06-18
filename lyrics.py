"""Lyric transcription utilities for Ultimate Chord Reader."""

from __future__ import annotations

from typing import List, Tuple

import whisper  # provided by the *openai-whisper* package (pip install openai-whisper)

import os, shutil, pathlib, imageio_ffmpeg  # provides self-contained binaries

for getter, canon in (
    (imageio_ffmpeg.get_ffmpeg_exe, "ffmpeg"),
    (getattr(imageio_ffmpeg, "get_ffprobe_exe", lambda: None), "ffprobe"),
):
    _exe = getter()
    if not _exe:          # get_ffprobe_exe may be missing on old versions
        continue
    _dir = os.path.dirname(_exe)
    os.environ["PATH"] = _dir + os.pathsep + os.environ.get("PATH", "")

    # If the binary name isn't the canonical one, make a symlink/copy
    if pathlib.Path(_exe).name != canon:
        target = pathlib.Path(_dir) / canon
        if not target.exists():
            try:
                target.symlink_to(_exe)    # best on Unix
            except (OSError, AttributeError):
                shutil.copy2(_exe, target) # fallback on filesystems w/o symlink


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
