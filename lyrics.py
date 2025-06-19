"""Lyric transcription utilities for Ultimate Chord Reader."""

from __future__ import annotations

from typing import List, Tuple
from pathlib import Path

import math
import re
from spellchecker import SpellChecker

import whisper  # provided by the *openai-whisper* package (pip install openai-whisper)

import os, shutil, pathlib, imageio_ffmpeg, tempfile, subprocess  # provides self-contained binaries

os.environ.setdefault("TORCH_HOME", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

_spell = SpellChecker()


def _spellcheck_line(text: str) -> str:
    """Return the text with simple spelling corrections applied."""
    tokens = re.findall(r"\w+|\W+", text)
    corrected = []
    for tok in tokens:
        if tok.isalpha():
            corrected.append(_spell.correction(tok) or tok)
        else:
            corrected.append(tok)
    return "".join(corrected)

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


def transcribe(vocal_path: str, work_dir: str) -> List[Tuple[float, float, str, float]]:
    """Transcribe the given vocal stem using Whisper.
    Returns a list of tuples of (start_time, end_time, text, confidence).
    """
    model = whisper.load_model("base")

    decoded = Path(work_dir) / "decoded.wav"
    try:
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i",
            vocal_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            str(decoded),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        result = model.transcribe(str(decoded))
    finally:
        decoded.exists() and decoded.unlink()

    lines = []
    for seg in result.get("segments", []):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = seg.get("text", "").strip()
        conf = float(seg.get("avg_logprob", 0.0))

        prob = math.exp(conf)
        if prob < 0.15:
            text = "???"
        else:
            text = _spellcheck_line(text)
            if prob < 0.5:
                text += " (???)"

        lines.append((start, end, text, conf))
    return lines
