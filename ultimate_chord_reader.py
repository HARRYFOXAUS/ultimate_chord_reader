"""Ultimate Chord Reader main application."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Auto-install dependencies
REQUIRED = [
    "torch",
    "whisper",
    "librosa",
    "numpy",
    "soundfile",
]
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except Exception:
        import subprocess
        subprocess.check_call(["python", "-m", "pip", "install", pkg])

# User configuration
INPUT_DIR = Path("input_songs")
OUTPUT_DIR = Path("output_charts")
TIME_SIGNATURE = "4/4"  # default time signature


from models.separation_manager import separate_and_score
from lyrics import transcribe
from chords import analyze_instrumental


def format_chart(
    title: str,
    bpm: float,
    key: str,
    time_sig: str,
    lyrics: List[Tuple[float, float, str, float]],
    chords: List[Tuple[str, float]],
    confidence: float,
) -> str:
    """Create a plain text chord chart."""
    header = [
        f"Title: {title}",
        f"BPM: {bpm:.1f}",
        f"Key: {key}",
        f"Time Signature: {time_sig}",
        f"Separation Confidence: {confidence:.2f}",
        "",
    ]

    chart_lines = []
    chord_idx = 0
    for start, end, line, conf in lyrics:
        while chord_idx < len(chords) and chords[chord_idx][1] <= start:
            chord_idx += 1
        chord = chords[chord_idx - 1][0] if chord_idx else ""
        chart_lines.append(f"{chord}\t{line}")

    return "\n".join(header + chart_lines)


def process_file(path: str) -> Path:
    """Process a single audio file and output chord chart path."""
    vocal, inst, conf = separate_and_score(path)
    lyric_lines = transcribe(str(vocal))
    bpm, key, chord_seq = analyze_instrumental(str(inst))

    title = Path(path).stem
    chart = format_chart(title, bpm, key, TIME_SIGNATURE, lyric_lines, chord_seq, conf)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{title}_chart.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(chart)

    # Remove stems
    stem_root = vocal.parent
    shutil.rmtree(stem_root, ignore_errors=True)

    return out_path


# TODO: Web GUI hook
# TODO: Cloud model hook
# TODO: Feedback loop for self-training

if __name__ == "__main__":
    for file in INPUT_DIR.glob("*.*"):
        print(f"Processing {file}")
        out = process_file(str(file))
        print(f"Saved chart to {out}")

