"""Entry point for Ultimate Chord Reader."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Validate dependencies early and provide a helpful error message instead of
# attempting implicit installation.  This keeps the runtime predictable and
# avoids long network operations when a package is missing.

REQUIRED = [
    "torch",
    "librosa",
    "numpy",
    "soundfile",
    "demucs",
    "dora-search",
    "treetable",
    "imageio-ffmpeg",
    "openai-whisper",
]

missing: list[str] = []
for _pkg in REQUIRED:
    try:
        __import__(_pkg)
    except Exception:  # pragma: no cover - import failure path
        missing.append(_pkg)

def ensure_dependencies() -> None:
    """Install any missing dependencies using pip."""
    if not missing:
        return
    print(f"Installing missing packages: {', '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


# User configuration
INPUT_DIR = Path("input_songs")
OUTPUT_DIR = Path("output_charts")
TIME_SIGNATURE = "4/4"  # default time signature


def format_chart(
    title: str,
    bpm: float,
    key: str,
    time_sig: str,
    lyrics: List[Tuple[float, float, str, float]],
    chords: List[Tuple[str, float, float]],
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

    beats_per_bar = int(time_sig.split("/")[0]) if "/" in time_sig else 4
    beat_len = 60.0 / bpm
    bar_len = beats_per_bar * beat_len

    chart_lines = []
    chord_idx = 0
    current_bar = -1
    for start, end, line, conf in lyrics:
        bar = int(start / bar_len)
        if bar != current_bar:
            chart_lines.append("|")
            current_bar = bar
        while chord_idx < len(chords) and chords[chord_idx][1] <= start:
            chord_idx += 1
        chord = chords[chord_idx - 1][0] if chord_idx else ""
        chart_lines.append(f"{chord}\t{line}")

    return "\n".join(header + chart_lines)


def process_file(path: str) -> Path:
    """Process a single audio file and output chord chart path."""
    from models.separation_manager import separate_and_score
    from lyrics import transcribe
    from chords import analyze_instrumental

    vocal, inst, conf = separate_and_score(path)
    lyric_lines = transcribe(str(vocal))
    bpm, key, chord_seq = analyze_instrumental(str(inst))

    title = Path(path).stem
    chart = format_chart(title, bpm, key, TIME_SIGNATURE, lyric_lines, chord_seq, conf)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{title}_chart.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(chart)

    # Remove temporary stems directory entirely
    stem_root = vocal.parent
    shutil.rmtree(stem_root.parent, ignore_errors=True)

    return out_path


# TODO: Web GUI hook
# TODO: Cloud model hook
# TODO: Feedback loop for self-training


def main() -> None:
    """Process all audio files in :data:`INPUT_DIR`."""
    ensure_dependencies()
    INPUT_DIR.mkdir(exist_ok=True)
    for file in INPUT_DIR.iterdir():
        if not file.is_file():
            continue
        if file.suffix.lower() not in {".mp3", ".wav", ".flac", ".m4a", ".ogg"}:
            continue
        print(f"Processing {file}")
        out = process_file(str(file))
        print(f"Saved chart to {out}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
