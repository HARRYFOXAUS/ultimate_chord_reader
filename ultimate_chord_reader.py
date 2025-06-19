"""Entry point for Ultimate Chord Reader."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import os, shutil, pathlib, imageio_ffmpeg
import argparse, textwrap

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

# Validate dependencies early and provide a helpful error message instead of
# attempting implicit installation.  This keeps the runtime predictable and
# avoids long network operations when a package is missing.

REQUIRED = [
    "torch", "librosa", "numpy", "soundfile",
    "openai-whisper",           # <- correct package
    "demucs", "dora-search", "treetable",
    "imageio-ffmpeg",           # <- brings ffprobe for Demucs API
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
            if chart_lines:
                chart_lines.append("")
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
    ensure_dependencies()

    audio_exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    files      = sorted(p for p in INPUT_DIR.iterdir()
                        if p.is_file() and p.suffix.lower() in audio_exts)

    if not files:
        print("No audio files found in", INPUT_DIR)
        return

    # -------- argparse: accept --all or explicit file paths ----------
    parser = argparse.ArgumentParser(
        description="Ultimate Chord Reader – select which songs to analyse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            If you run without arguments, you'll be asked to confirm each track.
            Examples:
              python ultimate_chord_reader.py --all
              python ultimate_chord_reader.py \"my song.mp3\" \"other.wav\"
        """),
    )
    parser.add_argument("tracks", nargs="*", metavar="TRACK",
                        help="one or more filenames in input_songs/")
    parser.add_argument("--all", action="store_true",
                        help="process every file in input_songs/")

    args = parser.parse_args()

    # 1) explicit list or --all  → skip menu
    if args.all:
        selection = files
    elif args.tracks:
        wanted = set(args.tracks)
        selection = [p for p in files if p.name in wanted]
        missing   = wanted - {p.name for p in selection}
        if missing:
            print("Not found in input_songs/:", *missing, sep="\n  • ")
            return
    # 2) no args  → interactive yes/no prompt
    else:
        print("Found the following tracks in", INPUT_DIR)
        for f in files:
            print("  •", f.name)
        choice = input("Select all? [y/N]: ").strip().lower()
        if choice.startswith("y"):
            selection = files
        else:
            selection = []
            for f in files:
                ans = input(f"Process '{f.name}'? [y/N]: ").strip().lower()
                if ans.startswith("y"):
                    selection.append(f)

    if not selection:
        print("Nothing selected. Exiting.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    for file in selection:
        print(f"\nProcessing {file}")
        try:
            out = process_file(str(file))
            print(f"Saved chart to {out}")
        except Exception as exc:
            print("⚠️  Failed on", file.name, "-", exc)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
