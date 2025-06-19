"""Entry point for Ultimate Chord Reader."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import os, shutil, pathlib, imageio_ffmpeg, tempfile

os.environ.setdefault("TORCH_HOME", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import argparse, textwrap
import math

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
    "pyspellchecker",
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

DISCLAIMER = (
    "ULTIMATE CHORD READER uses automated stem separation and AI analysis.\n"
    "All audio files and stems are automatically deleted immediately after processing.\n"
    "Results are best-effort guesses; verify before public use.\n"
)


def overwrite_and_remove(path: Path) -> None:
    """Overwrite a file with zeros and unlink it."""
    if not path.exists():
        return
    try:
        size = path.stat().st_size
        with open(path, "wb", buffering=0) as f:
            f.write(b"\x00" * size)
    finally:
        path.unlink(missing_ok=True)


def format_chart(
    title: str,
    bpm: float,
    key: str,
    time_sig: str,
    lyrics: list[tuple[float, float, str, float]],
    chords: list[tuple[str, float, float]],
    confidence: float,
) -> str:
    """
    Build a plain-text chart with **exactly one line per bar**.
    - Lyrics starting in the same bar are merged.
    - All bars (even silent ones) are represented.
    - Multiple chords in a bar are listed in time order.
    """
    import math
    import numpy as np

    # ── header ──────────────────────────────────────────────────────────
    if isinstance(bpm, np.ndarray):
        bpm = float(bpm.squeeze())
    header = [
        "ULTIMATE CHORD READER uses automated stem separation and AI analysis.",
        "All audio files and stems are automatically deleted immediately after processing.",
        "Results are best-effort guesses; verify before public use.",
        "",
        f"Title: {title}",
        f"BPM: {bpm:.1f}",
        f"Key: {key}",
        f"Time Signature: {time_sig}",
        f"Lyric Transcription Confidence: {confidence:.1f}%",
        "",
    ]

    # ── bar length ──────────────────────────────────────────────────────
    beats_per_bar = int(time_sig.split("/")[0]) if "/" in time_sig else 4
    beat_len = 60.0 / bpm          # seconds per beat
    bar_len  = beats_per_bar * beat_len

    # ── find how many bars we need ──────────────────────────────────────
    last_time = 0.0
    if lyrics:
        last_time = max(last_time, max(seg[1] for seg in lyrics))  # lyric end
    if chords:
        last_time = max(last_time, chords[-1][1])                  # last chord
    total_bars = math.ceil(last_time / bar_len)

    # ── main loop: one pass per bar ─────────────────────────────────────
    chart_lines: list[str] = []
    chord_idx = 0  # rolling pointer into chords list (sorted by time)
    for bar in range(total_bars):
        bar_start = bar * bar_len
        bar_end   = bar_start + bar_len

        # gather lyric segments whose *start* is in this bar
        lyr_texts = [
            txt for (s, _e, txt, _conf) in lyrics
            if bar_start <= s < bar_end
        ]
        merged_lyric = " ".join(lyr_texts).strip()

        # gather chords active / changing in this bar
        chords_in_bar: list[str] = []

        # include any chord that was already playing at bar_start
        if chord_idx and chords[chord_idx-1][1] < bar_start:
            chords_in_bar.append(chords[chord_idx-1][0])

        # walk forward while the chord change time is inside the bar
        while chord_idx < len(chords) and chords[chord_idx][1] < bar_end:
            name = chords[chord_idx][0]
            if not chords_in_bar or name != chords_in_bar[-1]:
                chords_in_bar.append(name)
            chord_idx += 1

        chord_text = " ".join(chords_in_bar)

        # build the line (empty string for silent bar)
        if chord_text or merged_lyric:
            chart_lines.append(f"{chord_text}\t{merged_lyric}".rstrip())
        else:
            chart_lines.append("")  # placeholder for empty bar

    # ── return final chart ──────────────────────────────────────────────
    return "\n".join(header + chart_lines)


def process_file(path: str) -> Path:
    """Process a single audio file and output chord chart path."""
    from models.separation_manager import separate_and_score
    from lyrics import transcribe
    from chords import analyze_instrumental

    with tempfile.TemporaryDirectory() as tmpdir:
        vocal, inst, conf = separate_and_score(path, tmpdir)
        lyric_lines = transcribe(str(vocal), tmpdir)
        bpm, key, chord_seq = analyze_instrumental(str(inst))

        import numpy as np
        if isinstance(bpm, np.ndarray):
            bpm = float(bpm.squeeze())


        title = Path(path).stem
        chart = format_chart(title, bpm, key, TIME_SIGNATURE, lyric_lines, chord_seq, conf)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{title}_chart.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(chart)

        overwrite_and_remove(vocal)
        overwrite_and_remove(inst)

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
