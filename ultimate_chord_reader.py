"""Entry point for Ultimate Chord Reader.

BPM detection: drums → no-vocals → mix (Librosa fallback)
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import imageio_ffmpeg  # brings self‑contained ffmpeg & ffprobe

# ---------------------------------------------------------------------------
# Environment defaults: keep large model caches off the repo filesystem
# ---------------------------------------------------------------------------
os.environ.setdefault("TORCH_HOME", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

# ---------------------------------------------------------------------------
# Ensure the bundled ffmpeg/ffprobe executables are visible to Whisper/Demucs
# ---------------------------------------------------------------------------
for getter, canon in (
    (imageio_ffmpeg.get_ffmpeg_exe, "ffmpeg"),
    (getattr(imageio_ffmpeg, "get_ffprobe_exe", lambda: None), "ffprobe"),
):
    _exe = getter()
    if not _exe:  # get_ffprobe_exe may be missing on very old versions
        continue
    _dir = os.path.dirname(_exe)
    os.environ["PATH"] = _dir + os.pathsep + os.environ.get("PATH", "")

    # On some platforms imageio-ffmpeg supplies "ffmpeg-imageio" → symlink it
    if pathlib.Path(_exe).name != canon:
        target = pathlib.Path(_dir) / canon
        if not target.exists():
            try:
                target.symlink_to(_exe)  # Unix‑like
            except (OSError, AttributeError):
                shutil.copy2(_exe, target)  # Fallback on FS w/o symlink support

# ---------------------------------------------------------------------------
# Hard dependencies – we fail fast with a helpful msg if any are missing
# ---------------------------------------------------------------------------
REQUIRED = [
    # runtime libs
    "torch", "librosa", "numpy", "soundfile",
    "openai-whisper", "demucs", "dora-search", "treetable",
    "imageio-ffmpeg", "pyspellchecker",
    "wheel",            # manylinux/OSX build helper
    "aubio",            # fallback beat tracker
]

_missing: list[str] = []
for _pkg in REQUIRED:
    try:
        __import__(_pkg)
    except Exception:  # pragma: no cover – import failure path
        _missing.append(_pkg)


def ensure_dependencies() -> None:
    """Install any missing dependencies using pip (rare in dev, handy for CI)."""
    if not _missing:
        return
    print("Installing missing packages:", ", ".join(_missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *_missing])


# ---------------------------------------------------------------------------
# User‑tweakable settings
# ---------------------------------------------------------------------------
INPUT_DIR = Path("input_songs")
OUTPUT_DIR = Path("output_charts")
TIME_SIGNATURE = "4/4"  # default TS until we add an onset‑detector later
MAX_CHANGES_PER_BAR = 2  # show at most N NEW chord names inside a single bar

DISCLAIMER = (
    "ULTIMATE CHORD READER uses automated stem separation and AI analysis.\n"
    "All audio files and stems are automatically deleted immediately after processing.\n"
    "Results are best‑effort guesses; verify before public use.\n"
)

# ---------------------------------------------------------------------------
# Helpers: secure delete (best‑effort) & chart formatter
# ---------------------------------------------------------------------------

def overwrite_and_remove(path: Path) -> None:
    """Overwrite a file with zeros and unlink it (best‑effort secure delete)."""
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
    beats: list[float] | None = None,
    **extra,
) -> str:
    """Return a human-readable text chart – **exactly one line per bar**."""

    from collections import defaultdict
    try:
        import numpy as np
    except Exception:  # pragma: no cover - fallback when numpy missing
        np = None

    if np is not None and isinstance(bpm, getattr(np, "ndarray", ())):
        bpm = float(bpm.squeeze())

    if beats is None:
        beats = extra.get("beat_times", [])
    beats_list = list(beats or [0.0])
    bar_of_beat = [i // 4 for i in range(len(beats_list))]
    last_bar = int(bar_of_beat[-1])

    def nearest_beat_idx(t: float) -> tuple[int, int | None]:
        diffs = [abs(b - t) for b in beats_list]
        i = diffs.index(min(diffs))
        return i, (None if diffs[i] > 0.1 else bar_of_beat[i])

    print(f"[ALIGN] using {len(beats_list)} beats → {last_bar+1} bars")

    bars_chords: defaultdict[int, list[str]] = defaultdict(list)
    bars_lyrics: defaultdict[int, list[str]] = defaultdict(list)

    for name, t, _ in chords:
        _idx, bar = nearest_beat_idx(t)
        if bar is None:
            continue
        if not bars_chords[bar] or bars_chords[bar][-1] != name:
            bars_chords[bar].append(name)

    for start, _end, text, _c in lyrics:
        _idx, bar = nearest_beat_idx(start)
        if bar is None:
            continue
        bars_lyrics[bar].append(text)

    header = [
        DISCLAIMER.rstrip(),
        "",
        f"Title: {title}",
        f"BPM: {bpm:.1f}",
        f"Key: {key}",
        f"Time Signature: {time_sig}",
        f"Lyric Transcription Confidence: {confidence:.1f}%",
        "",
    ]

    lines: list[str] = []
    for b in range(last_bar + 1):
        chord_text = " ".join(bars_chords[b][:4])
        lyric_text = " ".join(bars_lyrics[b])
        lines.append(f"{chord_text}\t{lyric_text}".rstrip())

    return "\n".join(header + lines)

# ---------------------------------------------------------------------------
# Main pipeline: separation → transcription → chord analyse → chart
# ---------------------------------------------------------------------------

def process_file(path: str) -> Path:
    """Run full pipeline on *path*.  Returns the saved chart path."""
    from models.separation_manager import separate_and_score
    from lyrics import transcribe
    from chords import analyze_instrumental
    from bpm_drums import get_bpm_from_drums, bpm_via_librosa

    with tempfile.TemporaryDirectory() as tmpdir:
        vocal, inst, _ = separate_and_score(path, tmpdir)
        lyric_lines = transcribe(str(vocal), tmpdir)

        # ---------------------------------------------------------------
        # Hierarchical BPM detection
        # ---------------------------------------------------------------
        bpm_source = "??"
        try:
            bpm, beat_times = get_bpm_from_drums(str(inst))
            bpm_source = "drums"
        except Exception as e1:
            try:
                bpm, beat_times = bpm_via_librosa(str(inst))    # no-vocals stem
                bpm_source = "no_vocals"
            except Exception as e2:
                bpm, beat_times = bpm_via_librosa(path)         # full mix
                bpm_source = "mix"

        print(f"[BPM] {bpm_source:9s} ➜ {bpm:.1f}")

        try:
            key, chord_seq = analyze_instrumental(str(inst), bpm=bpm, beats=beat_times)
        except TypeError:
            try:
                key, chord_seq = analyze_instrumental(str(inst), bpm=bpm)
            except TypeError:
                key, chord_seq = analyze_instrumental(str(inst))

        # average Whisper confidence – log‑probabilities → probabilities → %
        if lyric_lines:
            avg_conf = (sum(math.exp(conf) for *_rest, conf in lyric_lines)
                        / len(lyric_lines)) * 100.0
        else:
            avg_conf = 0.0

        title = Path(path).stem
        chart = format_chart(title, bpm, key, TIME_SIGNATURE,
                             lyric_lines, chord_seq, avg_conf, beat_times)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{title}_chart.txt"
        out_path.write_text(chart, encoding="utf‑8")

        # secure‑delete stems ASAP
        overwrite_and_remove(vocal)
        overwrite_and_remove(inst)

        return out_path

# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dependencies()

    audio_exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    files = sorted(p for p in INPUT_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() in audio_exts)
    if not files:
        print("No audio files found in", INPUT_DIR)
        return

    # ── argparse -------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Ultimate Chord Reader – choose which tracks to analyse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            If you run without arguments you'll be prompted for each file.
            Examples:
              python ultimate_chord_reader.py --all
              python ultimate_chord_reader.py "my song.mp3" "other.wav"
        """),
    )
    parser.add_argument("tracks", nargs="*", metavar="TRACK",
                        help="one or more filenames in input_songs/")
    parser.add_argument("--all", action="store_true",
                        help="process every file in input_songs/")
    args = parser.parse_args()

    # explicit list or --all  → skip menu
    if args.all:
        selection = files
    elif args.tracks:
        wanted = set(args.tracks)
        selection = [p for p in files if p.name in wanted]
        not_found = wanted - {p.name for p in selection}
        if not_found:
            print("Not found in input_songs/:", *not_found, sep="\n  • ")
            return
    else:
        # interactive picker
        print("Found the following tracks in", INPUT_DIR)
        for f in files:
            print("  •", f.name)
        if input("Select all? [y/N]: ").strip().lower().startswith("y"):
            selection = files
        else:
            selection = []
            for f in files:
                if input(f"Process '{f.name}'? [y/N]: ").strip().lower().startswith("y"):
                    selection.append(f)

    if not selection:
        print("Nothing selected. Exiting.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    for file in selection:
        print(f"\nProcessing {file}")
        try:
            out = process_file(str(file))
            print("Saved chart to", out)
        except Exception as exc:
            print("⚠️  Failed on", file.name, "-", exc)


if __name__ == "__main__":  # pragma: no cover – CLI entry point
    main()
