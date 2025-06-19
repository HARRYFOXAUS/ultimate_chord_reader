"""Entry point for Ultimate Chord Reader."""

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
    "torch",
    "librosa",
    "numpy",
    "soundfile",
    "openai-whisper",  # correct Whisper package
    "demucs",
    "dora-search",
    "treetable",
    "imageio-ffmpeg",
    "pyspellchecker",
    "madmom",
    "aubio",
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
    beat_times: list[float],
    confidence: float,
) -> str:
    """Return a human‑readable text chart – **exactly one line per bar**.

    * Lyrics whose *start* lies in the same bar are merged (0.25 s grace).
    * Every bar appears (empty line if nothing happens).
    * At most `MAX_CHANGES_PER_BAR` fresh chord changes per bar are displayed –
      continuing chords are omitted to avoid the “wall‑of‑chords”.
    """
    import numpy as np

    if isinstance(bpm, np.ndarray):
        bpm = float(bpm.squeeze())

    beats_per_bar = int(time_sig.split("/")[0]) if "/" in time_sig else 4
    bar_len = (60.0 / bpm) * beats_per_bar

    # ── header ────────────────────────────────────────────────────────────
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

    # ── total length in bars ─────────────────────────────────────────────
    last_time = 0.0
    if lyrics:
        last_time = max(last_time, max(e for _s, e, _t, _ in lyrics))
    if chords:
        last_time = max(last_time, chords[-1][1])
    if beat_times:
        last_time = max(last_time, beat_times[-1])

    bar_starts = [beat_times[i] for i in range(0, len(beat_times), beats_per_bar)] if beat_times else [0.0]
    # drop trailing bars that start after the song ends
    while len(bar_starts) > 1 and bar_starts[-1] >= last_time:
        bar_starts.pop()

    if beat_times and len(beat_times) > 1:
        diffs = [b - a for a, b in zip(beat_times, beat_times[1:])]
        import statistics
        bar_len_est = statistics.median(diffs) * beats_per_bar
    else:
        bar_len_est = bar_len
    while bar_starts[-1] + bar_len_est < last_time:
        bar_starts.append(bar_starts[-1] + bar_len_est)

    total_bars = len(bar_starts)

    # pre‑index lyric segs by bar (with 0.25 s grace so near‑boundary words join)
    grace = 0.25
    lyrics_by_bar: dict[int, list[str]] = {}
    for s, _e, txt, _c in lyrics:
        idx = 0
        while idx + 1 < len(bar_starts) and bar_starts[idx + 1] <= s + grace:
            idx += 1
        lyrics_by_bar.setdefault(idx, []).append(txt)

    chart_lines: list[str] = []
    chord_idx = 0  # running index into *sorted* chord list

    for bar in range(total_bars):
        bar_start = bar_starts[bar]
        bar_end = bar_starts[bar + 1] if bar + 1 < len(bar_starts) else bar_start + bar_len_est

        # ── lyrics ────────────────────────────────────────────────────
        merged_lyric = " ".join(lyrics_by_bar.get(bar, [])).strip()

        # ── chords ────────────────────────────────────────────────────
        chords_in_bar: list[str] = []

        # 1) chord already sounding at bar start
        carry_over = None
        if chord_idx and chords[chord_idx - 1][1] < bar_start:
            carry_over = chords[chord_idx - 1][0]
            chords_in_bar.append(carry_over)

        # 2) new changes inside the bar (capped)
        j = chord_idx
        while j < len(chords) and chords[j][1] < bar_end:
            name = chords[j][0]
            if (not chords_in_bar or name != chords_in_bar[-1]) and \
               len(chords_in_bar) < 1 + MAX_CHANGES_PER_BAR:
                chords_in_bar.append(name)
            j += 1
        chord_idx = j

        # hide carry‑over if absolutely nothing new happened
        if carry_over and len(chords_in_bar) == 1:
            chords_in_bar.clear()

        chord_text = " ".join(chords_in_bar)
        line = f"{chord_text}\t{merged_lyric}".rstrip()
        chart_lines.append(line)

    return "\n".join(header + chart_lines)

# ---------------------------------------------------------------------------
# Main pipeline: separation → transcription → chord analyse → chart
# ---------------------------------------------------------------------------

def process_file(path: str) -> Path:
    """Run full pipeline on *path*.  Returns the saved chart path."""
    from models.separation_manager import separate_and_score
    from lyrics import transcribe
    from chords import analyze_instrumental

    with tempfile.TemporaryDirectory() as tmpdir:
        vocal, inst, _ = separate_and_score(path, tmpdir)
        lyric_lines = transcribe(str(vocal), tmpdir)
        bpm, key, chord_seq, beat_times = analyze_instrumental(str(inst))

        # average Whisper confidence – log‑probabilities → probabilities → %
        if lyric_lines:
            avg_conf = (sum(math.exp(conf) for *_rest, conf in lyric_lines)
                        / len(lyric_lines)) * 100.0
        else:
            avg_conf = 0.0

        title = Path(path).stem
        chart = format_chart(title, bpm, key, TIME_SIGNATURE,
                             lyric_lines, chord_seq, beat_times, avg_conf)

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
