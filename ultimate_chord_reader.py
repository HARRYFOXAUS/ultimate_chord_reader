"""
Ultimate Chord Reader
–––––––––––––––––––––
Stem separation   → Whisper transcription
Drum-first BPM    → Librosa fall-back BPM
Chord analysis    → 1-bar-per-line text chart
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD LIB
# ─────────────────────────────────────────────────────────────────────────────
import argparse, math, os, pathlib, shutil, subprocess, sys, tempfile, textwrap
from inspect import signature
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY
# ─────────────────────────────────────────────────────────────────────────────
import imageio_ffmpeg                      # ships static ffmpeg/ffprobe
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from chords import analyze_instrumental
except Exception:  # heavy deps missing in tests
    analyze_instrumental = None  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# ENV / PATH
# ─────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("TORCH_HOME", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

for getter, canon in (
    (imageio_ffmpeg.get_ffmpeg_exe, "ffmpeg"),
    (getattr(imageio_ffmpeg, "get_ffprobe_exe", None), "ffprobe"),
):
    exe = getter() if getter else None
    if exe:
        bin_dir = os.path.dirname(exe)
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        if pathlib.Path(exe).name != canon:
            target = pathlib.Path(bin_dir) / canon
            if not target.exists():
                try:
                    target.symlink_to(exe)
                except (OSError, AttributeError):
                    shutil.copy2(exe, target)

# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME DEPENDENCIES – auto-install in dev/CI
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED = [
    "torch", "librosa", "numpy", "soundfile",
    "openai_whisper", "demucs", "dora_search", "treetable",
    "imageio_ffmpeg", "pyspellchecker",
    "wheel",
]
missing: list[str] = []
for pkg in REQUIRED:
    try:
        __import__(pkg.replace("-", "_"))
    except Exception:
        missing.append(pkg)


def ensure_dependencies() -> None:
    if missing:
        print("Installing missing packages:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


# ─────────────────────────────────────────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
INPUT_DIR           = Path("input_songs")
OUTPUT_DIR          = Path("output_charts")
TIME_SIGNATURE      = "4/4"
MAX_CHANGES_PER_BAR = 2          # beyond carry-over chord

DISCLAIMER = (
    "ULTIMATE CHORD READER uses automated stem separation and AI analysis.\n"
    "All audio files and stems are automatically deleted immediately after processing.\n"
    "Results are best-effort guesses; verify before public use.\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def overwrite_and_remove(path: Path) -> None:
    """Best-effort secure delete."""
    if not path.exists():
        return
    try:
        with open(path, "wb", buffering=0) as f:
            f.write(b"\x00" * path.stat().st_size)
    finally:
        path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Demucs wrapper – cope with old/new signatures
# ─────────────────────────────────────────────────────────────────────────────
def run_separation(src: str, workdir: str, *, model="htdemucs_6s", two_stems=None):
    from models.separation_manager import separate_and_score

    params = signature(separate_and_score).parameters
    if "model_name" in params:
        return separate_and_score(src, workdir, model_name=model, two_stems=two_stems)
    return separate_and_score(src, workdir)


def safe_analyze(path: str, *, bpm: float, beats: list[float]):
    from chords import analyze_instrumental
    sig = signature(analyze_instrumental).parameters
    kwargs = {}
    if "bpm" in sig:
        kwargs["bpm"] = bpm
    if "beats" in sig:
        kwargs["beats"] = beats
    ret = analyze_instrumental(path, **kwargs)
    if len(ret) == 3:
        _bpm, key, chords = ret
    elif len(ret) == 2:
        key, chords = ret
    else:
        raise RuntimeError("analyze_instrumental returned weird tuple")
    return key, chords



# ─────────────────────────────────────────────────────────────────────────────
# CHART FORMATTER – one line per bar
# ─────────────────────────────────────────────────────────────────────────────
def format_chart(title: str, bpm: float, key: str, time_sig: str,
                 lyrics, chords, confidence: float, beat_times=None):
    from collections import defaultdict
    import numpy as np

    if isinstance(bpm, np.ndarray):
        bpm = float(bpm.squeeze())

    beats = list(beat_times or [0.0])
    bar_of_beat = [i // 4 for i in range(len(beats))]
    last_bar = bar_of_beat[-1]

    def ts_to_bar(t: float):
        i = min(range(len(beats)), key=lambda k: abs(beats[k] - t))
        return bar_of_beat[i] if abs(beats[i] - t) < 0.10 else None

    chords_by_bar: dict[int, list[str]] = defaultdict(list)
    lyrics_by_bar: dict[int, list[str]] = defaultdict(list)

    for name, t, _ in chords:
        b = ts_to_bar(t)
        if b is not None and (not chords_by_bar[b] or chords_by_bar[b][-1] != name):
            if len(chords_by_bar[b]) < MAX_CHANGES_PER_BAR + 1:
                chords_by_bar[b].append(name)

    for start, _end, txt, _p in lyrics:
        b = ts_to_bar(start)
        if b is not None:
            lyrics_by_bar[b].append(txt)

    header = [
        DISCLAIMER.rstrip(), "",
        f"Title: {title}",
        f"BPM: {bpm:.1f}",
        f"Key: {key}",
        f"Time Signature: {time_sig}",
        f"Lyric Transcription Confidence: {confidence:.1f}%", "",
    ]

    lines = [
        f"{' '.join(chords_by_bar[b])}\t{' '.join(lyrics_by_bar[b])}".rstrip()
        for b in range(last_bar + 1)
    ]
    return "\n".join(header + lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def process_file(path: str) -> Path:
    from lyrics import transcribe
    from bpm_drums import get_bpm_from_drums, bpm_via_librosa

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. separation ------------------------------------------------------
        vocal, inst, _ = run_separation(path, tmpdir, model="htdemucs_6s")

        # 2. lyrics ----------------------------------------------------------
        lyric_lines = transcribe(str(vocal), tmpdir)

        # 3. BPM & beat times -----------------------------------------------
        try:
            bpm, beat_times = get_bpm_from_drums(str(inst))
            src = "drums"
        except Exception:
            try:
                bpm, beat_times = bpm_via_librosa(str(inst))
                src = "no-vocals"
            except Exception:
                bpm, beat_times = bpm_via_librosa(path)
                src = "mix"
        print(f"[BPM] {src:9s} → {bpm:.1f}")

        # 4. key + chords ----------------------------------------------------
        key, chord_seq = safe_analyze(str(inst), bpm=bpm, beats=beat_times)

        # 5. confidence ------------------------------------------------------
        if lyric_lines:
            avg_conf = sum(math.exp(c) for *_x, c in lyric_lines) / len(lyric_lines) * 100
        else:
            avg_conf = 0.0

        # 6. chart -----------------------------------------------------------
        title = Path(path).stem
        chart = format_chart(title, bpm, key, TIME_SIGNATURE,
                             lyric_lines, chord_seq, avg_conf, beat_times)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{title}_chart.txt"
        out_path.write_text(chart, encoding="utf-8")

        overwrite_and_remove(vocal)
        overwrite_and_remove(inst)
        return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ensure_dependencies()

    audio_exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    files = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in audio_exts)
    if not files:
        print("No audio files found in", INPUT_DIR)
        return

    p = argparse.ArgumentParser(
        description="Ultimate Chord Reader – choose tracks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            • Run without arguments to be prompted for each file
            • --all            → process every file
            • list of names    → process those specific files
        """),
    )
    p.add_argument("tracks", nargs="*", metavar="TRACK")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    selection: list[Path]

    if args.all:
        selection = files
    elif args.tracks:
        wanted = set(args.tracks)
        selection = [f for f in files if f.name in wanted]
        missing = wanted - {f.name for f in selection}
        if missing:
            print("Not found in input_songs/:", *missing, sep="\n  • ")
            return
    else:
        print("Tracks in", INPUT_DIR)
        for f in files:
            print("  •", f.name)
        if input("Process ALL files? [y/N]: ").lower().startswith("y"):
            selection = files
        else:
            selection = [f for f in files
                         if input(f"Process '{f.name}'? [y/N]: ").lower().startswith("y")]

        if not selection:
            print("Nothing selected. Exiting.")
            return

    for f in selection:
        print("\nProcessing", f.name)
        try:
            out = process_file(str(f))
            print("Saved chart to", out)
        except Exception as e:
            print("⚠️  Failed on", f.name, "–", e)


if __name__ == "__main__":
    main()
