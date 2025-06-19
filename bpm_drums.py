"""Estimate BPM by tracking kicks in a Demucs-extracted drum stem (Librosa version)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List
import tempfile
import subprocess
import sys
import numpy as np

import librosa

from ultimate_chord_reader import overwrite_and_remove


# ----------------------------------------------------------------------
# 1. Demucs separation – identical to your previous helper
# ----------------------------------------------------------------------
def _separate_drums(src: str, work_dir: str) -> Path:
    """
    Run Demucs (6-stem mode) and return the drums stem.
    Raises RuntimeError if no drum file is produced.
    """
    out = Path(work_dir)
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", "htdemucs_6s",         # model with 6 stem weights
        "--six-stems",               # 🔑 force 6-stem inference
        "-o", str(out),
        src,
    ]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    drum = next(out.rglob("drums.wav"), None)
    if drum is None:
        raise RuntimeError("drums.wav not produced – check Demucs install")
    return drum



# ----------------------------------------------------------------------
# 2. Beat tracking with Librosa
# ----------------------------------------------------------------------
def _librosa_beats(wav_path: str) -> list[float]:
    """
    Return beat times (in seconds) detected by Librosa on a *mono* drum stem.

    Steps
    -----
    1.  y, sr = librosa.load(wav_path, sr=44100, mono=True)
    2.  tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr,
            units="frames",
            start_bpm=90.0,      # good initial guess
            tightness=100,       # favour consistent tempo
        )
    3.  return librosa.frames_to_time(beat_frames, sr=sr).tolist()
    """

    y, sr = librosa.load(wav_path, sr=44100, mono=True)
    _, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        units="frames",
        start_bpm=90.0,
        tightness=100,
    )
    return librosa.frames_to_time(beat_frames, sr=sr).tolist()


# ----------------------------------------------------------------------
# 3. Public helper – unchanged signature
# ----------------------------------------------------------------------
def get_bpm_from_drums(src: str) -> Tuple[float, List[float]]:
    """
    Extract drum stem → run Librosa beat-tracker → return (bpm, beat_times).
    If fewer than two beats are found, raises RuntimeError.
    """
    with tempfile.TemporaryDirectory() as td:
        drum_path = _separate_drums(src, td)

        beat_times = _librosa_beats(str(drum_path))
        overwrite_and_remove(drum_path)

    if len(beat_times) < 2:
        raise RuntimeError("Insufficient beats detected")

    # median inter-beat-interval → BPM, then normalise to 60-160 range
    diffs = np.diff(beat_times)
    bpm   = 60.0 / float(np.median(diffs))

    while bpm > 160:
        bpm /= 2.0
    while bpm < 60:
        bpm *= 2.0

    return bpm, beat_times
