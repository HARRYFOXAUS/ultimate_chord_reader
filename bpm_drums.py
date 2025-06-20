"""Estimate BPM by tracking kicks in a Demucs-extracted drum stem.
Uses Librosa beat_track (tightness = 400) + robust median filtering;
raises RuntimeError if detection is unreliable or highly variable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List
import tempfile
import subprocess
import sys
import numpy as np

import soundfile as sf
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
    """Return (bpm, beat_times) from a drum stem using robust filtering."""

    with tempfile.TemporaryDirectory() as td:
        drum_path = _separate_drums(src, td)

        est_tempo, beat_times = _librosa_beats(str(drum_path))
        overwrite_and_remove(drum_path)

    if len(beat_times) < 4:
        raise RuntimeError("Too few beats detected")

    ibi = np.diff(beat_times)
    med = np.median(ibi)
    good = ibi[(ibi > 0.7 * med) & (ibi < 1.3 * med)]

    if len(good) < max(3, 0.5 * len(ibi)):
        raise RuntimeError("Inconsistent beat intervals")

    bpm = 60.0 / float(np.median(good))

    while bpm > 160:
        bpm /= 2
    while bpm < 60:
        bpm *= 2

    if not (0.9 * est_tempo <= bpm <= 1.1 * est_tempo):
        raise RuntimeError(
            f"Unreliable BPM (raw {est_tempo:.1f}, filtered {bpm:.1f})"
        )

    if __name__ == "__main__":
        print(
            f"[bpm_drums] beats={len(beat_times)}  raw={est_tempo:.1f}  filtered={bpm:.1f}"
        )

    return bpm, beat_times
