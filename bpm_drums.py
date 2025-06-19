"""Estimate BPM by tracking kicks in a Demucs-extracted drum stem (aubio version)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List
import tempfile
import subprocess
import sys
import numpy as np

import soundfile as sf
import aubio                                   # lightweight beat-tracker

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
# 2. Beat tracking with aubio
# ----------------------------------------------------------------------
def _aubio_beats(wav_path: str) -> List[float]:
    """Return a list of beat times (seconds) detected by aubio."""
    # --- read audio (mono float32) ------------------------------------
    y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)

    hop   = 512                     # hop-size in samples
    win   = 1024                    # analysis window

    tempo = aubio.tempo(            # default method = complexdomain
        samplerate=sr,
        hop_size=hop,
        win_size=win,
    )
    tempo.set_silence(-40)          # dBFS threshold
    tempo.set_threshold(0.3)        # onset threshold

    beat_times: list[float] = []
    for i in range(0, len(y), hop):
        frame = y[i : i + hop]
        if len(frame) < hop:                       # zero-pad last frame
            frame = np.pad(frame, (0, hop - len(frame)))
        if tempo(frame):
            beat_times.append(float(tempo.get_last_s()))

    return beat_times


# ----------------------------------------------------------------------
# 3. Public helper – unchanged signature
# ----------------------------------------------------------------------
def get_bpm_from_drums(src: str) -> Tuple[float, List[float]]:
    """
    Extract drum stem → run aubio beat-tracker → return (bpm, beat_times).
    If fewer than two beats are found, raises RuntimeError.
    """
    with tempfile.TemporaryDirectory() as td:
        drum_path = _separate_drums(src, td)

        beat_times = _aubio_beats(str(drum_path))
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
