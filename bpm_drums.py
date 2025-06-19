from __future__ import annotations

from pathlib import Path
from typing import Tuple, List
import tempfile
import subprocess
import sys
import numpy as np

from ultimate_chord_reader import overwrite_and_remove


def _separate_drums(src: str, work_dir: str) -> Path:
    """Run Demucs to extract the drum stem and return its path."""
    out = Path(work_dir)
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        "htdemucs_6s",
        "-o",
        str(out),
        src,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    drum = next(out.rglob("drums.wav"), None)
    if drum is None:
        raise RuntimeError("Drum stem not found")
    return drum


def get_bpm_from_drums(src: str) -> Tuple[float, List[float]]:
    """Return (bpm, beat_times) from *src* using a drum stem."""
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

    with tempfile.TemporaryDirectory() as td:
        drum_path = _separate_drums(src, td)
        proc = RNNBeatProcessor()(str(drum_path))
        beats = DBNBeatTrackingProcessor(fps=100)(proc)
        beat_times = beats.tolist()
        overwrite_and_remove(drum_path)

    if len(beat_times) < 2:
        raise RuntimeError("Insufficient beats")

    diffs = np.diff(beat_times)
    bpm = 60.0 / float(np.median(diffs))
    while bpm > 160:
        bpm /= 2.0
    while bpm < 60:
        bpm *= 2.0
    return bpm, beat_times

