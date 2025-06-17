"""Stem separation manager for Ultimate Chord Reader.

Both Demucs and UVR are attempted if available. If Demucs is missing, the
process continues with UVR only and logs a warning.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from .mvsep_loader import run_uvr
from .demucs_loader import run_demucs


def _rms(path: Path) -> float:
    """Return root-mean-square level for an audio file."""
    data, _ = sf.read(str(path))
    return float(np.sqrt(np.mean(np.square(data))))


def compare_stems(inst1: Path, inst2: Path) -> float:
    """Return similarity score between two instrumental stems."""
    if not inst1.exists() or not inst2.exists():
        return 0.0
    rms1 = _rms(inst1)
    rms2 = _rms(inst2)
    return 1.0 - abs(rms1 - rms2) / max(rms1, rms2, 1e-6)


def separate_and_score(input_path: str) -> Tuple[Path, Path, float]:
    """Run both separation methods and return best stems and a confidence score."""
    tempdir = Path(tempfile.mkdtemp())
    uvr_dir = tempdir / "uvr"
    demucs_dir = tempdir / "demucs"

    try:
        vocal_uvr, inst_uvr = run_uvr(input_path, str(uvr_dir))
    except (FileNotFoundError, RuntimeError):
        vocal_uvr, inst_uvr = None, None

    try:
        vocal_demucs, inst_demucs = run_demucs(input_path, str(demucs_dir))
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Demucs unavailable: {exc}")
        vocal_demucs, inst_demucs = None, None

    if inst_uvr is not None and inst_demucs is not None:
        score = compare_stems(inst_uvr, inst_demucs)
    else:
        score = 0.0

    if vocal_demucs is None or inst_demucs is None:
        if vocal_uvr is None or inst_uvr is None:
            raise RuntimeError("No separation method available")
        vocal, inst = vocal_uvr, inst_uvr
        confidence = 0.0
    else:
        if score >= 0.5 and vocal_uvr is not None and inst_uvr is not None:
            vocal, inst = vocal_uvr, inst_uvr
        else:
            vocal, inst = vocal_demucs, inst_demucs
        confidence = float(score)

    # Copy chosen stems to a stable location
    final_dir = tempdir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_vocal = final_dir / "vocals.wav"
    final_inst = final_dir / "instrumental.wav"
    shutil.copy2(vocal, final_vocal)
    shutil.copy2(inst, final_inst)

    # Clean up other dirs
    for p in [uvr_dir, demucs_dir]:
        shutil.rmtree(p, ignore_errors=True)

    return final_vocal, final_inst, confidence
