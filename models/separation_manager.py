"""Stem separation manager for Ultimate Chord Reader.

Flow
----
1. Try Demucs (preferred baseline).
2. If Demucs fails, try UVR.
3. If both succeed, compare instrumental RMS and choose the closer pair.
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
    data, _ = sf.read(str(path))
    return float(np.sqrt(np.mean(np.square(data))))


def _similarity(inst1: Path, inst2: Path) -> float:
    if not inst1.exists() or not inst2.exists():
        return 0.0
    rms1, rms2 = _rms(inst1), _rms(inst2)
    return 1.0 - abs(rms1 - rms2) / max(rms1, rms2, 1e-6)


def separate_and_score(input_path: str) -> Tuple[Path, Path, float]:
    tempdir = Path(tempfile.mkdtemp())
    uvr_dir, demucs_dir = tempdir / "uvr", tempdir / "demucs"

    try:
        vocal_demucs, inst_demucs = run_demucs(input_path, str(demucs_dir))
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[Demucs] unavailable → {exc}")
        vocal_demucs = inst_demucs = None

    try:
        vocal_uvr, inst_uvr = run_uvr(input_path, str(uvr_dir))
    except (FileNotFoundError, RuntimeError):
        vocal_uvr = inst_uvr = None

    # Decide which set to return
    if inst_uvr is not None and inst_demucs is not None:
        score = _similarity(inst_uvr, inst_demucs)
    else:
        score = 0.0

    if vocal_demucs is None or inst_demucs is None:
        if vocal_uvr is None or inst_uvr is None:
            raise RuntimeError("No separation method available")
        chosen_v, chosen_i, conf = vocal_uvr, inst_uvr, 0.0
    else:
        if score >= 0.5 and vocal_uvr is not None and inst_uvr is not None:
            chosen_v, chosen_i = vocal_uvr, inst_uvr
        else:
            chosen_v, chosen_i = vocal_demucs, inst_demucs
        conf = float(score)

    # Move selected stems to a stable folder
    final_dir = tempdir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_v, final_i = final_dir / "vocals.wav", final_dir / "instrumental.wav"
    shutil.copy2(chosen_v, final_v)
    shutil.copy2(chosen_i, final_i)

    # Clean temporary sub-folders
    for p in (uvr_dir, demucs_dir):
        shutil.rmtree(p, ignore_errors=True)

    return final_v, final_i, conf
