"""Demucs loader for Ultimate Chord Reader.

The module prefers the Demucs Python API and only falls back to invoking the
``demucs`` command-line tool found on ``PATH``.  This avoids relying on a
system-wide installation and keeps execution within the current virtual
environment.

If Demucs is not installed, :func:`run_demucs` raises ``FileNotFoundError`` with
instructions for installing the package or running separation manually.
"""

from __future__ import annotations

import sys
import shutil
import subprocess
from subprocess import CalledProcessError
from pathlib import Path
from typing import Tuple

try:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    import torch
except Exception:  # pragma: no cover - optional dependency
    apply_model = None  # type: ignore
    get_model = None  # type: ignore
    torch = None  # type: ignore


def run_demucs(input_path: str, output_dir: str) -> Tuple[Path, Path]:
    """Run Demucs and return paths to vocal and instrumental stems."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    demucs_out = output / Path(input_path).stem

    if apply_model is not None and torch is not None:
        try:
            model = get_model("htdemucs")
            wav = model.audio_loader.load_audio(input_path)[0]
            wav = torch.from_numpy(wav)
            sources = apply_model(model, wav[None])
            demucs_out.mkdir(parents=True, exist_ok=True)
            for name, src in zip(model.sources, sources[0]):
                dest = demucs_out / f"{name}.wav"
                model.audio_loader.save_audio(dest, src)
        except Exception as exc:
            raise RuntimeError("Demucs API failed to run") from exc
    else:
        exe = shutil.which("demucs")
        if exe is None:
            raise FileNotFoundError(
                "Demucs executable not found. Install it with 'pip install demucs' in your virtual environment."
            )
        cmd = [exe, "separate", str(input_path), "--out", str(output)]
        try:
            subprocess.run(cmd, check=True)
        except CalledProcessError as exc:
            raise RuntimeError("Demucs CLI failed") from exc

    vocal_path = demucs_out / "vocals.wav"
    instrumental_path = demucs_out / "no_vocals.wav"

    if not instrumental_path.exists():
        # Some models name the instrumental stem differently
        for p in demucs_out.glob("*.wav"):
            if "vocals" not in p.name:
                instrumental_path = p
                break

    return vocal_path, instrumental_path
