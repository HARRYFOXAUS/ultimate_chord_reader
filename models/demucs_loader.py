"""Demucs loader for Ultimate Chord Reader.
Uses the Demucs Python API to perform source separation.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Tuple

try:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    import torch
except Exception as exc:  # type: ignore
    apply_model = None  # type: ignore
    get_model = None  # type: ignore
    torch = None  # type: ignore


def run_demucs(input_path: str, output_dir: str) -> Tuple[Path, Path]:
    """Run Demucs and return paths to vocal and instrumental stems."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if apply_model is None:
        # Fallback to CLI if API unavailable
        cmd = ["demucs", str(input_path), "--out", str(output)]
        subprocess.run(cmd, check=True)
        demucs_out = output / Path(input_path).stem
    else:
        model = get_model("htdemucs")
        wav = torch.from_numpy(model.audio_loader.load_audio(input_path)[0])
        sources = apply_model(model, wav[None])
        demucs_out = output / Path(input_path).stem
        demucs_out.mkdir(parents=True, exist_ok=True)
        for name, src in zip(model.sources, sources[0]):
            dest = demucs_out / f"{name}.wav"
            model.audio_loader.save_audio(dest, src)

    vocal_path = demucs_out / "vocals.wav"
    instrumental_path = demucs_out / "no_vocals.wav"

    if not instrumental_path.exists():
        # Some models name the instrumental stem differently
        inst = [p for p in demucs_out.glob("*.wav") if "vocals" not in p.name]
        if inst:
            instrumental_path = inst[0]

    return vocal_path, instrumental_path