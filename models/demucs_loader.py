"""Demucs loader for Ultimate Chord Reader.

This module prefers the Demucs Python API for separation but falls back to the
``demucs`` CLI if the modules are unavailable.  Providing both options makes the
tool usable in more environments and gives clearer error messages when neither
is installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import shutil
import subprocess

try:  # pragma: no cover - optional dependency
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from demucs.audio import AudioFile, save_audio
    import torch
except Exception:  # pragma: no cover - optional dependency
    apply_model = None  # type: ignore
    get_model = None  # type: ignore
    AudioFile = None  # type: ignore
    save_audio = None  # type: ignore
    torch = None  # type: ignore


def run_demucs(input_path: str, output_dir: str) -> Tuple[Path, Path]:
    """Run Demucs and return paths to vocal and instrumental stems."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if (
        apply_model is None
        or get_model is None
        or AudioFile is None
        or save_audio is None
        or torch is None
    ):
        # Fallback to CLI if Python modules are unavailable
        demucs_exe = shutil.which("demucs")
        if not demucs_exe:
            raise FileNotFoundError(
                "Demucs executable not found. Install it with 'pip install demucs'."
            )

        cmd = [demucs_exe, "-o", str(output), input_path]
        try:
            subprocess.run(cmd, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("Demucs CLI failed") from exc

        # Locate produced stems
        demucs_out = next((p for p in output.rglob("vocals.wav")), None)
        if demucs_out is None:
            raise RuntimeError("Demucs output not found")
        vocal_path = demucs_out
        inst_candidate = next(
            (p for p in demucs_out.parent.glob("*.wav") if "vocals" not in p.name),
            None,
        )
        instrumental_path = inst_candidate or demucs_out.parent / "no_vocals.wav"
        return vocal_path, instrumental_path

    print("Using Demucs Python API for separation.")

    demucs_out = output / Path(input_path).stem

    try:
        model = get_model("htdemucs")
        wav = AudioFile(input_path).read(
            streams=0,
            samplerate=model.samplerate,
            channels=model.audio_channels,
        )
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        sources = apply_model(
            model, wav[None], split=True, overlap=0.25, progress=False
        )[0]
        sources = sources * ref.std() + ref.mean()
        demucs_out.mkdir(parents=True, exist_ok=True)
        for source, name in zip(sources, model.sources):
            dest = demucs_out / f"{name}.wav"
            save_audio(source, dest, model.samplerate)
    except Exception as exc:
        raise RuntimeError("Demucs API failed to run") from exc

    vocal_path = demucs_out / "vocals.wav"
    instrumental_path = demucs_out / "no_vocals.wav"

    if not instrumental_path.exists():
        # Some models name the instrumental stem differently
        for p in demucs_out.glob("*.wav"):
            if "vocals" not in p.name:
                instrumental_path = p
                break

    return vocal_path, instrumental_path
