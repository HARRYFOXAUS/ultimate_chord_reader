"""Demucs loader for Ultimate Chord Reader.

This module uses the Demucs Python API exclusively to perform source
separation.  No subprocess invocation of the ``demucs`` CLI is attempted.

If the required ``demucs`` or ``torch`` modules are missing, :func:`run_demucs`
raises ``FileNotFoundError`` with instructions for installing the package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

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
    if apply_model is None or get_model is None or AudioFile is None or save_audio is None or torch is None:
        raise FileNotFoundError(
            "Demucs or torch not installed. Install them with 'pip install demucs torch'."
        )

    print("Using Demucs Python API for separation.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

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
        sources = apply_model(model, wav[None], split=True, overlap=0.25, progress=False)[0]
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
