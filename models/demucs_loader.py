"""Demucs loader for Ultimate Chord Reader.

Order of attempts
-----------------
1.  Demucs **Python API**  (fast, no subprocess)
2.  CLI via **python -m demucs.separate**  (works even if `demucs` isn’t on $PATH)
3.  CLI via **demucs** binary on $PATH     (final fallback)

Returns
-------
Tuple[Path, Path]  →  (vocal_stem, instrumental_stem)
"""
from __future__ import annotations

import sys
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

# ----------------------------------------------------------------------
# Try to import the Python API. If anything fails we’ll drop to the CLI.
# ----------------------------------------------------------------------
try:  # pragma: no cover – optional dependency
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from demucs.audio import AudioFile, save_audio
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover
    print(f"[Demucs] Python API unavailable ({exc}); falling back to CLI.")
    apply_model = get_model = AudioFile = save_audio = None  # type: ignore


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _run(cmd: list[str]) -> None:
    """Run *cmd* and re-raise failures as RuntimeError."""
    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Demucs CLI failed") from exc


# ----------------------------------------------------------------------
# Main public entry-point
# ----------------------------------------------------------------------
def run_demucs(input_path: str, output_dir: str, *, model: str = "htdemucs") -> Tuple[Path, Path]:
    """Return (vocal_stem, instrumental_stem) for *input_path* using *model*."""
    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) ---------- Python-API fast path --------------------------------
    if all(obj is not None for obj in (apply_model, get_model, AudioFile, save_audio)):
        # Tell static type-checkers these symbols are non-None from here
        assert apply_model and get_model and AudioFile and save_audio

        try:
            demucs_model = get_model(model)
            wav = AudioFile(Path(input_path)).read(          # cast to Path
                streams=0,                                   # type: ignore[arg-type]
                samplerate=demucs_model.samplerate,
                channels=demucs_model.audio_channels,
            )
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / ref.std()

            sources = apply_model(
                demucs_model, wav[None], split=True, overlap=0.25, progress=False
            )[0]
            sources = sources * ref.std() + ref.mean()

            stems_dir = out_root / Path(input_path).stem
            stems_dir.mkdir(exist_ok=True)
            for source, name in zip(sources, demucs_model.sources):
                save_audio(source, stems_dir / f"{name}.wav", demucs_model.samplerate)

            vocal = stems_dir / "vocals.wav"
            inst = stems_dir / "no_vocals.wav"
            if not inst.exists():
                inst = next(stems_dir.glob("*accompaniment*.wav"), inst)

            if vocal.exists() and inst.exists():
                print("[Demucs] Separated with Python API")
                return vocal, inst

            raise RuntimeError("Demucs API produced no stems")

        except Exception as exc:
            print(f"[Demucs] API failed ({exc}); switching to CLI.")

    # 2) ---------- CLI via python -m demucs.separate -------------------
    cli_cmd = [
        sys.executable, "-m", "demucs.separate",
        "--two-stems=vocals",
        "-n", model,
        "-o", str(out_root),
        input_path,
    ]
    try:
        _run(cli_cmd)
    except RuntimeError:
        # 3) ---- Final fallback: standalone `demucs` binary ------------
        demucs_bin = shutil.which("demucs")
        if not demucs_bin:
            raise FileNotFoundError(
                "Demucs unavailable via API or CLI. "
                "Run `pip install demucs` in this environment."
            )
        _run([
            demucs_bin, "--two-stems=vocals",
            "-n", model,
            "-o", str(out_root), input_path,
        ])

    # -------- Locate stems the CLI just wrote --------------------------
    wav_files = list(out_root.rglob("*.wav"))
    print("[Demucs-debug] searched", out_root, "found", len(wav_files), "wav files")
    for p in out_root.rglob("*"):    # show the tree once
        print("   ", p.relative_to(out_root))
    if not wav_files:
        raise RuntimeError("Demucs CLI produced no wav files")

        # Pick stems robustly: ‘vocals.wav’ vs ‘no_vocals.wav’ (or accompaniment)
    vocal = next((p for p in wav_files if p.name.lower().startswith("vocals")), None)
    inst  = next(
        (p for p in wav_files if p is not vocal and (
            "no_vocals" in p.name.lower() or "accompaniment" in p.name.lower())),
        None,
    )
    if inst is None and vocal is not None and len(wav_files) == 2:
        # fallback: exactly two files, so the other one must be instrumental
        inst = next(p for p in wav_files if p is not vocal)

    if vocal is None or inst is None:
        raise RuntimeError("Couldn’t find expected stems in Demucs output")

    print("[Demucs] Separated with CLI →", vocal.relative_to(out_root))
    return vocal, inst

    # ------------------------------------------------------------------
    # Unreachable: every path above returns or raises.
    # Added to satisfy static type-checkers.
    # ------------------------------------------------------------------
    raise RuntimeError("Demucs failed on all attempted paths (unreachable)")
