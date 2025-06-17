"""MVSEP loader for Ultimate Chord Reader.

This wrapper launches Ultimate Vocal Remover (UVR) via ``uvr.py``.  The path to
``uvr.py`` is detected via the ``UVR_PY`` environment variable or ``PATH``.  The
function attempts to locate the resulting stems regardless of the exact file
names produced by the configured model.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from subprocess import CalledProcessError
from pathlib import Path
from typing import Tuple, Optional


def _find_stem(directory: Path, key: str) -> Optional[Path]:
    for p in directory.rglob("*.wav"):
        if key.lower() in p.stem.lower():
            return p
    return None


def run_uvr(input_path: str, output_dir: str) -> Tuple[Path, Path]:
    """Run UVR via subprocess and return paths to vocal and instrumental stems."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    uvr_exe = os.environ.get("UVR_PY") or shutil.which("uvr.py")
    if not uvr_exe or not Path(uvr_exe).exists():
        raise FileNotFoundError(
            "uvr.py not found. Install Ultimate Vocal Remover or set the UVR_PY environment variable."
        )

    cmd = [
        "python",
        uvr_exe,
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--model",
        "UVR-MDX",
    ]
    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, CalledProcessError) as exc:
        raise RuntimeError("UVR execution failed") from exc

    vocal_path = _find_stem(output, "vocals")
    instrumental_path = _find_stem(output, "instrumental")
    if vocal_path is None or instrumental_path is None:
        raise RuntimeError("UVR did not produce expected stems")
    return vocal_path, instrumental_path
