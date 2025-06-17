"""MVSEP loader for Ultimate Chord Reader.
Runs Ultimate Vocal Remover (UVR) using MVSEP-quality MDX models.
Assumes `uvr.py` is installed and available on the system path.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Tuple


def run_uvr(input_path: str, output_dir: str) -> Tuple[Path, Path]:
    """Run UVR via subprocess and return paths to vocal and instrumental stems."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    uvr_exe = shutil.which("uvr.py")
    if not uvr_exe or not Path(uvr_exe).exists():
        raise FileNotFoundError(
            "uvr.py not found. Install Ultimate Vocal Remover or set the path "
            "to the uvr.py script."
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
    subprocess.run(cmd, check=True)

    vocal_path = output / "vocals.wav"
    instrumental_path = output / "instrumental.wav"
    return vocal_path, instrumental_path
