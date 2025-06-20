"""Chord and tempo analysis utilities for Ultimate Chord Reader."""

from __future__ import annotations

from typing import Dict, List, Tuple

import librosa
import numpy as np

NOTE_NAMES = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]

CHORD_INTERVALS: Dict[str, List[int]] = {
    "": [0, 4, 7],  # major triad
    "m": [0, 3, 7],  # minor triad
    "6": [0, 4, 7, 9],  # major sixth
    "maj7": [0, 4, 7, 11],  # major seventh
    "7": [0, 4, 7, 10],  # dominant seventh
    "m7": [0, 3, 7, 10],  # minor seventh
    "maj9": [0, 4, 7, 11, 2],  # major ninth
    "9": [0, 4, 7, 10, 2],  # dominant ninth
    "m9": [0, 3, 7, 10, 2],  # minor ninth
}


def _build_chord_templates() -> Dict[str, List[int]]:
    """Generate chroma templates for common chord types."""
    templates: Dict[str, List[int]] = {}
    for root_idx, name in enumerate(NOTE_NAMES):
        for suffix, steps in CHORD_INTERVALS.items():
            vec = [0] * 12
            for step in steps:
                vec[(root_idx + step) % 12] = 1
            templates[name + suffix] = vec
    return templates


def analyze_instrumental(
    inst_path: str,
    sr: int = 44100,
) -> Tuple[str, List[Tuple[str, float, float]]]:
    """
    Analyze an *instrumental* stem and return:

        key   – e.g. "C", "G#m", "F#maj7" (string)
        chords – list of (chord_name, start_time_sec, similarity_score)

    BPM / beat-times are **not** computed here any more – those are handled
    earlier in the main pipeline.
    """

    import librosa
    import numpy as np

    y, sr = librosa.load(inst_path, sr=sr)

    # -------- key detection (same as before) -----------------------------
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key_idx = chroma.mean(axis=1).argmax()
    key_names = ["C", "C#", "D", "D#", "E", "F",
                 "F#", "G", "G#", "A", "A#", "B"]
    est_key = key_names[key_idx % 12]

    # -------- chord template matching (unchanged) ------------------------
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chord_templates = _build_chord_templates()

    chords: List[Tuple[str, float, float]] = []
    for i in range(chroma_cqt.shape[1]):
        frame = chroma_cqt[:, i]
        best, best_score = None, -1.0
        for name, tmpl in chord_templates.items():
            tmpl_vec = np.asarray(tmpl)
            score = float(
                np.dot(frame, tmpl_vec)
                / (np.linalg.norm(frame) * np.linalg.norm(tmpl_vec) + 1e-6)
            )
            if score > best_score:
                best, best_score = name, score
        if best:
            time_s = float(librosa.frames_to_time(i, sr=sr))
            chords.append((best, time_s, best_score))

    return est_key, chords

