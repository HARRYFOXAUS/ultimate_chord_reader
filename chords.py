"""Chord and tempo analysis utilities for Ultimate Chord Reader."""

from __future__ import annotations

from typing import Dict, List, Tuple

import librosa
import numpy as np
from bpm_drums import get_bpm_from_drums

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
    inst_path: str, sr: int = 44100
) -> Tuple[float, str, List[Tuple[str, float, float]], List[float]]:
    """Analyze BPM, key, and chords from an instrumental stem."""

    try:
        tempo, beat_times = get_bpm_from_drums(inst_path)
        y, sr = librosa.load(inst_path, sr=sr)
    except Exception:
        y, sr = librosa.load(inst_path, sr=sr)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
        while tempo > 160:
            tempo /= 2.0
        while tempo < 60:
            tempo *= 2.0

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    key_idx = key.argmax()
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    est_key = keys[key_idx % 12]

    chord_templates = _build_chord_templates()

    chords: List[Tuple[str, float, float]] = []
    for i in range(chroma.shape[1]):
        frame = chroma[:, i]
        best = None
        best_score = -1.0
        for name, tmpl in chord_templates.items():
            tmpl_vec = np.array(tmpl)
            score = float(
                np.dot(frame, tmpl_vec)
                / (np.linalg.norm(frame) * np.linalg.norm(tmpl_vec) + 1e-6)
            )
            if score > best_score:
                best = name
                best_score = score
        if best:
            time = float(librosa.frames_to_time(i, sr=sr))
            chords.append((best, time, best_score))

    return tempo, est_key, chords, beat_times

