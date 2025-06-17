"""Chord and tempo analysis utilities for Ultimate Chord Reader."""

from __future__ import annotations

from typing import List, Tuple
from pathlib import Path

import librosa
import numpy as np


def analyze_instrumental(inst_path: str, sr: int = 44100) -> Tuple[float, str, List[Tuple[str, float, float]]]:
    """Analyze BPM, key, and chords from an instrumental stem."""
    y, sr = librosa.load(inst_path, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    key_idx = key.argmax()
    keys = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    ]
    est_key = keys[key_idx % 12]

    chord_templates = {
        "C": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "G": [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        # TODO: expand chord templates
    }

    chords: List[Tuple[str, float, float]] = []
    for i in range(chroma.shape[1]):
        frame = chroma[:, i]
        best = None
        best_score = -1.0
        for name, tmpl in chord_templates.items():
            tmpl_vec = np.array(tmpl)
            score = float(np.dot(frame, tmpl_vec) / (np.linalg.norm(frame) * np.linalg.norm(tmpl_vec) + 1e-6))
            if score > best_score:
                best = name
                best_score = score
        if best:
            time = float(librosa.frames_to_time(i, sr=sr))
            chords.append((best, time, best_score))

    return tempo, est_key, chords
