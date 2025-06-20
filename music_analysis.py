from __future__ import annotations
from typing import Tuple, List
import librosa
import numpy as np
import soundfile as sf

# lazy-import Essentia only if librosa provides no key_estimate
try:
    from librosa import key_estimate
except ImportError:
    from essentia.standard import KeyExtractor
    def key_estimate(y, sr):
        ke = KeyExtractor()
        key, scale, _strength = ke(y, sr)
        return f"{key}{'' if scale=='major' else 'm'}"

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
CHORD_INTERVALS = {
    "":   [0,4,7],
    "m":  [0,3,7],
    "6":  [0,4,7,9],
    "maj7":[0,4,7,11],
    "7":  [0,4,7,10],
    "m7": [0,3,7,10],
    "maj9":[0,4,7,11,2],
    "9":  [0,4,7,10,2],
    "m9": [0,3,7,10,2],
}

def _build_templates():
    tmpls = {}
    for i, name in enumerate(NOTE_NAMES):
        for suf, steps in CHORD_INTERVALS.items():
            vec = np.zeros(12)
            vec[(i + np.array(steps)) % 12] = 1
            tmpls[name + suf] = vec
    return tmpls
_TEMPLATES = _build_templates()

def analyze_instrumental(
        wav_path: str, *, bpm: float|None=None, beats: List[float]|None=None
) -> Tuple[str, List[Tuple[str, float, float]]]:
    y, sr = sf.read(wav_path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)

    # --- key ---
    est_key = key_estimate(y, sr)

    # --- chords via simple template match on CQT ---
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chords: List[Tuple[str, float, float]] = []
    for i in range(chroma.shape[1]):
        frame = chroma[:, i]
        best, best_score = None, -1.0
        for name, tmpl in _TEMPLATES.items():
            score = float(np.dot(frame, tmpl) / (np.linalg.norm(frame) * np.linalg.norm(tmpl) + 1e-6))
            if score > best_score:
                best, best_score = name, score
        if best:
            time_s = float(librosa.frames_to_time(i, sr=sr))
            chords.append((best, time_s, best_score))

    return est_key, chords
