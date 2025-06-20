<h1 align="center">
  🌀  Ultimate Chord Reader
</h1>

<p align="center">
  <b>Stem separation&nbsp;⋅ BPM &amp; key detection&nbsp;⋅ AI chord + lyric alignment</b><br>
  <i>No audio cached, ever.</i>
  <i>All music licensed to this project by @lukae_music @harryfoxaus @spacerunnerthedj - using ultimate chord reader doesn't give you the license to reproduce these songs but feel free to keep them as example/test tracks in the code. No unlicensed music will ever be comitted to this project. Musicians and artists rights come first.</i>
</p>

<p align="center">
  <a href="https://github.com/HARRYFOXAUS/ultimate_chord_reader/actions">
    <img alt="CI" src="https://github.com/HARRYFOXAUS/ultimate_chord_reader/actions/workflows/ci.yml/badge.svg">
  </a>
  <img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue">
  <img alt="Python"  src="https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue">
</p>

---

## ⚡ Quick-start (local)

```bash
# 1. Get the code
git clone https://github.com/HARRYFOXAUS/ultimate_chord_reader.git
cd ultimate_chord_reader

# 2. (Optional but recommended) activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install deps – the script can auto-install,
#    but this is faster & CI-friendlier
pip install -r requirements.txt

# 4. Drop one or more .mp3/.wav/... files into input_songs/
python ultimate_chord_reader.py --all
# or let the script interactively ask which songs to analyze
Charts appear in output_charts/ – plain-text, one bar per line:

nginx
Copy
Edit
Am9 D#maj9 | Gm9 ........ desire drowns her thoughts
...
<details> <summary><b>What happens under the hood?</b></summary>
text
Copy
Edit
┌───────────────┐
│ input  audio  │
└───────┬───────┘
        │  Demucs (6-stem)      Whisper (lyrics)
        ▼                         ▲
 ┌───────────────┐            ┌────────┐
 │  drum  stem   │──Librosa──▶│  BPM   │
 │ music  stem   │──Chordino─▶│ chords │
 │ vocal  stem   │─Whisper──▶ │ lyrics │
 └───────────────┘            └────────┘
        │                       │
        └────── merge + align ──┘
                 ▼
          text chart ↑ confidence score
</details>

---

PRIVACY
No caching / no training – every stem (/tmp) is zero-filled then unlinked as soon as inference finishes.

BPM ANALYSIS
Hierarchical BPM finder

Drum stem → Librosa onset voting (fast & robust)

No-vocal stem (if drums fail)

Full mix (ultimate fallback)

CHORDS & LYRICS ANALYSIS with % CONFIDENCE SCORES
Chord analysis on music-only stem for cleaner voicings.

Lyric confidence: Whisper log-probs → % so you know when a line is shaky.

Runs fully offline; the only download is the Demucs checkpoint the very first time.

---

Contributing 🫶
Bug reports & PRs are welcome – just remember that no audio examples containing commercial music can be committed.

---

© 2025 Faux Sierra Music – Apache-2.0.
Technology should amplify musicians, not exploit them.
Free musicians from subscriptions.

All music licensed to this project by @lukae_music @harryfoxaus @spacerunnerthedj - using ultimate chord reader doesn't give you the license to reproduce these songs but feel free to keep them as example/test tracks in the code. No unlicensed music will ever be comitted to this project. Musicians and artists rights come first.