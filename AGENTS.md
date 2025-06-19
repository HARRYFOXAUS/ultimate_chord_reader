# Agent Instructions for Ultimate Chord Reader

This repository depends on heavy packages that are not available in Codex environments.
To keep verification lightweight, agents should **not** run the full application or
install large dependencies. Instead, use the included lightweight tests.

## Testing

1. Run a syntax check:
   ```bash
   python -m py_compile ultimate_chord_reader.py models/*.py chords.py lyrics.py
   ```
2. Run the minimal test suite:
   ```bash
   pytest -q tests/test_basic.py
   ```

These tests rely on stub modules under `tests/stubs` and avoid heavy
runtime requirements. Do **not** attempt to run other parts of the program.
