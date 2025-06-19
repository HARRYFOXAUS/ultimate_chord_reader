import sys
import importlib.util
from pathlib import Path

# Add stub modules to sys.path so imports in production code succeed
sys.path.insert(0, str(Path(__file__).resolve().parent / 'stubs'))

# Load ultimate_chord_reader as a module from its file
ucr_path = Path(__file__).resolve().parents[1] / 'ultimate_chord_reader.py'
spec = importlib.util.spec_from_file_location('ucr', ucr_path)
ucr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ucr)


def test_format_chart_basic(tmp_path):
    lyrics = [
        (0.0, 1.0, 'hello', 0.9),
        (2.0, 3.0, 'world', 0.9),
    ]
    chords = [
        ('C', 0.0, 0.9),
        ('G', 2.0, 0.9),
    ]
    text = ucr.format_chart(
        title='Test',
        bpm=60.0,
        key='C',
        time_sig='4/4',
        lyrics=lyrics,
        chords=chords,
        confidence=80.0,
    )
    lines = text.splitlines()
    assert lines[-1] == 'C G\thello world'


def test_overwrite_and_remove(tmp_path):
    f = tmp_path / 'sample.txt'
    f.write_text('data')
    ucr.overwrite_and_remove(f)
    assert not f.exists()
