## Ultimate Chord Reader - WORK IN PROGRESS, DO NOT INSTALL LOCALLY UNLESS YOU KNOW WHAT YOU'RE DOING

The `ultimate_chord_reader.py` script provides a local workflow for transcribing songs. Drop audio files into `input_songs/` and run `python ultimate_chord_reader.py` to generate text chord charts in `output_charts/`.

### Setup
Create and activate a virtual environment and install the requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Optional: Ultimate Vocal Remover
By default the separation step uses both [Demucs](https://github.com/facebookresearch/demucs) and Ultimate Vocal Remover (UVR). If `uvr.py` is not available on your system, the script will automatically fall back to using Demucs only. To enable UVR support, clone the UVR repository and ensure the `uvr.py` entry point is on your `PATH`.

### Demucs Usage
The project invokes Demucs through the local Python environment. Install it with:

```bash
pip install demucs
```

If the Python modules are unavailable, the script looks for the `demucs`
executable via ``shutil.which``. Install the package in your virtual
environment so the binary is on your ``PATH``.
If Demucs is missing, the separation step will log a warning and continue with
UVR only. You may also run Demucs manually and place the resulting stems next
to your input files.

