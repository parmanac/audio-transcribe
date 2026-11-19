# Repository guidelines

## Scope

This Python library and CLI transcribe speech from audio files and the audio track of videos.
The default exports are an SRT subtitle file and an editable DOCX for each recording. Keep these
defaults consistent in the CLI, Python API, examples, and tests. The default language is Spanish.

## Project layout

- `src/transcribe_audio/cli.py` defines command-line options and processes files and folders.
- `api.py` provides `Transcriber`, `TranscriberConfig`, `transcribe_file`, and `transcribe_dir`.
- `backends.py` selects local or OpenAI models, decodes recordings, and handles conversion.
- `discovery.py` discovers media and resolves output names and missing formats.
- `formats.py` writes SRT, DOCX, TXT, Markdown, JSON, and WebVTT from a single transcript.
- `models.py` and `text_utils.py` define transcript data and text helpers.
- `tests/` contains pytest tests with temporary files and simulated backends.
- `transcribir.ps1` runs the CLI using the project virtual environment on Windows.

## Development and verification

Use Python 3.10 or newer, four-space indentation, and the existing public interfaces. Keep device
selection and output defaults in shared code. Use native Word paragraphs and runs so transcript
text and speaker labels remain editable. Preserve existing output files unless the caller chooses
to overwrite them.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m transcribe_audio --help
```

Unit tests must run offline without model weights or API keys. For changes to recognition or
exports, also transcribe a short local recording. Inspect the subtitle timings, reopen the DOCX,
and compare its text with the SRT. If layout changes, render the Word document and inspect the pages.
Use an isolated temporary directory for smoke-test outputs.

## Repository contents

Commit source, tests, package metadata, and documentation. Keep recordings, generated transcripts,
credentials, model weights, virtual environments, and build caches outside Git. Set
`OPENAI_API_KEY` through the environment. Use a short imperative subject for commits and record
the verification commands when describing a change.
