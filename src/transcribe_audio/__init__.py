"""transcribe_audio: transcripcion de audio y video con faster-whisper o la API de OpenAI."""

from __future__ import annotations

from .api import Transcriber, TranscriberConfig, transcribe_dir, transcribe_file
from .formats import write_outputs
from .models import Segment, Transcript

__version__ = "0.4.0"

__all__ = [
    "Transcriber",
    "TranscriberConfig",
    "transcribe_file",
    "transcribe_dir",
    "write_outputs",
    "Transcript",
    "Segment",
    "__version__",
]
