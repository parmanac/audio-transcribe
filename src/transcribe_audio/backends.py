"""Backends de transcripcion: faster-whisper (local) y API de OpenAI."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .discovery import is_video_file, safe_filename
from .models import Segment, Transcript
from .text_utils import clean_text, get_value, join_speaker_text, join_transcript_text, to_jsonable

DEFAULT_LOCAL_MODEL = "large-v3"
CPU_FALLBACK_MODEL = "medium"
DEFAULT_OPENAI_MODEL = "gpt-transcribe"
DIARIZE_MODEL = "gpt-4o-transcribe-diarize"
DEFAULT_OPENAI_TIMESTAMP_MODEL = "whisper-1"

OPENAI_INPUT_EXTENSIONS = {
    ".flac",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".ogg",
    ".wav",
    ".webm",
}

OPENAI_MAX_BYTES = 25 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Resolucion de dispositivo y modelo local
# --------------------------------------------------------------------------- #
def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def choose_local_model(explicit_model: str | None, resolved_device: str) -> str:
    if explicit_model:
        return explicit_model
    if resolved_device == "cpu":
        return CPU_FALLBACK_MODEL
    return DEFAULT_LOCAL_MODEL


def load_local_model(
    model_name: str,
    device: str,
    compute_type: str,
    cpu_threads: int = 0,
    num_workers: int = 1,
) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Falta faster-whisper. Instala dependencias con: python -m pip install -e ."
        ) from exc

    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
    )


# --------------------------------------------------------------------------- #
# Backend local
# --------------------------------------------------------------------------- #
def transcribe_local(
    audio_path: Path,
    model_name: str,
    whisper_model: Any,
    language: str | None,
    prompt: str | None,
    beam_size: int,
    vad: bool,
    vad_min_silence_ms: int,
    temperature: float,
    condition_on_previous_text: bool,
    word_timestamps: bool,
    log_progress: bool = False,
) -> Transcript:
    from av.error import FFmpegError

    kwargs: dict[str, Any] = dict(
        language=language,
        task="transcribe",
        beam_size=beam_size,
        vad_filter=vad,
        vad_parameters={"min_silence_duration_ms": vad_min_silence_ms} if vad else None,
        initial_prompt=prompt,
        condition_on_previous_text=condition_on_previous_text,
        word_timestamps=word_timestamps,
        temperature=temperature,
        log_progress=log_progress,
    )

    try:
        segments_iter, info = whisper_model.transcribe(str(audio_path), **kwargs)
    except FFmpegError:
        # Algunos contenedores de video no se decodifican directamente: extraemos el
        # audio con ffmpeg (si esta disponible) y reintentamos.
        if is_video_file(audio_path) and shutil.which("ffmpeg"):
            with tempfile.TemporaryDirectory(prefix="transcribe_audio_") as tmp:
                wav = Path(tmp) / f"{safe_filename(audio_path.stem)}.wav"
                extract_audio(audio_path, wav)
                segments_iter, info = whisper_model.transcribe(str(wav), **kwargs)
                return _build_local_transcript(
                    audio_path, model_name, language, segments_iter, info, word_timestamps
                )
        raise
    return _build_local_transcript(audio_path, model_name, language, segments_iter, info, word_timestamps)


def _build_local_transcript(
    audio_path: Path,
    model_name: str,
    language: str | None,
    segments_iter: Any,
    info: Any,
    word_timestamps: bool,
) -> Transcript:
    segments: list[Segment] = []
    text_parts: list[str] = []
    for segment in segments_iter:
        text = clean_text(segment.text)
        text_parts.append(text)
        words = None
        if word_timestamps and getattr(segment, "words", None):
            words = [
                {
                    "start": float(word.start),
                    "end": float(word.end),
                    "word": word.word,
                    "probability": getattr(word, "probability", None),
                }
                for word in segment.words
            ]
        segments.append(
            Segment(start=float(segment.start), end=float(segment.end), text=text, words=words)
        )

    detected_language = getattr(info, "language", None) or language
    return Transcript(
        source=str(audio_path),
        backend="local",
        model=model_name,
        language=detected_language,
        text=join_transcript_text(text_parts),
        segments=segments,
        raw={
            "language": detected_language,
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
            "duration_after_vad": getattr(info, "duration_after_vad", None),
        },
    )


# --------------------------------------------------------------------------- #
# Backend OpenAI
# --------------------------------------------------------------------------- #
def select_openai_model(model: str, diarize: bool, need_timestamps: bool, timestamp_model: str) -> str:
    if diarize or model == DIARIZE_MODEL:
        return DIARIZE_MODEL
    return timestamp_model if need_timestamps else model


def transcribe_openai(
    audio_path: Path,
    model: str,
    language: str | None,
    prompt: str | None,
    convert_format: str,
    keep_temp: bool,
    diarize: bool,
    need_timestamps: bool,
    word_timestamps: bool,
    timestamp_model: str = DEFAULT_OPENAI_TIMESTAMP_MODEL,
    quiet: bool = False,
) -> Transcript:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Falta openai. Instala dependencias con: python -m pip install -e ."
        ) from exc

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Falta la variable OPENAI_API_KEY para usar --backend openai.")

    client = OpenAI()
    with temporary_audio_for_openai(audio_path, convert_format, keep_temp) as upload_path:
        selected_model = select_openai_model(model, diarize, need_timestamps, timestamp_model)
        request: dict[str, Any] = {"model": selected_model}
        if selected_model == DIARIZE_MODEL:
            request["response_format"] = "diarized_json"
            request["chunking_strategy"] = "auto"
        elif need_timestamps:
            if not quiet and model != timestamp_model:
                print(f"  Usando {timestamp_model} para obtener las marcas de tiempo solicitadas.")
            request["response_format"] = "verbose_json"
            granularities = ["segment"]
            if word_timestamps:
                granularities.append("word")
            request["timestamp_granularities"] = granularities
        elif selected_model != DEFAULT_OPENAI_MODEL:
            request["response_format"] = "json"
        if prompt and request["model"] != DIARIZE_MODEL:
            request["prompt"] = prompt
        if language:
            if request["model"] == DEFAULT_OPENAI_MODEL:
                # extra_body también funciona con versiones del SDK anteriores a este campo.
                request["extra_body"] = {"languages": [language]}
            else:
                request["language"] = language

        with upload_path.open("rb") as file_handle:
            response = client.audio.transcriptions.create(file=file_handle, **request)

    text, segments = parse_openai_response(response)
    detected_language = language or get_value(response, "language")
    languages = get_value(response, "languages") or []
    if languages:
        # El campo público es singular. La lista completa permanece en raw.
        detected_language = get_value(languages[0], "code") if len(languages) == 1 else None
    return Transcript(
        source=str(audio_path),
        backend="openai",
        model=request["model"],
        language=detected_language,
        text=text,
        segments=segments,
        raw=to_jsonable(response),
    )


def parse_openai_response(response: Any) -> tuple[str, list[Segment]]:
    if isinstance(response, str):
        return clean_text(response), []

    text = clean_text(get_value(response, "text", "") or "")
    raw_segments = get_value(response, "segments", []) or []
    segments: list[Segment] = []
    for item in raw_segments:
        segment_text = clean_text(get_value(item, "text", "") or "")
        speaker = get_value(item, "speaker", None)
        start = float(get_value(item, "start", 0.0) or 0.0)
        end = float(get_value(item, "end", 0.0) or 0.0)
        segments.append(Segment(start=start, end=end, text=segment_text, speaker=speaker))

    if segments:
        text = join_speaker_text(segments)
    return text, segments


class temporary_audio_for_openai:
    """Devuelve una ruta lista para subir a OpenAI, convirtiendo con ffmpeg si hace falta."""

    def __init__(self, source: Path, convert_format: str, keep_temp: bool) -> None:
        self.source = source
        self.convert_format = convert_format
        self.keep_temp = keep_temp
        self.temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self.path: Path | None = None

    def __enter__(self) -> Path:
        if self.can_upload_directly(self.source):
            self.path = self.source
            return self.path

        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "Este formato o tamano no se puede subir directamente a OpenAI. Instala ffmpeg para convertirlo."
            )

        self.temp_dir = tempfile.TemporaryDirectory(prefix="transcribe_audio_")
        temp_path = Path(self.temp_dir.name) / f"{self.source.stem}.openai.{self.convert_format}"
        convert_for_api(self.source, temp_path)
        if temp_path.stat().st_size > OPENAI_MAX_BYTES:
            raise RuntimeError(
                f"El audio convertido pesa {temp_path.stat().st_size / 1024 / 1024:.1f} MB; "
                "la API admite hasta 25 MB. Usa --backend local o divide/comprime el audio."
            )
        self.path = temp_path
        if self.keep_temp:
            keep_path = self.source.with_suffix(f".openai.{self.convert_format}")
            shutil.copy2(temp_path, keep_path)
            print(f"  Conversion conservada: {keep_path}")
        return temp_path

    def __exit__(self, *_: Any) -> None:
        if self.temp_dir is not None:
            self.temp_dir.cleanup()

    @staticmethod
    def can_upload_directly(path: Path) -> bool:
        return path.suffix.lower() in OPENAI_INPUT_EXTENSIONS and path.stat().st_size <= OPENAI_MAX_BYTES


# --------------------------------------------------------------------------- #
# ffmpeg helpers
# --------------------------------------------------------------------------- #
def convert_for_api(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    if target.suffix.lower() == ".mp3":
        command.extend(["-b:a", "64k"])
    command.append(str(target))
    _run_ffmpeg(command, source)


def extract_audio(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target),
    ]
    _run_ffmpeg(command, source)


def _run_ffmpeg(command: list[str], source: Path) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg no pudo procesar {source.name}: {detail or exc}") from exc
