"""API de alto nivel para transcribir audio y video desde codigo Python.

Ejemplo:

    from transcribe_audio import transcribe_file
    print(transcribe_file("entrevista.flac").text)

    from transcribe_audio import Transcriber, TranscriberConfig
    t = Transcriber(TranscriberConfig(backend="local", formats=["txt", "srt"]))
    transcript = t.transcribe("clase.mp4")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .backends import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_TIMESTAMP_MODEL,
    DIARIZE_MODEL,
    choose_local_model,
    load_local_model,
    resolve_device,
    select_openai_model,
    transcribe_local,
    transcribe_openai,
)
from .discovery import list_media_files, missing_formats, output_base_for
from .formats import DEFAULT_FORMATS, needs_timestamps, normalize_formats, write_outputs
from .models import Transcript


@dataclass
class TranscriberConfig:
    backend: str = "local"
    model: str | None = None
    language: str | None = "es"  # None = autodetectar
    prompt: str | None = None
    formats: list[str] = field(default_factory=lambda: list(DEFAULT_FORMATS))
    # backend local
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    vad: bool = True
    vad_min_silence_ms: int = 500
    temperature: float = 0.0
    cpu_threads: int = 0
    num_workers: int = 1
    condition_on_previous_text: bool = False
    word_timestamps: bool | None = None  # Por defecto, activados solo en local.
    # backend openai
    diarize: bool = False
    openai_timestamp_model: str = DEFAULT_OPENAI_TIMESTAMP_MODEL
    api_convert_format: str = "mp3"
    keep_temp: bool = False
    quiet: bool = False


def openai_model_name(config: TranscriberConfig) -> str:
    if config.model:
        return config.model
    return DIARIZE_MODEL if config.diarize else DEFAULT_OPENAI_MODEL


class Transcriber:
    """Transcribe ficheros reutilizando el modelo local (que se carga una sola vez)."""

    def __init__(self, config: TranscriberConfig | None = None) -> None:
        self.config = config or TranscriberConfig()
        self._local_model = None
        self._resolved_device: str | None = None
        self._local_model_name: str | None = None

    @property
    def resolved_device(self) -> str:
        if self._resolved_device is None:
            self._resolved_device = resolve_device(self.config.device)
        return self._resolved_device

    @property
    def local_model_name(self) -> str:
        if self._local_model_name is None:
            self._local_model_name = choose_local_model(self.config.model, self.resolved_device)
        return self._local_model_name

    def model_name(self) -> str:
        cfg = self.config
        if cfg.backend == "openai":
            return select_openai_model(
                openai_model_name(cfg),
                cfg.diarize,
                needs_timestamps(cfg.formats, bool(cfg.word_timestamps)),
                cfg.openai_timestamp_model,
            )
        return self.local_model_name

    def _ensure_local_model(self):
        if self._local_model is None:
            if not self.config.quiet:
                print(
                    f"  Cargando modelo local {self.local_model_name} "
                    f"({self.resolved_device}, {self.config.compute_type})..."
                )
            self._local_model = load_local_model(
                self.local_model_name,
                self.resolved_device,
                self.config.compute_type,
                self.config.cpu_threads,
                self.config.num_workers,
            )
        return self._local_model

    def transcribe(self, media_path: str | Path) -> Transcript:
        cfg = self.config
        path = Path(media_path)
        word_timestamps = cfg.word_timestamps if cfg.word_timestamps is not None else cfg.backend == "local"
        if cfg.backend == "local":
            model = self._ensure_local_model()
            return transcribe_local(
                audio_path=path,
                model_name=self.local_model_name,
                whisper_model=model,
                language=cfg.language,
                prompt=cfg.prompt,
                beam_size=cfg.beam_size,
                vad=cfg.vad,
                vad_min_silence_ms=cfg.vad_min_silence_ms,
                temperature=cfg.temperature,
                condition_on_previous_text=cfg.condition_on_previous_text,
                word_timestamps=word_timestamps,
                log_progress=not cfg.quiet,
            )
        if cfg.backend != "openai":
            raise ValueError(f"Backend desconocido: {cfg.backend}")
        return transcribe_openai(
            audio_path=path,
            model=openai_model_name(cfg),
            language=cfg.language,
            prompt=cfg.prompt,
            convert_format=cfg.api_convert_format,
            keep_temp=cfg.keep_temp,
            diarize=cfg.diarize,
            need_timestamps=needs_timestamps(cfg.formats, word_timestamps),
            word_timestamps=word_timestamps,
            timestamp_model=cfg.openai_timestamp_model,
            quiet=cfg.quiet,
        )


def transcribe_file(
    media_path: str | Path,
    *,
    config: TranscriberConfig | None = None,
    **overrides,
) -> Transcript:
    """Transcribe un unico fichero y devuelve el Transcript (no escribe ficheros)."""
    cfg = config or TranscriberConfig(**overrides)
    return Transcriber(cfg).transcribe(media_path)


def transcribe_dir(
    directory: str | Path,
    *,
    output_dir: str | Path = "transcripciones",
    recursive: bool = False,
    overwrite: bool = False,
    config: TranscriberConfig | None = None,
    **overrides,
) -> list[Transcript]:
    """Transcribe todos los medios de una carpeta y escribe las salidas configuradas."""
    cfg = config or TranscriberConfig(**overrides)
    files = list_media_files([Path(directory)], recursive=recursive)
    formats = normalize_formats(cfg.formats)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transcriber = Transcriber(cfg)

    results: list[Transcript] = []
    for media_path in files:
        base = output_base_for(media_path, out_dir, files)
        pending = formats if overwrite else missing_formats(base, formats)
        if not pending:
            continue
        transcript = transcriber.transcribe(media_path)
        write_outputs(transcript, base, pending)
        results.append(transcript)
    return results
