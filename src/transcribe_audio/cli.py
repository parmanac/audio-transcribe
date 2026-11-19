"""Interfaz de linea de comandos para transcribir audios y videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .api import Transcriber, TranscriberConfig
from .backends import DEFAULT_OPENAI_TIMESTAMP_MODEL
from .discovery import list_media_files, missing_formats, output_base_for
from .formats import ALL_FORMATS, DEFAULT_FORMATS, normalize_formats, write_outputs

DEFAULT_SPANISH_CONTEXT = (
    "Transcripción literal en español, con nombres propios, tecnicismos y puntuación natural."
)


def configure_console() -> None:
    """Fuerza UTF-8 en la consola para no fallar con acentos en Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcribe-audio",
        description="Transcribe archivos de audio y vídeo a Markdown y subtítulos SRT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("inputs", nargs="+", type=Path, help="Archivos de audio/video o carpetas.")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("transcripciones"), help="Carpeta de salida.")
    parser.add_argument("--backend", choices=("local", "openai"), default="local", help="Motor de transcripcion.")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Modelo local (large-v3, large-v3-turbo, medium, small) o de OpenAI (gpt-transcribe para texto). "
            "Si no se indica en local, se usa large-v3 (o medium si no hay GPU)."
        ),
    )
    parser.add_argument("--language", default="es", help="Codigo ISO del idioma. Usa 'auto' para autodetectar.")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=(*ALL_FORMATS, "all"),
        default=list(DEFAULT_FORMATS),
        help="Formatos de salida.",
    )
    parser.add_argument("--recursive", action="store_true", help="Buscar medios dentro de subcarpetas.")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribir transcripciones existentes.")
    parser.add_argument("--quiet", action="store_true", help="Mostrar solo errores y el resumen final.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Contexto para mejorar nombres propios o vocabulario.",
    )
    parser.add_argument("--prompt-file", type=Path, help="Archivo de texto con contexto adicional.")
    parser.add_argument("--no-default-prompt", action="store_true", help="Omitir el contexto predeterminado de OpenAI en español.")
    parser.add_argument(
        "--word-timestamps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Guardar tiempos por palabra en JSON. Activado por defecto en local y opcional en OpenAI.",
    )

    local = parser.add_argument_group("opciones del backend local")
    local.add_argument("--device", default="auto", help="Dispositivo para faster-whisper: auto, cpu o cuda.")
    local.add_argument("--compute-type", default="auto", help="Tipo de calculo: auto, int8, float16, etc.")
    local.add_argument("--beam-size", type=int, default=5, help="Beam size de Whisper.")
    local.add_argument("--temperature", type=float, default=0.0, help="Temperatura de muestreo (0 = determinista).")
    local.add_argument("--cpu-threads", type=int, default=0, help="Hilos de CPU para faster-whisper (0 = auto).")
    local.add_argument("--num-workers", type=int, default=1, help="Workers paralelos de faster-whisper.")
    local.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usar filtro de voz/silencio para reducir ruido y alucinaciones.",
    )
    local.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=500,
        help="Silencio minimo (ms) que el filtro VAD considera una pausa.",
    )
    local.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help="Mantener contexto entre segmentos. Puede mejorar continuidad, pero a veces repite frases.",
    )

    api = parser.add_argument_group("opciones del backend OpenAI")
    api.add_argument(
        "--api-convert-format",
        choices=("mp3", "wav"),
        default="mp3",
        help="Formato temporal para convertir medios no admitidos o demasiado grandes antes de subir a OpenAI.",
    )
    api.add_argument("--keep-temp", action="store_true", help="Conservar conversiones temporales hechas para la API.")
    api.add_argument(
        "--diarize",
        action="store_true",
        help="Usar gpt-4o-transcribe-diarize para separar hablantes en la API.",
    )
    api.add_argument(
        "--openai-timestamp-model",
        default=DEFAULT_OPENAI_TIMESTAMP_MODEL,
        help="Modelo de OpenAI usado para obtener tiempos (SRT/VTT) cuando no se usa --diarize.",
    )

    return parser.parse_args(argv)


def build_prompt(args: argparse.Namespace) -> str | None:
    parts: list[str] = []
    if args.backend == "openai" and args.language.lower() == "es" and not args.no_default_prompt:
        parts.append(DEFAULT_SPANISH_CONTEXT)
    if args.prompt_file:
        parts.append(args.prompt_file.read_text(encoding="utf-8").strip())
    if args.prompt:
        parts.append(args.prompt.strip())
    prompt = "\n\n".join(part for part in parts if part)
    return prompt or None


def build_config(args: argparse.Namespace) -> TranscriberConfig:
    language = None if args.language.lower() == "auto" else args.language
    return TranscriberConfig(
        backend=args.backend,
        model=args.model,
        language=language,
        prompt=build_prompt(args),
        formats=normalize_formats(args.formats),
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad=args.vad,
        vad_min_silence_ms=args.vad_min_silence_ms,
        temperature=args.temperature,
        cpu_threads=args.cpu_threads,
        num_workers=args.num_workers,
        condition_on_previous_text=args.condition_on_previous_text,
        word_timestamps=args.word_timestamps,
        diarize=args.diarize,
        openai_timestamp_model=args.openai_timestamp_model,
        api_convert_format=args.api_convert_format,
        keep_temp=args.keep_temp,
        quiet=args.quiet,
    )


def main(argv: list[str] | None = None) -> int:
    configure_console()
    args = parse_args(argv)
    quiet = args.quiet
    config = build_config(args)
    formats = config.formats

    media_files = list_media_files(args.inputs, recursive=args.recursive)
    if not media_files:
        print("No encontre archivos de audio/video en las rutas indicadas.", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transcriber = Transcriber(config)
    if not quiet:
        print(
            f"Encontrados {len(media_files)} fichero(s). "
            f"Backend: {config.backend}. Modelo: {transcriber.model_name()}"
        )

    ok = 0
    failed = 0
    for index, media_path in enumerate(media_files, start=1):
        if not quiet:
            print(f"[{index}/{len(media_files)}] {media_path}")
        try:
            output_base = output_base_for(media_path, args.output_dir, media_files)
            pending = formats if args.overwrite else missing_formats(output_base, formats)
            if not pending:
                if not quiet:
                    print(f"  Saltado: ya existen todas las salidas de {output_base.name}.")
                continue

            transcript = transcriber.transcribe(media_path)
            written = write_outputs(transcript, output_base, pending)
            if not quiet:
                for path in written:
                    print(f"  OK -> {path}")
            ok += 1
        except KeyboardInterrupt:
            print("\nCancelado por el usuario.", file=sys.stderr)
            return 130
        except Exception as exc:  # noqa: BLE001 - CLI: queremos continuar con el resto de ficheros.
            failed += 1
            print(f"  ERROR: {exc}", file=sys.stderr)

    print(f"Terminado. Correctos: {ok}. Errores: {failed}.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
