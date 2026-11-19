"""Descubrimiento de ficheros de medios y nombres de salida."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Iterable

AUDIO_EXTENSIONS = {
    ".aac",
    ".ac3",
    ".aif",
    ".aiff",
    ".amr",
    ".ape",
    ".au",
    ".caf",
    ".flac",
    ".m4a",
    ".m4b",
    ".mka",
    ".mp2",
    ".mp3",
    ".mpga",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
    ".wv",
}

VIDEO_EXTENSIONS = {
    ".avi",
    ".flv",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ts",
    ".webm",
    ".wmv",
}

MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def is_media_file(path: Path) -> bool:
    if path.suffix.lower() in MEDIA_EXTENSIONS:
        return True
    if not path.is_file():
        return False
    # Una grabadora puede dejar archivos sin extensión o con una extensión propia.
    import av

    try:
        with path.open("rb") as stream, av.open(stream, options={"protocol_whitelist": "file"}) as media:
            return bool(media.streams.audio)
    except (OSError, ValueError, av.error.FFmpegError):
        return False


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def list_media_files(inputs: Iterable[Path], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        path = raw.expanduser()
        if path.is_file():
            if is_media_file(path):
                files.append(path.resolve())
            else:
                print(f"Aviso: no se ha reconocido una pista de audio en {path}", file=sys.stderr)
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            files.extend(p.resolve() for p in path.glob(pattern) if p.is_file() and is_media_file(p))
        else:
            print(f"Aviso: ruta no encontrada: {path}", file=sys.stderr)
    return sorted(dict.fromkeys(files))


def safe_filename(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return safe or "audio"


def output_base_for(media_path: Path, output_dir: Path, all_files: list[Path]) -> Path:
    stem = safe_filename(media_path.stem)
    names = [safe_filename(path.stem).casefold() for path in all_files]
    if names.count(stem.casefold()) > 1:
        digest = hashlib.sha1(str(media_path).encode("utf-8")).hexdigest()[:8]
        stem = f"{stem}_{digest}"
    return output_dir / stem


def outputs_exist(output_base: Path, formats: list[str]) -> bool:
    return len(missing_formats(output_base, formats)) < len(formats)


def missing_formats(output_base: Path, formats: list[str]) -> list[str]:
    return [fmt for fmt in formats if not (output_base.parent / f"{output_base.name}.{fmt}").is_file()]
