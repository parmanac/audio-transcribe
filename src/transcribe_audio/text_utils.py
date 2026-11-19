"""Utilidades de texto y serializacion."""

from __future__ import annotations

import re
from typing import Any, Iterable

from .models import Segment


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def join_transcript_text(parts: Iterable[str]) -> str:
    text = " ".join(part.strip() for part in parts if part.strip())
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


def join_speaker_text(segments: list[Segment]) -> str:
    lines: list[str] = []
    for segment in segments:
        prefix = f"{segment.speaker}: " if segment.speaker else ""
        lines.append(f"{prefix}{segment.text.strip()}")
    return "\n".join(line for line in lines if line.strip()).strip()


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    return str(value)
