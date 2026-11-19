"""Estructuras de datos compartidas por los backends de transcripcion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[dict[str, Any]] | None = None


@dataclass
class Transcript:
    source: str
    backend: str
    model: str
    language: str | None
    text: str
    segments: list[Segment]
    raw: Any | None = None
