"""Exportación de transcripciones a subtítulos, documentos editables y texto."""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import asdict
from itertools import groupby
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator, Sequence

from .models import Segment, Transcript
from .text_utils import join_transcript_text, to_jsonable

DEFAULT_FORMATS = ("srt", "docx")
ALL_FORMATS = ["txt", "md", "json", "srt", "vtt", "docx"]
TIMESTAMP_FORMATS = {"srt", "vtt"}


def normalize_formats(values: Sequence[str]) -> list[str]:
    if "all" in values:
        return list(ALL_FORMATS)
    for value in values:
        if value not in ALL_FORMATS:
            raise ValueError(f"Formato de salida desconocido: {value}")
    return list(dict.fromkeys(values))


def needs_timestamps(formats: list[str], word_timestamps: bool) -> bool:
    return word_timestamps or any(fmt in TIMESTAMP_FORMATS for fmt in normalize_formats(formats))


def write_outputs(
    transcript: Transcript, output_base: Path, formats: Sequence[str] = DEFAULT_FORMATS
) -> list[Path]:
    """Guarda los formatos solicitados con la misma base de nombre. Devuelve sus rutas."""
    formats = normalize_formats(formats)
    if not transcript.segments and transcript.text.strip() and TIMESTAMP_FORMATS.intersection(formats):
        raise ValueError("El reconocimiento devolvió texto sin tiempos. Los subtítulos requieren segmentos temporales.")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.parent / f"{output_base.name}.{fmt}" for fmt in formats]
    # Preparar directamente en el destino conserva sus permisos heredados en Windows.
    with ExitStack() as cleanup:
        staged_paths: list[Path] = []
        for path in paths:
            with NamedTemporaryFile(dir=output_base.parent, prefix=".transcribe-", suffix=".tmp", delete=False) as tmp:
                staged = Path(tmp.name)
            cleanup.callback(staged.unlink, missing_ok=True)
            staged_paths.append(staged)
            if path.suffix == ".docx":
                write_docx(transcript, staged)
                continue
            if path.suffix == ".srt":
                text = render_srt(transcript.segments)
            elif path.suffix == ".vtt":
                text = render_vtt(transcript.segments)
            elif path.suffix == ".json":
                text = json.dumps(transcript_to_dict(transcript), ensure_ascii=False, indent=2)
            elif path.suffix == ".md":
                text = render_markdown(transcript)
            else:
                text = transcript.text.strip() + "\n"
            staged.write_text(text, encoding="utf-8")
        for staged, path in zip(staged_paths, paths):
            staged.replace(path)
    return paths


def transcript_to_dict(transcript: Transcript) -> dict[str, Any]:
    data = asdict(transcript)
    data["raw"] = to_jsonable(transcript.raw)
    return data


def timed_paragraphs(segments: list[Segment]) -> Iterator[tuple[str, str]]:
    groups = groupby(segments, key=lambda part: (int(part.start) // 60, part.speaker))
    for (_, speaker), parts in groups:
        paragraph = list(parts)
        stamp = format_srt_time(paragraph[0].start).split(",")[0]
        text = join_transcript_text(part.text for part in paragraph)
        yield stamp, f"{speaker}: {text}" if speaker else text


def render_markdown(transcript: Transcript) -> str:
    lines = [
        f"# {Path(transcript.source).name}",
        "",
        f"- Backend: {transcript.backend}",
        f"- Modelo: {transcript.model}",
        f"- Idioma: {transcript.language or 'auto'}",
        "",
        "## Transcripción",
        "",
    ]
    if len(transcript.segments) > 1 and transcript.segments[-1].end >= 120:
        for stamp, text in timed_paragraphs(transcript.segments):
            lines.extend([f"**{stamp}** {text}", ""])
    else:
        lines.extend([transcript.text.strip(), ""])
    return "\n".join(lines)


def write_docx(transcript: Transcript, path: Path) -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = section.right_margin = Cm(2.4)
    section.top_margin = section.bottom_margin = Cm(2.2)
    normal = document.styles["Normal"]
    normal.font.name, normal.font.size = "Calibri", Pt(11)
    normal.paragraph_format.line_spacing = 1.14
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.widow_control = True
    title = document.styles["Title"]
    title.font.size = Pt(25)
    title.font.color.rgb = RGBColor(0, 0, 0)
    title_properties = title.element.get_or_add_pPr()
    border = title_properties.find(qn("w:pBdr"))
    if border is not None:
        title_properties.remove(border)

    filename = Path(transcript.source).name
    document.core_properties.title = f"Transcripción de {filename}"
    document.core_properties.author = "audio-transcribe"
    document.add_paragraph("Transcripción", style="Title")
    source = document.add_paragraph(filename)
    source.paragraph_format.space_after = Pt(14)
    source.paragraph_format.keep_with_next = True
    if transcript.segments:
        for stamp, text in timed_paragraphs(transcript.segments):
            paragraph = document.add_paragraph()
            paragraph.add_run(stamp + "  ").bold = True
            paragraph.add_run(text)
    else:
        for line in transcript.text.strip().splitlines() or ["No se detectó voz en la grabación."]:
            document.add_paragraph(line)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    page = OxmlElement("w:fldSimple")
    page.set(qn("w:instr"), "PAGE")
    footer._p.append(page)
    document.save(path)


def render_srt(segments: list[Segment]) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        speaker = f"{segment.speaker}: " if segment.speaker else ""
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}",
                    f"{speaker}{segment.text.strip()}",
                ]
            )
        )
    return "\n\n".join(blocks) + "\n" if blocks else ""


def render_vtt(segments: list[Segment]) -> str:
    if not segments:
        return "WEBVTT\n\n"
    blocks = ["WEBVTT", ""]
    for segment in segments:
        speaker = f"{segment.speaker}: " if segment.speaker else ""
        blocks.append(
            "\n".join(
                [
                    f"{format_vtt_time(segment.start)} --> {format_vtt_time(segment.end)}",
                    f"{speaker}{segment.text.strip()}",
                ]
            )
        )
        blocks.append("")
    return "\n".join(blocks)


def format_srt_time(seconds: float) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_time(seconds: float) -> str:
    return format_srt_time(seconds).replace(",", ".")
