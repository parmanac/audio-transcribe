from transcribe_audio.formats import (
    format_srt_time,
    format_vtt_time,
    needs_timestamps,
    normalize_formats,
    render_markdown,
    render_srt,
    render_vtt,
)
from transcribe_audio.models import Segment, Transcript


def test_format_srt_time_basic():
    assert format_srt_time(0.0) == "00:00:00,000"
    assert format_srt_time(6.77) == "00:00:06,770"
    assert format_srt_time(3661.5) == "01:01:01,500"


def test_format_srt_time_rounding_does_not_overflow():
    # 6.9996 s redondea a 7000 ms: debe ser 00:00:07,000, no ...,1000.
    assert format_srt_time(6.9996) == "00:00:07,000"
    assert format_srt_time(59.9999) == "00:01:00,000"


def test_format_srt_time_clamps_negative():
    assert format_srt_time(-5.0) == "00:00:00,000"


def test_format_vtt_time_uses_dot():
    assert format_vtt_time(6.77) == "00:00:06.770"


def test_normalize_formats_expands_all_and_dedups():
    assert normalize_formats(["all"]) == ["txt", "md", "json", "srt", "vtt", "docx"]
    assert normalize_formats(["txt", "txt", "srt"]) == ["txt", "srt"]


def test_needs_timestamps():
    assert needs_timestamps(["srt"], False) is True
    assert needs_timestamps(["vtt"], False) is True
    assert needs_timestamps(["all"], False) is True
    assert needs_timestamps(["txt"], True) is True
    assert needs_timestamps(["txt", "md"], False) is False


def test_render_srt_and_vtt():
    segments = [Segment(start=0.0, end=1.5, text="Hola"), Segment(start=1.5, end=3.0, text="mundo")]
    srt = render_srt(segments)
    assert "1\n00:00:00,000 --> 00:00:01,500\nHola" in srt
    vtt = render_vtt(segments)
    assert vtt.startswith("WEBVTT")
    assert "00:00:01.500 --> 00:00:03.000" in vtt


def test_render_srt_with_speaker():
    segments = [Segment(start=0.0, end=1.0, text="Hola", speaker="A")]
    assert "A: Hola" in render_srt(segments)


def test_long_markdown_has_timed_paragraphs_and_preserves_speakers():
    segments = [
        Segment(3, 8, "Buenos días.", speaker="A"),
        Segment(9, 13, "Empezamos.", speaker="A"),
        Segment(14, 20, "De acuerdo.", speaker="B"),
        Segment(124, 126, "Seguimos.", speaker="B"),
    ]
    transcript = Transcript("clase.mp4", "openai", "modelo", "es", "", segments)
    body = render_markdown(transcript).split("## Transcripción\n\n")[1]
    assert body.splitlines() == [
        "**00:00:03** A: Buenos días. Empezamos.", "",
        "**00:00:14** B: De acuerdo.", "",
        "**00:02:04** B: Seguimos.",
    ]
