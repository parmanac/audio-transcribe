from unittest.mock import Mock

import pytest
from docx import Document

from transcribe_audio import Segment, Transcript, Transcriber, transcribe_dir, write_outputs
from transcribe_audio.cli import main


@pytest.fixture
def recording():
    return Transcript(
        "entrevista.flac", "local", "modelo", "es", "Buenos días. ¿Qué tal?",
        [Segment(2, 4, "Buenos días.", speaker="A"), Segment(6, 8, "¿Qué tal?", speaker="B")],
    )


def test_default_exports_are_markdown_and_subtitles(tmp_path, recording):
    markdown, srt = write_outputs(recording, tmp_path / "entrevista")
    assert [markdown.suffix, srt.suffix] == [".md", ".srt"]
    assert recording.text in markdown.read_text(encoding="utf-8")
    assert "00:00:06,000 --> 00:00:08,000\nB: ¿Qué tal?" in srt.read_text(encoding="utf-8")


def test_optional_word_export_is_editable(tmp_path, recording):
    [word] = write_outputs(recording, tmp_path / "entrevista", ["docx"])
    document = Document(word)
    assert [p.text for p in document.paragraphs][-2:] == [
        "00:00:02  A: Buenos días.", "00:00:06  B: ¿Qué tal?"
    ]
    document.paragraphs[-1].add_run(" Muy bien.")
    document.save(word)
    assert Document(word).paragraphs[-1].text.endswith("Muy bien.")


@pytest.mark.parametrize("entry", ["cli", "api"])
def test_default_pair_can_resume_with_one_existing_file(tmp_path, monkeypatch, recording, entry):
    audio = tmp_path / "entrevista.flac"
    audio.touch()
    output = tmp_path / "salidas"
    output.mkdir()
    srt = output / "entrevista.srt"
    srt.write_bytes(b"subtitle edited by user")
    recognize = Mock(return_value=recording)
    monkeypatch.setattr(Transcriber, "transcribe", recognize)
    for _ in range(2):
        if entry == "cli":
            assert main([str(audio), "--device", "cpu", "-o", str(output)]) == 0
        else:
            transcribe_dir(tmp_path, output_dir=output)
    recognize.assert_called_once()
    assert srt.read_bytes() == b"subtitle edited by user"
    assert {p.suffix for p in output.iterdir()} == {".md", ".srt"}
    assert recording.text in (output / "entrevista.md").read_text(encoding="utf-8")


def test_silence_and_text_without_times(tmp_path):
    base = tmp_path / "audio"
    silence = Transcript("audio.wav", "local", "modelo", "es", "", [])
    srt, word = write_outputs(silence, base, ["srt", "docx"])
    assert srt.read_bytes() == b""
    assert "No se detectó voz" in Document(word).paragraphs[-1].text
    text = Transcript("audio.wav", "openai", "modelo", "es", "Línea uno.\nLínea dos.", [])
    write_outputs(text, base, ["docx"])
    assert [p.text for p in Document(word).paragraphs][-2:] == ["Línea uno.", "Línea dos."]
    with pytest.raises(ValueError, match="tiempos"):
        write_outputs(text, base)


def test_failed_word_export_preserves_previous_subtitles(tmp_path, monkeypatch, recording):
    srt = tmp_path / "audio.srt"
    srt.write_bytes(b"previous subtitles")
    monkeypatch.setattr("transcribe_audio.formats.write_docx", Mock(side_effect=OSError("error")))
    with pytest.raises(OSError):
        write_outputs(recording, tmp_path / "audio", ["srt", "docx"])
    assert srt.read_bytes() == b"previous subtitles"
    assert list(tmp_path.iterdir()) == [srt]
