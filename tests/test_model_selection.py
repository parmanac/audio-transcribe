import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from transcribe_audio import Transcriber, TranscriberConfig
from transcribe_audio.cli import build_config, parse_args


@pytest.fixture
def create_transcription(monkeypatch):
    client = Mock()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: client))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    return client.audio.transcriptions.create


@pytest.mark.parametrize(
    ("options", "model", "response_format", "granularities"),
    [
        ([], "whisper-1", "verbose_json", ["segment"]),
        (["--formats", "docx"], "gpt-transcribe", "json", None),
        (["--formats", "json"], "gpt-transcribe", "json", None),
        (["--model", "gpt-4o-mini-transcribe", "--formats", "txt"], "gpt-4o-mini-transcribe", "json", None),
        (["--formats", "srt"], "whisper-1", "verbose_json", ["segment"]),
        (["--word-timestamps"], "whisper-1", "verbose_json", ["segment", "word"]),
        (["--diarize", "--formats", "srt"], "gpt-4o-transcribe-diarize", "diarized_json", None),
    ],
)
def test_openai_model_for_requested_output(
    tmp_path, create_transcription, options, model, response_format, granularities
):
    audio = tmp_path / "muestra.ogg"
    audio.write_bytes(b"test")
    create_transcription.return_value = {"text": "Hola", "languages": [{"code": "es"}]}

    config = build_config(parse_args([str(audio), "--backend", "openai", *options]))
    transcriber = Transcriber(config)
    result = transcriber.transcribe(audio)

    request = create_transcription.call_args.kwargs
    assert result.model == request["model"] == transcriber.model_name() == model
    assert request.get("response_format", "json") == response_format
    assert request.get("timestamp_granularities") == granularities
    if model == "gpt-transcribe":
        assert request["extra_body"] == {"languages": ["es"]}
        assert "language" not in request
    else:
        assert request["language"] == "es"
    if config.diarize:
        assert request["chunking_strategy"] == "auto"
        assert "prompt" not in request


def test_openai_autodetection_without_spanish_prompt(tmp_path, create_transcription):
    audio = tmp_path / "muestra.ogg"
    audio.touch()
    create_transcription.return_value = {"text": "Bonjour", "languages": [{"code": "fr"}]}
    config = build_config(parse_args([str(audio), "--backend", "openai", "--language", "auto", "--formats", "docx"]))
    assert Transcriber(config).transcribe(audio).language == "fr"
    assert not {"language", "extra_body", "prompt"}.intersection(create_transcription.call_args.kwargs)
    hint = TranscriberConfig(backend="openai", language="es", formats=["docx"])
    assert Transcriber(hint).transcribe(audio).language == "fr"


def test_local_defaults_keep_word_times_and_accept_custom_context(monkeypatch):
    transcribe = Mock()
    monkeypatch.setattr("transcribe_audio.api.transcribe_local", transcribe)
    monkeypatch.setattr("transcribe_audio.api.load_local_model", Mock())
    config = build_config(parse_args(["clase.mp4", "--device", "cpu", "--model", "large-v3-turbo"]))
    Transcriber(config).transcribe("clase.mp4")
    assert transcribe.call_args.kwargs["model_name"] == "large-v3-turbo"
    assert transcribe.call_args.kwargs["word_timestamps"] is True
    assert transcribe.call_args.kwargs["prompt"] is None

    custom = TranscriberConfig(device="cpu", prompt="Zaragoza", word_timestamps=False)
    Transcriber(custom).transcribe("clase.mp4")
    assert transcribe.call_args.kwargs["word_timestamps"] is False
    assert transcribe.call_args.kwargs["prompt"] == "Zaragoza"
