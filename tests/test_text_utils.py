from transcribe_audio.models import Segment
from transcribe_audio.text_utils import (
    clean_text,
    join_speaker_text,
    join_transcript_text,
    to_jsonable,
)


def test_clean_text_collapses_whitespace():
    assert clean_text("  hola\n  mundo \t ") == "hola mundo"
    assert clean_text(None) == ""


def test_join_transcript_text_fixes_punctuation_spacing():
    assert join_transcript_text(["Hola ", " mundo ."]) == "Hola mundo."
    assert join_transcript_text(["uno", "dos , tres"]) == "uno dos, tres"


def test_join_speaker_text():
    segments = [Segment(0.0, 1.0, "Hola", speaker="A"), Segment(1.0, 2.0, "Adios", speaker="B")]
    assert join_speaker_text(segments) == "A: Hola\nB: Adios"


def test_to_jsonable_handles_nested_and_objects():
    class Dummy:
        def __init__(self):
            self.a = 1
            self.b = [Segment(0.0, 1.0, "x")]

    out = to_jsonable(Dummy())
    assert out["a"] == 1
    assert out["b"][0]["text"] == "x"
    assert to_jsonable({"k": (1, 2)}) == {"k": [1, 2]}
