import wave
from pathlib import Path

from transcribe_audio.discovery import (
    is_media_file,
    is_video_file,
    list_media_files,
    output_base_for,
    outputs_exist,
    safe_filename,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_is_media_and_video():
    assert is_media_file(Path("a.ogg"))
    assert is_media_file(Path("a.MP4"))
    assert is_video_file(Path("a.mkv"))
    assert not is_video_file(Path("a.ogg"))
    assert not is_media_file(Path("a.txt"))


def test_list_media_files_filters_and_sorts(tmp_path):
    _touch(tmp_path / "b.ogg")
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "notes.txt")
    _touch(tmp_path / "sub" / "c.opus")

    flat = list_media_files([tmp_path], recursive=False)
    assert [p.name for p in flat] == ["a.mp4", "b.ogg"]

    deep = list_media_files([tmp_path], recursive=True)
    assert {p.name for p in deep} == {"a.mp4", "b.ogg", "c.opus"}


def test_discovers_audio_by_content_in_files_and_folders(tmp_path):
    audio = tmp_path / "grabacion"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", ""))
        stream.writeframes(b"\x00\x00" * 320)
    (tmp_path / "datos.bin").write_bytes(b"unrelated data")
    assert list_media_files([audio], recursive=False) == [audio.resolve()]
    assert list_media_files([tmp_path], recursive=True) == [audio.resolve()]


def test_safe_filename():
    assert safe_filename('a:b/c?d') == "a_b_c_d"
    assert safe_filename("   ") == "audio"


def test_output_base_for_disambiguates_collisions(tmp_path):
    f1 = tmp_path / "nota.ogg"
    f2 = tmp_path / "nota.opus"
    base1 = output_base_for(f1, tmp_path / "out", [f1, f2])
    base2 = output_base_for(f2, tmp_path / "out", [f1, f2])
    assert base1 != base2
    assert base1.name.startswith("nota_")
    upper = tmp_path / "NOTA.flac"
    assert output_base_for(upper, tmp_path / "out", [f1, upper]).name.startswith("NOTA_")


def test_outputs_exist(tmp_path):
    base = tmp_path / "out" / "nota"
    assert not outputs_exist(base, ["txt"])
    _touch(base.parent / "nota.txt")
    assert outputs_exist(base, ["txt", "srt"])
