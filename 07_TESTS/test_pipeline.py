import wave
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import main
from app.core import utils
from app.pipeline import transcription


def test_unsupported_audio_file_is_rejected(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not audio")
    with pytest.raises(ValueError, match="Unsupported audio file type"):
        transcription.validate_audio_file(path)


def test_empty_audio_file_is_rejected(tmp_path):
    path = tmp_path / "empty.wav"
    path.touch()
    with pytest.raises(ValueError, match="empty"):
        transcription.validate_audio_file(path)


def test_faster_whisper_preserves_timestamps_and_reuses_model(tmp_path, monkeypatch):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio")
    constructed = []

    class FakeWhisperModel:
        def __init__(self, model, device, compute_type):
            constructed.append((model, device, compute_type))

        def transcribe(self, file_path, language):
            assert file_path == str(audio)
            assert language == "en"
            return iter(
                [
                    SimpleNamespace(start=0.37, end=1.82, text=" Hello"),
                    SimpleNamespace(start=2.14, end=3.91, text=" meeting."),
                ]
            ), SimpleNamespace(language="en")

    transcription.get_whisper_model.cache_clear()
    monkeypatch.setattr(transcription, "WhisperModel", FakeWhisperModel)
    try:
        first = transcription.transcribe_audio(audio)
        second = transcription.transcribe_audio(audio)
    finally:
        transcription.get_whisper_model.cache_clear()

    assert constructed == [("small", "cpu", "int8")]
    assert first["text"] == "Hello meeting."
    assert first["text"] == second["text"]
    assert first["segments"] == [
        {"id": "S0001", "start": 0.37, "end": 1.82, "text": "Hello"},
        {"id": "S0002", "start": 2.14, "end": 3.91, "text": "meeting."},
    ]
    assert first["backend"] == "faster-whisper"
    assert first["language"] == "en"


def test_audio_decoder_accepts_pyav19_compatible_open_call(tmp_path):
    from faster_whisper.audio import decode_audio

    path = tmp_path / "short audio sample.wav"
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(b"\0\0" * 1600)

    decoded = decode_audio(str(path))
    assert decoded.size > 0


def test_transcribe_audio_accepts_real_wav_path_with_spaces(tmp_path, monkeypatch):
    path = tmp_path / "uploaded meeting audio.wav"
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(b"\0\0" * 1600)
    received_paths = []

    class FakeWhisperModel:
        def transcribe(self, file_path, language):
            received_paths.append(file_path)
            return iter(
                [SimpleNamespace(start=0.0, end=0.1, text=" Local transcript.")]
            ), SimpleNamespace(language=language)

    monkeypatch.setattr(transcription, "get_whisper_model", lambda: FakeWhisperModel())
    result = transcription.transcribe_audio(path)

    assert received_paths == [str(path)]
    assert result["text"] == "Local transcript."
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][0]["end"] == 0.1


def test_unreadable_audio_file_reports_error(tmp_path, monkeypatch):
    path = tmp_path / "meeting.wav"
    path.write_bytes(b"audio")
    original_open = Path.open

    def denied_open(self, *args, **kwargs):
        if self == path:
            raise PermissionError("denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_open)
    with pytest.raises(ValueError, match="cannot be read"):
        transcription.validate_audio_file(path)


def test_pipeline_runs_ordered_stages_and_returns_download(monkeypatch, tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio")
    calls = []
    raw = {"text": "Raw", "segments": []}
    refined = {"refined_text": "Refined", "refined_segments": []}
    record = {"meeting_title": "Meeting", "decisions": [], "action_items": []}
    stages = []
    partial_results = []

    monkeypatch.setattr(
        main, "transcribe_audio", lambda path: calls.append("transcribe") or raw
    )
    monkeypatch.setattr(
        main,
        "refine_transcription",
        lambda transcript, terms: calls.append("refine") or refined,
    )
    monkeypatch.setattr(
        main,
        "generate_record",
        lambda result: calls.append("document") or record,
    )
    monkeypatch.setattr(
        main, "validate_record", lambda result, transcript: result
    )
    monkeypatch.setattr(
        main,
        "save_outputs",
        lambda *args: (tmp_path / "run", tmp_path / "run.zip"),
    )

    result = main.process_meeting(
        audio,
        "API, CUDA",
        stage_callback=stages.append,
        partial_result_callback=partial_results.append,
    )
    assert calls == ["transcribe", "refine", "document"]
    assert stages == [
        "received",
        "validated",
        "transcribing",
        "raw_ready",
        "refining",
        "refined",
        "documenting",
        "validating",
        "exporting",
        "complete",
    ]
    assert result["raw_transcript"] == "Raw"
    assert result["refined_transcript"] == "Refined"
    assert result["archive_path"] == tmp_path / "run.zip"
    assert partial_results[0] == {"raw_transcript": "Raw", "raw_segments": []}
    assert partial_results[1] == {
        "raw_transcript": "Raw",
        "raw_segments": [],
        "refined_transcript": "Refined",
        "refined_segments": [],
    }


def test_pipeline_rejects_missing_audio_before_model_calls():
    with pytest.raises(ValueError, match="upload a meeting recording"):
        main.process_meeting("")


def test_saved_results_include_all_submission_exports(monkeypatch, tmp_path):
    output_dir = tmp_path / "generated"
    monkeypatch.setattr(utils, "OUTPUT_DIR", output_dir)
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"recording")
    transcription_result = {"text": "Unchanged raw transcript"}
    refined = {
        "refined_text": "Refined transcript",
        "glossary": [],
        "proposed_edits": [],
        "applied_edits": [],
        "rejected_edits": [],
    }
    record = {
        "meeting_title": "Test meeting",
        "summary": "A test.",
        "minutes": ["Discussed testing."],
        "decisions": [],
        "non_decisions": [],
        "action_items": [],
        "discussion_points": [],
        "open_questions": [],
    }

    run_dir, archive_path = utils.save_outputs(
        transcription_result, refined, record, audio
    )
    required = {
        "raw_transcript.txt",
        "refined_transcript.txt",
        "minutes.md",
        "decisions.json",
        "action_items.json",
        "meeting_record.json",
        "meeting_record.md",
        "meeting.wav",
    }
    assert required.issubset({path.name for path in run_dir.iterdir()})
    with zipfile.ZipFile(archive_path) as archive:
        assert required.issubset(set(archive.namelist()))
    assert (run_dir / "raw_transcript.txt").read_text() == "Unchanged raw transcript"
