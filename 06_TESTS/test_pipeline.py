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


def test_pipeline_preserves_raw_distinct_from_refined_and_final_record(monkeypatch, tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"recording")
    raw_content = "Raw transcript words spoken by team."
    refined_content = "Refined transcript words spoken by team."
    record_content = {
        "meeting_title": "Project Meeting",
        "summary": "Team met and decided next steps.",
        "minutes": ["Reviewed status."],
        "decisions": [{"text": "Ship v1", "status": "Confirmed", "evidence_quote": "Ship v1"}],
        "action_items": [{"task": "Deploy", "owner": "Unspecified", "deadline": "Unspecified"}],
    }

    monkeypatch.setattr(
        main, "transcribe_audio", lambda path: {"text": raw_content, "segments": [{"id": "S1", "start": 0.0, "end": 2.0, "text": raw_content}]}
    )
    monkeypatch.setattr(
        main, "refine_transcription", lambda raw, terms: {"refined_text": refined_content, "refined_segments": [{"id": "S1", "start": 0.0, "end": 2.0, "text": refined_content}]}
    )
    monkeypatch.setattr(
        main, "generate_record", lambda refined: record_content
    )
    monkeypatch.setattr(
        main, "validate_record", lambda record, refined: record
    )
    monkeypatch.setattr(
        main, "save_outputs", lambda *args: (tmp_path / "run", tmp_path / "run.zip")
    )

    result = main.process_meeting(audio)

    # 1. Raw transcript retained
    assert result["raw_transcript"] == raw_content
    # 2. Refined transcript retained separately
    assert result["refined_transcript"] == refined_content
    # 3. Raw != Refined
    assert result["raw_transcript"] != result["refined_transcript"]
    # 4. Final record separate from transcripts
    assert result["record"] != result["raw_transcript"]
    assert result["record"] != result["refined_transcript"]
    assert result["record"]["summary"] != result["raw_transcript"]
    assert result["record"]["summary"] != result["refined_transcript"]


def test_documentation_merge_records_deduplication_and_defaults():
    from app.pipeline.documentation import merge_records

    chunk1 = {
        "meeting_title": "Meeting",
        "summary": "First part of the discussion.",
        "minutes": ["Discussed project scope."],
        "decisions": [
            {"text": "Deploy to staging first", "status": "Confirmed", "evidence_quote": "Deploy to staging first"}
        ],
        "non_decisions": [
            {"text": "Proposal to use GraphQL instead of REST", "reason": "Deferred", "evidence_quote": "Maybe consider GraphQL"}
        ],
        "action_items": [
            {"task": "Prepare staging environment", "owner": "Alice", "deadline": "Friday", "evidence_quote": "Alice prepare staging by Friday"}
        ],
        "discussion_points": ["Budget review"],
        "open_questions": ["Who will manage DNS?"],
    }

    chunk2 = {
        "meeting_title": "Quarterly Planning & Architecture",
        "summary": "Second part covering deployments.",
        "minutes": ["Discussed project scope.", "Reviewed infrastructure requirements."],
        "decisions": [
            {"text": "Deploy to staging first", "status": "Confirmed", "evidence_quote": "Deploy to staging first"},
            {"text": "Migrate database at midnight", "status": "Confirmed", "evidence_quote": "Migrate database at midnight"}
        ],
        "non_decisions": [
            {"text": "Proposal to use GraphQL instead of REST", "reason": "Deferred", "evidence_quote": "Maybe consider GraphQL"},
            {"text": "Consider migrating to AWS", "reason": "Not approved", "evidence_quote": "We could move to AWS"}
        ],
        "action_items": [
            {"task": "Prepare staging environment", "owner": "", "deadline": "", "evidence_quote": "Alice prepare staging by Friday"},
            {"task": "Draft rollback plan", "owner": None, "deadline": None, "evidence_quote": "Draft rollback plan"}
        ],
        "discussion_points": ["Budget review", "Database indexing"],
        "open_questions": ["Who will manage DNS?", "What is the expected downtime?"],
    }

    merged = merge_records([chunk1, chunk2])

    # Title is descriptive, not generic "Meeting"
    assert merged["meeting_title"] == "Quarterly Planning & Architecture"

    # Summary is concatenated
    assert "First part" in merged["summary"]
    assert "Second part" in merged["summary"]

    # Minutes deduplicated
    assert len(merged["minutes"]) == 2
    assert merged["minutes"] == ["Discussed project scope.", "Reviewed infrastructure requirements."]

    # Decisions deduplicated
    assert len(merged["decisions"]) == 2
    assert merged["decisions"][0]["text"] == "Deploy to staging first"
    assert merged["decisions"][1]["text"] == "Migrate database at midnight"

    # Non-decisions deduplicated and preserved distinct from decisions
    assert len(merged["non_decisions"]) == 2
    assert merged["non_decisions"][0]["text"] == "Proposal to use GraphQL instead of REST"
    assert merged["non_decisions"][1]["text"] == "Consider migrating to AWS"

    # Action items deduplicated, never invent owners/deadlines (default "Unspecified")
    assert len(merged["action_items"]) == 2
    assert merged["action_items"][0]["task"] == "Prepare staging environment"
    assert merged["action_items"][0]["owner"] == "Alice"
    assert merged["action_items"][0]["deadline"] == "Friday"
    assert merged["action_items"][1]["task"] == "Draft rollback plan"
    assert merged["action_items"][1]["owner"] == "Unspecified"
    assert merged["action_items"][1]["deadline"] == "Unspecified"

    # Discussion points and open questions deduplicated
    assert len(merged["discussion_points"]) == 2
    assert len(merged["open_questions"]) == 2


def test_documentation_generate_record_token_aware_chunking(monkeypatch, capsys):
    from app.pipeline import documentation

    # Generate a long refined transcript that requires multiple chunks
    long_transcript = "In this meeting we discussed architecture and scaling. " * 300

    chunks_called = []

    def fake_groq_json(system_prompt, user_prompt, model, max_tokens, retries):
        assert model == "openai/gpt-oss-20b"
        assert max_tokens == 2048
        chunks_called.append(user_prompt)
        return {
            "meeting_title": f"Chunk Meeting {len(chunks_called)}",
            "summary": f"Summary for chunk {len(chunks_called)}.",
            "minutes": [f"Minute {len(chunks_called)}"],
            "decisions": [{"text": f"Decision {len(chunks_called)}", "status": "Confirmed", "evidence_quote": "quote"}],
            "non_decisions": [],
            "action_items": [{"task": f"Action {len(chunks_called)}", "owner": "Unspecified", "deadline": "Unspecified", "evidence_quote": "quote"}],
            "discussion_points": [],
            "open_questions": [],
        }

    monkeypatch.setattr(documentation, "groq_json", fake_groq_json)

    record = documentation.generate_record({"refined_text": long_transcript})

    captured = capsys.readouterr()
    assert len(chunks_called) > 1
    assert f"[DOC] chunk=1/{len(chunks_called)} input_tokens=" in captured.out
    assert f"[DOC] chunk={len(chunks_called)}/{len(chunks_called)} input_tokens=" in captured.out
    assert len(record["decisions"]) == len(chunks_called)
    assert len(record["action_items"]) == len(chunks_called)


