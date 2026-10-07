import wave
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import main
from app.core import utils
from app.pipeline import transcription


def create_valid_test_wav(path: Path, duration: float = 1.0, silent: bool = False) -> Path:
    import math
    import struct
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        if silent:
            wf.writeframes(b"\x00\x00" * num_samples)
        else:
            frames = bytearray()
            for i in range(num_samples):
                val = int(16000 * math.sin(2 * math.pi * 440.0 * i / sample_rate))
                frames.extend(struct.pack("<h", val))
            wf.writeframes(frames)
    return path


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
    audio = create_valid_test_wav(tmp_path / "meeting.wav", duration=1.0)
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
    path = create_valid_test_wav(tmp_path / "uploaded meeting audio.wav", duration=1.0)
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
    audio = create_valid_test_wav(tmp_path / "meeting.wav", duration=1.0)
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
    audio = create_valid_test_wav(tmp_path / "meeting.wav", duration=1.0)
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
        assert model == "openai/gpt-oss-120b"
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


def test_whisper_config_env_vars_and_cuda_detection(monkeypatch):
    from app.core import config

    # When CUDA is detected and no env vars set -> cuda + float16
    monkeypatch.setattr(config, "is_cuda_available", lambda: True)
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.delenv("WHISPER_COMPUTE_TYPE", raising=False)
    model, device, compute_type = config.get_whisper_config()
    assert model == "small"
    assert device == "cuda"
    assert compute_type == "float16"

    # When CUDA is not detected and no env vars set -> cpu + int8
    monkeypatch.setattr(config, "is_cuda_available", lambda: False)
    model, device, compute_type = config.get_whisper_config()
    assert model == "small"
    assert device == "cpu"
    assert compute_type == "int8"

    # Explicit environment variable overrides
    monkeypatch.setenv("WHISPER_MODEL", "medium")
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "float32")
    model, device, compute_type = config.get_whisper_config()
    assert model == "medium"
    assert device == "cuda"
    assert compute_type == "float32"

    # Explicit CPU device defaults to int8 compute
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    monkeypatch.delenv("WHISPER_COMPUTE_TYPE", raising=False)
    model, device, compute_type = config.get_whisper_config()
    assert device == "cpu"
    assert compute_type == "int8"


def test_transcribe_audio_enables_vad_filter_and_passes_initial_prompt(tmp_path, monkeypatch):
    audio = create_valid_test_wav(tmp_path / "meeting.wav", duration=1.0)
    recorded_kwargs = {}

    class FakeWhisperModelWithKwargs:
        def transcribe(self, file_path, **kwargs):
            recorded_kwargs.update(kwargs)
            return iter(
                [SimpleNamespace(start=0.0, end=1.5, text=" Testing speech.")]
            ), SimpleNamespace(language="en")

    monkeypatch.setattr(
        transcription, "get_whisper_model", lambda: FakeWhisperModelWithKwargs()
    )

    # 1. Comma-separated string glossary with duplicates and extra spaces
    result = transcription.transcribe_audio(audio, glossary="Docker, Kubernetes,  Docker , Helm")
    assert recorded_kwargs.get("vad_filter") is True
    assert recorded_kwargs.get("initial_prompt") == "Docker, Kubernetes, Helm"
    assert recorded_kwargs.get("language") == "en"
    assert result["text"] == "Testing speech."
    assert result["chunked"] is False
    assert result["backend"] == "faster-whisper"

    # 2. List glossary
    recorded_kwargs.clear()
    transcription.transcribe_audio(audio, glossary=["Kafka", "Redis"])
    assert recorded_kwargs.get("vad_filter") is True
    assert recorded_kwargs.get("initial_prompt") == "Kafka, Redis"

    # 3. Direct initial_prompt argument takes precedence
    recorded_kwargs.clear()
    transcription.transcribe_audio(audio, glossary="Ignored", initial_prompt="Direct prompt")
    assert recorded_kwargs.get("vad_filter") is True
    assert recorded_kwargs.get("initial_prompt") == "Direct prompt"

    # 4. Empty/None glossary does not set initial_prompt
    recorded_kwargs.clear()
    transcription.transcribe_audio(audio, glossary="")
    assert recorded_kwargs.get("vad_filter") is True
    assert "initial_prompt" not in recorded_kwargs


def test_transcribe_audio_uses_configured_model_and_device(tmp_path, monkeypatch):
    from app.core import config

    constructed = []

    class FakeWhisperModel:
        def __init__(self, model, device, compute_type):
            constructed.append((model, device, compute_type))

        def transcribe(self, file_path, **kwargs):
            return iter(
                [SimpleNamespace(start=0.0, end=1.0, text=" Test.")]
            ), SimpleNamespace(language="en")

    monkeypatch.setattr(config, "WHISPER_MODEL", "large-v3")
    monkeypatch.setattr(config, "WHISPER_DEVICE", "cuda")
    monkeypatch.setattr(config, "WHISPER_COMPUTE_TYPE", "float16")
    monkeypatch.setattr(transcription, "WhisperModel", FakeWhisperModel)

    transcription.get_whisper_model.cache_clear()
    try:
        model = transcription.get_whisper_model()
        assert constructed == [("large-v3", "cuda", "float16")]
    finally:
        transcription.get_whisper_model.cache_clear()


def test_process_meeting_passes_glossary_to_transcribe(tmp_path, monkeypatch):
    audio = create_valid_test_wav(tmp_path / "meeting.wav", duration=1.0)
    received_glossary = []
    raw = {"text": "Raw transcript", "segments": []}
    refined = {"refined_text": "Refined transcript", "refined_segments": []}
    record = {"meeting_title": "Meeting", "decisions": [], "action_items": []}

    monkeypatch.setattr(
        main,
        "transcribe_audio",
        lambda path, glossary=None: received_glossary.append(glossary) or raw,
    )
    monkeypatch.setattr(
        main,
        "refine_transcription",
        lambda raw, terms: refined,
    )
    monkeypatch.setattr(
        main,
        "generate_record",
        lambda ref: record,
    )
    monkeypatch.setattr(
        main,
        "validate_record",
        lambda rec, ref: rec,
    )
    monkeypatch.setattr(
        main,
        "save_outputs",
        lambda *args: (tmp_path / "run", tmp_path / "run.zip"),
    )

    main.process_meeting(audio, glossary="PostgreSQL, Celery")
    assert received_glossary == ["PostgreSQL, Celery"]


def test_audio_validation_corrupt_audio_with_valid_extension(tmp_path):
    path = tmp_path / "corrupt_recording.wav"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt corrupt header audio bytes")
    with pytest.raises(ValueError, match="corrupt or cannot be decoded"):
        transcription.validate_audio_file(path)


def test_audio_validation_empty_audio_rejected(tmp_path):
    path = tmp_path / "zero_bytes.wav"
    path.touch()
    with pytest.raises(ValueError, match="empty"):
        transcription.validate_audio_file(path)


def test_audio_validation_valid_audio_accepted(tmp_path):
    path = create_valid_test_wav(tmp_path / "clean_valid.wav", duration=1.0)
    result = transcription.validate_audio_file(path)
    assert result == path


def test_audio_validation_silent_audio_rejected(tmp_path):
    path = create_valid_test_wav(tmp_path / "silent_audio.wav", duration=1.0, silent=True)
    with pytest.raises(ValueError, match="completely silent or near-silent"):
        transcription.validate_audio_file(path)


def test_audio_validation_duration_boundary_checks(tmp_path, monkeypatch):
    from app.core import config

    short_path = create_valid_test_wav(tmp_path / "short_boundary.wav", duration=0.2)
    monkeypatch.setattr(config, "AUDIO_MIN_DURATION_SECONDS", 1.0)
    with pytest.raises(ValueError, match="below the minimum allowed duration"):
        transcription.validate_audio_file(short_path)

    long_path = create_valid_test_wav(tmp_path / "long_boundary.wav", duration=1.0)
    monkeypatch.setattr(config, "AUDIO_MAX_DURATION_SECONDS", 0.5)
    with pytest.raises(ValueError, match="exceeds the maximum allowed duration"):
        transcription.validate_audio_file(long_path)


def test_audio_validation_suspicious_file_size_warning(tmp_path, caplog, monkeypatch):
    import logging
    from app.core import config

    # Configure minimum threshold above small_path size (which is ~19KB for 0.6s 16kHz wav)
    monkeypatch.setattr(config, "AUDIO_SUSPICIOUS_MIN_SIZE_BYTES", 50000)
    small_path = create_valid_test_wav(tmp_path / "small_recording.wav", duration=0.6)
    with caplog.at_level(logging.WARNING):
        result = transcription.validate_audio_file(small_path)
    assert result == small_path
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("suspiciously small" in w for w in warnings)


def test_full_pipeline_end_to_end_run(tmp_path, monkeypatch):
    audio_path = create_valid_test_wav(tmp_path / "meeting_recording.wav", duration=1.0)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    monkeypatch.setattr("app.core.utils.OUTPUT_DIR", out_dir)

    raw_fixture = {
        "text": "Alice: We agreed to launch the API on Friday. Bob will deploy the microservices.",
        "segments": [
            {"id": "S0001", "start": 0.0, "end": 4.0, "text": "Alice: We agreed to launch the API on Friday."},
            {"id": "S0002", "start": 4.5, "end": 7.5, "text": "Bob will deploy the microservices."},
        ],
        "backend": "faster-whisper",
        "language": "en",
    }
    refined_fixture = {
        "refined_text": "Alice: We agreed to launch the API on Friday. Bob will deploy the microservices.",
        "refined_segments": [
            {"id": "S0001", "start": 0.0, "end": 4.0, "text": "Alice: We agreed to launch the API on Friday."},
            {"id": "S0002", "start": 4.5, "end": 7.5, "text": "Bob will deploy the microservices."},
        ],
        "changes": [],
    }
    record_fixture = {
        "meeting_title": "Sprint Launch Planning",
        "summary": "Team agreed to launch the API on Friday.",
        "minutes": ["Launch agreed for Friday.", "Bob assigned microservices deployment."],
        "decisions": [
            {
                "text": "Launch the API on Friday",
                "status": "Confirmed",
                "evidence_quote": "We agreed to launch the API on Friday.",
            }
        ],
        "non_decisions": [],
        "action_items": [
            {
                "task": "Deploy the microservices",
                "owner": "Bob",
                "deadline": "Unspecified",
                "status": "Confirmed",
                "evidence_quote": "Bob will deploy the microservices.",
            }
        ],
        "discussion_points": ["API launch readiness"],
        "open_questions": [],
    }

    monkeypatch.setattr(main, "transcribe_audio", lambda *args, **kwargs: raw_fixture)
    monkeypatch.setattr(main, "refine_transcription", lambda *args, **kwargs: refined_fixture)
    monkeypatch.setattr(main, "generate_record", lambda *args, **kwargs: record_fixture)

    recorded_stages = []
    recorded_partials = []

    def on_stage(stage: str):
        recorded_stages.append(stage)

    def on_partial(data: dict):
        recorded_partials.append(data)

    result = main.process_meeting(
        audio_path=audio_path,
        glossary="API, microservices",
        stage_callback=on_stage,
        partial_result_callback=on_partial,
    )

    expected_stages = [
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
    assert recorded_stages == expected_stages
    assert len(recorded_partials) == 2
    assert "raw_transcript" in recorded_partials[0]
    assert "refined_transcript" in recorded_partials[1]

    # Check return structure
    assert result["raw_transcript"] == raw_fixture["text"]
    assert result["refined_transcript"] == refined_fixture["refined_text"]
    assert result["record"]["meeting_title"] == "Sprint Launch Planning"
    assert len(result["record"]["decisions"]) == 1
    assert result["record"]["decisions"][0]["status"] == "Confirmed"
    assert result["record"]["decisions"][0]["evidence"]["found"] is True
    assert len(result["record"]["action_items"]) == 1
    assert result["record"]["action_items"][0]["owner"] == "Bob"
    assert result["record"]["action_items"][0]["status"] == "Confirmed"

    # Check disk outputs and archive
    run_output_dir = Path(result["output_dir"])
    archive_path = Path(result["archive_path"])
    assert run_output_dir.is_dir()
    assert (run_output_dir / "meeting_record.json").exists()
    assert (run_output_dir / "meeting_record.md").exists()
    assert (run_output_dir / "action_items.json").exists()
    assert (run_output_dir / "decisions.json").exists()
    assert archive_path.is_file()
    assert zipfile.is_zipfile(archive_path)


def test_pipeline_rejects_missing_or_empty_audio_input(tmp_path):
    with pytest.raises(ValueError, match="Please upload a meeting recording"):
        main.process_meeting("")

    missing_path = tmp_path / "nonexistent.wav"
    with pytest.raises(ValueError, match="missing or is not a file"):
        main.process_meeting(missing_path)


def test_merge_records_near_duplicate_decisions():
    from app.pipeline.documentation import merge_records

    chunk1 = {
        "decisions": [
            {
                "text": "Launch API on Friday",
                "status": "Confirmed",
                "evidence_quote": "We will launch API on Friday.",
            }
        ]
    }
    chunk2 = {
        "decisions": [
            # Near-duplicate of chunk1
            {
                "text": "Launch the new API on Friday",
                "status": "Confirmed",
                "evidence_quote": "We will launch the new API on Friday.",
            },
            # Genuinely different decision with different date
            {
                "text": "Launch the new API on Monday",
                "status": "Confirmed",
                "evidence_quote": "We will launch the new API on Monday.",
            },
        ]
    }

    merged = merge_records([chunk1, chunk2])
    # The near-duplicate was merged, and the different-date decision remained separate
    assert len(merged["decisions"]) == 2
    texts = [d["text"] for d in merged["decisions"]]
    assert any("Friday" in t for t in texts)
    assert any("Monday" in t for t in texts)


def test_merge_records_near_duplicate_actions():
    from app.pipeline.documentation import merge_records

    chunk1 = {
        "action_items": [
            {
                "task": "Deploy microservices",
                "owner": "Unspecified",
                "deadline": "Friday",
                "status": "Confirmed",
                "evidence_quote": "Deploy by Friday.",
            }
        ]
    }
    chunk2 = {
        "action_items": [
            # Near duplicate with more complete information (owner Bob)
            {
                "task": "Deploy the microservices",
                "owner": "Bob",
                "deadline": "Friday",
                "status": "Confirmed",
                "evidence_quote": "Bob will deploy the microservices by Friday.",
            },
            # Different action assigned to Alice
            {
                "task": "Deploy the microservices",
                "owner": "Alice",
                "deadline": "Friday",
                "status": "Confirmed",
                "evidence_quote": "Alice will deploy the microservices by Friday.",
            },
        ]
    }

    merged = merge_records([chunk1, chunk2])
    # Bob's item merged with chunk1's Unspecified item, while Alice's item remains separate
    assert len(merged["action_items"]) == 2
    owners = {a["owner"] for a in merged["action_items"]}
    assert owners == {"Bob", "Alice"}


def test_merge_records_splits_semicolon_minutes():
    from app.pipeline.documentation import merge_records

    chunk = {
        "minutes": [
            "Reviewed latency metrics; approved Q4 database migration; finalized security review"
        ]
    }
    merged = merge_records([chunk])
    assert len(merged["minutes"]) == 3
    assert merged["minutes"][0] == "Reviewed latency metrics"
    assert merged["minutes"][1] == "approved Q4 database migration"
    assert merged["minutes"][2] == "finalized security review"


def test_cross_chunk_context_recovery_in_generate_record(monkeypatch):
    from app.pipeline import documentation

    # Mock groq_json to inspect user prompts and return mock records
    recorded_prompts = []

    def fake_groq_json(system_prompt, user_prompt, model, **kwargs):
        recorded_prompts.append(user_prompt)
        if "Part two" in user_prompt:
            return {
                "meeting_title": "Project Sync",
                "summary": "Team agreed to deploy.",
                "minutes": ["Deployment agreed."],
                "decisions": [{"text": "Deploy service", "status": "Confirmed", "evidence_quote": "Deploy service"}],
                "non_decisions": [],
                "action_items": [
                    {
                        "task": "Finalize database migration",
                        "owner": "Unspecified",
                        "deadline": "Friday",
                        "status": "Confirmed",
                        "evidence_quote": "I'll finalize it by Friday.",
                    }
                ],
                "discussion_points": [],
                "open_questions": [],
            }
        return {
            "meeting_title": "Project Sync",
            "summary": "First part of discussion.",
            "minutes": ["Opening points."],
            "decisions": [],
            "non_decisions": [],
            "action_items": [],
            "discussion_points": [],
            "open_questions": [],
        }

    monkeypatch.setattr(documentation, "groq_json", fake_groq_json)

    # Force chunking by making fits_fn tight
    orig_split = documentation._split_text
    monkeypatch.setattr(
        documentation,
        "_split_text",
        lambda text, sp, mp, fits_fn=None: [
            "Part one discussion: Alice asked who will handle database migration.",
            "Part two discussion: Speaker said I'll finalize it by Friday.",
        ],
    )

    refined = {"refined_text": "Part one... Part two..."}
    record = documentation.generate_record(refined)

    # Chunk 2 prompt received preceding dialogue context from chunk 1
    assert len(recorded_prompts) == 2
    assert "Preceding dialogue context" in recorded_prompts[1]
    assert "Alice asked who will handle database migration" in recorded_prompts[1]
    # Final record recovered the action item with context
    assert len(record["action_items"]) == 1
    assert record["action_items"][0]["deadline"] == "Friday"


def test_markdown_and_json_action_item_consistency():
    from app.core.utils import render_markdown

    record = {
        "meeting_title": "Sprint Review",
        "summary": "Team aligned on release.",
        "minutes": ["Reviewed features."],
        "decisions": [
            {
                "text": "Ship on Monday",
                "status": "Confirmed",
                "evidence_quote": "We will ship on Monday.",
            }
        ],
        "non_decisions": [],
        "action_items": [
            {
                "task": "Prepare deployment script",
                "owner": "Priya",
                "deadline": "Sunday",
                "status": "Confirmed",
                "evidence_quote": "Priya will prepare deployment script by Sunday.",
            },
            {
                "task": "Complete load testing",
                "owner": "Unspecified",
                "deadline": "Unspecified",
                "status": "Needs Review",
                "evidence_quote": "We still need to complete testing.",
            },
        ],
    }
    md = render_markdown(record)
    for action in record["action_items"]:
        # Verify all important fields from JSON are represented in Markdown
        assert action["task"] in md
        assert action["owner"] in md
        assert action["deadline"] in md
        assert action["status"] in md
        assert action["evidence_quote"] in md


def test_refinement_api_failure_fallback_continues_pipeline(tmp_path, monkeypatch):
    import wave
    from types import SimpleNamespace
    from pathlib import Path
    from app.pipeline import transcription, refinement, documentation

    wav = tmp_path / "valid.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x05" * 16000)

    class FakeWhisper:
        def transcribe(self, file_path, language, **kwargs):
            return iter([
                SimpleNamespace(start=0.0, end=1.0, text="Good morning."),
                SimpleNamespace(start=1.0, end=2.0, text="We still need to complete testing."),
            ]), SimpleNamespace(language="en")

    monkeypatch.setattr(transcription, "get_whisper_model", lambda: FakeWhisper())

    # Simulate Groq API failure (e.g. RateLimit / Timeout / 500) during refinement
    def failing_groq(*args, **kwargs):
        raise RuntimeError("Groq 429 Too Many Requests - rate limit exceeded")

    monkeypatch.setattr(refinement, "groq_json", failing_groq)

    # Documentation succeeds
    def mock_doc_groq(*args, **kwargs):
        return {
            "meeting_title": "Daily Standup",
            "summary": "Team discussed testing.",
            "minutes": ["Testing is pending."],
            "decisions": [],
            "non_decisions": [],
            "action_items": [
                {
                    "task": "Complete testing",
                    "owner": "Unspecified",
                    "deadline": "Unspecified",
                    "status": "Needs Review",
                    "evidence_quote": "We still need to complete testing.",
                }
            ],
            "discussion_points": [],
            "open_questions": [],
        }

    monkeypatch.setattr(documentation, "groq_json", mock_doc_groq)

    # The entire pipeline should continue and complete gracefully without raising
    result = main.process_meeting(wav)
    assert result["refinement_failed"] is True
    assert result.get("refinement_warning") is not None
    assert "Good morning" in result["raw_transcript"]
    assert "complete testing" in result["refined_transcript"]
    assert len(result["record"]["action_items"]) == 1
    assert Path(result["archive_path"]).exists()
