import json

from app.core.config import GROQ_REFINEMENT_MODEL
from app.pipeline import refinement


def test_long_transcript_refinement_chunks_preserve_order_and_all_source_text(
    monkeypatch,
):
    monkeypatch.setattr(
        refinement,
        "load_prompt",
        lambda name: f"system:{name}",
    )
    glossary_chunks = []
    refinement_batches = []

    def fake_groq_json(system_prompt, user_prompt, model, max_tokens=1800):
        assert model == GROQ_REFINEMENT_MODEL
        assert refinement._fits_prompt(system_prompt, user_prompt)
        if "glossary_prompt.txt" in system_prompt:
            glossary_chunks.append(user_prompt.removeprefix("Extract terminology:\n"))
            return {"terms": ["Kubernetes"]}
        payload = json.loads(user_prompt.split("\nSEGMENTS:\n", 1)[1])
        refinement_batches.extend(payload)
        return {"edits": []}

    monkeypatch.setattr(refinement, "groq_json", fake_groq_json)

    segments = [
        {
            "id": f"S{index:04d}",
            "start": float(index),
            "end": float(index + 1),
            "text": f"segment {index} " + "technical wording " * 80,
        }
        for index in range(12)
    ]
    segments.insert(
        12,
        {
            "id": "S-LONG",
            "start": 24.0,
            "end": 13.0,
            "text": "verylongsegment " * 3000,
        },
    )
    transcript_text = " ".join(segment["text"] for segment in segments)

    result = refinement.refine_transcription(
        {"text": transcript_text, "segments": segments}
    )

    assert len(glossary_chunks) > 1
    assert "".join(glossary_chunks) == transcript_text
    reconstructed: dict[str, str] = {}
    order: list[str] = []
    for segment in refinement_batches:
        segment_id = segment["id"]
        if segment_id not in reconstructed:
            reconstructed[segment_id] = ""
            order.append(segment_id)
        reconstructed[segment_id] += segment["text"]

    assert order == [segment["id"] for segment in segments]
    assert reconstructed == {segment["id"]: segment["text"] for segment in segments}
    assert result["refined_segments"] == segments
    assert result["refined_text"] == transcript_text
    assert result["glossary"] == ["Kubernetes"]


def test_refinement_keeps_raw_input_immutable_and_applies_only_safe_segment_edits(
    monkeypatch,
):
    monkeypatch.setattr(
        refinement, "load_prompt", lambda name: f"system:{name}"
    )
    original_segments = [
        {
            "id": "S0001",
            "start": 0.0,
            "end": 4.0,
            "text": "Rahul said teh team will not deploy for ₹2.5 lakh.",
        }
    ]
    transcription = {
        "text": original_segments[0]["text"],
        "segments": original_segments,
    }

    def fake_groq_json(system_prompt, user_prompt, model, max_tokens=1800):
        if "glossary_prompt.txt" in system_prompt:
            return {"terms": ["Rahul"]}
        return {
            "edits": [
                {
                    "segment_id": "S0001",
                    "from": "teh",
                    "to": "the",
                    "confidence": 0.99,
                },
                {
                    "segment_id": "S0001",
                    "from": "will not",
                    "to": "will",
                    "confidence": 0.99,
                },
            ]
        }

    monkeypatch.setattr(refinement, "groq_json", fake_groq_json)
    result = refinement.refine_transcription(transcription)

    assert transcription["text"] == "Rahul said teh team will not deploy for ₹2.5 lakh."
    assert result["refined_text"] == "Rahul said the team will not deploy for ₹2.5 lakh."
    assert "Rahul" in result["refined_text"]
    assert "will not" in result["refined_text"]
    assert "₹2.5 lakh" in result["refined_text"]
    assert len(result["applied_edits"]) == 1
    assert len(result["rejected_edits"]) == 1


def test_extract_glossary_valid_response(monkeypatch):
    monkeypatch.setattr(
        refinement,
        "groq_json",
        lambda *args, **kwargs: {"terms": ["PyTorch", "PostgreSQL", "PGvector"]},
    )
    terms = refinement.extract_glossary(
        "We are using PyTorch and PostgreSQL with PGvector.",
        user_glossary=["CUDA"],
    )
    assert terms == ["CUDA", "PyTorch", "PostgreSQL", "PGvector"]


def test_extract_glossary_empty_response(monkeypatch):
    monkeypatch.setattr(
        refinement,
        "groq_json",
        lambda *args, **kwargs: {"terms": []},
    )
    terms_with_user = refinement.extract_glossary(
        "Nothing special here.",
        user_glossary=["CustomTerm"],
    )
    assert terms_with_user == ["CustomTerm"]

    terms_without_user = refinement.extract_glossary(
        "Nothing special here.",
        user_glossary=None,
    )
    assert terms_without_user == []

    # Empty text short-circuits
    assert refinement.extract_glossary("") == []
    assert refinement.extract_glossary("   ") == []


def test_extract_glossary_malformed_response(monkeypatch, caplog):
    # Tests handling of non-list, None, or unexpected structure
    responses = [
        {"terms": None},
        {"not_terms": ["Something"]},
        {"terms": "a single string not a list"},
    ]
    for malformed in responses:
        monkeypatch.setattr(
            refinement,
            "groq_json",
            lambda *args, **kwargs: malformed,
        )
        terms = refinement.extract_glossary(
            "Some transcript text",
            user_glossary=["Fallback"],
        )
        assert terms == ["Fallback"]


def test_extract_glossary_api_json_validation_failure(monkeypatch, caplog):
    def failing_groq_json(*args, **kwargs):
        raise RuntimeError(
            "groq.BadRequestError: 400 code: json_validate_failed message: Failed to validate JSON. failed_generation: ''"
        )

    monkeypatch.setattr(refinement, "groq_json", failing_groq_json)

    # Must not raise exception, logs warning, returns user glossary
    terms = refinement.extract_glossary(
        "Transcript mentioning PyTorch",
        user_glossary=["PreConfigured"],
    )
    assert terms == ["PreConfigured"]
    assert any("Glossary extraction failed" in record.message for record in caplog.records)


def test_refinement_continues_when_glossary_extraction_fails(monkeypatch):
    def groq_with_failing_glossary(system_prompt, user_prompt, model, **kwargs):
        if user_prompt.startswith("Extract terminology:") or "glossary_prompt.txt" in system_prompt:
            raise RuntimeError(
                "groq.BadRequestError: 400 code: json_validate_failed message: Failed to validate JSON."
            )
        return {
            "edits": [
                {
                    "segment_id": "S0001",
                    "from": "teh",
                    "to": "the",
                    "confidence": 0.95,
                }
            ]
        }

    monkeypatch.setattr(refinement, "groq_json", groq_with_failing_glossary)

    transcription = {
        "text": "teh model was deployed.",
        "segments": [{"id": "S0001", "start": 0.0, "end": 2.0, "text": "teh model was deployed."}],
    }
    result = refinement.refine_transcription(transcription, user_glossary=["UserTerm"])

    # Refinement must succeed using the transcript, falling back safely to user terms
    assert result["refined_text"] == "the model was deployed."
    assert result["glossary"] == ["UserTerm"]
    assert len(result["applied_edits"]) == 1


def test_documentation_still_works_with_groq_json(monkeypatch):
    from app.pipeline import documentation

    calls = []

    def mock_groq_json(system_prompt, user_prompt, model, max_tokens=2048, **kwargs):
        calls.append((system_prompt, user_prompt, model, max_tokens))
        return {
            "meeting_title": "Project Meeting",
            "summary": "Meeting summary.",
            "minutes": ["Reviewed status."],
            "decisions": [
                {
                    "text": "Will not deploy Friday",
                    "status": "Confirmed",
                    "evidence_quote": "We will not deploy Friday.",
                }
            ],
            "non_decisions": [],
            "action_items": [
                {
                    "task": "Prepare report",
                    "owner": "Priya",
                    "deadline": "Monday",
                    "status": "Needs Review",
                    "evidence_quote": "Priya will prepare report by Monday.",
                }
            ],
            "discussion_points": ["Review"],
            "open_questions": [],
        }

    monkeypatch.setattr(documentation, "groq_json", mock_groq_json)

    record = documentation.generate_record({"refined_text": "We will not deploy Friday. Priya will prepare report by Monday."})

    assert len(calls) == 1
    assert record["meeting_title"] == "Project Meeting"
    assert record["decisions"][0]["status"] == "Confirmed"
    assert record["action_items"][0]["owner"] == "Priya"


def test_sanitize_exception_redacts_api_keys():
    from app.core.utils import sanitize_exception

    err1 = Exception("Error code: 400 - API key gsk_1234567890abcdefABCDEF is invalid")
    sanitized1 = sanitize_exception(err1)
    assert "gsk_1234567890abcdefABCDEF" not in sanitized1
    assert "[REDACTED_API_KEY]" in sanitized1

    err2 = Exception("Authorization: Bearer secret_token_xyz failed")
    sanitized2 = sanitize_exception(err2)
    assert "secret_token_xyz" not in sanitized2
    assert "[REDACTED_API_KEY]" in sanitized2


def test_groq_json_sets_reasoning_effort_low(monkeypatch):
    captured_kwargs = []

    class MockCompletion:
        class Choice:
            class Message:
                content = '{"status": "ok"}'
            message = Message()
        choices = [Choice()]

    class MockClient:
        class Chat:
            class Completions:
                def create(self, **kwargs):
                    captured_kwargs.append(kwargs)
                    return MockCompletion()
            completions = Completions()
        chat = Chat()

    monkeypatch.setattr(refinement, "get_groq_client", lambda: MockClient())

    result = refinement.groq_json(
        system_prompt="system",
        user_prompt="user",
        model="openai/gpt-oss-120b",
    )
    assert result == {"status": "ok"}
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0].get("reasoning_effort") == "low"


def test_groq_json_retries_on_json_validate_failed(monkeypatch):
    attempts = []

    class MockCompletion:
        class Choice:
            class Message:
                content = '{"status": "recovered"}'
            message = Message()
        choices = [Choice()]

    class MockClient:
        class Chat:
            class Completions:
                def create(self, **kwargs):
                    attempts.append(kwargs)
                    if len(attempts) == 1:
                        raise RuntimeError("Error code: 400 json_validate_failed")
                    return MockCompletion()
            completions = Completions()
        chat = Chat()

    monkeypatch.setattr(refinement, "get_groq_client", lambda: MockClient())
    monkeypatch.setattr("time.sleep", lambda s: None)

    result = refinement.groq_json(
        system_prompt="system",
        user_prompt="user",
        model="openai/gpt-oss-120b",
        retries=3,
    )
    assert result == {"status": "recovered"}
    assert len(attempts) == 2


def test_stage_logging_prints_expected_markers(capsys, monkeypatch):
    monkeypatch.setattr(
        refinement,
        "load_prompt",
        lambda name: f"system:{name}",
    )
    monkeypatch.setattr(
        refinement,
        "groq_json",
        lambda *args, **kwargs: {"terms": ["Tech"], "edits": []},
    )

    refinement.refine_transcription(
        {
            "text": "Hello world",
            "segments": [{"id": "S0001", "start": 0.0, "end": 1.0, "text": "Hello world"}],
        }
    )
    captured = capsys.readouterr()
    assert "[STAGE] glossary" in captured.out
    assert "[STAGE] refinement chunk 1/1" in captured.out


def test_refinement_logs_stage_error_and_raises(capsys, monkeypatch):
    monkeypatch.setattr(
        refinement,
        "load_prompt",
        lambda name: f"system:{name}",
    )
    monkeypatch.setattr(
        refinement,
        "extract_glossary",
        lambda *args, **kwargs: [],
    )

    def failing_groq(*args, **kwargs):
        raise RuntimeError("groq.BadRequestError: 400 json_validate_failed gsk_supersecretkey")

    monkeypatch.setattr(refinement, "groq_json", failing_groq)

    import pytest
    with pytest.raises(RuntimeError):
        refinement.refine_transcription(
            {
                "text": "Hello world",
                "segments": [{"id": "S0001", "start": 0.0, "end": 1.0, "text": "Hello world"}],
            }
        )
    captured = capsys.readouterr()
    assert "[STAGE] refinement chunk 1/1" in captured.out
    assert "[ERROR] stage=refinement chunk=1/1 model=" in captured.out
    assert "[ERROR] exception=" in captured.out
    assert "gsk_supersecretkey" not in captured.out
    assert "[REDACTED_API_KEY]" in captured.out


def test_token_count_uses_tiktoken_when_available(monkeypatch):
    called = []
    real_encoding = refinement._token_encoding()

    class SpyEncoding:
        def encode(self, text):
            called.append(text)
            return real_encoding.encode(text)

    refinement._token_encoding.cache_clear()
    monkeypatch.setattr(refinement, "_token_encoding", lambda: SpyEncoding())
    sample_text = "The quick brown fox jumps over the lazy dog."
    count = refinement._token_count(sample_text)
    assert len(called) == 1
    assert called[0] == sample_text
    assert count == len(real_encoding.encode(sample_text))
    # Also verify that with actual tiktoken, it matches true token length
    # rather than fallback len // 4
    # "The quick brown fox jumps over the lazy dog." has length 44 -> len // 4 = 11, but 10 tokens
    assert len(sample_text) // 4 != count


def test_token_count_uses_fallback_when_tiktoken_fails(monkeypatch, caplog):
    refinement._token_encoding.cache_clear()

    def failing_encoding():
        raise RuntimeError("network failure: unable to download tokenizer data")

    monkeypatch.setattr(refinement, "_token_encoding", failing_encoding)

    import logging
    with caplog.at_level(logging.WARNING):
        count = refinement._token_count("Testing fallback when tiktoken is unavailable.")

    # Does not crash, returns valid integer count
    assert isinstance(count, int)
    assert count > 0

    # Warning is logged
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("tiktoken tokenizer unavailable" in w for w in warnings)

    # Secrets are not leaked in log
    for r in caplog.records:
        assert "Testing fallback when tiktoken is unavailable." not in r.message


import pytest

@pytest.mark.parametrize(
    "text",
    [
        "",
        "a",
        "abc",
        "abcd",
        "12345678",
        "The quick brown fox jumps over the lazy dog.",
        "Short",
        "A somewhat longer sample transcript text meant to verify mathematical correctness of fallback.",
    ],
)
def test_token_count_fallback_returns_max_1_len_div_4(monkeypatch, text):
    refinement._token_encoding.cache_clear()
    monkeypatch.setattr(
        refinement,
        "_token_encoding",
        lambda: (_ for _ in ()).throw(RuntimeError("offline network")),
    )

    expected = max(1, len(text) // 4)
    actual = refinement._token_count(text)
    assert actual == expected


def test_long_transcript_chunking_works_when_tiktoken_unavailable(monkeypatch):
    refinement._token_encoding.cache_clear()
    monkeypatch.setattr(
        refinement,
        "_token_encoding",
        lambda: (_ for _ in ()).throw(RuntimeError("restricted network: no tokenizer data")),
    )

    long_transcript = "This is a recurring meeting status update covering deliverables and blockers. " * 200
    system_prompt = "You are an expert meeting transcription refinement system."
    make_prompt = lambda text: "Extract terminology:\n" + text

    # Verifies _split_text succeeds and creates multiple chunks without crashing
    chunks = refinement._split_text(long_transcript, system_prompt, make_prompt)
    assert len(chunks) > 1
    assert "".join(chunks) == long_transcript

    # Verifies _chunk_segments also works end-to-end
    segments = [
        {"id": f"S{i:04d}", "start": float(i), "end": float(i + 1), "text": f"Segment line {i} text."}
        for i in range(100)
    ]
    batches = refinement._chunk_segments(
        segments,
        system_prompt,
        lambda segs: "Refine:\n" + str(segs),
    )
    assert len(batches) >= 1
    reconstructed_segments = [s for b in batches for s in b]
    assert len(reconstructed_segments) == len(segments)
