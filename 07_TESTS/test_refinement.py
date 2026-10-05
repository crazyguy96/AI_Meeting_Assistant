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
