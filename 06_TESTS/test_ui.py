from pathlib import Path

from app.ui import gradio_app


def test_gradio_ui_registers_processing_results_and_reset_events():
    dependencies = gradio_app.demo.get_config_file()["dependencies"]
    names = {dependency["api_name"] for dependency in dependencies}
    assert "_run_for_ui" in names
    assert "_result_component_values" in names
    assert "lambda" in names


def test_processing_status_reflects_current_pipeline_stage():
    status = gradio_app._status_view("refining")
    assert "Refining terminology" in status
    assert 'class="stage done"' in status
    assert 'class="stage active"' in status
    assert 'class="stage pending"' in status
    assert "percentage" not in status.lower()


def test_results_dashboard_renders_actual_record_and_timestamped_transcript(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    archive = output_dir / "run.zip"
    archive.write_bytes(b"zip")
    for _, filename in gradio_app.OUTPUT_FILES:
        (output_dir / filename).write_text(filename)
    result = {
        "record": {
            "meeting_title": "Actual meeting",
            "summary": "Discussed the deployment.",
            "decisions": [
                {
                    "text": "Delay deployment",
                    "status": "Confirmed",
                    "evidence_quote": "We will delay deployment.",
                }
            ],
            "non_decisions": [
                {"text": "Consider a second region", "evidence_quote": "Maybe later."}
            ],
            "action_items": [
                {
                    "task": "Send report",
                    "owner": "Not specified",
                    "deadline": "Not specified",
                    "status": "Needs Review",
                    "evidence_quote": "We should send it.",
                }
            ],
            "minutes": ["Deployment timing discussed."],
        },
        "output_dir": str(output_dir),
        "archive_path": str(archive),
        "raw_transcript": "Actual raw words.",
        "raw_segments": [
            {"start": 4.2, "end": 6.8, "text": "Actual raw words."}
        ],
        "refined_transcript": "Actual refined words.",
        "refined_segments": [
            {"start": 4.2, "end": 6.8, "text": "Actual refined words."}
        ],
    }

    values = gradio_app._result_component_values(result)
    assert len(values) == 9
    assert "Actual meeting" in values[4]
    # Raw transcript
    assert "[00:04 – 00:06] Actual raw words." in values[5]
    # Refined transcript
    assert "[00:04 – 00:06] Actual refined words." in values[6]
    # Raw != Refined check logic (they are distinct outputs)
    assert values[5] != values[7]
    # Final Output Markdown contains decisions and summary
    assert "Delay deployment" in values[7]
    assert "Discussed the deployment." in values[7]
    # Archive download path
    assert values[8] == str(archive)


def test_ui_error_preserves_technical_detail_and_reenables_processing():
    result = {
        "error": True,
        "friendly_error": "Local transcription failed.",
        "technical_error": "TypeError: decoder failure",
    }
    values = gradio_app._result_component_values(result)
    assert "Something went wrong" in values[0]
    assert "Local transcription failed." in values[0]
    assert "TypeError: decoder failure" in values[0]
    assert values[1]["visible"] is True
    assert values[2]["visible"] is False
    assert "error" in values[7].lower()


def test_refinement_failure_shows_complete_raw_transcript_without_false_completion():
    result = {
        "error": True,
        "stage": "refining",
        "friendly_error": "Refinement failed.",
        "technical_error": "HTTP 413",
        "raw_transcript": "Actual complete raw transcript",
        "raw_segments": [{"start": 0.0, "end": 1.0, "text": "Actual complete raw transcript"}],
    }

    values = gradio_app._result_component_values(result)
    assert "Refinement failed." in values[0]
    assert values[1]["visible"] is False
    assert values[2]["visible"] is True
    assert values[5] == "[00:00 – 00:01] Actual complete raw transcript"
    assert values[6] == ""
    assert "error" in values[7].lower()
    assert values[8] is None


def test_ui_processing_streams_backend_stages_and_final_pipeline_result(monkeypatch):
    def fake_process_meeting(
        audio_path, glossary, stage_callback, partial_result_callback
    ):
        assert audio_path == Path("uploaded meeting.wav")
        assert glossary == "API, CUDA"
        stage_callback("received")
        stage_callback("validated")
        stage_callback("transcribing")
        partial_result_callback(
            {"raw_transcript": "Actual raw transcript", "raw_segments": []}
        )
        stage_callback("raw_ready")
        stage_callback("refining")
        partial_result_callback(
            {
                "raw_transcript": "Actual raw transcript",
                "raw_segments": [],
                "refined_transcript": "Actual refined transcript",
                "refined_segments": [],
            }
        )
        stage_callback("refined")
        stage_callback("documenting")
        stage_callback("validating")
        stage_callback("exporting")
        stage_callback("complete")
        return {
            "raw_transcript": "Actual raw transcript",
            "refined_transcript": "Actual refined transcript",
            "record": {},
            "output_dir": "actual-output",
            "archive_path": "actual-output.zip",
        }

    monkeypatch.setattr(gradio_app, "process_meeting", fake_process_meeting)
    events = list(gradio_app._run_for_ui("uploaded meeting.wav", "API, CUDA"))

    assert any("Transcribing meeting" in event[0] for event in events)
    assert any("Raw transcript ready" in event[0] for event in events)
    assert any("Refining terminology" in event[0] for event in events)
    assert any("Refined transcript ready" in event[0] for event in events)
    assert any("Generating meeting record" in event[0] for event in events)
    assert "Processing complete" in events[-1][0]
    assert events[-1][1]["raw_transcript"] == "Actual raw transcript"
    assert events[-1][1]["refined_transcript"] == "Actual refined transcript"


def test_ui_retains_raw_transcript_and_marks_refinement_failure_incomplete(monkeypatch):
    def fail_during_refinement(
        audio_path, glossary, stage_callback, partial_result_callback
    ):
        stage_callback("received")
        stage_callback("validated")
        stage_callback("transcribing")
        partial_result_callback(
            {
                "raw_transcript": "Complete raw transcript",
                "raw_segments": [{"start": 0.0, "end": 2.0, "text": "Complete raw transcript"}],
            }
        )
        stage_callback("raw_ready")
        stage_callback("refining")
        raise RuntimeError("provider request failed")

    monkeypatch.setattr(gradio_app, "process_meeting", fail_during_refinement)
    events = list(gradio_app._run_for_ui("uploaded meeting.wav", ""))
    status, result, button = events[-1]
    values = gradio_app._result_component_values(result)

    assert result["error"] is True
    assert result["stage"] == "refining"
    assert result["raw_transcript"] == "Complete raw transcript"
    assert '<div class="progress-heading">Processing stopped</div>' in status
    assert '<div class="progress-heading">Processing complete</div>' not in status
    assert values[2]["visible"] is True
    assert values[5] == "[00:00 – 00:02] Complete raw transcript"
    assert values[6] == ""
    assert button["interactive"] is True


def test_ui_returns_readable_error_when_no_audio_is_selected():
    status, result, button = next(gradio_app._run_for_ui(None, ""))
    assert "Audio received" in status
    assert result["error"] is True
    assert "Upload a meeting recording" in result["friendly_error"]
    assert button["interactive"] is True
