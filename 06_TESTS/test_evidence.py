from app.pipeline.evidence import find_evidence, validate_action, validate_record


def test_evidence_matches_across_adjacent_segments():
    segments = [
        {"id": "S1", "start": 1.0, "text": "We approved the"},
        {"id": "S2", "start": 2.0, "text": "launch for Friday."},
    ]
    evidence = find_evidence("approved the launch for Friday", segments)
    assert evidence["found"]
    assert evidence["segment_ids"] == ["S1", "S2"]


def test_supported_owner_and_deadline_are_preserved():
    action = {
        "task": "Send the report",
        "owner": "Asha",
        "deadline": "Friday",
        "evidence_quote": "Asha will send the report by Friday.",
        "status": "Confirmed",
    }
    validated = validate_action(action)
    assert validated["owner"] == "Asha"
    assert validated["deadline"] == "Friday"
    assert validated["status"] == "Confirmed"


def test_supported_owner_and_deadline_are_retained():
    action = {
        "task": "Send the report",
        "owner": "Asha",
        "deadline": "Friday",
        "evidence_quote": "Asha will send the report by Friday.",
    }
    validated = validate_action(action)
    assert validated["owner"] == "Asha"
    assert validated["deadline"] == "Friday"


def test_unsupported_decision_is_retained_as_needs_review_and_proposal_is_non_decision():
    refined = {
        "refined_segments": [
            {"id": "S1", "start": 0.0, "text": "We could consider a June launch."}
        ]
    }
    source = {
        "decisions": [
            {
                "text": "Launch in May",
                "evidence_quote": "We decided to launch in May.",
                "status": "Confirmed",
            }
        ],
        "non_decisions": [
            {
                "text": "Consider a June launch",
                "evidence_quote": "We could consider a June launch.",
            }
        ],
    }
    validated = validate_record(source, refined)
    assert len(validated["decisions"]) == 1
    assert validated["decisions"][0]["text"] == "Launch in May"
    assert validated["decisions"][0]["status"] == "Needs Review"
    assert validated["decisions"][0]["evidence"]["found"] is False
    assert len(validated["non_decisions"]) == 1
    assert validated["non_decisions"][0]["text"] == "Consider a June launch"


def test_decision_with_missing_evidence_remains_in_output_as_needs_review():
    refined = {"refined_segments": [{"id": "S1", "start": 0.0, "text": "Unrelated meeting transcript."}]}
    source = {
        "decisions": [
            {
                "text": "Adopt microservices architecture",
                "status": "Confirmed",
                "evidence_quote": "We fully agreed to adopt microservices.",
            }
        ]
    }
    validated = validate_record(source, refined)
    assert len(validated["decisions"]) == 1
    assert validated["decisions"][0]["text"] == "Adopt microservices architecture"
    assert validated["decisions"][0]["status"] == "Needs Review"
    assert validated["decisions"][0]["evidence"]["found"] is False
    assert validated["decisions"][0]["evidence_quote"] == "We fully agreed to adopt microservices."


def test_valid_decision_and_action_statuses_are_preserved():
    refined = {
        "refined_segments": [
            {"id": "S1", "start": 0.0, "text": "We approved budget and Rahul will deploy tomorrow."}
        ]
    }
    source = {
        "decisions": [
            {
                "text": "Approved budget",
                "status": "Confirmed",
                "evidence_quote": "We approved budget",
            }
        ],
        "action_items": [
            {
                "task": "Deploy tomorrow",
                "owner": "Rahul",
                "deadline": "tomorrow",
                "status": "Confirmed",
                "evidence_quote": "Rahul will deploy tomorrow.",
            }
        ],
    }
    validated = validate_record(source, refined)
    assert validated["decisions"][0]["status"] == "Confirmed"
    assert validated["action_items"][0]["status"] == "Confirmed"


def test_invalid_llm_status_is_safely_normalized():
    refined = {
        "refined_segments": [
            {"id": "S1", "start": 0.0, "text": "We approved budget and Rahul will deploy tomorrow."}
        ]
    }
    source = {
        "decisions": [
            {
                "text": "Approved budget",
                "status": "Finalized_By_Management",
                "evidence_quote": "We approved budget",
            }
        ],
        "action_items": [
            {
                "task": "Deploy tomorrow",
                "owner": "Rahul",
                "deadline": "tomorrow",
                "status": "In_Progress",
                "evidence_quote": "Rahul will deploy tomorrow.",
            }
        ],
    }
    validated = validate_record(source, refined)
    assert validated["decisions"][0]["status"] == "Needs Review"
    assert validated["action_items"][0]["status"] == "Needs Review"


def test_confidently_supported_action_items_not_unnecessarily_marked_needs_review():
    refined = {
        "refined_segments": [
            {"id": "S1", "start": 10.0, "text": "Priya will finalize the API specification by end of day."}
        ]
    }
    source = {
        "action_items": [
            {
                "task": "Finalize the API specification",
                "owner": "Priya",
                "deadline": "end of day",
                "status": "Confirmed",
                "evidence_quote": "Priya will finalize the API specification by end of day.",
            }
        ]
    }
    validated = validate_record(source, refined)
    assert len(validated["action_items"]) == 1
    action = validated["action_items"][0]
    assert action["status"] == "Confirmed"
    assert action["owner"] == "Priya"
    assert action["deadline"] == "end of day"
    assert action["evidence"]["found"] is True


def test_proposal_not_incorrectly_classified_as_decision():
    refined = {
        "refined_segments": [
            {
                "id": "S1",
                "start": 0.0,
                "text": "Rahul suggested that we migrate the database to PostgreSQL 16. We discussed the idea, but we have not made a final decision.",
            }
        ]
    }
    source = {
        "decisions": [
            {
                "text": "Migrate database to PostgreSQL 16",
                "evidence_quote": "Rahul suggested that we migrate the database to PostgreSQL 16.",
            }
        ],
        "non_decisions": [],
    }
    validated = validate_record(source, refined)
    # Must NOT be classified as confirmed decision
    assert validated["decisions"] == []
    # Must be categorized as proposal / non-decision
    assert len(validated["non_decisions"]) == 1
    assert "PostgreSQL 16" in validated["non_decisions"][0]["text"]


def test_unspecified_owner_and_deadline_remain_unspecified():
    action = {
        "task": "Review latency metrics",
        "evidence_quote": "We should review latency metrics next week.",
    }
    validated = validate_action(action)
    assert validated["owner"] == "Unspecified"
    assert validated["deadline"] == "Unspecified"


def test_action_deadline_confirmed_in_adjacent_segment_is_retained():
    segments = [
        {"id": "S1", "start": 0.0, "text": "Can someone update the slide deck?"},
        {"id": "S2", "start": 3.0, "text": "I will handle the slide deck."},
        {"id": "S3", "start": 6.0, "text": "Sounds good, please confirm it by Friday."},
    ]
    source = {
        "action_items": [
            {
                "task": "Handle the slide deck",
                "owner": "Unspecified",
                "deadline": "Friday",
                "status": "Confirmed",
                "evidence_quote": "I will handle the slide deck.",
            }
        ]
    }
    refined = {"refined_segments": segments}
    validated = validate_record(source, refined)
    action = validated["action_items"][0]
    assert action["deadline"] == "Friday"
    assert action["status"] == "Confirmed"


def test_action_owner_spoken_in_adjacent_segment_is_retained():
    segments = [
        {"id": "S1", "start": 0.0, "text": "Alice, will you send the summary?"},
        {"id": "S2", "start": 3.0, "text": "Yes, I will send the summary."},
    ]
    source = {
        "action_items": [
            {
                "task": "Send the summary",
                "owner": "Alice",
                "deadline": "Unspecified",
                "status": "Confirmed",
                "evidence_quote": "Yes, I will send the summary.",
            }
        ]
    }
    refined = {"refined_segments": segments}
    validated = validate_record(source, refined)
    action = validated["action_items"][0]
    assert action["owner"] == "Alice"
    assert action["status"] == "Confirmed"


def test_first_person_commitment_sets_owner_note():
    action = {
        "task": "Send the deck",
        "owner": "Unspecified",
        "deadline": "Friday",
        "status": "Confirmed",
        "evidence_quote": "I'll send it by Friday.",
    }
    validated = validate_action(action)
    assert validated["owner"] == "Unspecified"
    assert validated["deadline"] == "Friday"
    assert validated["owner_note"] == "First-person commitment; speaker identity unavailable."


def test_cross_section_duplicates_handled_correctly():
    refined = {
        "refined_segments": [
            {"id": "S1", "start": 0.0, "text": "We agreed to adopt PostgreSQL 16."},
            {"id": "S2", "start": 5.0, "text": "Bob will deploy PostgreSQL 16 by Friday."},
        ]
    }
    source = {
        "decisions": [
            {
                "text": "Adopt PostgreSQL 16",
                "status": "Confirmed",
                "evidence_quote": "We agreed to adopt PostgreSQL 16.",
            }
        ],
        "action_items": [
            # 1. Unassigned verbatim duplicate of the decision -> should be dropped
            {
                "task": "Adopt PostgreSQL 16",
                "owner": "Unspecified",
                "deadline": "Unspecified",
                "status": "Confirmed",
                "evidence_quote": "We agreed to adopt PostgreSQL 16.",
            },
            # 2. Assigned implementation work -> should be preserved
            {
                "task": "Deploy PostgreSQL 16",
                "owner": "Bob",
                "deadline": "Friday",
                "status": "Confirmed",
                "evidence_quote": "Bob will deploy PostgreSQL 16 by Friday.",
            },
        ],
    }
    validated = validate_record(source, refined)
    assert len(validated["decisions"]) == 1
    assert validated["decisions"][0]["text"] == "Adopt PostgreSQL 16"
    assert len(validated["action_items"]) == 1
    assert validated["action_items"][0]["task"] == "Deploy PostgreSQL 16"
    assert validated["action_items"][0]["owner"] == "Bob"


def test_both_stages_configured_with_gpt_oss_120b():
    from app.core import config
    assert config.GROQ_REFINEMENT_MODEL == "openai/gpt-oss-120b"
    assert config.GROQ_DOCUMENTATION_MODEL == "openai/gpt-oss-120b"


def test_estimates_clarifications_and_tentative_items_routed_to_non_decisions():
    refined = {
        "refined_text": (
            "We discussed the infrastructure budget. The current estimate is 2.5 lakh rupees. "
            "We might deploy Friday."
        ),
        "refined_segments": [
            {"id": "S1", "start": 0.0, "text": "We discussed the infrastructure budget."},
            {"id": "S2", "start": 4.0, "text": "The current estimate is 2.5 lakh rupees."},
            {"id": "S3", "start": 8.0, "text": "We might deploy Friday."},
        ],
    }
    source = {
        "decisions": [
            {
                "text": "Infrastructure budget estimate is 2.5 lakh rupees",
                "status": "Confirmed",
                "evidence_quote": "The current estimate is 2.5 lakh rupees.",
            },
            {
                "text": "Deploy on Friday",
                "status": "Confirmed",
                "evidence_quote": "We might deploy Friday.",
            },
        ],
        "action_items": [],
    }
    validated = validate_record(source, refined)
    # Neither should remain a confirmed decision; both should be in non_decisions
    assert len(validated["decisions"]) == 0
    assert len(validated["non_decisions"]) == 2
    assert any("estimate" in nd["text"].lower() for nd in validated["non_decisions"])
    assert any("friday" in nd["text"].lower() for nd in validated["non_decisions"])


def test_unsupported_finalized_claim_downgrades_to_needs_review():
    refined = {
        "refined_text": "Today we need to finalize our machine learning deployment plan.",
        "refined_segments": [
            {"id": "S1", "start": 0.0, "text": "Today we need to finalize our machine learning deployment plan."}
        ],
    }
    source = {
        "decisions": [
            {
                "text": "Finalized the machine learning deployment plan",
                "status": "Confirmed",
                "evidence_quote": "Today we need to finalize our machine learning deployment plan.",
            }
        ],
        "action_items": [],
    }
    validated = validate_record(source, refined)
    # Since the text or evidence has "need to finalize", it is either in non_decisions or Needs Review
    if validated["decisions"]:
        assert validated["decisions"][0]["status"] == "Needs Review"
    else:
        assert len(validated["non_decisions"]) == 1


def test_summary_overclaiming_sanitization():
    refined = {
        "refined_text": "Good morning. Today we need to finalize our machine learning deployment plan.",
        "refined_segments": [
            {"id": "S1", "start": 0.0, "text": "Good morning. Today we need to finalize our machine learning deployment plan."}
        ],
    }
    source = {
        "summary": "Team finalized deployment schedule, assigned evaluation report, and set agenda.",
        "decisions": [],
        "action_items": [],
    }
    validated = validate_record(source, refined)
    assert "finalized deployment schedule" not in validated["summary"].lower()
    assert "discussed" in validated["summary"].lower() or "planned" in validated["summary"].lower()


def test_unassigned_outstanding_task_inferred_as_needs_review():
    action = {
        "task": "Complete testing before launch",
        "owner": "Unspecified",
        "deadline": "Unspecified",
        "status": "Confirmed",
        "evidence_quote": "We still need to complete testing.",
    }
    validated = validate_action(action)
    assert validated["status"] == "Needs Review"
    assert validated["owner"] == "Unspecified"
    assert validated["deadline"] == "Unspecified"


def test_explicit_assigned_task_preserves_confirmed_status():
    action = {
        "task": "Prepare evaluation report",
        "owner": "Priya",
        "deadline": "Monday",
        "status": "Confirmed",
        "evidence_quote": "Priya will prepare the evaluation report by Monday.",
    }
    validated = validate_action(action)
    assert validated["status"] == "Confirmed"
    assert validated["owner"] == "Priya"
    assert validated["deadline"] == "Monday"


def test_supported_decision_remains_confirmed():
    refined = {
        "refined_segments": [
            {
                "id": "S1",
                "start": 0.0,
                "text": "We decided to launch in May."
            }
        ]
    }

    source = {
        "decisions": [
            {
                "text": "Launch in May",
                "evidence_quote": "We decided to launch in May.",
                "status": "Confirmed",
            }
        ],
        "non_decisions": [],
    }

    validated = validate_record(source, refined)

    assert len(validated["decisions"]) == 1
    assert validated["decisions"][0]["text"] == "Launch in May"
    assert validated["decisions"][0]["status"] == "Confirmed"
    assert validated["decisions"][0]["evidence"]["found"] is True
