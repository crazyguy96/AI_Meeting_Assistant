from app.pipeline.evidence import find_evidence, validate_action, validate_record


def test_evidence_matches_across_adjacent_segments():
    segments = [
        {"id": "S1", "start": 1.0, "text": "We approved the"},
        {"id": "S2", "start": 2.0, "text": "launch for Friday."},
    ]
    evidence = find_evidence("approved the launch for Friday", segments)
    assert evidence["found"]
    assert evidence["segment_ids"] == ["S1", "S2"]


def test_unsupported_owner_and_deadline_are_removed():
    action = {
        "task": "Send the report",
        "owner": "Asha",
        "deadline": "Friday",
        "evidence_quote": "We should send the report.",
        "status": "Confirmed",
    }
    validated = validate_action(action)
    assert validated["owner"] == "Unspecified"
    assert validated["deadline"] == "Unspecified"
    assert validated["status"] == "Needs Review"


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


def test_unsupported_decision_is_excluded_but_proposal_remains_non_decision():
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
    assert validated["decisions"] == []
    assert len(validated["non_decisions"]) == 1


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


