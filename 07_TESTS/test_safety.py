from app.pipeline.evidence import protected_tokens, safe_edit


def test_protected_numbers_and_monetary_values_cannot_change():
    assert not safe_edit("budget is ₹2.5 lakh", "budget is ₹25 lakh")[0]
    assert not safe_edit("budget is 2.5 lakh", "budget is 2.5 crore")[0]
    assert not safe_edit("budget is $500", "budget is 500")[0]
    assert not safe_edit("budget is $500", "budget is €500")[0]
    assert not safe_edit("revenue is 12.5%", "revenue is 15.2%")[0]
    assert protected_tokens("cost is INR 1,25,000") == protected_tokens(
        "cost is INR 1,25,000"
    )


def test_negation_cannot_be_removed():
    assert not safe_edit("we will not deploy Friday", "we will deploy Friday")[0]


def test_modality_cannot_be_strengthened():
    assert not safe_edit("we might deploy Friday", "we will deploy Friday")[0]
    assert not safe_edit("we committed to deploy", "we planned to deploy")[0]


def test_safe_terminology_edit_preserves_protected_tokens():
    assert safe_edit("use kuberneties by 2027", "use Kubernetes by 2027")[0]
