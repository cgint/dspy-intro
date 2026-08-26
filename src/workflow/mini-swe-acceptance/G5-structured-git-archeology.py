import pytest
from pydantic import ValidationError

from workflow.schemas import CallTree, PrimaryBaseline


MARKER = "G5_HELD_OUT_GIT_ARCHEOLOGY"


def complete_baseline() -> dict[str, object]:
    return {
        "inquiry_id": "G5-HELD-OUT",
        "call_tree": CallTree(entry_point="API"),
        "git_archeology": {
            "relevant_commits": ["abc123"],
            "author_intent_summary": "intent",
            "intent_vs_code_gap": "gap",
        },
        "code_citations": [],
        "baseline_summary": "summary",
        "diagram_d2": "",
    }


def test_git_archeology_is_required_and_legacy_field_is_absent():
    assert "git_archeology" in PrimaryBaseline.model_fields, MARKER
    assert "git_intent_summary" not in PrimaryBaseline.model_fields
    assert PrimaryBaseline(**complete_baseline()).git_archeology
    missing = complete_baseline()
    missing.pop("git_archeology")
    with pytest.raises(ValidationError, match="git_archeology"):
        PrimaryBaseline(**missing)


@pytest.mark.parametrize("field", ["relevant_commits", "author_intent_summary", "intent_vs_code_gap"])
def test_each_git_archeology_field_is_required(field: str):
    candidate = complete_baseline()
    archeology = dict(candidate["git_archeology"])
    archeology.pop(field)
    candidate["git_archeology"] = archeology
    with pytest.raises(ValidationError, match=field):
        PrimaryBaseline(**candidate)
