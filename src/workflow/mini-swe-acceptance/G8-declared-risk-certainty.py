import pytest
from pydantic import ValidationError

from workflow.schemas import ReconciledVerdict


MARKER = "G8_UNVERIFIED_CANNOT_BE_FIVE"


def complete_verdict(**overrides: object) -> dict[str, object]:
    return {
        "inquiry_id": "G8-HELD-OUT",
        "adopted_proposals": [],
        "rejected_proposals": [],
        "certainty_score": 4,
        "residual_risks": [],
        "plain_language_verdict": "Declared-risk consistency check.",
        "actionable_decision_matrix": "Keep the declared risk visible.",
        "reconciled_diagram_d2": "",
    } | overrides


def test_score_five_rejects_one_or_more_declared_residual_risks():
    for risks in (["cache synchronization unverified"], ["cache synchronization unverified", "replica lag unverified"]):
        try:
            ReconciledVerdict(**complete_verdict(certainty_score=5, residual_risks=risks))
        except ValidationError as error:
            assert MARKER in str(error)
        else:
            pytest.fail(MARKER)


@pytest.mark.parametrize(
    "payload",
    [
        complete_verdict(certainty_score=5),
        complete_verdict(certainty_score=5, residual_risks=[]),
        complete_verdict(certainty_score=4, residual_risks=["cache synchronization unverified"]),
        complete_verdict(certainty_score=4, residual_risks=[]),
    ],
)
def test_valid_declared_risk_and_certainty_combinations_remain_valid(payload: dict[str, object]):
    assert ReconciledVerdict(**payload).certainty_score in {4, 5}
