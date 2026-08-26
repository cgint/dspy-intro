"""Protected acceptance for G10 unresolved-proposal failure semantics."""

from workflow.schemas import ReconciledVerdict

MARKER = "G10_UNRESOLVED_PROPOSAL_FAILURE"


def test_unresolved_proposal_failure_is_first_class_public_verdict_state():
    assert "status" in ReconciledVerdict.model_fields, MARKER
    assert "certainty_score" in ReconciledVerdict.model_fields, MARKER
    status = ReconciledVerdict.model_fields["status"]
    certainty = ReconciledVerdict.model_fields["certainty_score"]
    assert status.annotation is str, MARKER
    assert certainty.annotation is int | None, MARKER
