"""Protected public-runtime acceptance for G10 unresolved proposal semantics."""
from types import SimpleNamespace
import pytest
from pydantic import ValidationError
from workflow.flow import CodeGroundingWorkflow
from workflow.schemas import InquiryRecord, PrimaryBaseline, CallTree, GitArcheology, Proposal, ProposalLedger, ReconciledVerdict
MARKER = "G10_UNRESOLVED_PROPOSAL_FAILURE"


def proposal(pid):
    return Proposal(proposal_id=pid, category="race-condition", target_symbol="x", failing_scenario="x", code_evidence="x", suggested_revision="x")


def run(ledger, adopted, rejected, score=4, status="reconciled-and-calibrated"):
    inquiry = InquiryRecord(inquiry_id="I", raw_question="q", operational_context="c")
    baseline = PrimaryBaseline(inquiry_id="I", call_tree=CallTree(entry_point="x"), git_archeology=GitArcheology(relevant_commits=["a"], author_intent_summary="a", intent_vs_code_gap="a"), baseline_summary="b")
    verdict = ReconciledVerdict(inquiry_id="I", status=status, adopted_proposals=adopted, rejected_proposals=rejected, certainty_score=score, plain_language_verdict="v", actionable_decision_matrix="a")
    wf = CodeGroundingWorkflow()
    wf.ingress = lambda **_: SimpleNamespace(inquiry=inquiry)
    wf.discovery = lambda **_: SimpleNamespace(baseline=baseline)
    wf.adversarial_reviewer = lambda **_: SimpleNamespace(ledger=ledger)
    wf.reconciler = lambda **_: SimpleNamespace(verdict=verdict)
    return wf.forward("q").verdict


def assert_state(verdict, status, certainty):
    assert getattr(verdict, "status", None) == status, MARKER
    assert verdict.certainty_score == certainty, MARKER


def test_public_workflow_resolves_every_ledger_id_or_fails_with_null_certainty():
    two = ProposalLedger(inquiry_id="I", proposals=[proposal("P1"), proposal("P2")])
    assert_state(run(two, ["P1"], []), "rejected-unaddressed-proposals", None)
    assert_state(run(ProposalLedger(inquiry_id="I", proposals=[proposal("P1")]), ["UNKNOWN"], ["OTHER"]), "rejected-unaddressed-proposals", None)
    for ledger, adopted, rejected in [(two, ["P1"], ["P2"]), (two, ["P1", "P2"], []), (two, [], ["P1", "P2"]), (ProposalLedger(inquiry_id="I"), [], [])]:
        assert_state(run(ledger, adopted, rejected), "reconciled-and-calibrated", 4)


def assert_pairing_rejected(**values):
    try:
        ReconciledVerdict(**values)
    except ValidationError as error:
        assert "G10_FAILURE_STATUS_CERTAINTY_PAIRING" in str(error)
    else:
        pytest.fail("G10_FAILURE_STATUS_CERTAINTY_PAIRING")


def test_public_workflow_does_not_invent_certainty_for_resolved_ledger():
    try:
        run(ProposalLedger(inquiry_id="I"), [], [], score=None, status="reconciled-and-calibrated")
    except ValidationError as error:
        assert "G10_FAILURE_STATUS_CERTAINTY_PAIRING" in str(error), "G10_FAILURE_STATUS_CERTAINTY_PAIRING"
    else:
        pytest.fail("G10_FAILURE_STATUS_CERTAINTY_PAIRING")


def test_public_failure_status_and_certainty_pairing_is_enforced():
    assert_pairing_rejected(inquiry_id="I", status="rejected-unaddressed-proposals", certainty_score=4, plain_language_verdict="v", actionable_decision_matrix="a")
    assert_pairing_rejected(inquiry_id="I", status="reconciled-and-calibrated", certainty_score=None, plain_language_verdict="v", actionable_decision_matrix="a")
