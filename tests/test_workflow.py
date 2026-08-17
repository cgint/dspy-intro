"""Unit tests for the DSPy Code Grounding Workflow."""

import dspy
from workflow.schemas import (
    InquiryRecord,
    CallTree,
    PrimaryBaseline,
    Proposal,
    ProposalLedger,
    ReconciledVerdict,
)
from workflow.tools import search_code, read_code_slice, validate_d2_syntax
from workflow.flow import CodeGroundingWorkflow


def test_schema_instantiation_and_validation():
    """Verify all workflow Pydantic models instantiate with correct types."""
    inquiry = InquiryRecord(
        inquiry_id="INQ-100",
        stakeholder="Platform Lead",
        raw_question="Does pausing a campaign stop auctions immediately?",
        operational_context="Checking latency",
        target_systems=["campaign-service"],
        stakes_severity="high"
    )
    assert inquiry.inquiry_id == "INQ-100"
    assert inquiry.stakes_severity == "high"

    baseline = PrimaryBaseline(
        inquiry_id="INQ-100",
        call_tree=CallTree(
            entry_point="CampaignController.pause",
            intermediate_services=["CampaignService"],
            event_publishers=["CampaignPausedEvent"],
            leaf_mutations=["UPDATE campaigns SET status='PAUSED'"]
        ),
        git_intent_summary="Added atomic pause flag",
        code_citations=["CampaignController.java:42"],
        baseline_summary="Campaign is paused synchronously in DB",
        diagram_d2="controller -> service -> db"
    )
    assert len(baseline.call_tree.leaf_mutations) == 1

    proposal = Proposal(
        proposal_id="PROP-001",
        category="race-condition",
        target_symbol="CampaignController.pause",
        failing_scenario="Simultaneous unpause request may overwrite status",
        code_evidence="Line 45 lacks optimistic lock",
        suggested_revision="Add @Version check"
    )
    ledger = ProposalLedger(inquiry_id="INQ-100", proposals=[proposal])
    assert len(ledger.proposals) == 1

    verdict = ReconciledVerdict(
        inquiry_id="INQ-100",
        adopted_proposals=["PROP-001"],
        rejected_proposals=[],
        certainty_score=4,
        residual_risks=["External cache sync delay"],
        plain_language_verdict="Campaign pauses in DB immediately, but auction cache takes ~5s.",
        actionable_decision_matrix="Apply optimistic lock and check auction worker TTL.",
        reconciled_diagram_d2="controller -> service -> db"
    )
    assert verdict.certainty_score == 4


def test_tool_read_code_slice(tmp_path):
    """Verify read_code_slice correctly reads line ranges and handles missing files."""
    test_file = tmp_path / "sample.py"
    test_file.write_text("line1\nline2\nline3\nline4\nline5\n")

    res = read_code_slice(str(test_file), start_line=2, end_line=4)
    assert "2: line2" in res
    assert "3: line3"
    assert "4: line4"
    assert "5: line5" not in res

    # Missing file
    res_missing = read_code_slice(str(tmp_path / "non_existent.py"))
    assert "File not found" in res_missing


def test_tool_search_code(tmp_path):
    """Verify search_code finds symbols in files."""
    test_file = tmp_path / "service.py"
    test_file.write_text("class TargetService:\n    def execute(self):\n        pass\n")

    res = search_code("TargetService", directory=str(tmp_path))
    assert "TargetService" in res


def test_tool_validate_d2_syntax():
    """Verify D2 syntax checking detects unbalanced braces and template placeholders."""
    valid_d2 = "user -> api: request\napi -> db: query"
    assert "passed" in validate_d2_syntax(valid_d2) or "Valid" in validate_d2_syntax(valid_d2)

    unbalanced_d2 = "user -> api: request {\napi -> db"
    assert "Unbalanced" in validate_d2_syntax(unbalanced_d2)

    template_d2 = "user -> {node}: request"
    assert "Unescaped template" in validate_d2_syntax(template_d2)


def test_workflow_initialization():
    """Verify CodeGroundingWorkflow initializes all stages properly."""
    workflow = CodeGroundingWorkflow(use_react=False)
    assert hasattr(workflow, "ingress")
    assert hasattr(workflow, "discovery")
    assert hasattr(workflow, "adversarial_reviewer")
    assert hasattr(workflow, "reconciler")

    workflow_react = CodeGroundingWorkflow(use_react=True)
    assert hasattr(workflow_react, "discovery")
    assert hasattr(workflow_react, "adversarial_reviewer")


def test_workflow_forward_with_mocked_stages(monkeypatch):
    """Verify the deterministic sequential execution of CodeGroundingWorkflow."""
    workflow = CodeGroundingWorkflow(use_react=False)

    mock_inquiry = InquiryRecord(
        inquiry_id="INQ-TEST",
        raw_question="Does delete pause?",
        operational_context="Audit",
        target_systems=["billing"],
        stakes_severity="medium"
    )
    mock_baseline = PrimaryBaseline(
        inquiry_id="INQ-TEST",
        call_tree=CallTree(entry_point="API", leaf_mutations=["DELETE"]),
        git_intent_summary="Initial commit",
        code_citations=["test.py:1"],
        baseline_summary="Delete triggers DB removal",
        diagram_d2="api -> db"
    )
    mock_ledger = ProposalLedger(
        inquiry_id="INQ-TEST",
        proposals=[
            Proposal(
                proposal_id="PROP-1",
                category="intent-gap",
                target_symbol="DELETE",
                failing_scenario="Hard delete instead of soft delete",
                code_evidence="Line 1",
                suggested_revision="Use active=false"
            )
        ]
    )
    mock_verdict = ReconciledVerdict(
        inquiry_id="INQ-TEST",
        adopted_proposals=["PROP-1"],
        certainty_score=5,
        plain_language_verdict="Verified delete flow.",
        actionable_decision_matrix="Migrate to soft delete."
    )

    # Mock predictions
    monkeypatch.setattr(workflow, "ingress", lambda **kw: dspy.Prediction(inquiry=mock_inquiry))
    monkeypatch.setattr(workflow, "discovery", lambda **kw: dspy.Prediction(baseline=mock_baseline))
    monkeypatch.setattr(workflow, "adversarial_reviewer", lambda **kw: dspy.Prediction(ledger=mock_ledger))
    monkeypatch.setattr(workflow, "reconciler", lambda **kw: dspy.Prediction(verdict=mock_verdict))

    result = workflow(raw_question="Does delete pause?")
    assert result.inquiry.inquiry_id == "INQ-TEST"
    assert result.baseline.baseline_summary == "Delete triggers DB removal"
    assert len(result.ledger.proposals) == 1
    assert result.verdict.certainty_score == 5
