"""DSPy Signatures for the 4-Stage Code Grounding Workflow.

Each signature captures the intent, input parameters, and output contracts.
"""

import dspy
from workflow.schemas import (
    InquiryRecord,
    PrimaryBaseline,
    ProposalLedger,
    ReconciledVerdict,
)


class StakeholderIngressSignature(dspy.Signature):
    """Normalize and structure raw stakeholder technical questions into an actionable inquiry record."""

    raw_question: str = dspy.InputField(desc="Unstructured stakeholder question or incident report")
    operational_context: str = dspy.InputField(desc="Contextual background, urgency, systems involved")
    inquiry: InquiryRecord = dspy.OutputField(desc="Structured, normalized inquiry record")


class PrimaryEvidenceDiscoverySignature(dspy.Signature):
    """Trace forward execution paths from triggers to leaf mutations with exact citations and call tree."""

    inquiry: InquiryRecord = dspy.InputField(desc="The normalized stakeholder inquiry")
    code_context: str = dspy.InputField(desc="Discovered source code snippets, symbols, and git history")
    baseline: PrimaryBaseline = dspy.OutputField(desc="Verifiable baseline investigation with call tree and D2 diagram")


class AdversarialChallengeSignature(dspy.Signature):
    """Subject the primary baseline to independent adversarial challenge.
    Audit leaf mutations backward to uncover vacuous truths (e.g. allMatch on empty lists),
    race conditions, missing auth guards, and diagram syntax errors.
    Emit independent RFC proposals without modifying the baseline.
    """

    inquiry: InquiryRecord = dspy.InputField(desc="The original normalized inquiry")
    baseline: PrimaryBaseline = dspy.InputField(desc="The primary baseline report under review")
    code_context: str = dspy.InputField(desc="Source code guards, edge cases, and predicates")
    ledger: ProposalLedger = dspy.OutputField(desc="Adversarial challenge proposals ledger")


class OperationalReconciliationSignature(dspy.Signature):
    """Reconcile adversarial proposals into the primary baseline.
    Calibrate certainty score (0-5), account for operational metrics,
    and synthesize an unambiguous stakeholder verdict and actionable decision matrix.
    """

    inquiry: InquiryRecord = dspy.InputField(desc="Original normalized inquiry")
    baseline: PrimaryBaseline = dspy.InputField(desc="Primary baseline report")
    ledger: ProposalLedger = dspy.InputField(desc="Adversarial challenge proposals ledger")
    verdict: ReconciledVerdict = dspy.OutputField(desc="Final calibrated stakeholder verdict and reconciled diagram")
