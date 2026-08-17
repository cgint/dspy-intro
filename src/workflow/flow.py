"""End-to-End Code-Grounded Q&A Workflow in DSPy.

Orchestrates the 4-stage lifecycle:
1. Stakeholder Ingress (Normalization)
2. Primary Evidence Discovery (Call tree & citations)
3. Adversarial Challenge Review (Target-backward RFC auditing)
4. Operational Reconciliation (Calibrated stakeholder verdict)
"""

from typing import List, Optional, Callable
import dspy

from workflow.schemas import (
    InquiryRecord,
    PrimaryBaseline,
    ProposalLedger,
    ReconciledVerdict,
)
from workflow.signatures import (
    StakeholderIngressSignature,
    PrimaryEvidenceDiscoverySignature,
    AdversarialChallengeSignature,
    OperationalReconciliationSignature,
)
from workflow.tools import search_code, read_code_slice, validate_d2_syntax


class CodeGroundingWorkflow(dspy.Module):
    """Declarative 4-stage DSPy module for verifiable code-grounded Q&A."""

    def __init__(self, tools: Optional[List[Callable]] = None, use_react: bool = False):
        super().__init__()
        self.use_react = use_react
        active_tools = tools if tools is not None else [search_code, read_code_slice, validate_d2_syntax]

        # Stage 1: Ingress & Normalization
        self.ingress = dspy.Predict(StakeholderIngressSignature)

        # Stage 2: Forward Evidence Discovery
        if self.use_react:
            self.discovery = dspy.ReAct(PrimaryEvidenceDiscoverySignature, tools=active_tools)
            self.adversarial_reviewer = dspy.ReAct(AdversarialChallengeSignature, tools=active_tools)
        else:
            self.discovery = dspy.ChainOfThought(PrimaryEvidenceDiscoverySignature)
            self.adversarial_reviewer = dspy.ChainOfThought(AdversarialChallengeSignature)

        # Stage 4: Reconciliation & Calibrated Verdict
        self.reconciler = dspy.ChainOfThought(OperationalReconciliationSignature)

    def forward(
        self,
        raw_question: str,
        operational_context: str = "Standard investigation request",
        code_context: str = ""
    ) -> dspy.Prediction:
        """Execute the 4-stage pipeline sequentially."""

        # 1. Ingress
        ingress_pred = self.ingress(
            raw_question=raw_question,
            operational_context=operational_context
        )
        inquiry: InquiryRecord = ingress_pred.inquiry

        # 2. Primary Evidence Discovery (Forward tracing)
        effective_code_context = code_context if code_context else f"Repository search for inquiry {inquiry.inquiry_id}"
        discovery_pred = self.discovery(
            inquiry=inquiry,
            code_context=effective_code_context
        )
        baseline: PrimaryBaseline = discovery_pred.baseline

        # 3. Adversarial Challenge Review (Target-backward tracing)
        adversarial_pred = self.adversarial_reviewer(
            inquiry=inquiry,
            baseline=baseline,
            code_context=effective_code_context
        )
        ledger: ProposalLedger = adversarial_pred.ledger

        # 4. Operational Reconciliation & Calibrated Verdict
        reconcile_pred = self.reconciler(
            inquiry=inquiry,
            baseline=baseline,
            ledger=ledger
        )
        verdict: ReconciledVerdict = reconcile_pred.verdict

        return dspy.Prediction(
            inquiry=inquiry,
            baseline=baseline,
            ledger=ledger,
            verdict=verdict
        )
