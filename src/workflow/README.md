# Multi-Stage Code-Grounded Q&A Workflow (DSPy Implementation)

This package contains the architecture and implementation design for porting the OpenProse **4-stage code-grounded Q&A lifecycle** into a modular, typed, and optimizable **DSPy** pipeline.

For detailed background on the original OpenProse contracts and multi-run findings, see [`OpenProse-example.md`](OpenProse-example.md).

---

## 1. Conceptual Mapping: OpenProse to DSPy

| OpenProse Concept | DSPy Equivalent | Role in the Workflow |
| :--- | :--- | :--- |
| `*.prose.md` Contract | `dspy.Signature` + docstrings | Declares inputs, outputs, schemas, and prompting intent. |
| `Maintains` Schema | `pydantic.BaseModel` | Enforces strong typing for structured outputs. |
| `gateway` (Ingress) | `dspy.Predict(IngressSignature)` | Normalizes raw questions into structured inquiry records. |
| `responsibility` (Discovery) | `dspy.ReAct(BaselineSignature, tools=...)` | Explores repository, call graph, and git history using tool calls. |
| `responsibility` (Challenger) | `dspy.ReAct(AdversarialSignature, tools=...)` | Audits leaf side-effects backward and generates RFC proposals. |
| `responsibility` (Reconciler) | `dspy.ChainOfThought(ReconciliationSignature)` | Synthesizes proposals, verifies real data, and calibrates certainty. |
| Manual Prompt Iteration | `dspy.teleprompt.MIPROv2` / `BootstrapFewShot` | Automatically compiles and optimizes instructions against metrics. |

---

## 2. Core Workflow Architecture

```
                       ┌───────────────────────────────┐
                       │   Raw Stakeholder Question    │
                       └──────────────┬────────────────┘
                                      │
                                      ▼
                   ┌───────────────────────────────────────┐
                   │     Stage 1: Ingress Normalizer       │
                   │    (dspy.Predict[IngressSignature])   │
                   └──────────────────┬────────────────────┘
                                      │
                                      ▼
                   ┌───────────────────────────────────────┐
                   │    Stage 2: Primary Code Discovery    │ <───> [ Code Search Tools ]
                   │    (dspy.ReAct[BaselineSignature])    │       (rg, ctags, git log)
                   └──────────────────┬────────────────────┘
                                      │
                                      ▼
                   ┌───────────────────────────────────────┐
                   │   Stage 3: Adversarial RFC Auditor    │ <───> [ Guard Check Tools ]
                   │   (dspy.ReAct[AdversarialSignature])  │       (diagram compiler)
                   └──────────────────┬────────────────────┘
                                      │
                                      ▼
                   ┌───────────────────────────────────────┐
                   │ Stage 4: Reconciliation & Calibration │
                   │  (dspy.ChainOfThought[Reconciliation])│
                   └──────────────────┬────────────────────┘
                                      │
                                      ▼
                   ┌───────────────────────────────────────┐
                   │   Final Authoritative Verdict & SVG   │
                   └───────────────────────────────────────┘
```

---

## 3. Data Schemas (`pydantic.BaseModel`)

```python
from typing import List, Literal
from pydantic import BaseModel, Field

class InquiryRecord(BaseModel):
    inquiry_id: str
    stakeholder: str
    raw_question: str
    operational_context: str
    target_systems: List[str]
    stakes_severity: Literal["low", "medium", "high", "critical"]

class CallTree(BaseModel):
    entry_point: str
    intermediate_services: List[str]
    leaf_mutations: List[str]

class PrimaryBaseline(BaseModel):
    inquiry_id: str
    call_tree: CallTree
    git_intent_summary: str
    code_citations: List[str] = Field(description="Exact file:line references")
    baseline_summary: str
    diagram_d2: str

class Proposal(BaseModel):
    proposal_id: str
    category: Literal["vacuous-predicate", "race-condition", "inactive-bypass", "auth-blocker", "intent-gap"]
    target_symbol: str
    failing_scenario: str
    suggested_revision: str

class ProposalLedger(BaseModel):
    inquiry_id: str
    proposals: List[Proposal]

class ReconciledVerdict(BaseModel):
    inquiry_id: str
    adopted_proposals: List[str]
    rejected_proposals: List[str]
    certainty_score: int = Field(ge=0, le=5, description="Calibrated score 0-5")
    residual_risks: List[str]
    plain_language_verdict: str
    actionable_decision_matrix: str
    reconciled_diagram_d2: str
```

---

## 4. Signatures & Module Implementation

```python
import dspy

class IngressSignature(dspy.Signature):
    """Normalize and structure raw stakeholder technical questions."""
    raw_text: str = dspy.InputField(desc="Unstructured stakeholder question")
    inquiry: InquiryRecord = dspy.OutputField(desc="Structured inquiry record")


class BaselineDiscoverySignature(dspy.Signature):
    """Trace forward execution paths from triggers to leaf mutations with exact citations."""
    inquiry: InquiryRecord = dspy.InputField()
    code_context: str = dspy.InputField(desc="Relevant source files, symbols, and git history")
    baseline: PrimaryBaseline = dspy.OutputField(desc="Verifiable baseline investigation")


class AdversarialChallengeSignature(dspy.Signature):
    """Reverse-audit leaf mutations to uncover vacuous truths, race conditions, and bypassed guards.
    Strictly read-only on baseline; emit independent RFC proposals."""
    inquiry: InquiryRecord = dspy.InputField()
    baseline: PrimaryBaseline = dspy.InputField()
    code_context: str = dspy.InputField()
    ledger: ProposalLedger = dspy.OutputField(desc="Adversarial challenge proposals")


class ReconciliationSignature(dspy.Signature):
    """Reconcile adversarial proposals, calibrate certainty score, and deliver final stakeholder verdict."""
    inquiry: InquiryRecord = dspy.InputField()
    baseline: PrimaryBaseline = dspy.InputField()
    ledger: ProposalLedger = dspy.InputField()
    verdict: ReconciledVerdict = dspy.OutputField(desc="Authoritative calibrated verdict")


class CodeGroundingFlow(dspy.Module):
    def __init__(self, code_search_tools: list):
        super().__init__()
        self.ingress = dspy.Predict(IngressSignature)
        self.baseline_explorer = dspy.ReAct(BaselineDiscoverySignature, tools=code_search_tools)
        self.adversarial_auditor = dspy.ReAct(AdversarialChallengeSignature, tools=code_search_tools)
        self.reconciler = dspy.ChainOfThought(ReconciliationSignature)

    def forward(self, raw_question: str) -> dspy.Prediction:
        # Stage 1: Ingress
        inquiry_pred = self.ingress(raw_text=raw_question)
        inquiry = inquiry_pred.inquiry

        # Stage 2: Forward Baseline Discovery
        baseline_pred = self.baseline_explorer(inquiry=inquiry, code_context="Initial repository index")
        baseline = baseline_pred.baseline

        # Stage 3: Asymmetric Adversarial Audit
        challenge_pred = self.adversarial_auditor(
            inquiry=inquiry,
            baseline=baseline,
            code_context="Repository leaf mutations and boolean guards"
        )
        ledger = challenge_pred.ledger

        # Stage 4: Reconciliation & Calibration
        verdict_pred = self.reconciler(
            inquiry=inquiry,
            baseline=baseline,
            ledger=ledger
        )

        return dspy.Prediction(
            inquiry=inquiry,
            baseline=baseline,
            ledger=ledger,
            verdict=verdict_pred.verdict
        )
```

---

## 5. Automated Optimization with DSPy Teleprompters

Unlike manual prompt tweaks, DSPy can optimize instructions across all stages using validation metrics:

```python
from dspy.teleprompt import MIPROv2

def grounding_metric(example, prediction, trace=None) -> float:
    score = 0.0
    # 1. Exact citations present
    if prediction.baseline.code_citations and all(":" in c for c in prediction.baseline.code_citations):
        score += 0.3
    # 2. Meaningful adversarial proposals generated
    if len(prediction.ledger.proposals) > 0:
        score += 0.3
    # 3. Certainty score calibrated between 0 and 5
    if 0 <= prediction.verdict.certainty_score <= 5:
        score += 0.2
    # 4. Valid diagram syntax
    if prediction.verdict.reconciled_diagram_d2 and "{" in prediction.verdict.reconciled_diagram_d2:
        score += 0.2
    return score

# Compile optimized prompts and few-shot exemplars:
teleprompter = MIPROv2(metric=grounding_metric, auto="light")
# optimized_flow = teleprompter.compile(CodeGroundingFlow(tools), trainset=eval_dataset)
```
