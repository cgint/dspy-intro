# OpenProse Code-Grounded Q&A Workflow: Analysis & Findings

## 1. Overview & Problem Context

In complex software codebases, high-stakes domain or architectural questions cannot rely on surface-level static summaries or speculative LLM answers. OpenProse implements a contract-driven multi-stage lifecycle to ground inquiries in verifiable source code, commit history, and runtime realities while subjecting findings to independent adversarial critique.

This document synthesizes the architecture, contract specifications, and telemetry findings from the reference OpenProse implementation in `$HOME/dev/concepts/OpenProse/topics/question-answer-ground-by-code/`.

---

## 2. The 4-Stage OpenProse Lifecycle

```
[ Stakeholder Inquiry Ingress ]
             │
             ▼
[ Primary Evidence Discovery ] ───────┐
             │                         │ (Read-only baseline)
             ▼                         ▼
 [ Adversarial Challenge ]    [ Corrections Proposal Ledger ]
             │                         │ (RFC proposals)
             └───────────┬─────────────┘
                         ▼
      [ Operational Grounding & Reconciliation ]
                         │
                         ▼
        [ Authoritative Stakeholder Verdict ]
```

### Stage 1: Stakeholder Inquiry Ingress (`stakeholder-inquiry-ingress.prose.md`)
- **Kind:** `gateway` (ingress-driven)
- **Goal:** Ingress and structure high-stakes questions into normalized records.
- **Contract Schema:**
  ```yaml
  inquiry_id: string
  stakeholder: string
  raw_question: string
  operational_context: string
  target_systems: [string]
  stakes_severity: "low" | "medium" | "high" | "critical"
  received_at: ISO8601
  ```

### Stage 2: Primary Evidence Discovery (`primary-evidence-discovery.prose.md`)
- **Kind:** `responsibility` (input-driven)
- **Goal:** Establish a reproducible baseline by tracing execution paths forward from API triggers to leaf mutations, historical intent, and event subscriptions.
- **Key Invariants & Rules:**
  - Must cite exact source file paths, symbols, and commit hashes.
  - Must evaluate positive execution paths and no-op/filter guards.
  - Must generate and compile a visual architecture/flow diagram (`.d2` or `.puml` -> `.svg`) with zero compilation errors.
- **Contract Schema:**
  ```yaml
  inquiry_id: string
  call_tree:
    entry_point: string
    intermediate_services: [string]
    event_publishers: [string]
    event_subscribers: [string]
    leaf_mutations: [string]
  git_archeology:
    relevant_commits: [string]
    author_intent_summary: string
    intent_vs_code_gap: string
  baseline_report_path: string
  visual_diagram_source_path: string
  visual_diagram_svg_path: string
  status: "baseline-authored"
  ```

### Stage 3: Adversarial Challenge Review (`adversarial-challenge-review.prose.md`)
- **Kind:** `responsibility` (input-driven)
- **Goal:** Subject the primary baseline to independent adversarial challenge.
- **Key Invariants & Rules:**
  - **Asymmetric Boundary:** The challenger has strict read-only permissions on the primary baseline and emits proposals exclusively to an RFC ledger.
  - **Target-Backward Tracing:** Audits reachability starting at external side-effects (mutations) and traces upstream through boolean predicates.
  - **Edge-Case Verification:** Actively searches for vacuous truths (e.g. `allMatch` on empty collections returning true), race conditions, nulls, and inactive bypasses.
  - **Diagram Audit:** Validates diagram syntax and verifies SVG generation without unescaped template placeholders.
- **Contract Schema:**
  ```yaml
  inquiry_id: string
  baseline_ref: string
  proposals:
    - proposal_id: string
      category: "vacuous-predicate" | "race-condition" | "inactive-bypass" | "auth-blocker" | "intent-gap"
      target_leaf_symbol: string
      failing_scenario_description: string
      code_evidence_path: string
      line_numbers: [number]
      suggested_revision: string
  proposal_ledger_path: string
  status: "proposals-ready"
  ```

### Stage 4: Operational Grounding & Reconciliation (`operational-grounding-reconciliation.prose.md`)
- **Kind:** `responsibility` (input-driven)
- **Goal:** Reconcile verified adversarial proposals, ground theoretical code logic with runtime metrics, calibrate certainty, and deliver an actionable verdict.
- **Key Invariants & Rules:**
  - **Explicit Ledger Accounting:** Every proposal must be either adopted with updated citations or rejected with documented technical rationale.
  - **Empirical Grounding:** Quantify risk with production distributions (e.g. database query stats, feature flags, UI exposures).
  - **Calibrated Uncertainty:** If unverified flows or race conditions remain, certainty score cannot be 5/5.
  - **Visual Asset Delivery:** Produces compiled SVG diagrams embedded directly in the final report.
- **Contract Schema:**
  ```yaml
  inquiry_id: string
  reconciled_report_path: string
  updated_diagram_path: string
  updated_diagram_svg_path: string
  adopted_proposals: [string]
  rejected_proposals:
    - proposal_id: string
      rejection_reason: string
  operational_grounding:
    ui_origin_endpoint: string
    production_telemetry:
      total_entities_evaluated: number
      entities_with_flag_enabled: number
      percentage_exposed_to_risk: number
  calibration:
    certainty_score: number # Scale 0 to 5
    explicit_residual_risks: [string]
    unverified_scenarios: [string]
  stakeholder_summary:
    plain_language_verdict: string
    actionable_decision_matrix: string
  status: "reconciled-and-calibrated"
  ```

---

## 3. Findings & Multi-Run Telemetry

From experimental runs comparing execution models (`gemini-3.6-flash` vs. `gemini-3.5-flash-lite`) and prompt evolutions (Run 1 vs. Run 2):

1. **Prompt Compilation & Cache Efficiency:**
   - Structured multi-stage prompts achieved high prompt cache hit rates (~74% cached tokens), significantly lowering cost and latency across multi-step investigations.
2. **Asymmetric Ledger Value:**
   - Separating the baseline researcher from the adversarial challenger prevented confirmation bias. The challenger consistently caught vacuous truths (e.g., stream predicates passing on empty lists) and inactive feature flag overrides.
3. **Deterministic Diagram Compilation Gate:**
   - Enforcing strict compilation steps (e.g., verifying `.d2` -> `.svg` exit code 0) prevented unescaped template syntax errors (`{node}`) from polluting reports.
4. **Model Performance:**
   - `gemini-3.6-flash` exhibited deeper multi-step code graph reasoning, while `gemini-3.5-flash-lite` proved highly effective for focused classification and ingress normalization.
