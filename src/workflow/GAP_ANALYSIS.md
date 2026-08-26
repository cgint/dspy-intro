# Gap Analysis — DSPy Port vs. OpenProse Original

**As-of:** 2026-08-26. Original comparison and §7 verification were read-only;
this refresh also records subsequent gap-closing execution status.
**Scope:** Compare the **accepted committed** `src/workflow/` port against the
OpenProse evidence-grounded Q&A methodology it claims to port. Historical runner
experiments are tracked separately in §1.1 and do not close a parity gap unless
accepted and committed.

This document is a **parity / coverage map**, not a verdict. It records what each
side covers, where the accepted port diverges, and the open questions that determine
whether a given gap is a *bug to fix* or an *expected simplification*.

## Source pointers (verify before relying)

- **DSPy port (this package):**
  - `src/workflow/schemas.py` — pydantic stage contracts
  - `src/workflow/signatures.py` — DSPy signatures (stage I/O)
  - `src/workflow/flow.py` — `CodeGroundingWorkflow` orchestration
  - `src/workflow/tools.py` — `search_code`, `read_code_slice`, `validate_d2_syntax`
  - `src/workflow/main.py` — CLI demo
  - `src/workflow/README.md` — port's own design doc
  - `tests/test_workflow.py` — unit tests
- **OpenProse original:**
  - `~/dev/concepts/OpenProse-testing/topics/question-answer-ground-by-code/README.md`
  - `.../openprose/stakeholder-inquiry-ingress.prose.md` (Stage 1, gateway)
  - `.../openprose/primary-evidence-discovery.prose.md` (Stage 2, responsibility)
  - `.../openprose/adversarial-challenge-review.prose.md` (Stage 3, responsibility)
  - `.../openprose/operational-grounding-reconciliation.prose.md` (Stage 4, responsibility)
  - `.../openprose/patterns/evidence-grounded-qa-lifecycle.prose.md` (composite pattern)
  - `.../openprose/tests/unreconciled-proposal-rejected.prose.md` (contract test)
  - `.../openprose/reactor.yml` (Reactor config)
  - `.../HOW_AND_WHERE_IT_HAPPENED.md` (real execution breakdown)
  - `.../2026-*.md` (raw session transcripts — ground truth of what actually ran)

---

## 0. The crucial reframe

"OpenProse" is two different things, and the DSPy port relates to them differently:

1. **OpenProse-as-methodology** — what *actually ran* the bidder investigation:
   **two human-orchestrated pi agent sessions** following the OpenProse conventions
   *by hand*, with repository files as the durable ledger. The DSPy port is a port
   of **this**, and it is a **simplification that drops the load-bearing parts**.
2. **OpenProse-as-runtime** — the Reactor / compiled nodes / world-model / receipts.
   The DSPy port is **not** a port of this at all. (`reactor.yml` in the repo is a
   separate small self-description demo — "what are the main terms in OpenProse?" —
   *not* the bidder lifecycle wiring. `Unverified:` that the bidder flow was never
   wired to a compiled Reactor; inferred from transcripts + `reactor.yml` contents.)

**Consequence:** the port faithfully ports the **contract shapes** and little else.
Every gap below should be read against *which* OpenProse it is measured against.

---

## 1. Coverage map (dimension by dimension)

| #   | Dimension                                                    | OpenProse original                                                                          | DSPy port                                                            | Gap?                       |
| --- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | -------------------------- |
| G1  | 4-stage lifecycle shape                                      | ✅ 4 contracts + composite pattern                                                          | ✅ 4 stages in`flow.py`                                            | —                         |
| G2  | Output durability                                            | ✅ Path-based (`.md`, `.d2`, `.svg`, ledger paths)                                    | ❌ In-memory pydantic objects, die with process                      | **High**             |
| G3  | Asymmetric challenger (independence)                         | ✅ Human-enforced across 2 independent sessions                                             | ❌ One sequential same-context LLM call                              | **High**             |
| G4  | Correction-ledger status loop                                | ✅`PENDING → REFLECTED` discipline on a file                                             | ❌ Not represented                                                   | **High**             |
| G5  | Git archaeology (structured)                                 | ✅`relevant_commits`, `author_intent_summary`, `intent_vs_code_gap`                   | ✅ Present/closed in commit `6c38160`                              | **Closed**           |
| G6  | Operational telemetry                                        | ✅`production_telemetry` (3 fields) + `ui_origin_endpoint`                              | ❌ Absent from schema                                                | **Medium**           |
| G7  | Rejection reasons                                            | ✅`rejected_proposals: [{id, rejection_reason}]`                                          | ⚠️ Flat list of IDs, no reasons                                    | **Medium**           |
| G8  | No-5/5-if-declared-risk invariant                            | ✅ Hard rule in Stage 4 contract                                                            | ⚠️ Protected mission prepared; model enforcement not yet implemented | **Medium — in progress** |
| G9  | Certainty threshold gate                                     | ✅`certainty_threshold` pattern param (default 4/5)                                       | ❌ No threshold concept                                              | **Medium**           |
| G10 | First-class failure state                                    | ✅`kind: test` → `status: "rejected-unaddressed-proposals"`, `certainty_score: null` | ❌ No "inconclusive" terminal outcome                                | **Medium**           |
| G11 | Deterministic diagram gate                                   | ✅`d2to.sh` exit-0, SVG embedded                                                          | ⚠️ Heuristic brace/regex check; real compile only if`d2` on PATH | **Medium**           |
| G12 | `call_tree.event_subscribers`                              | ✅ Present                                                                                  | ✅ Present/closed in commit `2c7539a`                              | **Closed**           |
| G13 | Reactive re-execution (standing truth, subscriptions, wakes) | ✅ Reactor:`Requires`/`Continuity`, world-model, receipts                               | ❌ Single synchronous`forward()` pass                              | **High (by design)** |
| G14 | Typed stage-boundary validation                              | ⚠️ Loose YAML`Maintains` blocks                                                         | ✅ Pydantic                                                          | **Port advantage**   |
| G15 | Prompt/few-shot optimization                                 | ❌ None                                                                                     | ✅`MIPROv2` / `BootstrapFewShot`                                 | **Port advantage**   |
| G16 | Importable library API + unit-test seams                     | ⚠️ Contract Markdown, Reactor harness                                                     | ✅`dspy.Module`, pytest                                            | **Port advantage**   |
| G17 | `diagram-syntax` proposal category                         | ❌ (5 categories)                                                                           | ✅ (adds 6th)                                                        | Port addition              |

---

## 1.1 Gap-closing execution status (2026-08-26)

This is a **current execution record**, separate from the accepted-commit coverage
map above and from the historical read-only verdicts in §7.

- **G12 closed/committed.** Commit `2c7539a` adds
  `CallTree.event_subscribers` and its test coverage in
  `src/workflow/schemas.py` and `tests/test_workflow.py`.
- **G8 selected; protected implementation infrastructure prepared.** The selected
  scope is only declared-risk consistency: a future `ReconciledVerdict` model-level
  invariant must reject `certainty_score=5` with one or more nonempty
  `residual_risks` using `G8_UNVERIFIED_CANNOT_BE_FIVE`, while preserving score 5
  with default/empty risks and score 4 with either risk state. The protected oracle
  `src/workflow/mini-swe-acceptance/G8-declared-risk-certainty.py` is outside future
  editable paths and baseline-proven at pytest rc1 with that marker; mission
  `src/workflow/mini-swe-missions/G8-declared-risk-certainty.json` is ready for a
  future protected run. This does not discover risks, establish risk completeness,
  add fields, or close G8. One no-retry protected run from HEAD
  `c4599b25679acb910be2d7709905399a09c491d6` reached runner-mechanical `DONE` at
  terminal sequence 55: trace
  `/Users/cgint/dev/dspy-intro-mini-swe-runs/G8-declared-risk-certainty-1787770519-7026f8ad/trace.jsonl`
  (`a99b8e1ff4d8df72314d9314aab5177bc86d9e8c35f44779b78b67d3dc4f05f9`);
  fixture RED rc1 with `G8_UNVERIFIED_CANNOT_BE_FIVE`; focused GREEN rc0; ordinary
  GREEN rc0 (51 tests); and protected oracle baseline rc1 then candidate rc0 (5
  passed). Only `tests/test_workflow.py` and `src/workflow/schemas.py` changed;
  post-review scope remained exact. The candidate bound oracle SHA
  `0f498025e9f520b41c33c95a63501a205d41d516a8cc9979a4f22e30a0274d0f` to
  scoped-diff SHA `1e2f58b8642b2f9a6f7eeb7881d1720f9f2490726bccfc11578ce5b096654ee6`.
  Independent acceptance rejected this runner-DONE candidate: scoped Ruff reports
  `E402 Module level import not at top of file` at `src/workflow/schemas.py:72`.
  The two-path candidate was retained unchanged for one separate external learner
  cycle. Learner manifest
  `src/mini-swe-wf-learner/runs/G8-lint-rejected-learner-20260826/manifest.json`
  (`74d7b58c82147d8d5b503c1ef4424405baeab84fea423e90e3b8215dff2a7c85`)
  completed `DONE` against terminal sequence 55 and the bound scoped-diff SHA, with
  zero candidates (`candidates.json` SHA
  `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`). No
  learning was promoted. The rejected candidate was then rolled back exactly from
  `src/workflow/schemas.py` and `tests/test_workflow.py` to HEAD
  `c4599b25679acb910be2d7709905399a09c491d6`; restored hashes are
  `10f501a895ecbb2c3ba76a8b293eee06ddc4a29e68d531bb7a70197cd3cd615c` and
  `8df7ebb694de16a428b37490f7ba86bab01312f4a55a5765d564587e1ba74515`.
  Focused tests (6), ordinary tests (50), and Ruff are green; the protected oracle
  correctly returned baseline rc1 with `G8_UNVERIFIED_CANNOT_BE_FIVE`. G8 remains
  open and uncommitted. A generic mission-scoped Ruff gate is now TDD-verified:
  only exact repo-venv `python -m ruff check --no-cache` argv with existing
  repo-relative Python-file operands is accepted; baseline and candidate results bind
  terminal authority, review evidence, scope checks, and rollback. The G8 mission is
  rebound to lint `src/workflow/schemas.py` and `tests/test_workflow.py`; product
  targets remain clean and no re-entry is authorized yet.
- **G5 closed/committed.** Commit `6c38160` replaces free-text
  `git_intent_summary` with required structured `git_archeology` in
  `src/workflow/schemas.py` and migrates both existing fixtures in
  `tests/test_workflow.py`.
  The reviewed generic lesson from the vacuous negative test is now promoted into
  proposal guidance: expected-error/rejection tests must fail when no error occurs.
  A generic protected held-out acceptance gate is TDD-verified: baseline acceptance
  pytest must exit 1 and candidate acceptance pytest must exit 0; protected artifact
  and scoped candidate-diff hashes are rebound immediately before terminal success.
  The real protected G5 oracle now exists and its mission baseline rc1 marker is
  TDD-proven. Historical Mini-SWE experiments before the accepted run produced
  uncommitted candidates that did not close the gap:
  - `329313348afc65c0a10bfcaccbecf963dec54e9e5895e18fc45431ba5b746cab`:
    `RED_NOT_CONFIRMED` because the test-only edit imported a not-yet-created
    symbol and pytest collection exited 2.
  - `4a586f94bddb152a53ba401b4d448bc9306ded32119b5f1bbb2ef349cf058acd`
    (terminal sequence 50): runner-mechanical `DONE` rejected because
    `relevant_commits` was optional; its separate learner emitted zero candidates.
  - `0ad12952cca108d38cf449f8b65cb0fab8dda11d9890098fc17453c1f0d83b17`
    (terminal sequence 49): runner-mechanical `DONE` rejected because its negative
    requiredness test used `try/except` without failing when no exception occurred;
    its separate learner emitted zero candidates.
  - `aa5f6779e9720343402758ebb15a1474648d3aa9c5d4a2aabd60c98cdbcddbcf`
    (terminal sequence 49): runner-mechanical `DONE` rejected because production
    required all three nested fields but generated tests omitted every
    omission/rejection assertion; learner manifest
    `G5-missing-negative-test-learner-20260826/manifest.json`
    (`42983cb7b67223c91a3eb8fb8864a92b3da6579d73c3491347d06acb02460cd6`)
    recorded zero candidates.
  - `adc65d2f4b3a9e1a1af7d51a654d5e2a7286cfc64f4445f6b8de8a7909d569ef`
    (terminal sequence 21): protected acceptance baseline succeeded (rc1 with
    `G5_HELD_OUT_GIT_ARCHEOLOGY`), but runner blocked `INVALID_PROPOSAL` before
    writes because `tests/test_workflow.py` appeared in both phases. Learner
    candidate `candidate_proposal_target_exclusivity` was independently rejected
    as redundant; no runner change was promoted.
  - `db92aacba0fd07ae0cb2eaab56cabf17d1c9f927c5a86072519f1e238dc88946`
    (terminal sequence 33): path partition was valid, but test-phase import of
    not-yet-created `GitArcheology` caused collection rc2 and `RED_NOT_CONFIRMED`;
    rollback was clean. Learner candidate `cand-001` was independently rejected as
    redundant and unsafe if strict rc1 RED were relaxed; no runner change was
    promoted.

The prior fixture-only run
`e9f460091c677f6da38dbf30d1c7cdf18ed01a66aa873cc92ed68f53c81784cc`
(terminal sequence 47) reached genuine fixture RED and focused GREEN but rolled
back with `FULL_GREEN_FAILED`: its ordinary-suite readiness test incorrectly
asserted the protected oracle's baseline rc1 after a valid candidate made it rc0.
Its candidate-only learner manifest
`src/mini-swe-wf-learner/runs/G5-heldout-full-green-learner-20260826/manifest.json`
(`067996a2ddb6d0cc1646c83cb013a3979aeb2f425d0aa06a8eb470e82550c793`)
recorded zero candidates and an empty target diff. The readiness regression now
asserts only stable mission, protected-artifact hash, and derived-command facts;
generic runner coverage retains rc1-to-rc0 acceptance semantics and every real run
still executes the actual baseline oracle.

**G5 is accepted and committed in `6c38160`.** Protected run
`/Users/cgint/dev/dspy-intro-mini-swe-runs/G5-structured-git-archeology-1787768928-3f27151f/trace.jsonl`
(`1eb21d6dd6c5db885c3c9fec1a6a7ed8cbebbd797d01f76726feaaa51121e827`)
ended `DONE` at sequence 55. It established genuine fixture RED (rc1 with
`git_intent_summary`), then focused and ordinary GREEN; the protected oracle moved
from baseline rc1 with `G5_HELD_OUT_GIT_ARCHEOLOGY` to candidate rc0 (4 passed).
Only `tests/test_workflow.py` and `src/workflow/schemas.py` changed. Its acceptance
binding paired oracle SHA
`37ed232fa41c9a274412771659157ec322afb5a877b8208fa9e8a71aec03017e` with
scoped-diff SHA `639ff73ba25aeefbdaa56b08b6ddfde7bd787eefcb9ff46cd22797a0e0783a11`;
post-review scope remained exact. The advisory review was `INCOMPLETE` because it
cannot decide `DONE`; deterministic terminal authority was unaffected. Independent
ordinary tests (49), oracle tests (4), Ruff, and public runtime requiredness checks
are green. The successful-trace external learner cycle is complete: manifest
`src/mini-swe-wf-learner/runs/G5-heldout-success-learner-20260826/manifest.json`
(`9ff3c4fa83fe81c7b32584d2a4ee7ab826a009abcd711fbc775d3998969ca712`)
records learner `DONE` against terminal sequence 55, captured-diff SHA
`639ff73ba25aeefbdaa56b08b6ddfde7bd787eefcb9ff46cd22797a0e0783a11`, and zero
candidates; no learning was promoted. The accepted implementation and evidence are
committed; G5 is closed in the coverage map.

---

## 2. What the port DROPS (load-bearing, proven in transcripts)

The real investigation's trustworthiness came from capabilities the port does not
provide:

- **Independent cross-check.** The human enforced role separation across *two
  separate sessions*. That independent session found the vacuous `allMatch` bug,
  the sandbox/access-failure/cancelled-account blockers, the bid-sync-source
  hard-delete block, and the `findExisting` post-delete ordering risk. A sequential
  same-context `ChainOfThought` call cannot reproduce that independence — this is
  the single biggest loss.
- **Durable correction ledger.** `BIDDER_LIFECYCLE_INVESTIGATION_corrections.md`
  with per-entry `PENDING`/`REFLECTED` status was the control mechanism. The port
  has no file-based ledger or status loop.
- **Structured git evidence.** `git log -S` / `git show 3da8d08781` produced a real
  intent-vs-code gap finding. The port now records structured archeology fields, but
  does not provide the corresponding git-history discovery mechanism.
- **Operational grounding.** The "1 of 48 active bidders" fact was **human-supplied**
  in the real run. The port has no telemetry field or mechanism at all.

---

## 3. What the port ADDS (genuine improvements)

- Strictly typed pydantic contracts (cleaner than OpenProse's loose YAML).
- Teleprompter optimization (`MIPROv2`) — OpenProse has no equivalent.
- A repeatable, scriptable `dspy.Module` you can `compile` (reproducibility as code).
- Unit-testable seams (`tests/test_workflow.py` mocks each stage).

---

## 4. What OpenProse itself does NOT fully cover (both directions honest)

- `production_telemetry` has **no tool/mechanism** — the field is schema-covered but
  capability-uncovered; in practice a human typed the number.
- Git archaeology is **shallow** (commit subject + `// TODO`; no PR descriptions or
  diff analysis).
- **No stopping criterion / budget** on the discovery trace (context-limit risk).
- **Reactor runtime is beta**: non-cryptographic receipts, no timestamp/actor audit,
  deferred facet inference.
- The "insufficient evidence from repo" terminal state exists **only by analogy**
  (the unaddressed-proposal test) — "I couldn't find the code" is not a first-class
  status either.

---

## 5. Unverified items (must resolve before acting)

- `Unverified:` The bidder flow was never run through a compiled Reactor (inferred
  from raw-session transcripts + `reactor.yml` contents, not proven).
- `Unverified:` The port "isn't actually code-grounded in default mode" — inferred
  from `main.py` (`use_react=False`) + no-tool `ChainOfThought`; the port has **not
  been run** to observe it.

---

## 6. Open question that determines whether a gap is a bug

**What is the port's intended goal?**

- **(a) Faithfully reproduce the OpenProse methodology** → the gaps in §1/§2 are real
  defects to fix (restore durability, real asymmetry, telemetry, git evidence, the
  no-5/5 rule as an enforced gate, a first-class inconclusive outcome).
- **(b) Teach / prototype DSPy** → the gaps are expected; the port is roughly fine for
  a lightweight prompt-chain variant + optimizer, and should simply be *named and
  positioned* accordingly (not called "code-grounded" in the strong sense).

This single fact flips the "is it worse?" answer from *yes* to *not a useful question*.

---

## 7. Historical verification findings (independent read-only cross-check, 2026-08-26)

**Method:** Each coverage-map row (§1, G1–G17) was handed to an independent read-only
worker that read the cited OpenProse and DSPy files and had to ground or falsify the
specific claim. No worker proposed fixes. Findings below are findings only.

### Verdicts

| #   | Claim                                | Verdict                                                             |
| --- | ------------------------------------ | ------------------------------------------------------------------- |
| G1  | 4-stage lifecycle shape              | ✅ VERIFIED                                                         |
| G2  | Output durability                    | ✅ VERIFIED                                                         |
| G3  | Asymmetric challenger (independence) | ⚠️ PARTIALLY-VERIFIED                                             |
| G4  | Correction-ledger status loop        | ❌ FALSIFIED (as-stated)                                            |
| G5  | Git archaeology (structured)         | ✅ VERIFIED                                                         |
| G6  | Operational telemetry                | ✅ VERIFIED                                                         |
| G7  | Rejection reasons                    | ✅ VERIFIED                                                         |
| G8  | No-5/5-if-unverified invariant       | ✅ VERIFIED                                                         |
| G9  | Certainty threshold gate             | ⚠️ PARTIALLY-VERIFIED                                             |
| G10 | First-class failure state            | ✅ VERIFIED                                                         |
| G11 | Deterministic diagram gate           | ✅ VERIFIED (report format non-standard; evidence supports)         |
| G12 | `call_tree.event_subscribers`      | ✅ VERIFIED                                                         |
| G13 | Reactive re-execution                | ⚠️ UNDETERMINED by worker → ✅ VERIFIED by supervisor (see note) |
| G14 | Typed stage-boundary validation      | ⚠️ PARTIALLY-VERIFIED                                             |
| G15 | Prompt/few-shot optimization         | ✅ VERIFIED (scoped to the 5 named contracts)                       |
| G16 | Importable library API + test seams  | ⚠️ PARTIALLY-VERIFIED                                             |
| G17 | `diagram-syntax` proposal category | ✅ VERIFIED                                                         |

**Tally:** 11 VERIFIED · 4 PARTIALLY-VERIFIED · 1 FALSIFIED (as-stated) · 1 UNDETERMINED
(supervisor-resolved to VERIFIED).

### Corrections to the claims (findings, not fixes)

- **G4 — evidence pointer was wrong, but the phenomenon is real.** The worker could not
  find a per-entry `PENDING → REFLECTED` loop or a "main `.md` + `.d2` + `.svg` must
  agree" rule in `HOW_AND_WHERE_IT_HAPPENED.md` (a summary). That language actually lives
  in the **raw Phase-4 session transcript** (`2026-08-14T10-15-58-779Z_…md`), which the
  supervisor read directly: it shows entries marked "PENDING MAIN-DOCUMENT REFLECTION" and
  the rule that an item is marked `REFLECTED` only after main `.md`, `.d2`, and `.svg`
  agree. So the gap claim holds; the cited source should be the transcript, not the summary.
- **G13 — worker pointed at a non-existent path.** It looked for
  `…/question-answer-ground-by-code/AGENTS.md` and `…/PROJECT_OVERVIEW.md` (ENOENT).
  The Reactor concepts actually live at the **repo root**
  `~/dev/concepts/OpenProse-testing/AGENTS.md` and `PROJECT_OVERVIEW.md`, which the
  supervisor read: they document standing-truth `responsibility`, `### Requires` /
  `### Continuity` wake semantics, world-model + receipt trail, and deterministic
  reconcile on wakes. `flow.py` is a single synchronous pass. ⇒ effectively VERIFIED.
- **G3 — partially verified.** Two independent, human-orchestrated sessions are confirmed
  (separate session IDs; the user explicitly mandated the ledger split). But the claim
  that independence was *not* runtime-enforced is **not established**: the contract says
  "strict read-only permissions" without naming the enforcement mechanism. The DSPy side
  (single sequential `forward()`, no separate agent/session/permission boundary) is
  confirmed.
- **G9 — partially verified.** The `certainty_threshold` parameter (default 4/5) is
  present in the pattern's Parameters, but the pattern's choreography returns
  `final_verdict` **unconditionally** — it does not express a conditional/withholding
  branch on the score. So OpenProse *declares* the gate but does not show it enforced in
  the pattern. The DSPy port has no threshold at all. The "gate" framing in the map is
  slightly stronger than what the pattern text demonstrates.
- **G14 — partially verified.** Pydantic `BaseModel` stage I/O with `Literal`/`Field`/
  `ge`/`le` is confirmed. But "enforces at **each** stage boundary" is not fully
  established from `schemas.py` alone — `flow.py` passes model objects through between
  stages without an explicit validation call shown at each handoff (validation happens
  at model construction, which the cited files do not show being re-invoked per stage).
- **G16 — partially verified.** The DSPy side is fully confirmed (clean `__init__` API,
  importable `dspy.Module`, mockable stages, pytest). The OpenProse side is confirmed
  only up to the Reactor YAML config; the broader "contract-Markdown + `npx` CLI, no
  importable library" contrast relied on an `AGENTS.md` at a path the worker could not
  find.

### No changes during the original read-only verification pass

During the original §7 pass, §1 remained the *claims-under-test* map: no claim wording
was altered and only findings were added (per the "findings only" constraint). The
subsequent execution refresh in §1.1 and the G12/G5 status corrections above are not
part of that historical pass; they distinguish current closure status from whether the
original comparison claims were grounded.

---

## Net

OpenProse covers the **methodology end-to-end** — including failure states,
durability, and reactive re-execution. The port covers the **prompt shapes** of that
methodology plus engineering niceties (typing, compiler, test seams). The port's
missing surface is exactly the surface that made the original investigation
trustworthy: the durable asymmetric ledger, structured telemetry, structured git
evidence, postcondition tests, and the no-5/5 gate as a rule rather than a
suggestion.

**Recommendation (pending §6 answer):** if the goal is (a), close the **High** gaps
first — independence, durability, and the no-5/5 / inconclusive outcome — because
those are what make the output *defensible to a PO*, and they are the parts a
prompt-only pipeline cannot fake.
