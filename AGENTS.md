# AGENTS.md — dspy-intro

Durable repository rules. Current implementation and gap status must be checked
live; do not infer them from chat or old reports.

## Workflow parity status is mandatory

`src/workflow/GAP_ANALYSIS.md` is the canonical parity and gap-closing status map
for `src/workflow/` against the OpenProse source.

Every change that alters a G1–G17 status, accepted implementation, rejected
completion, evidence basis, or closure/reopening decision **must update
`src/workflow/GAP_ANALYSIS.md` in the same milestone**. A status-changing change
is incomplete while that file is stale.

Keep these states distinct:

- accepted/committed implementation;
- uncommitted or rejected experiments;
- historical claim-verification findings;
- open questions and evidence edges.

Before reporting or committing a status change, compare the map with the live
producer code/tests and the relevant frozen evidence. Never mark a gap closed
from green tests or runner `DONE` alone when an explicit mission criterion is
uncovered or contradicted.
