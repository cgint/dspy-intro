# Mini SWE workflow (V0)

A contained, mission-driven DSPy workflow. DSPy investigates and proposes; Python owns path containment, edits, verification, trace evidence, and `DONE`/`BLOCKED`.

## Run a mission

```sh
.venv/bin/python src/mini-swe-wf/mini_swe_workflow.py \
  --mission-file src/workflow/mini-swe-missions/G12-event-subscribers.json \
  --repo-root "$PWD" \
  --run-root /absolute/external/mini-swe-runs \
  --model gemini/gemini-3.5-flash-lite \
  --reviewed-learning "optional reviewed input"
```

The JSON mission declares its goal, existing allow-listed paths, focused/full repo-venv pytest commands, expected RED text, and timeout. `--run-root` is required and must be outside the repository. Repeat `--reviewed-learning` to pass reviewed, human-supplied context to investigation.

## States and gates

`PREFLIGHT → baseline GREEN → investigation → typed proposal → test edit → RED → production edit → focused GREEN → full GREEN → review → DONE`.

Preflight rejects dirty allow-listed files and unsafe commands. Edits require existing regular files, allow-list containment, an exact preimage hash, and exactly one old-text match. RED requires pytest exit code `1` and the mission marker. Scope snapshots, revision checks, and targeted rollback block unsafe outcomes. The post-GREEN review receives frozen investigation, proposal, scoped diff, changed paths, and command evidence, but its judgment is non-gating.

## Durable evidence and limits

Each run writes a sanitized append-only JSONL trace under the external run root before DSPy calls. It records provenance, stage transitions, bounded tools, proposals, commands, scope checks, review input/hash, review, and terminal evidence.

V0 has no shell tool or model write tool, permits exact edits only to existing allow-listed files, and allows only repo-venv pytest commands. Tests inject DSPy sources; they do not call providers. This runner does not promote learner candidates or replace mechanical terminal gates with model judgment.
