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

`PREFLIGHT → baseline GREEN → held-out baseline RED → investigation → typed proposal → test edit → RED → production edit → focused GREEN → full GREEN → held-out candidate GREEN → review → DONE`.

Preflight rejects dirty allow-listed files and unsafe commands. Every observational validation command (baseline/focused/full pytest, optional Ruff, and protected acceptance) is immutable: Python snapshots before and after it, records the ordinary `command` result first, then a `command_mutation` event when any nonignored workspace entry changes, and blocks as `COMMAND_MUTATION` regardless of exit code or timeout. Baseline mutations stop before sources without restoration; candidate mutations restore only applied allow-listed preimages and leave unowned effects visible in trace evidence. An optional `ruff_check_argv` accepts only the exact repo-venv command shape `python -m ruff check --no-cache` followed by existing repo-relative `.py` files; it runs before provider/acceptance baseline work and after ordinary GREEN before protected acceptance, blocking or rolling back on denial, timeout, nonzero exit, or mutation. An optional paired `acceptance_test_path` / `acceptance_expected_red_text` config derives a fixed repo-venv pytest command; its existing regular artifact must be outside editable paths and cannot be a symlink or hardlink alias. Its baseline must exit `1` with the marker, and its candidate must exit `0`; timeout, mutation, invalid baseline infrastructure, or candidate failure blocks (and candidate failure rolls back edits). Edits require existing regular files, allow-list containment, an exact preimage hash, and exactly one old-text match. RED requires pytest exit code `1` and the mission marker. For expected-error test edits, proposals must place that exact marker on the failing path—prefer `pytest.raises(match=Expected RED text)` or an explicit failure message—rather than an assertion after the context manager, which cannot run when no exception is raised. Every introduced identifier needs a matching existing or proposed module-top import before use; `from package import name` uses `name`, while `import package` uses `package.name`. Reuse existing module-top bindings; do not add a function-local import for a name already used in that function, because Python makes it local throughout the scope. Any pytest-provided name must be declared as a test-function fixture parameter or explicitly constructed; never use an undeclared fixture name. Scope snapshots, revision checks, and targeted rollback block unsafe outcomes. After acceptance passes, the runner records the scoped candidate-diff and protected-artifact hashes, supplies that binding to review, then recomputes both immediately before terminal success; review-time drift blocks as stale acceptance and rolls back editable paths. The post-GREEN review receives frozen investigation, proposal, scoped diff, changed paths, ordinary command evidence, both acceptance results, and the binding, but its judgment is non-gating.

## Durable evidence and limits

Each run writes a sanitized append-only JSONL trace under the external run root before DSPy calls. It records provenance, stage transitions, bounded tools, proposals, commands, scope checks, review input/hash, review, and terminal evidence.

V0 has no shell tool or model write tool, permits exact edits only to existing allow-listed files, and allows only repo-venv pytest commands. Tests inject DSPy sources; they do not call providers. This runner does not promote learner candidates or replace mechanical terminal gates with model judgment.
