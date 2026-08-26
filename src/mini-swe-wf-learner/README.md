# Mini SWE learner

This is a **separate, candidate-only** system for learning from frozen mini-SWE evidence. It does not import, invoke, or edit the runner.

Its inputs are hash-verified frozen artifacts: a terminal-`DONE` or reasoned terminal-`BLOCKED` trace, mission JSON, runner source, and scoped diff. Its outputs are a candidates JSON file and a manifest with provenance hashes, terminal state, terminal reason, and candidate hash.

Run the CLI with `--repo-root`. It hash-verifies the mission before deriving its `allowed_paths`, then captures `git diff HEAD -- <allowed paths>` as `target.diff`. An empty `target.diff` after rollback is valid evidence and is hash-verified as an input. Candidate generation compares proposed runner changes with the supplied runner source, excludes already-implemented or redundant advice, and cites only trace sequence numbers.

Candidates are unpromoted observations, not instructions or code changes. This learner cannot promote a candidate, edit runner/workflow code, choose a next mission, or certify a terminal outcome. A separate human review decides whether a candidate is rejected, accepted for a bounded experiment, or later invalidated.

`runs/` is intentionally local-only. It stores frozen captures and review records for inspection but is ignored by Git.
