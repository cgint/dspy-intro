# Mini SWE learner

This is a **separate, candidate-only** system for learning from frozen mini-SWE evidence. It does not import, invoke, or edit the runner.

Its inputs are hash-verified frozen artifacts: a terminal-`DONE` trace, mission JSON, runner source, and scoped diff. Its outputs are a candidates JSON file and a manifest with provenance hashes.

Candidates are unpromoted observations, not instructions or code changes. This learner cannot promote a candidate, edit runner/workflow code, choose a next mission, or certify a terminal outcome. A separate human review decides whether a candidate is rejected, accepted for a bounded experiment, or later invalidated.

`runs/` is intentionally local-only. It stores frozen captures and review records for inspection but is ignored by Git.
