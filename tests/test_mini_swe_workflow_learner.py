"""Contract tests for the separate, read-only mini-SWE learner."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

LEARNER_PATH = Path(__file__).resolve().parents[1] / "src" / "mini-swe-wf-learner" / "mini_swe_workflow_learner.py"


def learner_module():
    spec = importlib.util.spec_from_file_location("mini_swe_workflow_learner", LEARNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inputs(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join(json.dumps({"sequence": n, "event": event, "payload": {"state": "DONE"} if event == "terminal" else {}}) for n, event in [(1, "preflight"), (2, "terminal")]) + "\n")
    mission = tmp_path / "mission.json"
    mission.write_text('{"mission_id": "G12", "goal": "add field"}')
    runner = tmp_path / "runner.py"
    runner.write_text("# frozen runner\n")
    diff = tmp_path / "g12.diff"
    diff.write_text("diff --git a/a b/a\n")
    return trace, mission, runner, diff


def request(module, tmp_path):
    trace, mission, runner, diff = inputs(tmp_path)
    return module.LearnerRequest(
        trace_path=trace, trace_sha256=module.sha256_file(trace),
        mission_path=mission, mission_sha256=module.sha256_file(mission),
        runner_path=runner, runner_sha256=module.sha256_file(runner),
        diff_path=diff, diff_sha256=module.sha256_file(diff),
        model="gemini/gemini-3.5-flash-lite", output_dir=tmp_path / "out",
    )


def test_preflight_accepts_done_or_reasoned_blocked_and_rejects_invalid_terminal(tmp_path):
    learner = learner_module()
    req = request(learner, tmp_path)
    evidence = learner.preflight(req)
    assert evidence.event_numbers == [1, 2]
    assert evidence.terminal_sequence == 2
    assert evidence.terminal_state == "DONE"
    assert evidence.terminal_reason is None

    req.trace_path.write_text(json.dumps({"sequence": 1, "event": "terminal", "payload": {"state": "BLOCKED", "reason": "EDIT_VALIDATION_FAILED"}}) + "\n")
    blocked = learner.preflight(req.model_copy(update={"trace_sha256": learner.sha256_file(req.trace_path)}))
    assert blocked.terminal_state == "BLOCKED"
    assert blocked.terminal_reason == "EDIT_VALIDATION_FAILED"

    with pytest.raises(learner.PreflightError, match="hash"):
        learner.preflight(req.model_copy(update={"trace_sha256": "0" * 64}))
    req.trace_path.write_text('{"sequence": 1, "event": "terminal", "payload": {"state": "BLOCKED"}}\n')
    with pytest.raises(learner.PreflightError, match="reason"):
        learner.preflight(req.model_copy(update={"trace_sha256": learner.sha256_file(req.trace_path)}))
    req.trace_path.write_text('{"sequence": 1, "event": "terminal"}\n')
    with pytest.raises(learner.PreflightError, match="DONE or BLOCKED"):
        learner.preflight(req.model_copy(update={"trace_sha256": learner.sha256_file(req.trace_path)}))
    req.trace_path.write_text('{bad json}\n')
    with pytest.raises(learner.PreflightError, match="malformed"):
        learner.preflight(req.model_copy(update={"trace_sha256": learner.sha256_file(req.trace_path)}))


def test_load_allowed_paths_and_capture_diff_use_verified_mission_and_repo_root(tmp_path):
    learner = learner_module()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess = pytest.importorskip("subprocess")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    target = repo / "nested" / "target.py"
    target.parent.mkdir()
    target.write_text("before\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base"], check=True)
    target.write_text("after\n")
    mission = tmp_path / "mission.json"
    mission.write_text(json.dumps({"mission_id": "generic", "allowed_paths": ["nested/target.py"]}))

    assert learner.load_allowed_paths(mission, learner.sha256_file(mission)) == ["nested/target.py"]
    diff = tmp_path / "target.diff"
    learner.capture_target_diff(repo, ["nested/target.py"], diff)
    assert "nested/target.py" in diff.read_text()
    assert learner.sha256_file(diff) == learner.sha256_file(diff)


def test_candidate_validation_requires_candidate_status_and_known_atomic_event_references():
    learner = learner_module()
    candidate = {
        "id": "L-1", "statement": "Pass actual diff to review.", "rationale": "review evidence lacks it.",
        "evidence_event_numbers": [2], "applies_to": "later #1 runs", "invalidation": "review input includes diff",
        "proposed_runner_change": "include diff", "status": "candidate",
    }
    assert learner.validate_candidates([candidate], {1, 2})[0].status == "candidate"
    with pytest.raises(learner.CandidateValidationError):
        learner.validate_candidates([{**candidate, "evidence_event_numbers": [3]}], {1, 2})
    with pytest.raises(learner.CandidateValidationError):
        learner.validate_candidates([{**candidate, "status": "promoted"}], {1, 2})


def test_run_persists_provenance_and_never_changes_runner_or_target_inputs(tmp_path):
    learner = learner_module()
    req = request(learner, tmp_path)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (req.trace_path, req.mission_path, req.runner_path, req.diff_path)}
    result = learner.run(req, candidate_source=lambda _evidence: [{
        "id": "L-1", "statement": "Pass observable review inputs.", "rationale": "Evidence is incomplete.",
        "evidence_event_numbers": [1], "applies_to": "later #1 runs", "invalidation": "review inputs are recorded",
        "proposed_runner_change": "record review inputs", "status": "candidate",
    }])
    assert result.status == "DONE"
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["model"] == "gemini/gemini-3.5-flash-lite"
    assert manifest["terminal_state"] == "DONE"
    assert manifest["terminal_reason"] is None
    assert manifest["content_provenance"]["candidate_source"] == "injected"
    assert json.loads(result.candidates_path.read_text())[0]["status"] == "candidate"
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}


def test_run_records_blocked_terminal_provenance(tmp_path):
    learner = learner_module()
    req = request(learner, tmp_path)
    req.trace_path.write_text(json.dumps({"sequence": 1, "event": "terminal", "payload": {"state": "BLOCKED", "reason": "EDIT_VALIDATION_FAILED"}}) + "\n")
    req = req.model_copy(update={"trace_sha256": learner.sha256_file(req.trace_path)})
    result = learner.run(req, candidate_source=lambda _evidence: [])
    manifest = json.loads(result.manifest_path.read_text())
    assert result.status == "DONE"
    assert manifest["terminal_state"] == "BLOCKED"
    assert manifest["terminal_reason"] == "EDIT_VALIDATION_FAILED"
