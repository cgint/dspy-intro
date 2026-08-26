"""Mechanical-contract tests for the contained mini SWE DSPy workflow."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mini-swe-wf"
    / "mini_swe_workflow.py"
)


def runner_module():
    spec = importlib.util.spec_from_file_location("mini_swe_workflow", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def make_repo(tmp_path: Path, check_source: str | None = None) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".venv" / "bin").mkdir(parents=True)
    repo_python = repo / ".venv" / "bin" / "python"
    repo_python.write_text(f"#!/bin/sh\nexec '{sys.executable}' \"$@\"\n")
    repo_python.chmod(0o755)
    (repo / "source.py").write_text("class CallTree:\n    event_publishers = []\n")
    (repo / "check.py").write_text(
        check_source
        or "def test_baseline():\n"
        "    assert True\n"
    )
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "mini-swe-test")
    git(repo, "add", "source.py", "check.py")
    git(repo, "commit", "-qm", "baseline")
    return repo


def make_mission(runner, repo: Path, expected_red_text: str = "EVENT_SUBSCRIBERS_MISSING"):
    python = str(repo / ".venv" / "bin" / "python")
    command = [python, "-m", "pytest", "-p", "no:cacheprovider", "check.py", "-q"]
    return runner.Mission(
        mission_id="mission-test",
        goal="restore event subscribers",
        repo_root=str(repo),
        allowed_paths=["check.py", "source.py"],
        focused_test_argv=command,
        full_test_argv=command,
        expected_red_text=expected_red_text,
        expected_red_returncode=1,
        max_attempts=1,
        mission_provenance="test-mission.json",
    )


def successful_proposal(runner, repo: Path):
    test_before = (repo / "check.py").read_text()
    source_before = (repo / "source.py").read_text()
    return runner.ChangeProposal(
        test_edits=[
            runner.ExactTextEdit(
                path="check.py",
                expected_sha256=runner.sha256_text(test_before),
                old_text=test_before,
                new_text=(
                    "from pathlib import Path\n\n"
                    "def test_event_subscribers():\n"
                    "    assert 'event_subscribers' in Path('source.py').read_text(), "
                    "'EVENT_SUBSCRIBERS_MISSING'\n"
                ),
            )
        ],
        production_edits=[
            runner.ExactTextEdit(
                path="source.py",
                expected_sha256=runner.sha256_text(source_before),
                old_text="    event_publishers = []\n",
                new_text="    event_publishers = []\n    event_subscribers = []\n",
            )
        ],
    )


def sources(runner, proposal, concerns: list[str] | None = None):
    return {
        "investigation_source": lambda _mission: runner.Investigation(
            facts=["CallTree lacks event_subscribers"], category="schema-parity"
        ),
        "proposal_source": lambda _mission, _investigation: proposal,
        "review_source": lambda _mission, _investigation, _proposal, _evidence: runner.PostGreenReview(
            judgment="fresh-context review completed",
            concerns=concerns or [],
            evidence_requests=[],
        ),
    }


def test_repo_root_containment_rejects_escape_and_symlink(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("secret = 1\n")
    (repo / "linked.py").symlink_to(outside)

    assert runner.resolve_repo_file(repo, "source.py") == (repo / "source.py").resolve()
    with pytest.raises(runner.ScopeViolation):
        runner.resolve_repo_file(repo, "../outside.py")
    with pytest.raises(runner.ScopeViolation):
        runner.resolve_repo_file(repo, str(outside))
    with pytest.raises(runner.ScopeViolation):
        runner.resolve_repo_file(repo, "linked.py")


def test_bounded_read_and_search_exclude_sensitive_and_runtime_paths(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    (repo / "visible.py").write_text("needle\n" * 20)
    (repo / ".env").write_text("needle=secret\n")
    (repo / "credentials.txt").write_text("needle=secret\n")
    (repo / ".mini-swe-runs").mkdir()
    (repo / ".mini-swe-runs" / "trace.jsonl").write_text("needle\n")

    read = runner.bounded_read(repo, "visible.py", max_bytes=12)
    assert read.truncated is True
    search = runner.bounded_search(repo, "needle", max_results=10, max_bytes=100)
    assert {match.path for match in search.matches} == {"visible.py"}


def test_exact_hash_edit_applies_once_and_rejects_stale_content(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    edit = runner.ExactTextEdit(
        path="source.py",
        expected_sha256=runner.sha256_text("class CallTree:\n    event_publishers = []\n"),
        old_text="    event_publishers = []\n",
        new_text="    event_publishers = []\n    event_subscribers = []\n",
    )
    applied = runner.apply_exact_text_edit(repo, ["source.py"], edit)
    assert applied.after_sha256 == runner.sha256_text((repo / "source.py").read_text())
    with pytest.raises(runner.EditValidationError):
        runner.apply_exact_text_edit(repo, ["source.py"], edit)


def test_append_only_ledger_redacts_credentials_and_project_identifiers(tmp_path):
    runner = runner_module()
    path = tmp_path / "runs" / "trace.jsonl"
    ledger = runner.EventLedger(path)
    ledger.append(
        "preflight",
        {
            "token": "shh",
            "message": "GEMINI_API_KEY=abc123",
            "provider_error": "projects/gen-lang-client-0910640178/locations/europe-west1",
        },
    )
    ledger.append("transition", {"authorization": "Bearer very-secret"})
    lines = path.read_text().splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["preflight", "transition"]
    persisted = path.read_text()
    assert "abc123" not in persisted
    assert "very-secret" not in persisted
    assert "gen-lang-client-0910640178" not in persisted
    assert "[REDACTED]" in persisted


def test_command_validation_allows_only_repo_venv_pytest_and_timeout(tmp_path, monkeypatch):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission = make_mission(runner, repo)
    allowed = [mission.focused_test_argv, mission.full_test_argv]

    ok = runner.run_allowed_command(mission.focused_test_argv, repo, allowed, 1, repo)
    assert ok.allowed is True and ok.returncode == 0
    denied = runner.run_allowed_command([sys.executable, "-c", "print('no')"], repo, allowed, 1, repo)
    assert denied.allowed is False

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(mission.focused_test_argv, 0.01)

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    timed_out = runner.run_allowed_command(mission.focused_test_argv, repo, allowed, 0.01, repo)
    assert timed_out.allowed is True and timed_out.timed_out is True


def test_terminal_gate_requires_only_mechanical_evidence():
    runner = runner_module()
    evidence = runner.GateEvidence(
        preflight_ok=True,
        live_proposal_valid=True,
        baseline_green=True,
        red_confirmed=True,
        production_applied=True,
        focused_green=True,
        full_green=True,
        scope_ok=True,
        revision_stable=True,
        rollback_performed=False,
    )
    assert runner.decide_terminal(evidence).status == runner.TerminalStatus.DONE
    assert runner.decide_terminal(evidence.model_copy(update={"baseline_green": False})).reason == "BASELINE_FAILED"
    assert runner.decide_terminal(evidence.model_copy(update={"production_applied": False})).reason == "PRODUCTION_NOT_APPLIED"


def test_load_mission_file_binds_external_repo_model_and_reviewed_learning(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission_file = tmp_path / "mission.json"
    mission_file.write_text(
        json.dumps(
            {
                "mission_id": "generic",
                "goal": "one bounded goal",
                "allowed_paths": ["source.py"],
                "focused_test_argv": ["PLACEHOLDER"],
                "full_test_argv": ["PLACEHOLDER"],
                "expected_red_text": "marker",
                "expected_red_returncode": 1,
            }
        )
    )
    mission = runner.load_mission_file(
        mission_file,
        repo_root=repo,
        model_name="gemini/gemini-3.5-flash-lite",
        reviewed_learning=["review-1"],
    )
    assert mission.repo_root == str(repo.resolve())
    assert mission.model_name == "gemini/gemini-3.5-flash-lite"
    assert mission.reviewed_learning == ["review-1"]
    assert mission.mission_provenance == str(mission_file.resolve())


def test_runner_blocks_dirty_allowed_file_before_any_source(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission = make_mission(runner, repo)
    (repo / "source.py").write_text("dirty\n")
    called = False

    def investigation(_mission):
        nonlocal called
        called = True
        return runner.Investigation(facts=[], category="unused")

    result = runner.MiniSWEWorkflow(
        mission=mission,
        run_root=tmp_path / "runs",
        investigation_source=investigation,
    ).run()
    assert result.status == runner.TerminalStatus.BLOCKED
    assert result.reason == "DIRTY_ALLOWED_PATHS"
    assert called is False


def test_runner_executes_baseline_red_green_and_nonblocking_review(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission = make_mission(runner, repo)
    result = runner.MiniSWEWorkflow(
        mission=mission,
        run_root=tmp_path / "runs",
        **sources(runner, successful_proposal(runner, repo), concerns=["consider documenting the field"]),
    ).run()

    assert result.status == runner.TerminalStatus.DONE
    assert "event_subscribers" in (repo / "source.py").read_text()
    events = [json.loads(line) for line in result.ledger_path.read_text().splitlines()]
    starts = {event["payload"]["stage"] for event in events if event["event"] == "stage_start"}
    ends = {event["payload"]["stage"] for event in events if event["event"] == "stage_end"}
    required = {"baseline", "investigation", "proposal", "test_edit", "red", "production_edit", "focused_green", "full_green", "review"}
    assert required <= starts <= ends
    assert any(event["event"] == "review" and event["payload"]["output"]["concerns"] for event in events)
    assert events[0]["event"] == "preflight"
    assert events[0]["payload"]["run_root"] == str((tmp_path / "runs").resolve())
    assert events[-1]["event"] == "terminal"


def test_scoped_git_diff_includes_staged_changes(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    source = repo / "source.py"
    source.write_text("staged review evidence\n")
    git(repo, "add", "source.py")

    diff = runner._scoped_git_diff(repo, ["source.py"])

    assert "diff --git a/source.py b/source.py" in diff
    assert "+staged review evidence" in diff


def test_review_receives_frozen_investigation_proposal_diff_and_command_evidence(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission = make_mission(runner, repo)
    proposal = successful_proposal(runner, repo)
    seen: dict[str, object] = {}

    def review(_mission, investigation, received_proposal, evidence):
        payload = json.loads(evidence)
        seen.update(payload)
        assert investigation.facts == ["CallTree is a schema"]
        assert received_proposal.model_dump() == proposal.model_dump()
        assert payload["investigation"]["facts"] == ["CallTree is a schema"]
        assert payload["proposal"] == proposal.model_dump()
        assert payload["changed_paths"] == ["check.py", "source.py"]
        assert "diff --git a/check.py b/check.py" in payload["scoped_git_diff"]
        assert "diff --git a/source.py b/source.py" in payload["scoped_git_diff"]
        assert payload["focused_command"]["returncode"] == 0
        assert payload["full_command"]["returncode"] == 0
        return runner.PostGreenReview(judgment="reviewed", concerns=[], evidence_requests=[])

    result = runner.MiniSWEWorkflow(
        mission=mission,
        run_root=tmp_path / "runs",
        investigation_source=lambda _mission: runner.Investigation(facts=["CallTree is a schema"], category="schema-parity"),
        proposal_source=lambda _mission, _investigation: proposal,
        review_source=review,
    ).run()

    assert result.status == runner.TerminalStatus.DONE
    events = [json.loads(line) for line in result.ledger_path.read_text().splitlines()]
    review_input = next(event for event in events if event["event"] == "review_input")
    assert review_input["payload"]["sha256"] == runner.sha256_text(json.dumps(seen, sort_keys=True))
    assert review_input["payload"]["input"] == seen


def test_runner_rejects_returncode_two_even_when_expected_text_is_present(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission = make_mission(runner, repo)
    test_before = (repo / "check.py").read_text()
    proposal = runner.ChangeProposal(
        test_edits=[
            runner.ExactTextEdit(
                path="check.py",
                expected_sha256=runner.sha256_text(test_before),
                old_text=test_before,
                new_text="raise RuntimeError('EVENT_SUBSCRIBERS_MISSING')\n",
            )
        ],
        production_edits=[],
    )
    result = runner.MiniSWEWorkflow(
        mission=mission,
        run_root=tmp_path / "runs",
        **sources(runner, proposal),
    ).run()
    assert result.status == runner.TerminalStatus.BLOCKED
    assert result.reason == "RED_NOT_CONFIRMED"


def test_runner_preserves_mode_and_rolls_back_only_modified_path(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission = make_mission(runner, repo)
    check = repo / "check.py"
    check.chmod(0o744)
    git(repo, "add", "check.py")
    git(repo, "commit", "-qm", "make check executable")
    original_test = check.read_text()
    original_source = (repo / "source.py").read_text()
    proposal = runner.ChangeProposal(
        test_edits=[
            runner.ExactTextEdit(
                path="check.py",
                expected_sha256=runner.sha256_text(original_test),
                old_text=original_test,
                new_text="def test_wrong_reason():\n    assert False, 'WRONG_REASON'\n",
            )
        ],
        production_edits=[],
    )
    result = runner.MiniSWEWorkflow(
        mission=mission,
        run_root=tmp_path / "runs",
        **sources(runner, proposal),
    ).run()
    assert result.reason == "RED_NOT_CONFIRMED"
    assert check.read_text() == original_test
    assert (repo / "source.py").read_text() == original_source
    assert check.stat().st_mode & 0o777 == 0o744
    rollback = next(json.loads(line) for line in result.ledger_path.read_text().splitlines() if json.loads(line)["event"] == "rollback")
    assert rollback["payload"]["paths"] == ["check.py"]


def test_runner_blocks_malformed_proposal_without_fallback(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission = make_mission(runner, repo)
    result = runner.MiniSWEWorkflow(
        mission=mission,
        run_root=tmp_path / "runs",
        investigation_source=lambda _mission: runner.Investigation(facts=[], category="schema-parity"),
        proposal_source=lambda _mission, _investigation: {"not": "a proposal"},
        review_source=lambda *_args: pytest.fail("review must not run"),
    ).run()
    assert result.status == runner.TerminalStatus.BLOCKED
    assert result.reason == "INVALID_PROPOSAL"


def test_runner_blocks_out_of_scope_command_mutation(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission = make_mission(runner, repo)
    test_before = (repo / "check.py").read_text()
    source_before = (repo / "source.py").read_text()
    proposal = runner.ChangeProposal(
        test_edits=[
            runner.ExactTextEdit(
                path="check.py",
                expected_sha256=runner.sha256_text(test_before),
                old_text=test_before,
                new_text=(
                    "from pathlib import Path\n\n"
                    "def test_outside_mutation():\n"
                    "    Path('outside.txt').write_text('mutation')\n"
                    "    assert False, 'EVENT_SUBSCRIBERS_MISSING'\n"
                ),
            )
        ],
        production_edits=[
            runner.ExactTextEdit(
                path="source.py",
                expected_sha256=runner.sha256_text(source_before),
                old_text="    event_publishers = []\n",
                new_text="    event_publishers = []\n    event_subscribers = []\n",
            )
        ],
    )
    result = runner.MiniSWEWorkflow(
        mission=mission,
        run_root=tmp_path / "runs",
        **sources(runner, proposal),
    ).run()
    assert result.status == runner.TerminalStatus.BLOCKED
    assert result.reason == "OUT_OF_SCOPE_MUTATION"
