"""Mechanical-contract tests for the contained mini SWE DSPy workflow."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError


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


def test_v1_sanity_signature_constrains_and_describes_model_output():
    runner = runner_module()
    fields = runner.EvidenceSanitySignature.fields

    assert fields["status"].annotation == Literal[
        "PROCEED", "NEEDS_EVIDENCE", "BLOCKED"
    ]
    assert fields["status"].json_schema_extra["desc"] == (
        "Evidence sufficiency status. Use PROCEED when the packet is sufficient; "
        "NEEDS_EVIDENCE only with one bounded-read/search capability; BLOCKED "
        "when evidence cannot support proceeding. Examples: PROCEED, "
        "NEEDS_EVIDENCE, BLOCKED."
    )
    assert fields["reason"].json_schema_extra["desc"] == (
        "Short evidence-based explanation for the selected status."
    )
    assert fields["capability"].json_schema_extra["desc"] == (
        "One bounded_read or bounded_search request only for NEEDS_EVIDENCE; "
        "otherwise null."
    )


def test_v1_live_sanity_rejects_invalid_raw_status(tmp_path, monkeypatch):
    runner = runner_module()
    repo = make_repo(tmp_path)
    workflow = runner.MiniSWEWorkflow(make_mission(runner, repo), tmp_path / "runs")

    class FakePredict:
        def __init__(self, signature):
            assert signature is runner.EvidenceSanitySignature

        def __call__(self, **_kwargs):
            return type(
                "Prediction",
                (),
                {"status": "success", "reason": "not a sanctioned status", "capability": None},
            )()

    monkeypatch.setattr(workflow, "_configure_live", lambda: None)
    monkeypatch.setattr(runner.dspy, "Predict", FakePredict)

    with pytest.raises(ValidationError, match="invalid sanity status"):
        workflow._live_sanity({"facts": []})


def test_v1_sanity_proceed_and_one_follow_up_are_hash_pinned(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission = make_mission(runner, repo)
    calls = {"proposal": 0, "sanity": 0}

    def proposal(_mission, _investigation):
        calls["proposal"] += 1
        return successful_proposal(runner, repo)

    def sanity(_mission, _investigation, _packet):
        calls["sanity"] += 1
        return runner.SanityDecision(status="NEEDS_EVIDENCE", reason="inspect source", capability=runner.EvidenceCapability(kind="bounded_read", path="source.py"))

    result = runner.MiniSWEWorkflow(mission, tmp_path / "runs", investigation_source=sources(runner, successful_proposal(runner, repo))["investigation_source"], proposal_source=proposal, review_source=sources(runner, successful_proposal(runner, repo))["review_source"], sanity_source=sanity).run()
    assert result.status is runner.TerminalStatus.DONE
    assert calls == {"proposal": 1, "sanity": 1}
    events = [json.loads(line) for line in result.ledger_path.read_text().splitlines()]
    assert {"sanity_packet", "sanity", "evidence_follow_up", "preview"} <= {event["event"] for event in events}
    assert all(event["payload"].get("sha256") for event in events if event["event"] in {"sanity_packet", "evidence_follow_up", "preview"})


def test_v1_sanity_packet_binds_redacted_persisted_investigator_trace(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    workflow = runner.MiniSWEWorkflow(make_mission(runner, repo), tmp_path / "runs")
    ledger = runner.EventLedger(tmp_path / "trace.jsonl")
    ledger.append("tool", {"tool": "read_repo_file", "output": {"token": "private-token"}})
    ledger.append("tool_error", {"tool": "search_repo", "credential": "private-key"})
    ledger.append("react_trajectory", {"trajectory": {"step": "searched"}})
    ledger.append("proposal", {"ignored": True})

    packet = workflow._evidence_packet(runner.Investigation(facts=["fact"], category="test"), ledger)

    assert packet["investigator_trace"] == ledger.records({"tool", "tool_error", "react_trajectory"})
    encoded = json.dumps(packet, sort_keys=True)
    assert "private-token" not in encoded and "private-key" not in encoded
    assert "[REDACTED]" in encoded


def test_v1_live_investigator_tools_never_expose_out_of_mission_content(tmp_path, monkeypatch):
    runner = runner_module()
    repo = make_repo(tmp_path)
    (repo / "protected_acceptance.py").write_text("protected-needle\n")
    mission = make_mission(runner, repo).model_copy(update={"allowed_paths": ["source.py"]})
    workflow = runner.MiniSWEWorkflow(mission, tmp_path / "runs")
    observed: dict[str, str] = {}

    class FakeReAct:
        def __init__(self, signature, tools):
            assert signature is runner.InvestigateMissionSignature
            self.tools = tools

        def __call__(self, **_kwargs):
            observed["read"] = self.tools[0]("protected_acceptance.py")
            observed["search"] = self.tools[1]("protected-needle")
            return type("Prediction", (), {"facts": [], "category": "test"})()

    monkeypatch.setattr(workflow, "_configure_live", lambda: None)
    monkeypatch.setattr(runner.dspy, "ReAct", FakeReAct)

    workflow._live_investigation(runner.EventLedger(tmp_path / "trace.jsonl"))

    assert observed["read"].startswith("TOOL_ERROR: Outside mission allow-list")
    assert "protected_acceptance.py" not in observed["search"]


def test_v1_live_proposer_receives_exact_final_hash_pinned_packet(tmp_path, monkeypatch):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission = make_mission(runner, repo)
    captured: dict[str, str] = {}

    class FakePredict:
        def __init__(self, signature):
            assert signature is runner.ProposeEditsSignature

        def __call__(self, **kwargs):
            captured["packet"] = kwargs["evidence_packet"]
            assert set(kwargs) == {"mission", "evidence_packet"}
            return type("Prediction", (), {"proposal": successful_proposal(runner, repo)})()

    workflow = runner.MiniSWEWorkflow(
        mission,
        tmp_path / "runs",
        investigation_source=lambda _mission: runner.Investigation(facts=["fact"], category="test"),
        sanity_source=lambda _mission, _investigation, _packet: runner.SanityDecision(
            status="NEEDS_EVIDENCE",
            reason="read source",
            capability=runner.EvidenceCapability(kind="bounded_read", path="source.py"),
        ),
        review_source=lambda *_args: runner.PostGreenReview(judgment="reviewed", concerns=[], evidence_requests=[]),
    )
    monkeypatch.setattr(workflow, "_configure_live", lambda: None)
    monkeypatch.setattr(runner.dspy, "Predict", FakePredict)

    result = workflow.run()

    assert result.status is runner.TerminalStatus.DONE
    events = [json.loads(line) for line in result.ledger_path.read_text().splitlines()]
    packet = next(event["payload"] for event in events if event["event"] == "proposal_packet")
    assert captured["packet"] == json.dumps(packet["packet"], sort_keys=True)
    assert packet["sha256"] == runner.sha256_text(captured["packet"])
    final_packet = json.loads(captured["packet"])
    assert final_packet["sanity"] == {
        "status": "NEEDS_EVIDENCE",
        "reason": "read source",
        "capability": {"kind": "bounded_read", "path": "source.py", "query": None},
    }
    assert final_packet["follow_up"]["path"] == "source.py"


def test_v1_out_of_allowed_follow_up_blocks_before_proposal_without_content_exposure(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    protected = repo / "protected_acceptance.py"
    protected.write_text("protected-follow-up-content\n")
    called = False

    def proposal(_mission, _investigation):
        nonlocal called
        called = True
        return successful_proposal(runner, repo)

    result = runner.MiniSWEWorkflow(
        make_mission(runner, repo),
        tmp_path / "runs",
        investigation_source=lambda _mission: runner.Investigation(facts=[], category="test"),
        proposal_source=proposal,
        sanity_source=lambda _mission, _investigation, _packet: runner.SanityDecision(
            status="NEEDS_EVIDENCE",
            reason="inspect protected path",
            capability=runner.EvidenceCapability(kind="bounded_read", path="protected_acceptance.py"),
        ),
    ).run()

    assert result.reason == "NEEDS_EVIDENCE_BLOCKED"
    assert called is False
    assert "protected-follow-up-content" not in result.ledger_path.read_text()


def test_v1_preview_copy_excludes_dotenv_files(tmp_path, monkeypatch):
    runner = runner_module()
    repo = make_repo(tmp_path)
    (repo / ".env").write_text("secret\n")
    (repo / ".env.local").write_text("secret\n")
    (repo / "nested").mkdir()
    (repo / "nested" / ".env.production").write_text("secret\n")
    original_copytree = runner.shutil.copytree

    def checked_copytree(source, destination, *args, **kwargs):
        copied = original_copytree(source, destination, *args, **kwargs)
        destination = Path(destination)
        if destination.name == "workspace":
            assert not (destination / ".env").exists()
            assert not (destination / ".env.local").exists()
            assert not (destination / "nested" / ".env.production").exists()
        return copied

    monkeypatch.setattr(runner.shutil, "copytree", checked_copytree)
    result = runner.MiniSWEWorkflow(make_mission(runner, repo), tmp_path / "runs", **sources(runner, successful_proposal(runner, repo))).run()
    assert result.status is runner.TerminalStatus.DONE


@pytest.mark.parametrize("decision, reason", [
    (lambda runner: runner.SanityDecision(status="BLOCKED", reason="insufficient"), "SANITY_BLOCKED"),
    (lambda runner: {"status": "NEEDS_EVIDENCE", "reason": "missing"}, "INVALID_SANITY"),
])
def test_v1_sanity_blocks_before_proposal(tmp_path, decision, reason):
    runner = runner_module()
    repo = make_repo(tmp_path)
    called = False

    def proposal(_mission, _investigation):
        nonlocal called
        called = True
        return successful_proposal(runner, repo)

    result = runner.MiniSWEWorkflow(make_mission(runner, repo), tmp_path / "runs", investigation_source=sources(runner, successful_proposal(runner, repo))["investigation_source"], proposal_source=proposal, sanity_source=lambda _m, _i, _p: decision(runner)).run()
    assert result.reason == reason
    assert called is False


def test_v1_preview_failure_preserves_real_paths_and_does_not_enter_real_gates(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    before = {path: (repo / path).read_bytes() for path in ("check.py", "source.py")}
    bad = successful_proposal(runner, repo)
    bad.test_edits[0].new_text = "def test_bad():\n    assert False, 'WRONG'\n"
    proposal_calls = 0
    review_calls = 0

    def proposal(_mission, _investigation):
        nonlocal proposal_calls
        proposal_calls += 1
        return bad

    def review(*_args):
        nonlocal review_calls
        review_calls += 1
        return runner.PostGreenReview(judgment="unused")

    result = runner.MiniSWEWorkflow(make_mission(runner, repo), tmp_path / "runs", investigation_source=sources(runner, bad)["investigation_source"], proposal_source=proposal, review_source=review).run()
    assert result.reason == "PREVIEW_RED_NOT_CONFIRMED"
    assert {path: (repo / path).read_bytes() for path in before} == before
    assert proposal_calls == 1
    assert review_calls == 0


def test_v1_preview_success_enters_unchanged_v0_gates(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    result = runner.MiniSWEWorkflow(make_mission(runner, repo), tmp_path / "runs", **sources(runner, successful_proposal(runner, repo))).run()
    assert result.status is runner.TerminalStatus.DONE
    events = [json.loads(line) for line in result.ledger_path.read_text().splitlines()]
    assert next(index for index, event in enumerate(events) if event["event"] == "preview") < next(index for index, event in enumerate(events) if event["event"] == "test_edit")


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


def test_live_proposal_guides_assertion_red_existing_surface_and_receives_only_packet(tmp_path, monkeypatch):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission = make_mission(runner, repo)
    workflow = runner.MiniSWEWorkflow(mission, tmp_path / "runs")
    captured: dict[str, object] = {}

    class FakePredict:
        def __init__(self, signature):
            assert signature is runner.ProposeEditsSignature

        def __call__(self, **kwargs):
            captured.update(kwargs)
            return type("Prediction", (), {"proposal": runner.ChangeProposal()})()

    assert set(runner.ProposeEditsSignature.input_fields) == {"mission", "evidence_packet"}
    packet_json = json.dumps({"frozen": "evidence"}, sort_keys=True)
    monkeypatch.setattr(workflow, "_configure_live", lambda: None)
    monkeypatch.setattr(runner.dspy, "Predict", FakePredict)
    workflow._live_proposal(packet_json)

    instruction = runner.ProposeEditsSignature.__doc__
    assert "assertion-level RED" in instruction
    assert "existing import/module/public API surface" in instruction
    assert "not-yet-created production symbol" in instruction
    assert "expected-error/rejection test must fail if the expected error is absent" in instruction
    assert captured["evidence_packet"] == packet_json
    assert set(captured) == {"mission", "evidence_packet"}


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
    assert result.reason == "PREVIEW_RED_NOT_CONFIRMED"


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
    assert result.reason == "PREVIEW_RED_NOT_CONFIRMED"
    assert check.read_text() == original_test
    assert (repo / "source.py").read_text() == original_source
    assert check.stat().st_mode & 0o777 == 0o744
    assert not any(json.loads(line)["event"] == "rollback" for line in result.ledger_path.read_text().splitlines())


def test_grouped_same_file_test_edits_write_once_and_produce_red(tmp_path):
    runner = runner_module()
    check_source = (
        "def test_alpha():\n    assert True\n\n"
        "def test_beta():\n    assert True\n"
    )
    repo = make_repo(tmp_path, check_source)
    mission = make_mission(runner, repo, expected_red_text="GROUPED_RED")
    check_before = (repo / "check.py").read_text()
    source_before = (repo / "source.py").read_text()
    proposal = runner.ChangeProposal(
        test_edits=[
            runner.ExactTextEdit(path="check.py", expected_sha256=runner.sha256_text(check_before), old_text="def test_alpha():\n    assert True\n", new_text="def test_alpha():\n    assert 'alpha' in open('source.py').read(), 'GROUPED_RED'\n"),
            runner.ExactTextEdit(path="check.py", expected_sha256=runner.sha256_text(check_before), old_text="def test_beta():\n    assert True\n", new_text="def test_beta():\n    assert 'beta' in open('source.py').read(), 'GROUPED_RED'\n"),
        ],
        production_edits=[runner.ExactTextEdit(path="source.py", expected_sha256=runner.sha256_text(source_before), old_text="    event_publishers = []\n", new_text="    event_publishers = []\n    alpha = True\n    beta = True\n")],
    )
    result = runner.MiniSWEWorkflow(mission, tmp_path / "runs", **sources(runner, proposal)).run()

    assert result.status == runner.TerminalStatus.DONE
    events = [json.loads(line) for line in result.ledger_path.read_text().splitlines()]
    assert [event["payload"]["edit"]["path"] for event in events if event["event"] == "test_edit"] == ["check.py"]
    assert any(event["event"] == "red_confirmed" for event in events)
    assert any(event["event"] == "proposal_received" for event in events)
    assert any(event["event"] == "proposal_validated" for event in events)


def test_grouped_normalization_rejects_ambiguous_edits_before_writes(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    before = (repo / "source.py").read_text()
    valid = runner.ExactTextEdit(path="source.py", expected_sha256=runner.sha256_text(before), old_text="    event_publishers = []\n", new_text="    one = []\n")
    cases = [
        runner.ChangeProposal(test_edits=[runner.ExactTextEdit(path="./source.py", expected_sha256=valid.expected_sha256, old_text=valid.old_text, new_text=valid.new_text)]),
        runner.ChangeProposal(test_edits=[valid, runner.ExactTextEdit(path="source.py", expected_sha256="0" * 64, old_text="class CallTree:\n", new_text="class Other:\n")]),
        runner.ChangeProposal(test_edits=[runner.ExactTextEdit(path="source.py", expected_sha256=valid.expected_sha256, old_text="", new_text="x")]),
        runner.ChangeProposal(test_edits=[valid, valid]),
        runner.ChangeProposal(test_edits=[runner.ExactTextEdit(path="source.py", expected_sha256=valid.expected_sha256, old_text="class CallTree:\n    event_publishers", new_text="x"), runner.ExactTextEdit(path="source.py", expected_sha256=valid.expected_sha256, old_text="    event_publishers = []\n", new_text="y")]),
        runner.ChangeProposal(test_edits=[valid], production_edits=[valid]),
    ]

    for proposal in cases:
        with pytest.raises(runner.ProposalValidationError):
            runner.normalize_change_proposal(repo, ["source.py"], proposal)
        assert (repo / "source.py").read_text() == before

    alias = repo / "source-alias.py"
    alias.hardlink_to(repo / "source.py")
    alias_edit = runner.ExactTextEdit(path="source-alias.py", expected_sha256=valid.expected_sha256, old_text=valid.old_text, new_text=valid.new_text)
    with pytest.raises(runner.ProposalValidationError):
        runner.normalize_change_proposal(
            repo,
            ["source.py", "source-alias.py"],
            runner.ChangeProposal(test_edits=[valid], production_edits=[alias_edit]),
        )


def test_grouped_normalization_is_order_independent(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path, "first = 1\nsecond = 2\n")
    before = (repo / "check.py").read_text()
    first = runner.ExactTextEdit(path="check.py", expected_sha256=runner.sha256_text(before), old_text="first = 1", new_text="first = 10")
    second = runner.ExactTextEdit(path="check.py", expected_sha256=runner.sha256_text(before), old_text="second = 2", new_text="second = 20")
    forward = runner.normalize_change_proposal(repo, ["check.py"], runner.ChangeProposal(test_edits=[first, second]))
    reverse = runner.normalize_change_proposal(repo, ["check.py"], runner.ChangeProposal(test_edits=[second, first]))

    assert forward.test_groups[0].after_bytes == reverse.test_groups[0].after_bytes
    (repo / "check.py").write_text("abcdef")
    adjacent_before = (repo / "check.py").read_text()
    adjacent = runner.normalize_change_proposal(
        repo,
        ["check.py"],
        runner.ChangeProposal(test_edits=[
            runner.ExactTextEdit(path="check.py", expected_sha256=runner.sha256_text(adjacent_before), old_text="abc", new_text="ABC"),
            runner.ExactTextEdit(path="check.py", expected_sha256=runner.sha256_text(adjacent_before), old_text="def", new_text="DEF"),
        ]),
    )
    assert adjacent.test_groups[0].after_bytes == b"ABCDEF"


def test_grouped_second_write_failure_rolls_back_first(tmp_path, monkeypatch):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission = make_mission(runner, repo)
    check_before = (repo / "check.py").read_text()
    source_before = (repo / "source.py").read_text()
    proposal = runner.ChangeProposal(
        test_edits=[
            runner.ExactTextEdit(path="check.py", expected_sha256=runner.sha256_text(check_before), old_text=check_before, new_text="def test_failure():\n    assert False, 'EVENT_SUBSCRIBERS_MISSING'\n"),
            runner.ExactTextEdit(path="source.py", expected_sha256=runner.sha256_text(source_before), old_text="    event_publishers = []\n", new_text="    event_publishers = ['pending']\n"),
        ]
    )
    original_replace = runner.os.replace
    failed = False

    def fail_second(source, destination):
        nonlocal failed
        if Path(destination).name == "source.py" and not failed:
            failed = True
            raise OSError("second write failed")
        return original_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", fail_second)
    result = runner.MiniSWEWorkflow(mission, tmp_path / "runs", **sources(runner, proposal)).run()

    assert result.reason == "PREVIEW_EDIT_INVALID"
    assert (repo / "check.py").read_text() == check_before
    assert (repo / "source.py").read_text() == source_before
    assert not any(json.loads(line)["event"] == "rollback" for line in result.ledger_path.read_text().splitlines())


def test_runner_rejects_duplicate_path_proposal_before_any_write(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission = make_mission(runner, repo)
    check_before = (repo / "check.py").read_text()
    source_before = (repo / "source.py").read_text()
    duplicate_edit = runner.ExactTextEdit(
        path="check.py",
        expected_sha256=runner.sha256_text(check_before),
        old_text=check_before,
        new_text="def test_duplicate_path():\n    assert False\n",
    )
    proposal = runner.ChangeProposal(test_edits=[duplicate_edit], production_edits=[duplicate_edit])

    result = runner.MiniSWEWorkflow(
        mission=mission,
        run_root=tmp_path / "runs",
        **sources(runner, proposal),
    ).run()

    assert result.status == runner.TerminalStatus.BLOCKED
    assert result.reason == "INVALID_PROPOSAL"
    assert (repo / "check.py").read_text() == check_before
    assert (repo / "source.py").read_text() == source_before
    events = [json.loads(line) for line in result.ledger_path.read_text().splitlines()]
    invalid = next(event for event in events if event["event"] == "proposal_invalid")
    assert invalid["payload"]["path"] == "check.py"
    assert not any(event["event"] == "test_edit" for event in events)
    assert not any(event["event"] == "rollback" for event in events)


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
    assert result.reason == "PREVIEW_COMMAND_MUTATION"


def acceptance_mission(runner, repo: Path, source: str | None = None):
    artifact = repo / "acceptance.py"
    artifact.write_text(source or "from pathlib import Path\n\ndef test_held_out():\n    assert 'event_subscribers' in Path('source.py').read_text(), 'HELD_OUT_MISSING'\n")
    git(repo, "add", "acceptance.py")
    git(repo, "commit", "-qm", "add acceptance oracle")
    return make_mission(runner, repo).model_copy(update={"acceptance_test_path": "acceptance.py", "acceptance_expected_red_text": "HELD_OUT_MISSING"})


def test_held_out_acceptance_contract(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    with pytest.raises(ValueError):
        runner.Mission(**(make_mission(runner, repo).model_dump() | {"acceptance_test_path": "acceptance.py"}))
    mission = acceptance_mission(runner, repo)
    assert runner.validate_acceptance_artifact(repo, mission.allowed_paths, mission.acceptance_test_path).sha256
    with pytest.raises(runner.ScopeViolation):
        runner.validate_acceptance_artifact(repo, mission.allowed_paths, "check.py")
    (repo / "alias.py").hardlink_to(repo / "acceptance.py")
    with pytest.raises(runner.ScopeViolation):
        runner.validate_acceptance_artifact(repo, mission.allowed_paths, "alias.py")


def test_held_out_acceptance_orders_gates_and_binds_review(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission = acceptance_mission(runner, repo)
    seen = {}
    def review(_m, _i, _p, evidence):
        seen.update(json.loads(evidence))
        return runner.PostGreenReview(judgment="reviewed", concerns=[], evidence_requests=[])
    result = runner.MiniSWEWorkflow(mission, tmp_path / "runs", investigation_source=lambda _m: runner.Investigation(facts=[], category="test"), proposal_source=lambda _m, _i: successful_proposal(runner, repo), review_source=review).run()
    assert result.status is runner.TerminalStatus.DONE
    assert seen["acceptance_baseline_command"]["returncode"] == 1
    assert seen["acceptance_candidate_command"]["returncode"] == 0
    events = [json.loads(line) for line in result.ledger_path.read_text().splitlines()]
    assert [e["payload"]["phase"] for e in events if e["event"] == "command"] == ["baseline_focused", "baseline_full", "acceptance_baseline", "focused_red", "focused_green", "full_green", "acceptance_candidate"]


@pytest.mark.parametrize("source, reason", [("def test_held_out():\n    assert True\n", "ACCEPTANCE_ALREADY_PASSES"), ("def test_held_out():\n    assert False, 'WRONG'\n", "ACCEPTANCE_BASELINE_INVALID")])
def test_held_out_baseline_blocks_before_sources(tmp_path, source, reason):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission = acceptance_mission(runner, repo, source)
    result = runner.MiniSWEWorkflow(mission, tmp_path / "runs", investigation_source=lambda _m: pytest.fail("source called")).run()
    assert result.reason == reason


def test_held_out_candidate_failure_rolls_back_and_terminal_requires_evidence(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    mission = acceptance_mission(runner, repo, "def test_held_out():\n    assert False, 'HELD_OUT_MISSING'\n")
    before = (repo / "source.py").read_text()
    result = runner.MiniSWEWorkflow(mission, tmp_path / "runs", **sources(runner, successful_proposal(runner, repo))).run()
    assert result.reason == "PREVIEW_ACCEPTANCE_CANDIDATE_FAILED"
    assert (repo / "source.py").read_text() == before
    evidence = runner.GateEvidence(preflight_ok=True, live_proposal_valid=True, baseline_green=True, red_confirmed=True, production_applied=True, focused_green=True, full_green=True, scope_ok=True, revision_stable=True, rollback_performed=False, acceptance_configured=True, acceptance_baseline=None, acceptance_candidate=None)
    assert runner.decide_terminal(evidence).reason == "ACCEPTANCE_FAILED"


def test_held_out_baseline_rc2_timeout_and_mutation_block_before_sources(tmp_path, monkeypatch):
    runner = runner_module()
    for source, reason in [("def test_held_out(:\n", "ACCEPTANCE_BASELINE_INVALID"), ("from pathlib import Path\n\ndef test_held_out():\n    Path('outside.txt').write_text('x')\n    assert False, 'HELD_OUT_MISSING'\n", "COMMAND_MUTATION")]:
        case_root = tmp_path / reason
        case_root.mkdir()
        repo = make_repo(case_root)
        mission = acceptance_mission(runner, repo, source)
        result = runner.MiniSWEWorkflow(mission, tmp_path / f"runs-{reason}", investigation_source=lambda _m: pytest.fail("source called")).run()
        assert result.reason == reason
    timeout_root = tmp_path / "timeout"
    timeout_root.mkdir()
    repo = make_repo(timeout_root)
    mission = acceptance_mission(runner, repo)
    original = runner.run_allowed_command
    def timeout(argv, *args):
        if argv[-2] == "acceptance.py":
            return runner.CommandResult(argv=argv, allowed=True, returncode=None, stdout="", stderr="", timed_out=True)
        return original(argv, *args)
    monkeypatch.setattr(runner, "run_allowed_command", timeout)
    result = runner.MiniSWEWorkflow(mission, tmp_path / "runs-timeout", investigation_source=lambda _m: pytest.fail("source called")).run()
    assert result.reason == "ACCEPTANCE_BASELINE_INVALID"


def test_held_out_candidate_mutation_and_review_staleness_roll_back(tmp_path):
    runner = runner_module()
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    repo = make_repo(candidate_root)
    mission = acceptance_mission(runner, repo, "from pathlib import Path\n\ndef test_held_out():\n    if 'event_subscribers' in Path('source.py').read_text():\n        Path('outside.txt').write_text('x')\n        return\n    assert False, 'HELD_OUT_MISSING'\n")
    before = (repo / "source.py").read_text()
    result = runner.MiniSWEWorkflow(mission, tmp_path / "runs-candidate", **sources(runner, successful_proposal(runner, repo))).run()
    assert result.reason == "PREVIEW_COMMAND_MUTATION"
    assert (repo / "source.py").read_text() == before
    stale_root = tmp_path / "stale"
    stale_root.mkdir()
    repo = make_repo(stale_root)
    mission = acceptance_mission(runner, repo)
    before = (repo / "source.py").read_text()
    result = runner.MiniSWEWorkflow(mission, tmp_path / "runs-stale", investigation_source=lambda _m: runner.Investigation(facts=[], category="test"), proposal_source=lambda _m, _i: successful_proposal(runner, repo), review_source=lambda *_args: (((repo / "source.py").write_text("stale\n"), runner.PostGreenReview(judgment="reviewed", concerns=[], evidence_requests=[]))[1])).run()
    assert result.reason == "ACCEPTANCE_STALE"
    assert (repo / "source.py").read_text() == before


def test_load_mission_file_acceptance_fields_are_paired_and_backward_compatible(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    base = make_mission(runner, repo).model_dump(exclude={"repo_root", "model_name", "reviewed_learning", "mission_provenance"})
    mission_file = tmp_path / "mission.json"
    mission_file.write_text(json.dumps(base | {"acceptance_test_path": "oracle.py", "acceptance_expected_red_text": "marker"}))
    loaded = runner.load_mission_file(mission_file, repo, "gemini/gemini-3.5-flash-lite")
    assert loaded.acceptance_test_path == "oracle.py" and loaded.acceptance_expected_red_text == "marker"
    mission_file.write_text(json.dumps(base))
    assert runner.load_mission_file(mission_file, repo, "gemini/gemini-3.5-flash-lite").acceptance_test_path is None
    mission_file.write_text(json.dumps(base | {"acceptance_test_path": "oracle.py"}))
    with pytest.raises(ValueError):
        runner.load_mission_file(mission_file, repo, "gemini/gemini-3.5-flash-lite")


def test_real_g5_mission_binds_stable_protected_oracle_contract():
    runner = runner_module()
    root = RUNNER_PATH.parents[2]
    mission = runner.load_mission_file(root / "src/workflow/mini-swe-missions/G5-structured-git-archeology.json", root, "gemini/gemini-3.5-flash-lite")
    assert mission.acceptance_test_path == "src/workflow/mini-swe-acceptance/G5-structured-git-archeology.py"
    assert mission.acceptance_expected_red_text == "G5_HELD_OUT_GIT_ARCHEOLOGY"
    assert mission.full_test_argv == [str(root / ".venv/bin/python"), "-m", "pytest", "-p", "no:cacheprovider", "tests", "-q"]
    artifact = runner.validate_acceptance_artifact(root, mission.allowed_paths, mission.acceptance_test_path)
    assert artifact.sha256 == "37ed232fa41c9a274412771659157ec322afb5a877b8208fa9e8a71aec03017e"
    assert runner.acceptance_pytest_argv(root, mission.acceptance_test_path) == [
        str(root / ".venv/bin/python"), "-m", "pytest", "-p", "no:cacheprovider",
        "src/workflow/mini-swe-acceptance/G5-structured-git-archeology.py", "-q",
    ]


def test_real_g8_mission_binds_protected_declared_risk_oracle_contract():
    runner = runner_module()
    root = RUNNER_PATH.parents[2]
    mission_path = root / "src/workflow/mini-swe-missions/G8-declared-risk-certainty.json"
    oracle_path = root / "src/workflow/mini-swe-acceptance/G8-declared-risk-certainty.py"
    assert mission_path.is_file(), "G8_MISSION_MISSING"
    assert oracle_path.is_file(), "G8_DECLARED_RISK_ORACLE_MISSING"
    mission = runner.load_mission_file(mission_path, root, "gemini/gemini-3.5-flash-lite")
    assert mission.allowed_paths == ["src/workflow/schemas.py", "tests/test_workflow.py"]
    assert mission.expected_red_text == "G8_UNVERIFIED_CANNOT_BE_FIVE"
    assert mission.acceptance_test_path == "src/workflow/mini-swe-acceptance/G8-declared-risk-certainty.py"
    assert mission.acceptance_expected_red_text == "G8_UNVERIFIED_CANNOT_BE_FIVE"
    assert mission.full_test_argv == [str(root / ".venv/bin/python"), "-m", "pytest", "-p", "no:cacheprovider", "tests", "-q"]
    assert mission.ruff_check_argv == [
        str(root / ".venv/bin/python"), "-m", "ruff", "check", "--no-cache",
        "src/workflow/schemas.py", "tests/test_workflow.py",
    ]
    artifact = runner.validate_acceptance_artifact(root, mission.allowed_paths, mission.acceptance_test_path)
    assert artifact.sha256 == "0f498025e9f520b41c33c95a63501a205d41d516a8cc9979a4f22e30a0274d0f"
    assert runner.acceptance_pytest_argv(root, mission.acceptance_test_path) == [
        str(root / ".venv/bin/python"), "-m", "pytest", "-p", "no:cacheprovider",
        "src/workflow/mini-swe-acceptance/G8-declared-risk-certainty.py", "-q",
    ]


def test_ruff_gate_mission_expansion_and_exact_safe_shape(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    ruff = [str(repo / ".venv/bin/python"), "-m", "ruff", "check", "--no-cache", "source.py", "check.py"]
    mission = make_mission(runner, repo).model_copy(update={"ruff_check_argv": ruff})
    assert runner._is_safe_ruff(mission.ruff_check_argv, repo)
    for argv in [
        [str(repo / ".venv/bin/python"), "-m", "ruff", "check", "source.py"],
        [str(repo / ".venv/bin/python"), "-m", "ruff", "check", "--no-cache", "--fix", "source.py"],
        [str(repo / ".venv/bin/python"), "-m", "ruff", "check", "--no-cache", "../outside.py"],
        [str(repo / ".venv/bin/python"), "-m", "ruff", "check", "--no-cache", "."],
    ]:
        assert not runner._is_safe_ruff(argv, repo)
    linked = repo / "linked.py"
    linked.symlink_to(repo / "source.py")
    assert not runner._is_safe_ruff([str(repo / ".venv/bin/python"), "-m", "ruff", "check", "--no-cache", "linked.py"], repo)


def test_ruff_gate_baseline_blocks_before_sources_and_candidate_failure_rolls_back(tmp_path, monkeypatch):
    runner = runner_module()
    repo = make_repo(tmp_path)
    ruff = [str(repo / ".venv/bin/python"), "-m", "ruff", "check", "--no-cache", "source.py", "check.py"]
    mission = make_mission(runner, repo).model_copy(update={"ruff_check_argv": ruff})
    calls = []
    original = runner.run_allowed_command
    def baseline_fail(argv, *args):
        calls.append(argv)
        if argv == ruff:
            return runner.CommandResult(argv=argv, allowed=True, returncode=1, stdout="", stderr="ruff", timed_out=False)
        return original(argv, *args)
    monkeypatch.setattr(runner, "run_allowed_command", baseline_fail)
    result = runner.MiniSWEWorkflow(mission, tmp_path / "runs-baseline", investigation_source=lambda _m: pytest.fail("source called")).run()
    assert result.reason == "RUFF_BASELINE_FAILED" and calls.count(ruff) == 1
    monkeypatch.setattr(runner, "run_allowed_command", original)
    before = (repo / "source.py").read_text()
    seen = []
    def candidate_fail(argv, *args):
        seen.append(argv)
        if argv == ruff and len([item for item in seen if item == ruff]) == 2:
            return runner.CommandResult(argv=argv, allowed=True, returncode=1, stdout="", stderr="ruff", timed_out=False)
        return original(argv, *args)
    monkeypatch.setattr(runner, "run_allowed_command", candidate_fail)
    result = runner.MiniSWEWorkflow(mission, tmp_path / "runs-candidate", **sources(runner, successful_proposal(runner, repo))).run()
    assert result.reason == "PREVIEW_CANDIDATE_RUFF_FAILED" and (repo / "source.py").read_text() == before


def test_ruff_gate_orders_commands_binds_terminal_and_review(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    ruff = [str(repo / ".venv/bin/python"), "-m", "ruff", "check", "--no-cache", "source.py", "check.py"]
    mission = make_mission(runner, repo).model_copy(update={"ruff_check_argv": ruff})
    seen = {}
    result = runner.MiniSWEWorkflow(mission, tmp_path / "runs", investigation_source=lambda _m: runner.Investigation(facts=[], category="test"), proposal_source=lambda _m, _i: successful_proposal(runner, repo), review_source=lambda _m, _i, _p, evidence: (seen.update(json.loads(evidence)) or runner.PostGreenReview(judgment="reviewed", concerns=[], evidence_requests=[]))).run()
    assert result.status is runner.TerminalStatus.DONE
    assert seen["baseline_ruff_command"]["returncode"] == 0
    assert seen["candidate_ruff_command"]["returncode"] == 0
    evidence = runner.GateEvidence(preflight_ok=True, live_proposal_valid=True, baseline_green=True, red_confirmed=True, production_applied=True, focused_green=True, full_green=True, scope_ok=True, revision_stable=True, rollback_performed=False, ruff_configured=True, ruff_argv=ruff, ruff_baseline=runner.CommandResult(argv=ruff, allowed=True, returncode=0, stdout="", stderr="", timed_out=False), ruff_candidate=runner.CommandResult(argv=ruff, allowed=True, returncode=0, stdout="", stderr="", timed_out=False))
    assert runner.decide_terminal(evidence).status is runner.TerminalStatus.DONE
    assert runner.decide_terminal(evidence.model_copy(update={"ruff_candidate": None})).reason == "RUFF_CHECK_FAILED"


def test_ruff_audit_loader_expands_placeholder_and_omission_is_compatible(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    raw = runner.MissionSpec.model_validate(make_mission(runner, repo).model_dump(exclude={"repo_root", "model_name", "reviewed_learning", "mission_provenance"})).model_dump()
    raw["ruff_check_argv"] = ["${REPO_VENV_PYTHON}", "-m", "ruff", "check", "--no-cache", "source.py"]
    path = tmp_path / "with-ruff.json"
    path.write_text(json.dumps(raw))
    loaded = runner.load_mission_file(path, repo, "test")
    assert loaded.ruff_check_argv[0] == str(repo / ".venv/bin/python")
    raw.pop("ruff_check_argv")
    path.write_text(json.dumps(raw))
    assert runner.load_mission_file(path, repo, "test").ruff_check_argv is None


def test_ruff_audit_rejects_all_unsafe_shapes_and_non_allowlisted_command(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    safe = [str(repo / ".venv/bin/python"), "-m", "ruff", "check", "--no-cache", "source.py"]
    linked = repo / "linked.py"
    linked.symlink_to(repo / "source.py")
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n")
    bad = [
        [str(repo / "python"), *safe[1:]],
        [safe[0], "-m", "not_ruff", *safe[3:]],
        [safe[0], "-m", "ruff", "check", "source.py"],
        [*safe[:5], "--fix", "source.py"], [*safe[:5], "--output-file", "x", "source.py"],
        [*safe[:5], "--config", "x", "source.py"], [*safe[:5], str(outside)],
        [*safe[:5], "../outside.py"], [*safe[:5], "."], [*safe[:5], "linked.py"],
    ]
    assert all(not runner._is_safe_ruff(argv, repo) for argv in bad)
    result = runner.run_allowed_command(safe, repo, [], 1, repo)
    assert not result.allowed


@pytest.mark.parametrize("phase", ["baseline_timeout", "baseline_mutation", "candidate_timeout", "candidate_mutation"])
def test_ruff_audit_timeouts_and_mutations_block_with_required_rollback(tmp_path, monkeypatch, phase):
    runner = runner_module()
    repo = make_repo(tmp_path)
    ruff = [str(repo / ".venv/bin/python"), "-m", "ruff", "check", "--no-cache", "source.py", "check.py"]
    mission = make_mission(runner, repo).model_copy(update={"ruff_check_argv": ruff})
    original = runner.run_allowed_command
    calls = 0
    source_before, check_before = (repo / "source.py").read_text(), (repo / "check.py").read_text()
    def controlled(argv, *args):
        nonlocal calls
        if argv == ruff:
            calls += 1
            if phase.startswith("baseline") and calls == 1 or phase.startswith("candidate") and calls == 2:
                if phase.endswith("timeout"):
                    return runner.CommandResult(argv=argv, allowed=True, returncode=None, stdout="", stderr="", timed_out=True)
                (repo / "source.py").write_text("mutated = True\n")
                return runner.CommandResult(argv=argv, allowed=True, returncode=0, stdout="", stderr="", timed_out=False)
        return original(argv, *args)
    monkeypatch.setattr(runner, "run_allowed_command", controlled)
    result = runner.MiniSWEWorkflow(mission, tmp_path / phase, investigation_source=lambda _m: pytest.fail("source called") if phase.startswith("baseline") else runner.Investigation(facts=[], category="test"), **({} if phase.startswith("baseline") else {"proposal_source": lambda _m, _i: successful_proposal(runner, repo), "review_source": lambda *_args: runner.PostGreenReview(judgment="ok", concerns=[], evidence_requests=[])})).run()
    if phase.startswith("baseline"):
        assert result.reason == ("RUFF_BASELINE_FAILED" if phase.endswith("timeout") else "COMMAND_MUTATION")
    else:
        assert result.reason == ("PREVIEW_CANDIDATE_RUFF_FAILED" if phase.endswith("timeout") else "PREVIEW_COMMAND_MUTATION")
        assert (repo / "source.py").read_text() == source_before and (repo / "check.py").read_text() == check_before


def test_ruff_audit_terminal_rejects_all_bad_evidence(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    ruff = [str(repo / ".venv/bin/python"), "-m", "ruff", "check", "--no-cache", "source.py"]
    good = runner.CommandResult(argv=ruff, allowed=True, returncode=0, stdout="", stderr="", timed_out=False)
    base = runner.GateEvidence(preflight_ok=True, live_proposal_valid=True, baseline_green=True, red_confirmed=True, production_applied=True, focused_green=True, full_green=True, scope_ok=True, revision_stable=True, rollback_performed=False, ruff_configured=True, ruff_argv=ruff, ruff_baseline=good, ruff_candidate=good)
    assert runner.decide_terminal(base).status is runner.TerminalStatus.DONE
    variants = [
        {"ruff_candidate": None}, {"ruff_candidate": good.model_copy(update={"argv": [*ruff, "check.py"]})},
        {"ruff_candidate": good.model_copy(update={"allowed": False})}, {"ruff_candidate": good.model_copy(update={"timed_out": True})},
        {"ruff_candidate": good.model_copy(update={"returncode": 1})},
    ]
    assert all(runner.decide_terminal(base.model_copy(update=update)).reason == "RUFF_CHECK_FAILED" for update in variants)


def test_real_g10_mission_binds_unresolved_proposal_failure_oracle_contract():
    root = RUNNER_PATH.parents[2]
    mission_path = root / "src/workflow/mini-swe-missions/G10-unresolved-proposal-failure.json"
    oracle_path = root / "src/workflow/mini-swe-acceptance/G10-unresolved-proposal-failure.py"
    assert mission_path.is_file(), "G10_MISSION_MISSING"
    assert oracle_path.is_file(), "G10_UNRESOLVED_PROPOSAL_ORACLE_MISSING"
    oracle_source = oracle_path.read_text()
    assert "score=None, status=\"reconciled-and-calibrated\"" in oracle_source
    assert "G10_FAILURE_STATUS_CERTAINTY_PAIRING" in oracle_source
    assert __import__("hashlib").sha256(oracle_path.read_bytes()).hexdigest() == "7c2cfbc1eb22ac9b55a9d0f2d53bbd43b124916257f89ef15805152db68d0aa9"
    runner = runner_module()
    mission = runner.load_mission_file(mission_path, root, "test")
    assert mission.allowed_paths == ["src/workflow/schemas.py", "src/workflow/flow.py", "tests/test_workflow.py"]
    assert mission.acceptance_expected_red_text == "G10_UNRESOLVED_PROPOSAL_FAILURE"
    assert mission.ruff_check_argv == [str(root / ".venv/bin/python"), "-m", "ruff", "check", "--no-cache", "src/workflow/schemas.py", "src/workflow/flow.py", "tests/test_workflow.py"]


def test_proposal_guidance_distinguishes_returned_failure_states():
    runner = runner_module()
    guidance = runner.ProposeEditsSignature.__doc__ or ""
    assert "Reserve pytest.raises for contracts that actually raise" in guidance
    assert "assert the returned state" in guidance
    assert "constructor/call exception" in guidance


def test_proposal_guidance_requires_pytest_fixture_bindings():
    runner = runner_module()
    guidance = runner.ProposeEditsSignature.__doc__ or ""
    assert "pytest-provided identifier" in guidance
    assert "fixture parameters" in guidance
    assert "undeclared fixture name" in guidance


def test_proposal_guidance_prevents_local_import_shadowing():
    runner = runner_module()
    guidance = runner.ProposeEditsSignature.__doc__ or ""
    assert "reuse existing module-top bindings" in guidance
    assert "function-local import" in guidance
    assert "shadows the entire local scope" in guidance


def test_proposal_guidance_requires_import_bindings_before_use():
    runner = runner_module()
    guidance = runner.ProposeEditsSignature.__doc__ or ""
    assert "identifier" in guidance
    assert "matching import" in guidance
    assert "from package import name" in guidance
    assert "package.name" in guidance
    assert "collection-time NameError" in guidance


def test_proposal_guidance_requires_marker_visible_expected_error_failures():
    runner = runner_module()
    guidance = runner.ProposeEditsSignature.__doc__ or ""
    assert "Expected RED text" in guidance
    assert "pytest.raises" in guidance
    assert "match=" in guidance
    assert "unreachable" in guidance


def test_command_mutation_audit_snapshot_detects_file_and_symlink_identity(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    before = runner._snapshot_workspace(repo)
    (repo / "source.py").write_text("changed\n")
    assert runner._changed_paths(before, runner._snapshot_workspace(repo)) == {"source.py"}
    before = runner._snapshot_workspace(repo)
    (repo / "source.py").chmod(0o700)
    assert runner._changed_paths(before, runner._snapshot_workspace(repo)) == {"source.py"}
    before = runner._snapshot_workspace(repo)
    (repo / "created.py").write_text("created\n")
    (repo / "check.py").unlink()
    (repo / "source.py").rename(repo / "renamed.py")
    assert runner._changed_paths(before, runner._snapshot_workspace(repo)) == {"check.py", "created.py", "source.py", "renamed.py"}
    before = runner._snapshot_workspace(repo)
    (repo / "link.py").symlink_to("created.py")
    assert runner._changed_paths(before, runner._snapshot_workspace(repo)) == {"link.py"}
    before = runner._snapshot_workspace(repo)
    (repo / "link.py").unlink()
    (repo / "link.py").symlink_to("renamed.py")
    assert runner._changed_paths(before, runner._snapshot_workspace(repo)) == {"link.py"}


@pytest.mark.parametrize("kind, returncode, timed_out", [("rc1", 1, False), ("timeout", None, True)])
def test_command_mutation_audit_preserves_result_and_event_order(tmp_path, monkeypatch, kind, returncode, timed_out):
    runner = runner_module()
    repo = make_repo(tmp_path)
    ruff = [str(repo / ".venv/bin/python"), "-m", "ruff", "check", "--no-cache", "source.py", "check.py"]
    mission = make_mission(runner, repo).model_copy(update={"ruff_check_argv": ruff})
    original = runner.run_allowed_command
    def mutate(argv, *args):
        if argv == ruff:
            (repo / "source.py").write_text("command mutation\n")
            return runner.CommandResult(argv=argv, allowed=True, returncode=returncode, stdout="", stderr="", timed_out=timed_out)
        return original(argv, *args)
    monkeypatch.setattr(runner, "run_allowed_command", mutate)
    result = runner.MiniSWEWorkflow(mission, tmp_path / kind, investigation_source=lambda _m: pytest.fail("source called")).run()
    assert result.reason == "COMMAND_MUTATION"
    events = [json.loads(line) for line in result.ledger_path.read_text().splitlines()]
    command = next(i for i, event in enumerate(events) if event["event"] == "command" and event["payload"]["phase"] == "baseline_ruff")
    mutation = events[command + 1]
    assert mutation["event"] == "command_mutation"
    assert mutation["payload"] == {"phase": "baseline_ruff", "argv": ruff, "changed_paths": ["source.py"], "returncode": returncode, "timed_out": timed_out}


def test_command_mutation_audit_candidate_rollback_order_and_residual_outside_file(tmp_path, monkeypatch):
    runner = runner_module()
    repo = make_repo(tmp_path)
    ruff = [str(repo / ".venv/bin/python"), "-m", "ruff", "check", "--no-cache", "source.py", "check.py"]
    mission = make_mission(runner, repo).model_copy(update={"ruff_check_argv": ruff})
    source_before = (repo / "source.py").read_text()
    original, calls = runner.run_allowed_command, 0
    def mutate_candidate(argv, *args):
        nonlocal calls
        if argv == ruff:
            calls += 1
            if calls == 2:
                (repo / "source.py").write_text("command mutation\n")
                (repo / "outside.txt").write_text("unowned\n")
        return original(argv, *args)
    monkeypatch.setattr(runner, "run_allowed_command", mutate_candidate)
    result = runner.MiniSWEWorkflow(mission, tmp_path / "candidate", **sources(runner, successful_proposal(runner, repo))).run()
    assert result.reason == "PREVIEW_COMMAND_MUTATION"
    assert (repo / "source.py").read_text() == source_before
    assert (repo / "outside.txt").read_text() == "unowned\n"
    events = [json.loads(line) for line in result.ledger_path.read_text().splitlines()]
    preview = next(event for event in events if event["event"] == "preview")
    assert preview["payload"]["transcript"][-1]["phase"] == "candidate_ruff"
    assert preview["payload"]["transcript"][-1]["real_changed_paths"] == ["outside.txt", "source.py"]
    assert not any(event["event"] == "command" and event["payload"]["phase"] == "candidate_ruff" for event in events)
    assert any(event["event"] == "rollback" for event in events)


def _v2_delta(runner, repo: Path, trace: Path, phase: str = "focused_green") -> dict[str, object]:
    proposal_hash = "a" * 64
    preview_payload = {"passed": False, "transcript": [{"phase": phase, "result": {"stdout": "public failure", "stderr": ""}, "real_changed_paths": []}]}
    records = [
        {"sequence": 1, "event": "preflight", "payload": {"mission_id": "mission-test"}},
        {"sequence": 2, "event": "preview", "payload": {"passed": False, "proposal_sha256": proposal_hash, "sha256": runner.sha256_text(json.dumps({"transcript": preview_payload["transcript"]}, sort_keys=True)), **preview_payload}},
        {"sequence": 3, "event": "terminal", "payload": {"state": "BLOCKED", "reason": "PREVIEW_FOCUSED_GREEN_FAILED"}},
    ]
    trace.write_text("".join(json.dumps(record) + "\n" for record in records))
    return {
        "failure_reference": {"source_trace_path": str(trace), "source_trace_sha256": runner.sha256_bytes(trace.read_bytes()), "source_mission_id": "mission-test", "terminal_sequence": 3, "prior_proposal_sha256": proposal_hash, "preview_payload_sha256": records[1]["payload"]["sha256"], "failed_phase": phase, "terminal_reason": "PREVIEW_FOCUSED_GREEN_FAILED"},
        "coordinator_judgement": {"changed_premise": "The public failure falsifies the original premise.", "next_strategy": "Inspect the public contract before proposing once.", "public_evidence": [{"path": "source.py", "sha256": runner.sha256_bytes((repo / "source.py").read_bytes()), "observation": "CallTree is the public source under investigation."}]},
    }


def test_v2_delta_models_forbid_extra_and_score_fields(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    trace = tmp_path / "prior.jsonl"
    delta = _v2_delta(runner, repo, trace)
    assert runner.LearningDelta.model_validate(delta).coordinator_judgement.next_strategy
    with pytest.raises(ValidationError):
        runner.LearningDelta.model_validate({**delta, "score": 1})
    with pytest.raises(ValidationError):
        runner.LearningDelta.model_validate({**delta, "coordinator_judgement": {**delta["coordinator_judgement"], "semantic_score": 1}})


@pytest.mark.parametrize("mutation", ["trace_hash", "sequence", "phase", "protected", "stale", "outside", "raw"])
def test_v2_admission_rejects_falsified_inputs_before_sources(tmp_path, mutation):
    runner = runner_module()
    repo = make_repo(tmp_path)
    trace = tmp_path / "prior.jsonl"
    delta = _v2_delta(runner, repo, trace, "acceptance_candidate" if mutation == "protected" else "focused_green")
    if mutation == "trace_hash":
        delta["failure_reference"]["source_trace_sha256"] = "0" * 64
    elif mutation == "sequence":
        delta["failure_reference"]["terminal_sequence"] = 2
    elif mutation == "phase":
        delta["failure_reference"]["failed_phase"] = "full_green"
    elif mutation == "stale":
        delta["coordinator_judgement"]["public_evidence"][0]["sha256"] = "0" * 64
    elif mutation == "outside":
        delta["coordinator_judgement"]["public_evidence"][0]["path"] = "outside.py"
    delta_path = tmp_path / "delta.json"
    delta_path.write_text(json.dumps(delta))
    mission = make_mission(runner, repo).model_copy(update={"learning_delta_file": str(delta_path), "reviewed_learning": ["raw"] if mutation == "raw" else []})
    before = trace.read_bytes()
    result = runner.MiniSWEWorkflow(mission, tmp_path / "runs", investigation_source=lambda _m: pytest.fail("investigator called"), proposal_source=lambda _m, _i: pytest.fail("proposer called")).run()
    assert result.status is runner.TerminalStatus.BLOCKED
    assert trace.read_bytes() == before
    events = [json.loads(line) for line in result.ledger_path.read_text().splitlines()]
    assert not any(event["event"] == "learning_delta_admitted" for event in events)


def test_v2_admission_orders_event_and_carries_identical_safe_view(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    trace = tmp_path / "prior.jsonl"
    delta_path = tmp_path / "delta.json"
    delta_path.write_text(json.dumps(_v2_delta(runner, repo, trace)))
    seen: dict[str, object] = {"proposals": 0}
    mission = make_mission(runner, repo).model_copy(update={"learning_delta_file": str(delta_path)})
    def investigate(current):
        seen["investigation"] = current.learning_context
        return runner.Investigation(facts=[], category="test")
    def propose(current, _investigation):
        seen["proposals"] += 1
        seen["proposal"] = current.learning_context
        return successful_proposal(runner, repo)
    result = runner.MiniSWEWorkflow(mission, tmp_path / "runs", investigation_source=investigate, proposal_source=propose, review_source=lambda *_args: runner.PostGreenReview(judgment="ok", concerns=[], evidence_requests=[])).run()
    assert result.status is runner.TerminalStatus.DONE
    events = [json.loads(line) for line in result.ledger_path.read_text().splitlines()]
    admitted = next(event for event in events if event["event"] == "learning_delta_admitted")
    packet = next(event["payload"]["packet"] for event in events if event["event"] == "proposal_packet")
    assert events.index(admitted) < next(index for index, event in enumerate(events) if event["event"] == "stage_start" and event["payload"]["stage"] == "investigation")
    assert seen["proposals"] == 1
    assert seen["investigation"] == seen["proposal"] == packet["learning_context"]
    assert str(trace) not in json.dumps(packet)


def test_v2_safe_output_excerpt_has_visible_truncation_and_hashes(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    trace = tmp_path / "prior.jsonl"
    delta = _v2_delta(runner, repo, trace)
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    output = "x" * 2001
    records[1]["payload"]["transcript"][0]["result"]["stdout"] = output
    preview_hash = runner.sha256_text(json.dumps({"transcript": records[1]["payload"]["transcript"]}, sort_keys=True))
    records[1]["payload"]["sha256"] = preview_hash
    trace.write_text("".join(json.dumps(record) + "\n" for record in records))
    delta["failure_reference"]["preview_payload_sha256"] = preview_hash
    delta["failure_reference"]["source_trace_sha256"] = runner.sha256_bytes(trace.read_bytes())
    delta_path = tmp_path / "delta.json"
    delta_path.write_text(json.dumps(delta))
    seen: dict[str, object] = {}
    def investigate(mission):
        seen["safe_view"] = mission.learning_context
        return runner.Investigation(facts=[], category="test")
    result = runner.MiniSWEWorkflow(make_mission(runner, repo).model_copy(update={"learning_delta_file": str(delta_path)}), tmp_path / "runs", investigation_source=investigate, proposal_source=lambda _m, _i: successful_proposal(runner, repo), review_source=lambda *_args: runner.PostGreenReview(judgment="ok", concerns=[], evidence_requests=[])).run()
    assert result.status is runner.TerminalStatus.DONE
    failure = seen["safe_view"]["failure"]
    assert failure["failed_command_excerpt"] == output[:2000]
    assert failure["excerpt_truncated"] is True
    assert failure["full_redacted_output_sha256"] == runner.sha256_text(f"{output}\n")
    admitted = next(json.loads(line) for line in result.ledger_path.read_text().splitlines() if json.loads(line)["event"] == "learning_delta_admitted")
    assert admitted["payload"]["safe_view_sha256"] == runner.sha256_text(json.dumps(seen["safe_view"], sort_keys=True))


def test_v2_malformed_delta_file_blocks_before_sources_and_admission(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    delta_path = tmp_path / "malformed-delta.json"
    delta_path.write_text("{malformed")
    mission = make_mission(runner, repo).model_copy(update={"learning_delta_file": str(delta_path)})
    result = runner.MiniSWEWorkflow(mission, tmp_path / "runs", investigation_source=lambda _m: pytest.fail("investigator called"), proposal_source=lambda _m, _i: pytest.fail("proposer called")).run()
    assert result.status is runner.TerminalStatus.BLOCKED
    events = [json.loads(line) for line in result.ledger_path.read_text().splitlines()]
    assert not any(event["event"] == "learning_delta_admitted" for event in events)


@pytest.mark.parametrize("falsification", ["malformed", "nonterminal", "nonmonotonic", "source_symlink", "mission", "proposal", "preview_hash", "reason", "passed", "real_escape", "multiple_preview", "final_item", "unknown_phase"])
def test_v2_admission_falsifies_frozen_trace_contract(tmp_path, falsification):
    runner = runner_module()
    repo = make_repo(tmp_path)
    trace = tmp_path / "prior.jsonl"
    delta = _v2_delta(runner, repo, trace)
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    if falsification == "malformed":
        trace.write_text("{not-json}\n")
    elif falsification == "nonterminal":
        records[-1]["event"] = "review"
    elif falsification == "nonmonotonic":
        records[1]["sequence"] = 1
    elif falsification == "source_symlink":
        target = tmp_path / "trace-target.jsonl"
        target.write_bytes(trace.read_bytes())
        trace.unlink()
        trace.symlink_to(target)
    elif falsification == "mission":
        records[0]["payload"]["mission_id"] = "wrong"
    elif falsification == "proposal":
        records[1]["payload"]["proposal_sha256"] = "b" * 64
    elif falsification == "preview_hash":
        records[1]["payload"]["sha256"] = "b" * 64
        delta["failure_reference"]["preview_payload_sha256"] = "b" * 64
    elif falsification == "reason":
        records[-1]["payload"]["reason"] = "PREVIEW_COMMAND_MUTATION"
        delta["failure_reference"]["terminal_reason"] = "PREVIEW_COMMAND_MUTATION"
    elif falsification == "passed":
        records[1]["payload"]["passed"] = True
    elif falsification == "real_escape":
        records[1]["payload"]["transcript"][0]["real_changed_paths"] = ["source.py"]
    elif falsification == "multiple_preview":
        records.insert(2, {**records[1], "sequence": 3})
        records[-1]["sequence"] = 4
        delta["failure_reference"]["terminal_sequence"] = 4
    elif falsification == "final_item":
        records[1]["payload"]["transcript"].append({"phase": "full_green", "result": {}, "real_changed_paths": []})
        preview_hash = runner.sha256_text(json.dumps({"transcript": records[1]["payload"]["transcript"]}, sort_keys=True))
        records[1]["payload"]["sha256"] = preview_hash
        delta["failure_reference"]["preview_payload_sha256"] = preview_hash
    elif falsification == "unknown_phase":
        records[1]["payload"]["transcript"][0]["phase"] = "unknown_phase"
        delta["failure_reference"]["failed_phase"] = "unknown_phase"
    if falsification not in {"malformed", "source_symlink"}:
        trace.write_text("".join(json.dumps(record) + "\n" for record in records))
    if falsification not in {"source_symlink"}:
        delta["failure_reference"]["source_trace_sha256"] = runner.sha256_bytes(trace.read_bytes())
    mission = make_mission(runner, repo).model_copy(update={"learning_delta": runner.LearningDelta.model_validate(delta)})
    with pytest.raises(runner.LearningAdmissionError):
        runner._admit_learning_delta(mission)


def test_v2_runtime_typed_and_file_ambiguity_and_lowercase_hashes(tmp_path):
    runner = runner_module()
    repo = make_repo(tmp_path)
    trace = tmp_path / "prior.jsonl"
    delta = _v2_delta(runner, repo, trace)
    delta_path = tmp_path / "delta.json"
    delta_path.write_text(json.dumps(delta))
    mission = make_mission(runner, repo).model_copy(update={"learning_delta": runner.LearningDelta.model_validate(delta), "learning_delta_file": str(delta_path)})
    with pytest.raises(runner.LearningAdmissionError):
        runner._admit_learning_delta(mission)
    delta["failure_reference"]["source_trace_sha256"] = "A" * 64
    with pytest.raises(ValidationError):
        runner.LearningDelta.model_validate(delta)


def test_v2_admission_blocks_in_place_source_drift_and_live_input_omits_trace_path(tmp_path, monkeypatch):
    runner = runner_module()
    repo = make_repo(tmp_path)
    trace = tmp_path / "prior.jsonl"
    delta = _v2_delta(runner, repo, trace)
    mission = make_mission(runner, repo).model_copy(update={"learning_delta": runner.LearningDelta.model_validate(delta)})
    original = runner._read_regular_file_once
    reads = 0
    def drift(path):
        nonlocal reads
        reads += 1
        if reads == 2:
            trace.write_text(trace.read_text() + "\n")
        return original(path)
    monkeypatch.setattr(runner, "_read_regular_file_once", drift)
    with pytest.raises(runner.LearningAdmissionError, match="changed during admission"):
        runner._admit_learning_delta(mission)

    admitted = runner._admit_learning_delta(make_mission(runner, repo).model_copy(update={"learning_delta": runner.LearningDelta.model_validate(_v2_delta(runner, repo, trace))}))
    assert admitted is not None
    safe_view = admitted[1]
    workflow = runner.MiniSWEWorkflow(make_mission(runner, repo).model_copy(update={"learning_context": safe_view}), tmp_path / "runs")
    captured: dict[str, str] = {}
    class FakeReAct:
        def __init__(self, _signature, tools):
            self.tools = tools
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return type("Prediction", (), {"facts": [], "category": "test"})()
    monkeypatch.setattr(workflow, "_configure_live", lambda: None)
    monkeypatch.setattr(runner.dspy, "ReAct", FakeReAct)
    workflow._live_investigation(runner.EventLedger(tmp_path / "live-trace.jsonl"))
    assert str(trace) not in captured["reviewed_learning"]


def test_v2_cli_rejects_repeated_delta_file_flag(monkeypatch):
    runner = runner_module()
    monkeypatch.setattr(sys, "argv", ["runner", "--mission-file", "mission.json", "--repo-root", "repo", "--run-root", "runs", "--learning-delta-file", "one.json", "--learning-delta-file", "two.json"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
