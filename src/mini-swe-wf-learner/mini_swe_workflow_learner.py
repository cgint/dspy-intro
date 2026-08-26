"""Separate, read-only learner for frozen mini-SWE evidence.

It emits only unpromoted candidate learnings; it never imports, invokes, or edits the
mini-SWE runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import dspy
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PreflightError(ValueError):
    pass


class CandidateValidationError(ValueError):
    pass


class CandidateLearning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    evidence_event_numbers: list[int] = Field(min_length=1)
    applies_to: str = Field(min_length=1)
    invalidation: str = Field(min_length=1)
    proposed_runner_change: str = Field(min_length=1)
    status: Literal["candidate"]


class LearnerRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    trace_path: Path
    trace_sha256: str = Field(min_length=64, max_length=64)
    mission_path: Path
    mission_sha256: str = Field(min_length=64, max_length=64)
    runner_path: Path
    runner_sha256: str = Field(min_length=64, max_length=64)
    diff_path: Path
    diff_sha256: str = Field(min_length=64, max_length=64)
    model: Literal["gemini/gemini-3.5-flash-lite"]
    output_dir: Path


class PreflightEvidence(BaseModel):
    event_numbers: list[int]
    terminal_sequence: int
    sanitized_events: list[dict[str, Any]]


@dataclass(frozen=True)
class LearnerResult:
    status: str
    reason: str | None
    candidates_path: Path
    manifest_path: Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise PreflightError(f"known {label} path is not a regular file: {path}")
    if sha256_file(path) != expected:
        raise PreflightError(f"{label} hash mismatch")


def preflight(request: LearnerRequest) -> PreflightEvidence:
    for path, digest, label in (
        (request.trace_path, request.trace_sha256, "trace"),
        (request.mission_path, request.mission_sha256, "mission"),
        (request.runner_path, request.runner_sha256, "runner"),
        (request.diff_path, request.diff_sha256, "diff"),
    ):
        _verify_file(path, digest, label)
    events: list[dict[str, Any]] = []
    try:
        for line in request.trace_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("sequence"), int):
                raise PreflightError("malformed trace event")
            events.append(record)
    except json.JSONDecodeError as error:
        raise PreflightError("malformed trace JSON") from error
    sequences = [event["sequence"] for event in events]
    if not sequences or sequences != sorted(sequences) or len(set(sequences)) != len(sequences):
        raise PreflightError("trace sequences must be monotonic and unique")
    terminal = events[-1]
    if terminal.get("event") != "terminal" or terminal.get("payload", {}).get("state") != "DONE":
        raise PreflightError("trace final terminal is not DONE")
    sanitized = [{"sequence": e["sequence"], "event": e.get("event"), "payload": e.get("payload", {})} for e in events]
    return PreflightEvidence(event_numbers=sequences, terminal_sequence=sequences[-1], sanitized_events=sanitized)


def validate_candidates(raw: Any, event_numbers: set[int]) -> list[CandidateLearning]:
    if not isinstance(raw, list):
        raise CandidateValidationError("candidates must be a list")
    try:
        candidates = [item if isinstance(item, CandidateLearning) else CandidateLearning.model_validate(item) for item in raw]
    except ValidationError as error:
        raise CandidateValidationError("candidate schema validation failed") from error
    for candidate in candidates:
        if candidate.status != "candidate" or not set(candidate.evidence_event_numbers) <= event_numbers:
            raise CandidateValidationError("candidate has invalid status or evidence references")
    return candidates


class LearnFromFrozenEvidence(dspy.Signature):
    """Derive zero or more atomic candidate learnings only from supplied observable evidence.
    Do not promote, certify, modify code, choose a next task, or claim hidden reasoning.
    Each candidate must cite trace event sequence numbers and concern improving later #1 runs.
    """

    observable_evidence: str = dspy.InputField()
    candidates: list[CandidateLearning] = dspy.OutputField()


def _observable_evidence(request: LearnerRequest, evidence: PreflightEvidence) -> str:
    return json.dumps({
        "trace_events": evidence.sanitized_events,
        "mission": json.loads(request.mission_path.read_text(encoding="utf-8")),
        "runner": {"path": str(request.runner_path), "sha256": request.runner_sha256, "content": request.runner_path.read_text(encoding="utf-8")},
        "g12_diff": {"sha256": request.diff_sha256, "content": request.diff_path.read_text(encoding="utf-8")},
    }, sort_keys=True)


def _live_candidates(observable_evidence: str, model: str) -> Any:
    from common.utils import dspy_configure, get_lm_for_model_name

    dspy_configure(get_lm_for_model_name(model, reasoning_effort="disable"))
    return dspy.Predict(LearnFromFrozenEvidence)(observable_evidence=observable_evidence).candidates


def run(request: LearnerRequest, candidate_source: Callable[[str], Any] | None = None) -> LearnerResult:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = request.output_dir / "candidates.json"
    manifest_path = request.output_dir / "manifest.json"
    protected_before = {label: sha256_file(path) for label, path in (("trace", request.trace_path), ("mission", request.mission_path), ("runner", request.runner_path), ("diff", request.diff_path))}
    manifest: dict[str, Any] = {"timestamp": time.time(), "model": request.model, "input_hashes": protected_before, "content_provenance": {"candidate_source": "injected" if candidate_source else "dspy-direct", "trace": str(request.trace_path), "mission": str(request.mission_path), "runner": str(request.runner_path), "diff": str(request.diff_path)}}
    try:
        evidence = preflight(request)
        raw = candidate_source(_observable_evidence(request, evidence)) if candidate_source else _live_candidates(_observable_evidence(request, evidence), request.model)
        candidates = validate_candidates(raw, set(evidence.event_numbers))
        if protected_before != {label: sha256_file(path) for label, path in (("trace", request.trace_path), ("mission", request.mission_path), ("runner", request.runner_path), ("diff", request.diff_path))}:
            raise PreflightError("protected input changed during learner execution")
        candidates_path.write_text(json.dumps([c.model_dump(mode="json") for c in candidates], indent=2) + "\n", encoding="utf-8")
        manifest.update({"status": "DONE", "terminal_sequence": evidence.terminal_sequence, "candidate_count": len(candidates), "candidates_sha256": sha256_file(candidates_path)})
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return LearnerResult("DONE", None, candidates_path, manifest_path)
    except Exception as error:
        manifest.update({"status": "BLOCKED", "reason": repr(error)})
        candidates_path.write_text("[]\n", encoding="utf-8")
        manifest["candidates_sha256"] = sha256_file(candidates_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return LearnerResult("BLOCKED", repr(error), candidates_path, manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Learn candidates from a frozen mini-SWE trace")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--trace-sha256", required=True)
    parser.add_argument("--mission", required=True)
    parser.add_argument("--mission-sha256", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gemini/gemini-3.5-flash-lite")
    args = parser.parse_args()
    output = Path(args.output_dir)
    diff_path = output / "g12.diff"
    completed = subprocess.run(["git", "diff", "--", "src/workflow/schemas.py", "tests/test_workflow.py"], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit("could not capture G12 diff")
    output.mkdir(parents=True, exist_ok=True)
    diff_path.write_text(completed.stdout, encoding="utf-8")
    request = LearnerRequest(trace_path=Path(args.trace), trace_sha256=args.trace_sha256, mission_path=Path(args.mission), mission_sha256=args.mission_sha256, runner_path=Path(args.runner), runner_sha256=args.runner_sha256, diff_path=diff_path, diff_sha256=sha256_file(diff_path), model=args.model, output_dir=output)
    result = run(request)
    print(json.dumps({"status": result.status, "reason": result.reason, "candidates_path": str(result.candidates_path), "manifest_path": str(result.manifest_path)}))
    return 0 if result.status == "DONE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
