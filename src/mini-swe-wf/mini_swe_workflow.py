"""Contained mission-driven DSPy software-development workflow (V0).

DSPy investigates and proposes; Python owns repository safety, state transitions,
verification, evidence, and DONE/BLOCKED. Artifacts are append-only JSONL outside
the mission repository. This file is generic: missions arrive as validated JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import dspy
from pydantic import BaseModel, ConfigDict, Field

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ScopeViolation(ValueError):
    pass


class DirtyAllowedPaths(ValueError):
    pass


class EditValidationError(ValueError):
    pass


class TerminalStatus(str, Enum):
    DONE = "DONE"
    BLOCKED = "BLOCKED"


class WorkflowState(str, Enum):
    PREFLIGHT = "PREFLIGHT"
    INVESTIGATED = "INVESTIGATED"
    PROPOSED = "PROPOSED"
    RED_CONFIRMED = "RED_CONFIRMED"
    GREEN_CONFIRMED = "GREEN_CONFIRMED"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


class MissionSpec(BaseModel):
    """Portable mission data. Repository/model/run-root arrive separately at invocation."""

    model_config = ConfigDict(extra="forbid")

    mission_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    allowed_paths: list[str] = Field(min_length=1)
    focused_test_argv: list[str] = Field(min_length=1)
    full_test_argv: list[str] = Field(min_length=1)
    expected_red_text: str = Field(min_length=1)
    expected_red_returncode: int = 1
    max_attempts: int = Field(default=1, ge=1, le=3)
    timeout_seconds: float = Field(default=120.0, gt=0, le=600)


class Mission(MissionSpec):
    model_config = ConfigDict(extra="forbid")

    repo_root: str = Field(min_length=1)
    model_name: str = "gemini/gemini-3.5-flash-lite"
    reviewed_learning: list[str] = Field(default_factory=list)
    mission_provenance: str = "in-memory"


class ExactTextEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    expected_sha256: str = Field(min_length=64, max_length=64)
    old_text: str
    new_text: str


class ChangeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_edits: list[ExactTextEdit] = Field(default_factory=list)
    production_edits: list[ExactTextEdit] = Field(default_factory=list)


class Investigation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[str] = Field(default_factory=list)
    category: str = Field(min_length=1)


class PostGreenReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgment: str
    concerns: list[str] = Field(default_factory=list)
    evidence_requests: list[str] = Field(default_factory=list)


class ReadResult(BaseModel):
    path: str
    content: str
    truncated: bool


class SearchMatch(BaseModel):
    path: str
    line: int
    text: str


class SearchResult(BaseModel):
    matches: list[SearchMatch]
    truncated: bool


class AppliedEdit(BaseModel):
    path: str
    before_sha256: str
    after_sha256: str


class CommandResult(BaseModel):
    argv: list[str]
    allowed: bool
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool


class GateEvidence(BaseModel):
    preflight_ok: bool
    live_proposal_valid: bool
    baseline_green: bool
    red_confirmed: bool
    production_applied: bool
    focused_green: bool
    full_green: bool
    scope_ok: bool
    revision_stable: bool
    rollback_performed: bool


class TerminalOutcome(BaseModel):
    status: TerminalStatus
    reason: str | None = None


@dataclass(frozen=True)
class RunResult:
    status: TerminalStatus
    reason: str | None
    ledger_path: Path
    run_directory: Path


_SECRET_KEY_PARTS = ("token", "secret", "password", "api_key", "authorization", "credential")
_SECRET_PATTERNS = (
    re.compile(r"((?:GEMINI_API_KEY|OPENAI_API_KEY|VERTEXAI_[A-Z_]+)\s*=\s*)[^\s,;]+", re.I),
    re.compile(r"(Bearer\s+)[^\s,;]+", re.I),
    re.compile(r"(projects/)[^/\s,;]+", re.I),
)
_IGNORED_SNAPSHOT_PARTS = {".git", ".venv", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache", ".mini-swe-runs"}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normal_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or "." in path.parts or ".." in path.parts:
        raise ScopeViolation(f"Expected normalized repository-relative path: {value!r}")
    return path


def _is_env_path(path: str) -> bool:
    return any(part == ".env" or part.startswith(".env.") for part in PurePosixPath(path).parts)


def _is_sensitive_path(path: Path) -> bool:
    lowered = path.name.lower()
    return _is_env_path(path.as_posix()) or any(word in lowered for word in ("credential", "secret", "token", "password"))


def resolve_repo_file(repo_root: Path | str, relative_path: str) -> Path:
    root = Path(repo_root).resolve(strict=True)
    relative = _normal_relative_path(relative_path)
    candidate = root.joinpath(*relative.parts)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ScopeViolation(f"Symlink mission path is forbidden: {relative_path}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ScopeViolation(f"Mission path does not exist: {relative_path}") from error
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ScopeViolation(f"Mission path is not a regular file: {relative_path}")
    return resolved


def _is_allowed(relative_path: str, allowed_paths: list[str]) -> bool:
    target = _normal_relative_path(relative_path)
    return any(target == _normal_relative_path(item) or _normal_relative_path(item) in target.parents for item in allowed_paths)


def _validate_allowed_file(repo_root: Path | str, allowed_paths: list[str], relative_path: str) -> Path:
    if _is_env_path(relative_path) or not _is_allowed(relative_path, allowed_paths):
        raise ScopeViolation(f"Outside mission allow-list: {relative_path}")
    return resolve_repo_file(repo_root, relative_path)


def bounded_read(repo_root: Path | str, relative_path: str, max_bytes: int = 24_000) -> ReadResult:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    target = resolve_repo_file(repo_root, relative_path)
    data = target.read_bytes()
    return ReadResult(
        path=relative_path,
        content=data[:max_bytes].decode("utf-8", errors="replace"),
        truncated=len(data) > max_bytes,
    )


def bounded_search(repo_root: Path | str, query: str, max_results: int = 20, max_bytes: int = 48_000) -> SearchResult:
    if not query:
        raise ValueError("query must not be empty")
    root = Path(repo_root).resolve(strict=True)
    matches: list[SearchMatch] = []
    bytes_seen = 0
    truncated = False
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            any(part in _IGNORED_SNAPSHOT_PARTS for part in relative.parts)
            or path.is_symlink()
            or not path.is_file()
            or _is_sensitive_path(relative)
        ):
            continue
        remaining = max_bytes - bytes_seen
        if remaining <= 0:
            return SearchResult(matches=matches, truncated=True)
        data = path.read_bytes()
        visible = data[:remaining]
        bytes_seen += len(visible)
        truncated = truncated or len(visible) < len(data)
        for line_number, line in enumerate(visible.decode("utf-8", errors="replace").splitlines(), start=1):
            if query in line:
                matches.append(SearchMatch(path=relative.as_posix(), line=line_number, text=line))
                if len(matches) >= max_results:
                    return SearchResult(matches=matches, truncated=True)
    return SearchResult(matches=matches, truncated=truncated)


def _redact(value: Any, key: str | None = None) -> Any:
    if key and any(part in key.lower() for part in _SECRET_KEY_PARTS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, BaseModel):
        return _redact(value.model_dump(mode="json"))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        for pattern in _SECRET_PATTERNS:
            value = pattern.sub(r"\1[REDACTED]", value)
    return value


class EventLedger:
    """Sanitized append-only JSONL. DSPy history/callback capture is intentionally excluded."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sequence = len(self.path.read_text(encoding="utf-8").splitlines()) if self.path.exists() else 0

    def append(self, event: str, payload: dict[str, Any]) -> int:
        self.sequence += 1
        record = {"sequence": self.sequence, "timestamp": time.time(), "event": event, "payload": _redact(payload)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return self.sequence


def apply_exact_text_edit(repo_root: Path | str, allowed_paths: list[str], edit: ExactTextEdit) -> AppliedEdit:
    target = _validate_allowed_file(repo_root, allowed_paths, edit.path)
    before = target.read_text(encoding="utf-8")
    before_hash = sha256_text(before)
    if before_hash != edit.expected_sha256:
        raise EditValidationError(f"Stale preimage hash: {edit.path}")
    if before.count(edit.old_text) != 1:
        raise EditValidationError(f"Expected exactly one old-text match: {edit.path}")
    after = before.replace(edit.old_text, edit.new_text, 1)
    mode = target.stat().st_mode
    temporary = target.with_name(f".{target.name}.mini-swe-{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(after, encoding="utf-8")
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    actual_hash = sha256_text(target.read_text(encoding="utf-8"))
    if actual_hash != sha256_text(after):
        raise EditValidationError(f"Post-write hash mismatch: {edit.path}")
    return AppliedEdit(path=edit.path, before_sha256=before_hash, after_sha256=actual_hash)


def _is_safe_pytest(argv: list[str], repo_root: Path) -> bool:
    expected_python = str(repo_root / ".venv" / "bin" / "python")
    return len(argv) >= 3 and argv[0] == expected_python and argv[1:3] == ["-m", "pytest"]


def run_allowed_command(argv: list[str], cwd: Path | str, allowed_argvs: list[list[str]], timeout_seconds: float, repo_root: Path | str) -> CommandResult:
    root = Path(repo_root).resolve()
    if not any(argv == allowed for allowed in allowed_argvs) or not _is_safe_pytest(argv, root):
        return CommandResult(argv=argv, allowed=False, returncode=None, stdout="", stderr="command not allow-listed safe pytest", timed_out=False)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(argv, cwd=Path(cwd), shell=False, env=env, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        return CommandResult(argv=argv, allowed=True, returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr, timed_out=False)
    except subprocess.TimeoutExpired as error:
        return CommandResult(argv=argv, allowed=True, returncode=None, stdout=error.stdout or "", stderr=error.stderr or "", timed_out=True)


def decide_terminal(evidence: GateEvidence) -> TerminalOutcome:
    checks = (
        (evidence.preflight_ok, "PREFLIGHT_FAILED"),
        (evidence.live_proposal_valid, "INVALID_PROPOSAL"),
        (evidence.baseline_green, "BASELINE_FAILED"),
        (evidence.red_confirmed, "RED_NOT_CONFIRMED"),
        (evidence.production_applied, "PRODUCTION_NOT_APPLIED"),
        (evidence.focused_green, "FOCUSED_GREEN_FAILED"),
        (evidence.full_green, "FULL_GREEN_FAILED"),
        (evidence.scope_ok, "OUT_OF_SCOPE_MUTATION"),
        (evidence.revision_stable, "REVISION_DRIFT"),
        (not evidence.rollback_performed, "ROLLBACK_PERFORMED"),
    )
    for passed, reason in checks:
        if not passed:
            return TerminalOutcome(status=TerminalStatus.BLOCKED, reason=reason)
    return TerminalOutcome(status=TerminalStatus.DONE)


def _git_revision(root: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, shell=False, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _dirty_allowed_paths(root: Path, allowed_paths: list[str]) -> list[str]:
    result = subprocess.run(["git", "status", "--porcelain", "--", *allowed_paths], cwd=root, shell=False, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ScopeViolation("Repository must support git status for contained missions")
    return [line for line in result.stdout.splitlines() if line]


def _scoped_git_diff(root: Path, allowed_paths: list[str]) -> str:
    result = subprocess.run(["git", "diff", "HEAD", "--no-ext-diff", "--", *allowed_paths], cwd=root, shell=False, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ScopeViolation("Repository must support scoped git diff for contained missions")
    return result.stdout


def _snapshot_workspace(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in _IGNORED_SNAPSHOT_PARTS for part in relative.parts) or path.is_symlink() or not path.is_file():
            continue
        result[relative.as_posix()] = sha256_bytes(path.read_bytes())
    return result


def _changed_paths(before: dict[str, str], after: dict[str, str]) -> set[str]:
    return {path for path in set(before) | set(after) if before.get(path) != after.get(path)}


def _scope_clean(changed: set[str], allowed_paths: list[str]) -> bool:
    return all(_is_allowed(path, allowed_paths) and not _is_env_path(path) for path in changed)


def _restore_preimages(root: Path, preimages: dict[str, tuple[bytes, int]]) -> None:
    for relative_path, (content, mode) in preimages.items():
        target = resolve_repo_file(root, relative_path)
        temporary = target.with_name(f".{target.name}.mini-swe-rollback-{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(content)
            os.chmod(temporary, mode)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()


class InvestigateMissionSignature(dspy.Signature):
    """Use only supplied read-only repository tools. Return facts and one category; do not propose edits."""

    mission: str = dspy.InputField()
    allowed_paths: str = dspy.InputField()
    reviewed_learning: str = dspy.InputField()
    facts: list[str] = dspy.OutputField()
    category: str = dspy.OutputField()


class ProposeEditsSignature(dspy.Signature):
    """Produce exact-hash test edits first and production edits second, using only supplied evidence."""

    mission: str = dspy.InputField()
    investigation: Investigation = dspy.InputField()
    repository_evidence: str = dspy.InputField()
    proposal: ChangeProposal = dspy.OutputField()


class PostGreenReviewSignature(dspy.Signature):
    """Fresh-context review of frozen investigation, proposal, diff, and verification evidence; it cannot decide DONE."""

    mission: str = dspy.InputField()
    investigation: Investigation = dspy.InputField()
    proposal: ChangeProposal = dspy.InputField()
    verification_evidence: str = dspy.InputField()
    judgment: str = dspy.OutputField()
    concerns: list[str] = dspy.OutputField()
    evidence_requests: list[str] = dspy.OutputField()


InvestigationSource = Callable[[Mission], Investigation]
ProposalSource = Callable[[Mission, Investigation], ChangeProposal]
ReviewSource = Callable[[Mission, Investigation, ChangeProposal, str], PostGreenReview]


class MiniSWEWorkflow:
    def __init__(self, mission: Mission, run_root: Path | str, investigation_source: InvestigationSource | None = None, proposal_source: ProposalSource | None = None, review_source: ReviewSource | None = None):
        self.mission = mission
        self.repo_root = Path(mission.repo_root).resolve()
        self.run_root = Path(run_root).resolve()
        self.investigation_source = investigation_source
        self.proposal_source = proposal_source
        self.review_source = review_source
        self.live_configured = False

    def _run_directory(self) -> Path:
        return self.run_root / f"{self.mission.mission_id}-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    def _stage_start(self, ledger: EventLedger, stage: str) -> None:
        ledger.append("stage_start", {"stage": stage})

    def _stage_end(self, ledger: EventLedger, stage: str, payload: dict[str, Any]) -> None:
        ledger.append("stage_end", {"stage": stage, **payload})

    def _stage_error(self, ledger: EventLedger, stage: str, error: Exception) -> None:
        ledger.append("stage_error", {"stage": stage, "error": repr(error)})

    def _preflight(self) -> dict[str, Any]:
        if not self.repo_root.is_dir() or self.run_root.is_relative_to(self.repo_root):
            raise ScopeViolation("Repository root must exist and run root must be external")
        if self.mission.expected_red_returncode != 1:
            raise ScopeViolation("V0 RED must require pytest return code 1")
        for path in self.mission.allowed_paths:
            _validate_allowed_file(self.repo_root, self.mission.allowed_paths, path)
        for command in (self.mission.focused_test_argv, self.mission.full_test_argv):
            if not _is_safe_pytest(command, self.repo_root):
                raise ScopeViolation("Mission commands must invoke pytest through repo .venv/bin/python")
        dirty = _dirty_allowed_paths(self.repo_root, self.mission.allowed_paths)
        if dirty:
            raise DirtyAllowedPaths("; ".join(dirty))
        return {
            "mission_id": self.mission.mission_id,
            "repo_root": str(self.repo_root),
            "workspace_root": str(self.repo_root.parent),
            "run_root": str(self.run_root),
            "mission_provenance": self.mission.mission_provenance,
            "model": self.mission.model_name,
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "dspy_version": dspy.__version__,
            "git_revision": _git_revision(self.repo_root),
            "allowed_paths": self.mission.allowed_paths,
            "reviewed_learning": self.mission.reviewed_learning,
        }

    def _configure_live(self) -> None:
        if self.live_configured:
            return
        from common.utils import dspy_configure, get_lm_for_model_name

        dspy_configure(get_lm_for_model_name(self.mission.model_name, reasoning_effort="disable"))
        self.live_configured = True

    def _live_investigation(self, ledger: EventLedger) -> Investigation:
        self._configure_live()

        def read_repo_file(path: str) -> str:
            try:
                result = bounded_read(self.repo_root, path)
                ledger.append("tool", {"tool": "read_repo_file", "input": {"path": path}, "output": result})
                return result.model_dump_json()
            except Exception as error:
                ledger.append("tool_error", {"tool": "read_repo_file", "input": {"path": path}, "error": repr(error)})
                return f"TOOL_ERROR: {error}"

        def search_repo(query: str) -> str:
            try:
                result = bounded_search(self.repo_root, query)
                ledger.append("tool", {"tool": "search_repo", "input": {"query": query}, "output": result})
                return result.model_dump_json()
            except Exception as error:
                ledger.append("tool_error", {"tool": "search_repo", "input": {"query": query}, "error": repr(error)})
                return f"TOOL_ERROR: {error}"

        prediction = dspy.ReAct(InvestigateMissionSignature, tools=[read_repo_file, search_repo])(
            mission=self.mission.goal,
            allowed_paths=", ".join(self.mission.allowed_paths),
            reviewed_learning=json.dumps(self.mission.reviewed_learning),
        )
        trajectory = getattr(prediction, "trajectory", None)
        if trajectory is not None:
            ledger.append("react_trajectory", {"trajectory": trajectory})
        return Investigation(facts=list(prediction.facts), category=str(prediction.category))

    def _repository_evidence(self) -> str:
        records = []
        for path in self.mission.allowed_paths:
            target = _validate_allowed_file(self.repo_root, self.mission.allowed_paths, path)
            content = target.read_text(encoding="utf-8")
            records.append({"path": path, "sha256": sha256_text(content), "content": content})
        return json.dumps(records, sort_keys=True)

    def _live_proposal(self, investigation: Investigation) -> ChangeProposal:
        self._configure_live()
        prediction = dspy.Predict(ProposeEditsSignature)(
            mission=f"{self.mission.goal}\nExpected RED text: {self.mission.expected_red_text}",
            investigation=investigation,
            repository_evidence=self._repository_evidence(),
        )
        return prediction.proposal if isinstance(prediction.proposal, ChangeProposal) else ChangeProposal.model_validate(prediction.proposal)

    def _live_review(self, investigation: Investigation, proposal: ChangeProposal, evidence: str) -> PostGreenReview:
        self._configure_live()
        prediction = dspy.Predict(PostGreenReviewSignature)(
            mission=self.mission.goal,
            investigation=investigation,
            proposal=proposal,
            verification_evidence=evidence,
        )
        return PostGreenReview(judgment=str(prediction.judgment), concerns=list(prediction.concerns), evidence_requests=list(prediction.evidence_requests))

    def _blocked(self, ledger: EventLedger, run_directory: Path, reason: str, preimages: dict[str, tuple[bytes, int]] | None, details: dict[str, Any] | None = None) -> RunResult:
        rollback = bool(preimages)
        if preimages:
            _restore_preimages(self.repo_root, preimages)
            ledger.append("rollback", {"paths": sorted(preimages)})
        ledger.append("terminal", {"state": WorkflowState.BLOCKED, "reason": reason, "rollback_performed": rollback, "details": details or {}})
        return RunResult(TerminalStatus.BLOCKED, reason, ledger.path, run_directory)

    def run(self) -> RunResult:
        run_directory = self._run_directory()
        ledger = EventLedger(run_directory / "trace.jsonl")
        preimages: dict[str, tuple[bytes, int]] = {}
        applied_paths: set[str] = set()
        baseline_snapshot: dict[str, str] = {}
        initial_revision: str | None = None

        def rollback_preimages() -> dict[str, tuple[bytes, int]] | None:
            return {path: preimages[path] for path in applied_paths} if applied_paths else None

        def scope_gate(phase: str) -> tuple[bool, set[str]]:
            changed = _changed_paths(baseline_snapshot, _snapshot_workspace(self.repo_root))
            clean = _scope_clean(changed, self.mission.allowed_paths)
            ledger.append("scope_check", {"phase": phase, "changed_paths": sorted(changed), "scope_ok": clean})
            return clean, changed

        allowed_commands = [self.mission.focused_test_argv, self.mission.full_test_argv]
        try:
            preflight = self._preflight()
            initial_revision = preflight["git_revision"]
            ledger.append("preflight", preflight)
            ledger.append("transition", {"from": None, "to": WorkflowState.PREFLIGHT, "reason": "preflight passed"})
            baseline_snapshot = _snapshot_workspace(self.repo_root)
            preimages = {path: (_validate_allowed_file(self.repo_root, self.mission.allowed_paths, path).read_bytes(), _validate_allowed_file(self.repo_root, self.mission.allowed_paths, path).stat().st_mode) for path in self.mission.allowed_paths}

            self._stage_start(ledger, "baseline")
            baseline_focused = run_allowed_command(self.mission.focused_test_argv, self.repo_root, allowed_commands, self.mission.timeout_seconds, self.repo_root)
            ledger.append("command", {"phase": "baseline_focused", "result": baseline_focused})
            clean, changed = scope_gate("after_baseline_focused")
            baseline_full = run_allowed_command(self.mission.full_test_argv, self.repo_root, allowed_commands, self.mission.timeout_seconds, self.repo_root)
            ledger.append("command", {"phase": "baseline_full", "result": baseline_full})
            clean = clean and scope_gate("after_baseline_full")[0]
            baseline_green = clean and all(result.allowed and not result.timed_out and result.returncode == 0 for result in (baseline_focused, baseline_full))
            if not baseline_green:
                self._stage_error(ledger, "baseline", RuntimeError("baseline verification failed"))
                return self._blocked(ledger, run_directory, "BASELINE_FAILED", None)
            self._stage_end(ledger, "baseline", {"focused": baseline_focused, "full": baseline_full})

            self._stage_start(ledger, "investigation")
            try:
                investigation = self.investigation_source(self.mission) if self.investigation_source else self._live_investigation(ledger)
            except Exception as error:
                self._stage_error(ledger, "investigation", error)
                return self._blocked(ledger, run_directory, "INVESTIGATION_FAILED", None, {"error": repr(error)})
            if not isinstance(investigation, Investigation):
                return self._blocked(ledger, run_directory, "INVALID_INVESTIGATION", None)
            ledger.append("investigation", {"origin": "injected" if self.investigation_source else "live-react", "output": investigation})
            self._stage_end(ledger, "investigation", {"category": investigation.category})
            ledger.append("transition", {"from": WorkflowState.PREFLIGHT, "to": WorkflowState.INVESTIGATED, "reason": "investigation persisted"})

            self._stage_start(ledger, "proposal")
            try:
                proposal = self.proposal_source(self.mission, investigation) if self.proposal_source else self._live_proposal(investigation)
            except Exception as error:
                self._stage_error(ledger, "proposal", error)
                return self._blocked(ledger, run_directory, "PROPOSAL_FAILED", None, {"error": repr(error)})
            if not isinstance(proposal, ChangeProposal) or not proposal.test_edits:
                return self._blocked(ledger, run_directory, "INVALID_PROPOSAL", None)
            ledger.append("proposal", {"origin": "injected" if self.proposal_source else "live-predict", "output": proposal})
            self._stage_end(ledger, "proposal", {"test_edit_count": len(proposal.test_edits), "production_edit_count": len(proposal.production_edits)})
            ledger.append("transition", {"from": WorkflowState.INVESTIGATED, "to": WorkflowState.PROPOSED, "reason": "typed proposal validated"})

            self._stage_start(ledger, "test_edit")
            for edit in proposal.test_edits:
                applied = apply_exact_text_edit(self.repo_root, self.mission.allowed_paths, edit)
                applied_paths.add(edit.path)
                ledger.append("test_edit", {"edit": applied})
            clean, changed = scope_gate("after_test_edit")
            if not clean:
                self._stage_error(ledger, "test_edit", RuntimeError("scope violation"))
                return self._blocked(ledger, run_directory, "OUT_OF_SCOPE_MUTATION", rollback_preimages(), {"changed_paths": sorted(changed)})
            self._stage_end(ledger, "test_edit", {"paths": sorted(applied_paths)})

            self._stage_start(ledger, "red")
            red = run_allowed_command(self.mission.focused_test_argv, self.repo_root, allowed_commands, self.mission.timeout_seconds, self.repo_root)
            ledger.append("command", {"phase": "focused_red", "result": red})
            clean, changed = scope_gate("after_red")
            red_confirmed = clean and red.allowed and not red.timed_out and red.returncode == self.mission.expected_red_returncode and self.mission.expected_red_text in f"{red.stdout}\n{red.stderr}"
            if not red_confirmed:
                self._stage_error(ledger, "red", RuntimeError("expected pytest assertion failure not observed"))
                return self._blocked(ledger, run_directory, "RED_NOT_CONFIRMED" if clean else "OUT_OF_SCOPE_MUTATION", rollback_preimages(), {"command": red.model_dump(), "changed_paths": sorted(changed)})
            ledger.append("red_confirmed", {"returncode": red.returncode, "expected_text": self.mission.expected_red_text})
            self._stage_end(ledger, "red", {"returncode": red.returncode})
            ledger.append("transition", {"from": WorkflowState.PROPOSED, "to": WorkflowState.RED_CONFIRMED, "reason": "expected pytest failure"})

            if not proposal.production_edits:
                return self._blocked(ledger, run_directory, "INVALID_PROPOSAL", rollback_preimages(), {"error": "production_edits is empty"})
            self._stage_start(ledger, "production_edit")
            production_paths: list[str] = []
            for edit in proposal.production_edits:
                applied = apply_exact_text_edit(self.repo_root, self.mission.allowed_paths, edit)
                applied_paths.add(edit.path)
                production_paths.append(edit.path)
                ledger.append("production_edit", {"edit": applied})
            clean, changed = scope_gate("after_production_edit")
            if not clean:
                self._stage_error(ledger, "production_edit", RuntimeError("scope violation"))
                return self._blocked(ledger, run_directory, "OUT_OF_SCOPE_MUTATION", rollback_preimages(), {"changed_paths": sorted(changed)})
            self._stage_end(ledger, "production_edit", {"paths": production_paths})

            self._stage_start(ledger, "focused_green")
            focused = run_allowed_command(self.mission.focused_test_argv, self.repo_root, allowed_commands, self.mission.timeout_seconds, self.repo_root)
            ledger.append("command", {"phase": "focused_green", "result": focused})
            clean, changed = scope_gate("after_focused_green")
            if not (clean and focused.allowed and not focused.timed_out and focused.returncode == 0):
                self._stage_error(ledger, "focused_green", RuntimeError("focused verification failed"))
                return self._blocked(ledger, run_directory, "FOCUSED_GREEN_FAILED" if clean else "OUT_OF_SCOPE_MUTATION", rollback_preimages(), {"command": focused.model_dump()})
            self._stage_end(ledger, "focused_green", {"returncode": focused.returncode})

            self._stage_start(ledger, "full_green")
            full = run_allowed_command(self.mission.full_test_argv, self.repo_root, allowed_commands, self.mission.timeout_seconds, self.repo_root)
            ledger.append("command", {"phase": "full_green", "result": full})
            clean, changed = scope_gate("after_full_green")
            if not (clean and full.allowed and not full.timed_out and full.returncode == 0):
                self._stage_error(ledger, "full_green", RuntimeError("full verification failed"))
                return self._blocked(ledger, run_directory, "FULL_GREEN_FAILED" if clean else "OUT_OF_SCOPE_MUTATION", rollback_preimages(), {"command": full.model_dump()})
            self._stage_end(ledger, "full_green", {"returncode": full.returncode})
            ledger.append("transition", {"from": WorkflowState.RED_CONFIRMED, "to": WorkflowState.GREEN_CONFIRMED, "reason": "focused and full pytest green"})

            self._stage_start(ledger, "review")
            review_input = _redact({
                "investigation": investigation.model_dump(mode="json"),
                "proposal": proposal.model_dump(mode="json"),
                "scoped_git_diff": _scoped_git_diff(self.repo_root, self.mission.allowed_paths),
                "changed_paths": sorted(changed),
                "focused_command": focused.model_dump(mode="json"),
                "full_command": full.model_dump(mode="json"),
            })
            review_evidence = json.dumps(review_input, sort_keys=True)
            ledger.append("review_input", {"sha256": sha256_text(review_evidence), "input": review_input})
            try:
                review = self.review_source(self.mission, investigation, proposal, review_evidence) if self.review_source else self._live_review(investigation, proposal, review_evidence)
            except Exception as error:
                self._stage_error(ledger, "review", error)
                return self._blocked(ledger, run_directory, "REVIEW_FAILED", rollback_preimages(), {"error": repr(error)})
            if not isinstance(review, PostGreenReview):
                return self._blocked(ledger, run_directory, "INVALID_REVIEW", rollback_preimages())
            ledger.append("review", {"origin": "injected" if self.review_source else "live-predict", "output": review})
            self._stage_end(ledger, "review", {"concern_count": len(review.concerns), "evidence_request_count": len(review.evidence_requests)})

            outcome = decide_terminal(GateEvidence(preflight_ok=True, live_proposal_valid=True, baseline_green=True, red_confirmed=True, production_applied=bool(production_paths), focused_green=True, full_green=True, scope_ok=clean, revision_stable=_git_revision(self.repo_root) == initial_revision, rollback_performed=False))
            if outcome.status is TerminalStatus.BLOCKED:
                return self._blocked(ledger, run_directory, outcome.reason or "TERMINAL_GATE_FAILED", rollback_preimages())
            ledger.append("terminal", {"state": WorkflowState.DONE, "changed_paths": sorted(changed), "review_concerns": review.concerns})
            return RunResult(TerminalStatus.DONE, None, ledger.path, run_directory)
        except DirtyAllowedPaths as error:
            ledger.append("stage_error", {"stage": "preflight", "error": str(error)})
            return self._blocked(ledger, run_directory, "DIRTY_ALLOWED_PATHS", None, {"dirty": str(error)})
        except ScopeViolation as error:
            ledger.append("stage_error", {"stage": "preflight", "error": str(error)})
            return self._blocked(ledger, run_directory, "PREFLIGHT_FAILED", rollback_preimages(), {"error": str(error)})
        except EditValidationError as error:
            ledger.append("stage_error", {"stage": "edit", "error": str(error)})
            return self._blocked(ledger, run_directory, "EDIT_VALIDATION_FAILED", rollback_preimages(), {"error": str(error)})
        except Exception as error:
            ledger.append("stage_error", {"stage": "unhandled", "error": repr(error)})
            return self._blocked(ledger, run_directory, "UNHANDLED_ERROR", rollback_preimages(), {"error": repr(error)})


def load_mission_file(path: Path | str, repo_root: Path | str, model_name: str, reviewed_learning: list[str] | None = None) -> Mission:
    mission_path = Path(path).resolve(strict=True)
    raw = json.loads(mission_path.read_text(encoding="utf-8"))
    root = Path(repo_root).resolve()
    for command_key in ("focused_test_argv", "full_test_argv"):
        raw[command_key] = [str(root / ".venv" / "bin" / "python") if item == "${REPO_VENV_PYTHON}" else item for item in raw[command_key]]
    spec = MissionSpec.model_validate(raw)
    return Mission(**spec.model_dump(), repo_root=str(root), model_name=model_name, reviewed_learning=reviewed_learning or [], mission_provenance=str(mission_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one contained DSPy mini SWE mission")
    parser.add_argument("--mission-file", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--run-root", required=True, help="Durable external artifact root; never inside the mission repository.")
    parser.add_argument("--model", default="gemini/gemini-3.5-flash-lite")
    parser.add_argument("--reviewed-learning", action="append", default=[])
    args = parser.parse_args()
    mission = load_mission_file(args.mission_file, args.repo_root, args.model, args.reviewed_learning)
    result = MiniSWEWorkflow(mission, args.run_root).run()
    print(json.dumps({"status": result.status.value, "reason": result.reason, "ledger_path": str(result.ledger_path)}))
    return 0 if result.status is TerminalStatus.DONE else 2


if __name__ == "__main__":
    raise SystemExit(main())
