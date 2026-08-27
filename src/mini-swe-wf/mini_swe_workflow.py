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
import shutil
import stat
import subprocess
import sys
import time
import tempfile
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import dspy
from pydantic import BaseModel, ConfigDict, Field, model_validator

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ScopeViolation(ValueError):
    pass


class DirtyAllowedPaths(ValueError):
    pass


class EditValidationError(ValueError):
    pass


class ProposalValidationError(ValueError):
    def __init__(self, path: str, reason: str):
        self.path = path
        super().__init__(f"{path}: {reason}")


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
    acceptance_test_path: str | None = None
    acceptance_expected_red_text: str | None = None
    ruff_check_argv: list[str] | None = None

    @model_validator(mode="after")
    def _acceptance_fields_are_paired(self) -> "MissionSpec":
        if (self.acceptance_test_path is None) != (self.acceptance_expected_red_text is None):
            raise ValueError("acceptance_test_path and acceptance_expected_red_text must be configured together")
        return self


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


class EvidenceCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    path: str | None = None
    query: str | None = None

    @model_validator(mode="after")
    def _one_bounded_target(self) -> "EvidenceCapability":
        if self.kind == "bounded_read" and bool(self.path) and self.query is None:
            return self
        if self.kind == "bounded_search" and bool(self.query) and self.path is None:
            return self
        raise ValueError("capability must be one bounded_read path or bounded_search query")


class SanityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    reason: str
    capability: EvidenceCapability | None = None

    @model_validator(mode="after")
    def _valid_status(self) -> "SanityDecision":
        if self.status not in {"PROCEED", "NEEDS_EVIDENCE", "BLOCKED"}:
            raise ValueError("invalid sanity status")
        if (self.status == "NEEDS_EVIDENCE") != (self.capability is not None):
            raise ValueError("only NEEDS_EVIDENCE supplies one capability")
        return self


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


@dataclass(frozen=True)
class GroupedEdit:
    path: str
    target: Path
    before_bytes: bytes
    after_bytes: bytes
    mode: int
    before_sha256: str
    after_sha256: str


@dataclass(frozen=True)
class GroupedProposal:
    test_groups: list[GroupedEdit]
    production_groups: list[GroupedEdit]


class CommandResult(BaseModel):
    argv: list[str]
    allowed: bool
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool


class AcceptanceArtifact(BaseModel):
    path: str
    sha256: str


class AcceptanceBinding(BaseModel):
    scoped_diff_sha256: str
    artifact_sha256: str


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
    acceptance_configured: bool = False
    acceptance_baseline: CommandResult | None = None
    acceptance_candidate: CommandResult | None = None
    acceptance_binding: AcceptanceBinding | None = None
    acceptance_final_binding: AcceptanceBinding | None = None
    ruff_configured: bool = False
    ruff_argv: list[str] | None = None
    ruff_baseline: CommandResult | None = None
    ruff_candidate: CommandResult | None = None


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


def validate_acceptance_artifact(repo_root: Path | str, allowed_paths: list[str], relative_path: str | None) -> AcceptanceArtifact:
    if relative_path is None or _is_env_path(relative_path) or _is_allowed(relative_path, allowed_paths):
        raise ScopeViolation("Acceptance artifact must be outside editable allow-list")
    root = Path(repo_root).resolve(strict=True)
    target = resolve_repo_file(root, relative_path)
    identity = (target.stat().st_dev, target.stat().st_ino)
    for candidate in root.rglob("*"):
        if candidate == target or candidate.is_symlink() or not candidate.is_file():
            continue
        if (candidate.stat().st_dev, candidate.stat().st_ino) == identity:
            raise ScopeViolation("Acceptance artifact aliases another repository path")
    return AcceptanceArtifact(path=relative_path, sha256=sha256_bytes(target.read_bytes()))


def acceptance_pytest_argv(repo_root: Path, relative_path: str) -> list[str]:
    return [str(repo_root / ".venv" / "bin" / "python"), "-m", "pytest", "-p", "no:cacheprovider", relative_path, "-q"]


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

    def records(self, events: set[str] | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        return [record for record in records if events is None or record["event"] in events]

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


def _canonical_edit_path(path: str) -> str:
    try:
        canonical = _normal_relative_path(path).as_posix()
    except ScopeViolation as error:
        raise ProposalValidationError(path, str(error)) from error
    if path != canonical:
        raise ProposalValidationError(path, "path alias is forbidden")
    return canonical


def _normalize_phase(repo_root: Path | str, allowed_paths: list[str], edits: list[ExactTextEdit]) -> list[GroupedEdit]:
    grouped: dict[str, list[ExactTextEdit]] = {}
    targets: dict[str, Path] = {}
    identities: dict[tuple[int, int], str] = {}
    for edit in edits:
        canonical = _canonical_edit_path(edit.path)
        try:
            target = _validate_allowed_file(repo_root, allowed_paths, canonical)
        except ScopeViolation as error:
            raise ProposalValidationError(edit.path, str(error)) from error
        identity = (target.stat().st_dev, target.stat().st_ino)
        if identity in identities and identities[identity] != canonical:
            raise ProposalValidationError(edit.path, "path alias resolves to an existing proposal target")
        identities[identity] = canonical
        grouped.setdefault(canonical, []).append(edit)
        targets[canonical] = target

    normalized: list[GroupedEdit] = []
    for path in sorted(grouped):
        entries = grouped[path]
        target = targets[path]
        before = target.read_bytes()
        try:
            before.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProposalValidationError(path, "file is not UTF-8") from error
        before_hash = sha256_bytes(before)
        if any(entry.expected_sha256 != before_hash for entry in entries):
            raise ProposalValidationError(path, "mixed or stale original full-file hash")
        spans: list[tuple[int, int, bytes]] = []
        seen_old: set[bytes] = set()
        for entry in entries:
            old_bytes = entry.old_text.encode("utf-8")
            if not old_bytes:
                raise ProposalValidationError(path, "empty old_text is forbidden")
            if old_bytes in seen_old:
                raise ProposalValidationError(path, "old_text must be unique within a grouped path")
            seen_old.add(old_bytes)
            if before.count(old_bytes) != 1:
                raise ProposalValidationError(path, "old_text must occur exactly once in original bytes")
            start = before.find(old_bytes)
            spans.append((start, start + len(old_bytes), entry.new_text.encode("utf-8")))
        spans.sort(key=lambda span: (span[0], span[1]))
        for previous, current in zip(spans, spans[1:]):
            if current[0] < previous[1]:
                raise ProposalValidationError(path, "old_text spans overlap")
        pieces: list[bytes] = []
        cursor = 0
        for start, end, replacement in spans:
            pieces.extend((before[cursor:start], replacement))
            cursor = end
        pieces.append(before[cursor:])
        after = b"".join(pieces)
        normalized.append(GroupedEdit(path, target, before, after, stat.S_IMODE(target.stat().st_mode), before_hash, sha256_bytes(after)))
    return normalized


def normalize_change_proposal(repo_root: Path | str, allowed_paths: list[str], proposal: ChangeProposal) -> GroupedProposal:
    """Validate and stage every final file before any grouped edit is written."""
    test_groups = _normalize_phase(repo_root, allowed_paths, proposal.test_edits)
    production_groups = _normalize_phase(repo_root, allowed_paths, proposal.production_edits)
    test_identities = {(group.target.stat().st_dev, group.target.stat().st_ino): group.path for group in test_groups}
    for group in production_groups:
        identity = (group.target.stat().st_dev, group.target.stat().st_ino)
        if identity in test_identities:
            raise ProposalValidationError(group.path, "target cannot appear in both test and production phases")
    return GroupedProposal(test_groups=test_groups, production_groups=production_groups)


def apply_grouped_edit(group: GroupedEdit) -> AppliedEdit:
    """Atomically write one fully staged file and prove its byte-level postimage."""
    temporary = group.target.with_name(f".{group.target.name}.mini-swe-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(group.after_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, group.mode)
        os.replace(temporary, group.target)
        actual = group.target.read_bytes()
        if sha256_bytes(actual) != group.after_sha256:
            raise EditValidationError(f"Post-write byte hash mismatch: {group.path}")
    except OSError as error:
        raise EditValidationError(f"Atomic grouped write failed: {group.path}: {error}") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return AppliedEdit(path=group.path, before_sha256=group.before_sha256, after_sha256=group.after_sha256)


def _bounded_proposal_evidence(proposal: Any, max_edits: int = 32, max_text_chars: int = 512) -> dict[str, Any]:
    if not isinstance(proposal, ChangeProposal):
        return {"type": type(proposal).__name__}

    def project(edits: list[ExactTextEdit]) -> dict[str, Any]:
        return {
            "count": len(edits),
            "edits": [
                {
                    "path": edit.path,
                    "expected_sha256": edit.expected_sha256,
                    "old_text_sha256": sha256_text(edit.old_text),
                    "new_text_sha256": sha256_text(edit.new_text),
                    "old_text_chars": len(edit.old_text),
                    "new_text_chars": len(edit.new_text),
                    "old_text_preview": edit.old_text[:max_text_chars],
                    "new_text_preview": edit.new_text[:max_text_chars],
                }
                for edit in edits[:max_edits]
            ],
            "truncated": len(edits) > max_edits,
        }

    return {"test_edits": project(proposal.test_edits), "production_edits": project(proposal.production_edits)}


def _is_safe_pytest(argv: list[str], repo_root: Path) -> bool:
    expected_python = str(repo_root / ".venv" / "bin" / "python")
    return len(argv) >= 3 and argv[0] == expected_python and argv[1:3] == ["-m", "pytest"]


def _is_safe_ruff(argv: list[str], repo_root: Path) -> bool:
    root = repo_root.resolve()
    expected = [str(root / ".venv" / "bin" / "python"), "-m", "ruff", "check", "--no-cache"]
    if argv[:5] != expected or len(argv) == 5:
        return False
    for operand in argv[5:]:
        candidate = Path(operand)
        if candidate.is_absolute() or ".." in candidate.parts or candidate.suffix != ".py":
            return False
        raw = root / candidate
        if raw.is_symlink() or not raw.is_file():
            return False
        try:
            resolve_repo_file(root, operand)
        except ScopeViolation:
            return False
    return True


def run_allowed_command(argv: list[str], cwd: Path | str, allowed_argvs: list[list[str]], timeout_seconds: float, repo_root: Path | str) -> CommandResult:
    root = Path(repo_root).resolve()
    if not any(argv == allowed for allowed in allowed_argvs) or not (_is_safe_pytest(argv, root) or _is_safe_ruff(argv, root)):
        return CommandResult(argv=argv, allowed=False, returncode=None, stdout="", stderr="command not allow-listed safe pytest or ruff", timed_out=False)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(argv, cwd=Path(cwd), shell=False, env=env, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        return CommandResult(argv=argv, allowed=True, returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr, timed_out=False)
    except subprocess.TimeoutExpired as error:
        return CommandResult(argv=argv, allowed=True, returncode=None, stdout=error.stdout or "", stderr=error.stderr or "", timed_out=True)


def decide_terminal(evidence: GateEvidence) -> TerminalOutcome:
    if evidence.ruff_configured:
        baseline = evidence.ruff_baseline
        candidate = evidence.ruff_candidate
        if baseline is None or candidate is None or baseline.argv != evidence.ruff_argv or candidate.argv != evidence.ruff_argv or not baseline.allowed or not candidate.allowed or baseline.timed_out or candidate.timed_out or baseline.returncode != 0 or candidate.returncode != 0:
            return TerminalOutcome(status=TerminalStatus.BLOCKED, reason="RUFF_CHECK_FAILED")
    if evidence.acceptance_configured:
        baseline = evidence.acceptance_baseline
        candidate = evidence.acceptance_candidate
        if baseline is None or candidate is None or evidence.acceptance_binding is None or evidence.acceptance_final_binding != evidence.acceptance_binding or baseline.timed_out or candidate.timed_out or baseline.returncode != 1 or candidate.returncode != 0:
            return TerminalOutcome(status=TerminalStatus.BLOCKED, reason="ACCEPTANCE_FAILED")
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
        if any(part in _IGNORED_SNAPSHOT_PARTS for part in relative.parts):
            continue
        stat = path.lstat()
        if path.is_symlink():
            result[relative.as_posix()] = f"symlink:{stat.st_mode:o}:{os.readlink(path)}"
        elif path.is_file():
            result[relative.as_posix()] = f"file:{stat.st_mode:o}:{sha256_bytes(path.read_bytes())}"
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


class EvidenceSanitySignature(dspy.Signature):
    """Return one evidence sufficiency decision; never propose or invoke tools."""

    packet: str = dspy.InputField()
    status: str = dspy.OutputField()
    reason: str = dspy.OutputField()
    capability: EvidenceCapability | None = dspy.OutputField()


class ProposeEditsSignature(dspy.Signature):
    """Produce exact-hash test edits first and production edits second using only supplied evidence. Multiple same-file replacements are permitted only within one phase when non-overlapping, non-empty, and sharing the original full-file hash; cross-phase reuse is forbidden. When production introduces a new symbol, obtain assertion-level RED through an existing import/module/public API surface and do not import the not-yet-created production symbol in the test-only phase. An expected-error/rejection test must fail if the expected error is absent and its failing path must include the exact mission Expected RED text: use pytest.raises(match=Expected RED text) or an explicit failure message containing it. Do not rely on an assertion after the context manager because it is unreachable when no exception occurs; silent try/except fallthrough is forbidden. Every identifier introduced by an edit must be bound by an existing or proposed matching import before use; match import form (`from package import name` uses `name`, while `import package` uses `package.name`), keep imports at module top, and prevent collection-time NameError or unbound decorator references. Before proposing an import, inspect and reuse existing module-top bindings; never add a function-local import for a name already bound at module scope, especially after earlier use in that function, because it shadows the entire local scope. Any pytest-provided identifier must be declared in test function fixture parameters (or explicitly constructed); never use an undeclared fixture name. Reserve pytest.raises for contracts that actually raise. If a mission or public contract returns a failure status/object, call the existing public surface and assert the returned state; never expect a constructor/call exception instead."""

    mission: str = dspy.InputField()
    investigation: Investigation = dspy.InputField()
    repository_evidence: str = dspy.InputField()
    reviewed_learning: str = dspy.InputField()
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
SanitySource = Callable[[Mission, Investigation, dict[str, Any]], SanityDecision]


class MiniSWEWorkflow:
    def __init__(self, mission: Mission, run_root: Path | str, investigation_source: InvestigationSource | None = None, proposal_source: ProposalSource | None = None, review_source: ReviewSource | None = None, sanity_source: SanitySource | None = None):
        self.mission = mission
        self.repo_root = Path(mission.repo_root).resolve()
        self.run_root = Path(run_root).resolve()
        self.investigation_source = investigation_source
        self.proposal_source = proposal_source
        self.review_source = review_source
        self.sanity_source = sanity_source
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
        if self.mission.ruff_check_argv is not None and not _is_safe_ruff(self.mission.ruff_check_argv, self.repo_root):
            raise ScopeViolation("Ruff command must use the exact repo-local --no-cache check shape")
        acceptance_artifact = (
            validate_acceptance_artifact(self.repo_root, self.mission.allowed_paths, self.mission.acceptance_test_path)
            if self.mission.acceptance_test_path is not None
            else None
        )
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
            "acceptance_artifact": acceptance_artifact,
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

    def _live_sanity(self, packet: dict[str, Any]) -> SanityDecision:
        self._configure_live()
        prediction = dspy.Predict(EvidenceSanitySignature)(packet=json.dumps(packet, sort_keys=True))
        return SanityDecision(status=str(prediction.status), reason=str(prediction.reason), capability=prediction.capability)

    def _live_proposal(self, investigation: Investigation) -> ChangeProposal:
        self._configure_live()
        prediction = dspy.Predict(ProposeEditsSignature)(
            mission=f"{self.mission.goal}\nExpected RED text: {self.mission.expected_red_text}",
            investigation=investigation,
            repository_evidence=self._repository_evidence(),
            reviewed_learning=json.dumps(self.mission.reviewed_learning),
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

    def _evidence_packet(self, investigation: Investigation, ledger: EventLedger) -> dict[str, Any]:
        evidence = json.loads(self._repository_evidence())
        investigator_trace = ledger.records({"tool", "tool_error", "react_trajectory"})
        return _redact({
            "mission_boundary": {"goal": self.mission.goal, "allowed_paths": self.mission.allowed_paths},
            "investigation": investigation.model_dump(mode="json"),
            "investigator_trace": investigator_trace,
            "repository_evidence": evidence,
            "reviewed_learning": self.mission.reviewed_learning,
        })

    def _preview(self, grouped: GroupedProposal, allowed_commands: list[list[str]], acceptance_argv: list[str] | None, ruff_argv: list[str] | None) -> tuple[bool, str, dict[str, Any]]:
        transcript: list[dict[str, Any]] = []
        ignored = set(_IGNORED_SNAPSHOT_PARTS)
        with tempfile.TemporaryDirectory(prefix="mini-swe-preview-") as directory:
            copy_root = Path(directory) / "workspace"
            shutil.copytree(self.repo_root, copy_root, ignore=lambda _root, names: {name for name in names if name in ignored or name == ".env" or name.startswith(".env.")})

            def copied(group: GroupedEdit) -> GroupedEdit:
                target = resolve_repo_file(copy_root, group.path)
                return GroupedEdit(group.path, target, group.before_bytes, group.after_bytes, group.mode, group.before_sha256, group.after_sha256)

            def command(phase: str, argv: list[str]) -> tuple[CommandResult, set[str]]:
                before = _snapshot_workspace(copy_root)
                real_before = _snapshot_workspace(self.repo_root)
                result = run_allowed_command(argv, copy_root, allowed_commands, self.mission.timeout_seconds, self.repo_root)
                copy_changed = _changed_paths(before, _snapshot_workspace(copy_root))
                real_changed = _changed_paths(real_before, _snapshot_workspace(self.repo_root))
                changed = copy_changed | real_changed
                transcript.append({"phase": phase, "result": result.model_dump(mode="json"), "changed_paths": sorted(changed), "real_changed_paths": sorted(real_changed)})
                return result, changed

            try:
                for group in grouped.test_groups:
                    apply_grouped_edit(copied(group))
                red, changed = command("focused_red", self.mission.focused_test_argv)
                if changed or not (red.allowed and not red.timed_out and red.returncode == self.mission.expected_red_returncode and self.mission.expected_red_text in f"{red.stdout}\n{red.stderr}"):
                    return False, "PREVIEW_COMMAND_MUTATION" if changed else "PREVIEW_RED_NOT_CONFIRMED", {"transcript": transcript}
                if not grouped.production_groups:
                    return False, "INVALID_PROPOSAL", {"transcript": transcript}
                for group in grouped.production_groups:
                    apply_grouped_edit(copied(group))
                for phase, argv in (("focused_green", self.mission.focused_test_argv), ("full_green", self.mission.full_test_argv)):
                    result, changed = command(phase, argv)
                    if changed or not (result.allowed and not result.timed_out and result.returncode == 0):
                        return False, "PREVIEW_COMMAND_MUTATION" if changed else f"PREVIEW_{phase.upper()}_FAILED", {"transcript": transcript}
                for phase, argv in (("candidate_ruff", ruff_argv), ("acceptance_candidate", acceptance_argv)):
                    if argv is None:
                        continue
                    result, changed = command(phase, argv)
                    if changed or not (result.allowed and not result.timed_out and result.returncode == 0):
                        return False, "PREVIEW_COMMAND_MUTATION" if changed else f"PREVIEW_{phase.upper()}_FAILED", {"transcript": transcript}
            except (EditValidationError, ScopeViolation) as error:
                return False, "PREVIEW_EDIT_INVALID", {"error": repr(error), "transcript": transcript}
        return True, "", {"transcript": transcript}

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

        def validation_command(phase: str, argv: list[str]) -> tuple[CommandResult, set[str]]:
            before = _snapshot_workspace(self.repo_root)
            result = run_allowed_command(argv, self.repo_root, allowed_commands, self.mission.timeout_seconds, self.repo_root)
            after = _snapshot_workspace(self.repo_root)
            changed = _changed_paths(before, after)
            ledger.append("command", {"phase": phase, "result": result})
            if changed:
                ledger.append("command_mutation", {"phase": phase, "argv": argv, "changed_paths": sorted(changed), "returncode": result.returncode, "timed_out": result.timed_out})
            return result, changed

        acceptance_baseline: CommandResult | None = None
        acceptance_candidate: CommandResult | None = None
        acceptance_binding: AcceptanceBinding | None = None
        ruff_baseline: CommandResult | None = None
        ruff_candidate: CommandResult | None = None
        ruff_argv = self.mission.ruff_check_argv
        acceptance_argv = acceptance_pytest_argv(self.repo_root, self.mission.acceptance_test_path) if self.mission.acceptance_test_path else None
        allowed_commands = [self.mission.focused_test_argv, self.mission.full_test_argv] + ([ruff_argv] if ruff_argv else []) + ([acceptance_argv] if acceptance_argv else [])
        try:
            preflight = self._preflight()
            initial_revision = preflight["git_revision"]
            ledger.append("preflight", preflight)
            ledger.append("transition", {"from": None, "to": WorkflowState.PREFLIGHT, "reason": "preflight passed"})
            baseline_snapshot = _snapshot_workspace(self.repo_root)
            preimages = {path: (_validate_allowed_file(self.repo_root, self.mission.allowed_paths, path).read_bytes(), _validate_allowed_file(self.repo_root, self.mission.allowed_paths, path).stat().st_mode) for path in self.mission.allowed_paths}

            self._stage_start(ledger, "baseline")
            baseline_focused, mutation = validation_command("baseline_focused", self.mission.focused_test_argv)
            if mutation:
                return self._blocked(ledger, run_directory, "COMMAND_MUTATION", None, {"phase": "baseline_focused", "changed_paths": sorted(mutation), "command": baseline_focused.model_dump()})
            clean, changed = scope_gate("after_baseline_focused")
            baseline_full, mutation = validation_command("baseline_full", self.mission.full_test_argv)
            if mutation:
                return self._blocked(ledger, run_directory, "COMMAND_MUTATION", None, {"phase": "baseline_full", "changed_paths": sorted(mutation), "command": baseline_full.model_dump()})
            clean = clean and scope_gate("after_baseline_full")[0]
            baseline_green = clean and all(result.allowed and not result.timed_out and result.returncode == 0 for result in (baseline_focused, baseline_full))
            if not baseline_green:
                self._stage_error(ledger, "baseline", RuntimeError("baseline verification failed"))
                return self._blocked(ledger, run_directory, "BASELINE_FAILED", None)
            if ruff_argv:
                ruff_baseline, mutation = validation_command("baseline_ruff", ruff_argv)
                if mutation:
                    return self._blocked(ledger, run_directory, "COMMAND_MUTATION", None, {"phase": "baseline_ruff", "changed_paths": sorted(mutation), "command": ruff_baseline.model_dump()})
                clean, changed = scope_gate("after_baseline_ruff")
                if not clean:
                    return self._blocked(ledger, run_directory, "OUT_OF_SCOPE_MUTATION", None, {"changed_paths": sorted(changed)})
                if not ruff_baseline.allowed or ruff_baseline.timed_out or ruff_baseline.returncode != 0:
                    return self._blocked(ledger, run_directory, "RUFF_BASELINE_FAILED", None, {"command": ruff_baseline.model_dump()})
            if acceptance_argv:
                acceptance_baseline, mutation = validation_command("acceptance_baseline", acceptance_argv)
                if mutation:
                    return self._blocked(ledger, run_directory, "COMMAND_MUTATION", None, {"phase": "acceptance_baseline", "changed_paths": sorted(mutation), "command": acceptance_baseline.model_dump()})
                clean, changed = scope_gate("after_acceptance_baseline")
                output = f"{acceptance_baseline.stdout}\n{acceptance_baseline.stderr}"
                if not clean:
                    return self._blocked(ledger, run_directory, "OUT_OF_SCOPE_MUTATION", None, {"changed_paths": sorted(changed)})
                if acceptance_baseline.timed_out or acceptance_baseline.returncode not in (0, 1):
                    return self._blocked(ledger, run_directory, "ACCEPTANCE_BASELINE_INVALID", None, {"command": acceptance_baseline.model_dump()})
                if acceptance_baseline.returncode == 0:
                    return self._blocked(ledger, run_directory, "ACCEPTANCE_ALREADY_PASSES", None, {"command": acceptance_baseline.model_dump()})
                if self.mission.acceptance_expected_red_text not in output:
                    return self._blocked(ledger, run_directory, "ACCEPTANCE_BASELINE_INVALID", None, {"command": acceptance_baseline.model_dump()})
            self._stage_end(ledger, "baseline", {"focused": baseline_focused, "full": baseline_full, "acceptance": acceptance_baseline})

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

            packet = self._evidence_packet(investigation, ledger)
            packet_json = json.dumps(packet, sort_keys=True)
            ledger.append("sanity_packet", {"sha256": sha256_text(packet_json), "packet": packet})
            self._stage_start(ledger, "sanity")
            try:
                decision = self.sanity_source(self.mission, investigation, packet) if self.sanity_source else (SanityDecision(status="PROCEED", reason="injected sources require no live sanity") if self.investigation_source else self._live_sanity(packet))
            except Exception as error:
                self._stage_error(ledger, "sanity", error)
                return self._blocked(ledger, run_directory, "INVALID_SANITY", None, {"error": repr(error)})
            if not isinstance(decision, SanityDecision):
                return self._blocked(ledger, run_directory, "INVALID_SANITY", None)
            ledger.append("sanity", {"decision": decision.model_dump(mode="json")})
            if decision.status == "BLOCKED":
                return self._blocked(ledger, run_directory, "SANITY_BLOCKED", None, {"reason": decision.reason})
            if decision.status == "NEEDS_EVIDENCE":
                try:
                    follow_up = bounded_read(self.repo_root, decision.capability.path) if decision.capability and decision.capability.kind == "bounded_read" else bounded_search(self.repo_root, decision.capability.query) if decision.capability else None
                    if follow_up is None or follow_up.truncated:
                        return self._blocked(ledger, run_directory, "NEEDS_EVIDENCE_BLOCKED", None)
                except Exception as error:
                    return self._blocked(ledger, run_directory, "NEEDS_EVIDENCE_BLOCKED", None, {"error": repr(error)})
                follow_up_data = follow_up.model_dump(mode="json")
                packet["follow_up"] = follow_up_data
                ledger.append("evidence_follow_up", {"sha256": sha256_text(json.dumps(follow_up_data, sort_keys=True)), "result": follow_up_data})
            self._stage_end(ledger, "sanity", {"status": decision.status})

            self._stage_start(ledger, "proposal")
            try:
                proposal = self.proposal_source(self.mission, investigation) if self.proposal_source else self._live_proposal(investigation)
            except Exception as error:
                self._stage_error(ledger, "proposal", error)
                return self._blocked(ledger, run_directory, "PROPOSAL_FAILED", None, {"error": repr(error)})
            ledger.append("proposal_received", {"origin": "injected" if self.proposal_source else "live-predict", "proposal": _bounded_proposal_evidence(proposal)})
            if not isinstance(proposal, ChangeProposal) or not proposal.test_edits:
                return self._blocked(ledger, run_directory, "INVALID_PROPOSAL", None)
            try:
                grouped_proposal = normalize_change_proposal(self.repo_root, self.mission.allowed_paths, proposal)
            except ProposalValidationError as error:
                ledger.append("proposal_invalid", {"path": error.path, "error": str(error)})
                return self._blocked(ledger, run_directory, "INVALID_PROPOSAL", None, {"path": error.path, "error": str(error)})
            ledger.append("proposal_validated", {"test_paths": [group.path for group in grouped_proposal.test_groups], "production_paths": [group.path for group in grouped_proposal.production_groups]})
            proposal_hash = sha256_text(json.dumps(proposal.model_dump(mode="json"), sort_keys=True))
            preview_ok, preview_reason, preview_details = self._preview(grouped_proposal, allowed_commands, acceptance_argv, ruff_argv)
            preview_payload = {"sha256": sha256_text(json.dumps(preview_details, sort_keys=True)), "proposal_sha256": proposal_hash, "passed": preview_ok, **preview_details}
            ledger.append("preview", preview_payload)
            if not preview_ok:
                real_escape = any(item.get("real_changed_paths") for item in preview_details.get("transcript", []))
                return self._blocked(ledger, run_directory, preview_reason, preimages if real_escape else None, preview_payload)
            ledger.append("proposal", {"origin": "injected" if self.proposal_source else "live-predict", "sha256": proposal_hash, "output": proposal})
            self._stage_end(ledger, "proposal", {"test_edit_count": len(proposal.test_edits), "production_edit_count": len(proposal.production_edits), "test_group_count": len(grouped_proposal.test_groups), "production_group_count": len(grouped_proposal.production_groups)})
            ledger.append("transition", {"from": WorkflowState.INVESTIGATED, "to": WorkflowState.PROPOSED, "reason": "typed proposal validated"})

            self._stage_start(ledger, "test_edit")
            for group in grouped_proposal.test_groups:
                applied_paths.add(group.path)
                applied = apply_grouped_edit(group)
                ledger.append("test_edit", {"edit": applied})
            clean, changed = scope_gate("after_test_edit")
            if not clean:
                self._stage_error(ledger, "test_edit", RuntimeError("scope violation"))
                return self._blocked(ledger, run_directory, "OUT_OF_SCOPE_MUTATION", rollback_preimages(), {"changed_paths": sorted(changed)})
            self._stage_end(ledger, "test_edit", {"paths": sorted(applied_paths)})

            self._stage_start(ledger, "red")
            red, mutation = validation_command("focused_red", self.mission.focused_test_argv)
            if mutation:
                return self._blocked(ledger, run_directory, "COMMAND_MUTATION", rollback_preimages(), {"phase": "focused_red", "changed_paths": sorted(mutation), "command": red.model_dump()})
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
            for group in grouped_proposal.production_groups:
                applied_paths.add(group.path)
                applied = apply_grouped_edit(group)
                production_paths.append(group.path)
                ledger.append("production_edit", {"edit": applied})
            clean, changed = scope_gate("after_production_edit")
            if not clean:
                self._stage_error(ledger, "production_edit", RuntimeError("scope violation"))
                return self._blocked(ledger, run_directory, "OUT_OF_SCOPE_MUTATION", rollback_preimages(), {"changed_paths": sorted(changed)})
            self._stage_end(ledger, "production_edit", {"paths": production_paths})

            self._stage_start(ledger, "focused_green")
            focused, mutation = validation_command("focused_green", self.mission.focused_test_argv)
            if mutation:
                return self._blocked(ledger, run_directory, "COMMAND_MUTATION", rollback_preimages(), {"phase": "focused_green", "changed_paths": sorted(mutation), "command": focused.model_dump()})
            clean, changed = scope_gate("after_focused_green")
            if not (clean and focused.allowed and not focused.timed_out and focused.returncode == 0):
                self._stage_error(ledger, "focused_green", RuntimeError("focused verification failed"))
                return self._blocked(ledger, run_directory, "FOCUSED_GREEN_FAILED" if clean else "OUT_OF_SCOPE_MUTATION", rollback_preimages(), {"command": focused.model_dump()})
            self._stage_end(ledger, "focused_green", {"returncode": focused.returncode})

            self._stage_start(ledger, "full_green")
            full, mutation = validation_command("full_green", self.mission.full_test_argv)
            if mutation:
                return self._blocked(ledger, run_directory, "COMMAND_MUTATION", rollback_preimages(), {"phase": "full_green", "changed_paths": sorted(mutation), "command": full.model_dump()})
            clean, changed = scope_gate("after_full_green")
            if not (clean and full.allowed and not full.timed_out and full.returncode == 0):
                self._stage_error(ledger, "full_green", RuntimeError("full verification failed"))
                return self._blocked(ledger, run_directory, "FULL_GREEN_FAILED" if clean else "OUT_OF_SCOPE_MUTATION", rollback_preimages(), {"command": full.model_dump()})
            self._stage_end(ledger, "full_green", {"returncode": full.returncode})
            if ruff_argv:
                ruff_candidate, mutation = validation_command("candidate_ruff", ruff_argv)
                if mutation:
                    return self._blocked(ledger, run_directory, "COMMAND_MUTATION", rollback_preimages(), {"phase": "candidate_ruff", "changed_paths": sorted(mutation), "command": ruff_candidate.model_dump()})
                clean, changed = scope_gate("after_candidate_ruff")
                if not clean:
                    return self._blocked(ledger, run_directory, "OUT_OF_SCOPE_MUTATION", rollback_preimages(), {"changed_paths": sorted(changed)})
                if not ruff_candidate.allowed or ruff_candidate.timed_out or ruff_candidate.returncode != 0:
                    return self._blocked(ledger, run_directory, "RUFF_CHECK_FAILED", rollback_preimages(), {"command": ruff_candidate.model_dump()})
            if acceptance_argv:
                acceptance_candidate, mutation = validation_command("acceptance_candidate", acceptance_argv)
                if mutation:
                    return self._blocked(ledger, run_directory, "COMMAND_MUTATION", rollback_preimages(), {"phase": "acceptance_candidate", "changed_paths": sorted(mutation), "command": acceptance_candidate.model_dump()})
                clean, changed = scope_gate("after_acceptance_candidate")
                if not clean:
                    return self._blocked(ledger, run_directory, "OUT_OF_SCOPE_MUTATION", rollback_preimages(), {"changed_paths": sorted(changed)})
                if acceptance_candidate.timed_out or acceptance_candidate.returncode != 0:
                    return self._blocked(ledger, run_directory, "ACCEPTANCE_FAILED", rollback_preimages(), {"command": acceptance_candidate.model_dump()})
                acceptance_binding = AcceptanceBinding(
                    scoped_diff_sha256=sha256_text(_scoped_git_diff(self.repo_root, self.mission.allowed_paths)),
                    artifact_sha256=sha256_bytes(resolve_repo_file(self.repo_root, self.mission.acceptance_test_path).read_bytes()),
                )
                ledger.append("acceptance_binding", acceptance_binding)
            ledger.append("transition", {"from": WorkflowState.RED_CONFIRMED, "to": WorkflowState.GREEN_CONFIRMED, "reason": "focused and full pytest green"})

            self._stage_start(ledger, "review")
            review_input = _redact({
                "investigation": investigation.model_dump(mode="json"),
                "proposal": proposal.model_dump(mode="json"),
                "scoped_git_diff": _scoped_git_diff(self.repo_root, self.mission.allowed_paths),
                "changed_paths": sorted(changed),
                "focused_command": focused.model_dump(mode="json"),
                "full_command": full.model_dump(mode="json"),
                "baseline_ruff_command": ruff_baseline.model_dump(mode="json") if ruff_baseline else None,
                "candidate_ruff_command": ruff_candidate.model_dump(mode="json") if ruff_candidate else None,
                "acceptance_baseline_command": acceptance_baseline.model_dump(mode="json") if acceptance_baseline else None,
                "acceptance_candidate_command": acceptance_candidate.model_dump(mode="json") if acceptance_candidate else None,
                "acceptance_binding": acceptance_binding.model_dump(mode="json") if acceptance_binding else None,
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

            clean, changed = scope_gate("after_review")
            if not clean:
                return self._blocked(ledger, run_directory, "OUT_OF_SCOPE_MUTATION", rollback_preimages(), {"changed_paths": sorted(changed)})
            acceptance_final_binding = (
                AcceptanceBinding(
                    scoped_diff_sha256=sha256_text(_scoped_git_diff(self.repo_root, self.mission.allowed_paths)),
                    artifact_sha256=sha256_bytes(resolve_repo_file(self.repo_root, self.mission.acceptance_test_path).read_bytes()),
                )
                if acceptance_argv
                else None
            )
            if acceptance_argv and acceptance_final_binding != acceptance_binding:
                return self._blocked(ledger, run_directory, "ACCEPTANCE_STALE", rollback_preimages())
            outcome = decide_terminal(GateEvidence(preflight_ok=True, live_proposal_valid=True, baseline_green=True, red_confirmed=True, production_applied=bool(production_paths), focused_green=True, full_green=True, scope_ok=clean, revision_stable=_git_revision(self.repo_root) == initial_revision, rollback_performed=False, ruff_configured=ruff_argv is not None, ruff_argv=ruff_argv, ruff_baseline=ruff_baseline, ruff_candidate=ruff_candidate, acceptance_configured=acceptance_argv is not None, acceptance_baseline=acceptance_baseline, acceptance_candidate=acceptance_candidate, acceptance_binding=acceptance_binding, acceptance_final_binding=acceptance_final_binding))
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
    for command_key in ("focused_test_argv", "full_test_argv", "ruff_check_argv"):
        if raw.get(command_key) is not None:
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
