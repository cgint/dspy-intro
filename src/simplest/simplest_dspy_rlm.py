"""DSPy RLM example (Recursive Language Model).

This demo mirrors the structure of other runnable examples in this repo, but uses
`dspy.RLM` to let the model write Python code in a sandboxed REPL.

The agent analyzes a small set of sample log files and counts how many lines
contain the substring "ERROR".

Run:
    uv run simplestdspyrlm
"""

from __future__ import annotations

from pathlib import Path

import dspy

from common.constants import MODEL_NAME_GEMINI_2_5_FLASH
from common.utils import dspy_configure, get_lm_for_model_name


SAMPLE_LOG_DIR = Path(__file__).resolve().parent / "sample_logs"


# -----------------------------------------------------------------------------
# Tools exposed to the RLM sandbox
# -----------------------------------------------------------------------------

def get_available_files() -> str:
    """List available sample log files.

    Returns:
        Sorted list of filenames (e.g., ["app.log", "db.log"]).
    """
    if not SAMPLE_LOG_DIR.exists():
        return "No files found"

    sorted_names = sorted(p.name for p in SAMPLE_LOG_DIR.glob("*.log") if p.is_file())

    return "\n".join(sorted_names)


def fetch_log_data(path: str) -> str:
    """Read the content of one sample log file.

    Args:
        path: Filename returned by get_available_files(), e.g. "app.log".

    Returns:
        The full file content as a string.

    Raises:
        ValueError: If `path` is not a plain filename.
        FileNotFoundError: If the file does not exist.
    """
    # Prevent path traversal / directory access.
    if Path(path).name != path:
        raise ValueError("path must be a filename from get_available_files() (no directories)")

    full_path = SAMPLE_LOG_DIR / path
    if not full_path.is_file():
        raise FileNotFoundError(f"File not found in sample_logs/: {path}")

    return full_path.read_text(encoding="utf-8", errors="replace")


# -----------------------------------------------------------------------------
# RLM Signature
# -----------------------------------------------------------------------------


class LogAnalysis(dspy.Signature):
    """You are a log analysis agent.

    Goal:
    - Analyze the available local sample log files.

    Tools:
    - get_available_files() -> list[str]
    - fetch_log_data(path: str) -> str

    Task:
    - For each file, count the number of lines that contain the substring "ERROR".
    - Compute the total across all files.

    Output policy:
    - Do NOT hardcode counts into SUBMIT(). Always compute in Python and pass
      variables (e.g., SUBMIT(total_errors=total_errors, ...)).
    - Avoid printing huge amounts of log text; print small samples if needed.

    When finished, call:
      SUBMIT(summary_text=..., file_counts=..., total_errors=...)

    Note:
    - SUBMIT must include *all* output fields.
    """

    question: str = dspy.InputField()
    summary_text: str = dspy.OutputField(desc="Human-readable summary of the findings")
    file_counts: dict[str, int] = dspy.OutputField(
        desc='Mapping from filename to number of lines containing "ERROR"'
    )
    total_errors: int = dspy.OutputField(desc='Total number of lines containing "ERROR"')


# -----------------------------------------------------------------------------
# Module wrapper + main
# -----------------------------------------------------------------------------


class LogAgentRLMModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.agent = dspy.RLM(
            signature=LogAnalysis,
            tools={
                "get_available_files": get_available_files,
                "fetch_log_data": fetch_log_data,
            },
            max_iterations=10,
            verbose=True,
        )

    def forward(self, question: str) -> dspy.Prediction:
        return self.agent(question=question)


def main() -> None:
    dspy_configure(get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, "disable"))

    agent = LogAgentRLMModule()

    q = (
        "Read the available local sample log files. "
        'For each file, compute how many lines contain the substring "ERROR". '
        "Also compute the total across all files. "
        "Return a short explanation plus the per-file counts and the total."
    )

    print(f"\nQuestion:\n -> {q}\n")

    pred = agent(question=q)

    print("\nFinal Summary:")
    print(pred.summary_text)
    print("\nFile Counts:")
    print(pred.file_counts)
    print("\nTotal Errors:")
    print(pred.total_errors)


if __name__ == "__main__":
    main()
