"""Tools for repository exploration, code reading, and diagram syntax validation.

These functions are exposed as standard Python callables that can be utilized
by dspy.ReAct modules or called deterministically within the workflow.
"""

import re
import subprocess
from pathlib import Path
from typing import List


def search_code(pattern: str, directory: str = ".") -> str:
    """Search for regex or keyword pattern across source code files.

    Args:
        pattern: Regex or string pattern to find (e.g. 'class BidderService', 'fun is_active').
        directory: Root directory to search within (defaults to current directory).

    Returns:
        Formatted matches with filepath and line numbers, or a not-found message.
    """
    # Prefer ripgrep (rg) if available
    try:
        res = subprocess.run(
            ["rg", "-n", "--max-count", "15", "--ignore-case", pattern, directory],
            capture_output=True,
            text=True,
            timeout=5
        )
        if res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass

    # Fallback to pure Python search
    matches: List[str] = []
    regex = re.compile(pattern, re.IGNORECASE)
    base_path = Path(directory)
    
    for path in base_path.rglob("*"):
        if path.is_file() and not any(part.startswith(".") for part in path.parts):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for idx, line in enumerate(f, start=1):
                        if regex.search(line):
                            matches.append(f"{path}:{idx}: {line.strip()}")
                            if len(matches) >= 15:
                                break
            except Exception:
                continue
        if len(matches) >= 15:
            break

    return "\n".join(matches) if matches else f"No matches found for pattern '{pattern}' in {directory}."


def read_code_slice(filepath: str, start_line: int = 1, end_line: int = 50) -> str:
    """Read a specific slice of lines from a source file.

    Args:
        filepath: Relative or absolute path to the file.
        start_line: 1-indexed line number to start reading from.
        end_line: 1-indexed line number to stop reading at.

    Returns:
        Numbered lines of code or error description.
    """
    path = Path(filepath)
    if not path.is_file():
        return f"File not found: {filepath}"

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        
        selected = [f"{i+1}: {line}" for i, line in enumerate(lines[start_idx:end_idx], start=start_idx)]
        return "".join(selected) if selected else f"Line range {start_line}-{end_line} out of bounds (file has {len(lines)} lines)."
    except Exception as e:
        return f"Error reading {filepath}: {str(e)}"


def validate_d2_syntax(d2_code: str) -> str:
    """Validate D2 diagram syntax and check for unescaped placeholder errors.

    Args:
        d2_code: Source D2 markup string.

    Returns:
        Validation status message.
    """
    if not d2_code or not d2_code.strip():
        return "Warning: Empty D2 diagram."

    # Heuristic check for common syntax errors
    open_braces = d2_code.count("{")
    close_braces = d2_code.count("}")
    if open_braces != close_braces:
        return f"D2 Syntax Error: Unbalanced curly braces ({open_braces} open vs {close_braces} close)."

    # Check for unescaped template placeholders (e.g. {node})
    if re.search(r"\{[a-zA-Z_]+\}", d2_code):
        return "D2 Syntax Error: Unescaped template placeholder detected (e.g. '{node}')."

    # If d2 CLI is available, run a test compile
    try:
        res = subprocess.run(
            ["d2", "-", "/dev/null"],
            input=d2_code,
            capture_output=True,
            text=True,
            timeout=3
        )
        if res.returncode == 0:
            return "Valid D2 syntax (verified with d2 compiler)."
        else:
            return f"D2 Compilation Warning: {res.stderr.strip()}"
    except Exception:
        pass

    return "D2 syntax check passed (heuristic validation)."
