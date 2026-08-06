"""Probe implicit context caching usage stats via DSPy.

Goal
----
Make sequential requests with a large shared prefix and print DSPy usage stats,
including whether provider-side implicit caching was used.

We test both providers explicitly by using model ids:
- Gemini API:  gemini/gemini-3.5-flash
- Vertex AI:   vertex_ai/gemini-3.5-flash

Notes
-----
- Implicit caching is opportunistic (not guaranteed). Even with identical prefixes,
  `cached_tokens` may not show up on exactly the 2nd call every time.
- This script defaults to 2 rounds ("at least 2"), but supports more rounds to
  increase the chance of observing a cache hit.

Usage
-----
  uv run python src/simplest/cached_tokens_probe_gemini_vertex.py
  uv run python src/simplest/cached_tokens_probe_gemini_vertex.py --rounds 5
  uv run python src/simplest/cached_tokens_probe_gemini_vertex.py --provider gemini
  uv run python src/simplest/cached_tokens_probe_gemini_vertex.py --provider vertex
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from typing import Any

import dspy


def _configure_dspy_for_model(model_id: str) -> dspy.LM:
    """Configure DSPy for a given provider/model without mutating env vars."""
    lm = dspy.LM(
        model=model_id,
        max_tokens=96,
        temperature=0.0,
        reasoning_effort="disable",
    )

    dspy.settings.configure(lm=lm, track_usage=True, adapter=dspy.JSONAdapter())
    # Disable DSPy-local caches; we want provider-side caching/usage.
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    return lm


def _has_gemini_credentials() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def _has_vertex_credentials() -> bool:
    return bool(os.getenv("VERTEXAI_PROJECT") and os.getenv("VERTEXAI_LOCATION"))


def _extract_usage_fields(pred: dspy.Prediction) -> dict[str, Any]:
    """Return a normalized dict of usage fields from prediction.get_lm_usage()."""
    usage_all = pred.get_lm_usage() or {}
    if not usage_all:
        return {}

    # DSPy returns a dict keyed by lm name. Here we expect a single LM.
    _, usage = next(iter(usage_all.items()))

    prompt_details = usage.get("prompt_tokens_details")
    cached_tokens = None
    text_tokens = None
    if isinstance(prompt_details, dict):
        cached_tokens = prompt_details.get("cached_tokens")
        text_tokens = prompt_details.get("text_tokens")

    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": cached_tokens,
        "text_tokens": text_tokens,
        "raw": usage,
    }


def _build_large_shared_prefix() -> str:
    # Add a unique marker so consecutive script runs don't accidentally benefit
    # from previous (ephemeral) provider caches, but keep it stable within the run.
    marker = str(uuid.uuid4())

    # Create a very long shared prefix (well above 1024 tokens for Flash).
    # Repetition keeps the prefix stable and large.
    repeated = "This is a shared prefix for implicit caching. " * 2200
    return f"UNIQUE_MARKER={marker} {repeated}".strip()


def run_probe(model_id: str, rounds: int, sleep_s: float) -> int:
    print("=" * 88)
    print(f"Provider probe: {model_id}")

    _configure_dspy_for_model(model_id)

    shared_prefix = _build_large_shared_prefix()
    predictor = dspy.Predict("context, question -> answer")

    saw_hit = False
    for i in range(1, rounds + 1):
        pred = predictor(context=shared_prefix, question="Return the marker value only.")
        usage = _extract_usage_fields(pred)

        cached = usage.get("cached_tokens")
        if isinstance(cached, int) and cached > 0:
            saw_hit = True

        print(
            f"{i:>2} | prompt_tokens={usage.get('prompt_tokens')} "
            f"cached_tokens={cached} text_tokens={usage.get('text_tokens')} "
            f"completion_tokens={usage.get('completion_tokens')} total_tokens={usage.get('total_tokens')}"
        )

        if sleep_s:
            time.sleep(sleep_s)

    if not saw_hit:
        print(
            "\nNote: cached_tokens was not observed in these rounds. "
            "Implicit caching is opportunistic; try re-running or increasing --rounds."
        )
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=["both", "gemini", "vertex"],
        default="both",
        help="Which provider(s) to test.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of sequential requests per provider (min 2 recommended).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Seconds to sleep between calls (helps keep requests close in time).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if cached_tokens is not observed for a provider (not recommended; implicit caching is opportunistic).",
    )
    args = parser.parse_args()

    if args.rounds < 2:
        raise SystemExit("--rounds must be >= 2")

    exit_code = 0

    if args.provider in ("both", "gemini"):
        if not _has_gemini_credentials():
            print("Skipping Gemini API probe: GEMINI_API_KEY not set")
            exit_code = max(exit_code, 2)
        else:
            result = run_probe(model_id="gemini/gemini-3.5-flash", rounds=args.rounds, sleep_s=args.sleep)
            if args.strict:
                exit_code = max(exit_code, result)

    if args.provider in ("both", "vertex"):
        if not _has_vertex_credentials():
            print("Skipping Vertex AI probe: VERTEXAI_PROJECT and/or VERTEXAI_LOCATION not set")
            exit_code = max(exit_code, 2)
        else:
            result = run_probe(model_id="vertex_ai/gemini-3.5-flash", rounds=args.rounds, sleep_s=args.sleep)
            if args.strict:
                exit_code = max(exit_code, result)

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
