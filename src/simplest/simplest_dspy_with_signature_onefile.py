#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["dspy", "pydantic", "google-cloud-aiplatform"]
# ///
"""
Self-contained DSPy example: Extract Q&A pairs from text for dataset creation.

This single file demonstrates:
- Configuring DSPy with Gemini or Vertex AI
- Using Pydantic models for structured output
- Extracting question/answer pairs from text

Requirements:
- Set GEMINI_API_KEY or (VERTEXAI_PROJECT + VERTEXAI_LOCATION) as environment variables
"""

import os
from typing import Literal

import dspy
import pydantic


# =============================================================================
# Sample Story Text
# =============================================================================

STORY_TEXT = """
The Clockmaker's Gift

In the small village of Millbrook, there lived an old clockmaker named Eleanor. 
She had crafted timepieces for over fifty years, but her most treasured creation 
was a pocket watch she made for her grandson, Marcus, on his tenth birthday.

The watch was no ordinary timepiece. Its face was made of mother-of-pearl, and 
inside, Eleanor had engraved the words: "Time is the thread that weaves our 
memories together."

Marcus carried the watch everywhere. When he left for university in London, it 
reminded him of home. When his grandmother passed away three years later, it 
became his most precious possession.

Years later, Marcus became a renowned surgeon. Before every operation, he would 
touch the watch in his pocket—a ritual that connected him to Eleanor's steady 
hands and patient heart.

On his fiftieth birthday, Marcus gave the watch to his own granddaughter, Sophie, 
passing on not just a timepiece, but a legacy of love and craftsmanship that had 
now spanned four generations.
"""


# =============================================================================
# DSPy Signature and Models for Q&A Extraction
# =============================================================================


class QAPair(pydantic.BaseModel):
    """A question and its gold answer extracted from text."""

    question: str = pydantic.Field(
        description="A question that can be answered from the text"
    )
    gold_answer: str = pydantic.Field(
        description="The correct answer based on the text"
    )


class QAExtractionSignature(dspy.Signature):
    """
    Extract question/answer pairs from the given text for dataset creation.
    Generate diverse questions covering facts, reasoning, and comprehension.
    Each answer must be directly supported by the text.
    """

    text: str = dspy.InputField(desc="The source text to extract Q&A pairs from")
    qa_pairs: list[QAPair] = dspy.OutputField(
        desc="List of question/gold-answer pairs"
    )


def extract_qa_pairs(text: str) -> list[QAPair]:
    """Extract Q&A pairs from text using DSPy."""
    extractor = dspy.Predict(QAExtractionSignature)
    result = extractor(text=text)
    return result.qa_pairs


# =============================================================================
# DSPy Configuration Utilities
# =============================================================================


class DSPyGeminiConfig:
    """Configuration utilities for DSPy with Gemini/Vertex AI."""

    MODEL_GEMINI_2_5_FLASH = "gemini-2.5-flash"

    _PROVIDER_GEMINI = "gemini"
    _PROVIDER_VERTEX_AI = "vertex_ai"
    _PROVIDER_LIST = [_PROVIDER_GEMINI, _PROVIDER_VERTEX_AI]

    @classmethod
    def configure(
        cls,
        model_name: str,
        reasoning_effort: Literal["low", "medium", "high", "disable"] | None = "disable",
        max_tokens: int = 8192,
        temperature: float = 0.3,
        track_usage: bool = True,
    ) -> None:
        """Configure DSPy with the given model."""
        lm = cls._get_lm(model_name, reasoning_effort, max_tokens, temperature)
        dspy.settings.configure(lm=lm, track_usage=track_usage, adapter=dspy.JSONAdapter())
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    @classmethod
    def _get_lm(
        cls,
        model_name: str,
        reasoning_effort: Literal["low", "medium", "high", "disable"] | None,
        max_tokens: int,
        temperature: float,
    ) -> dspy.LM:
        """Create a DSPy LM instance for the given model."""
        prefix = cls._get_model_access_prefix_or_fail(model_name)
        return dspy.LM(
            model=f"{prefix}{model_name}",
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

    @classmethod
    def _get_model_access_prefix_or_fail(cls, model_name: str) -> str:
        """
        Determine the access prefix based on model name and available credentials.
        Returns e.g. "" or "gemini/" or "vertex_ai/"
        """
        provider_from_name: str | None = (
            model_name.split("/")[0] if "/" in model_name else None
        )
        if provider_from_name is not None and provider_from_name not in cls._PROVIDER_LIST:
            return ""

        has_vertex = os.getenv("VERTEXAI_PROJECT") and os.getenv("VERTEXAI_LOCATION")
        has_gemini = os.getenv("GEMINI_API_KEY")

        selected: str | None = None

        if provider_from_name == cls._PROVIDER_VERTEX_AI:
            if not has_vertex:
                raise ValueError(
                    f"VERTEXAI_PROJECT and VERTEXAI_LOCATION must be set for {cls._PROVIDER_VERTEX_AI}"
                )
            selected = provider_from_name
        elif provider_from_name == cls._PROVIDER_GEMINI:
            if not has_gemini:
                raise ValueError(f"GEMINI_API_KEY must be set for {cls._PROVIDER_GEMINI}")
            selected = provider_from_name

        if not selected:
            if has_vertex:
                selected = cls._PROVIDER_VERTEX_AI
            elif has_gemini:
                selected = cls._PROVIDER_GEMINI
            else:
                raise ValueError(
                    "Either (VERTEXAI_PROJECT + VERTEXAI_LOCATION) or GEMINI_API_KEY must be set."
                )

        if selected == cls._PROVIDER_VERTEX_AI:
            os.unsetenv("GEMINI_API_KEY")
        elif selected == cls._PROVIDER_GEMINI:
            os.unsetenv("VERTEXAI_PROJECT")
            os.unsetenv("VERTEXAI_LOCATION")

        print(f"Using model access prefix: {selected}")
        return f"{selected}/"


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    DSPyGeminiConfig.configure(DSPyGeminiConfig.MODEL_GEMINI_2_5_FLASH)

    print("\n" + "=" * 80)
    print("Q&A Dataset Extraction Example")
    print("=" * 80)

    print("\n📖 Source Text:")
    print("-" * 40)
    print(STORY_TEXT.strip())
    print("-" * 40)

    print("\n→ Extracting Q&A pairs...")
    qa_pairs = extract_qa_pairs(STORY_TEXT)

    print(f"\n✓ Extracted {len(qa_pairs)} Q&A pairs:\n")

    for i, qa in enumerate(qa_pairs, 1):
        print(f"  [{i}] Q: {qa.question}")
        print(f"      A: {qa.gold_answer}")
        print()

    print("=" * 80)
    print("Done!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
