from __future__ import annotations

import asyncio
from typing import Literal

import dspy
from dspy import streaming
from pydantic import BaseModel, Field

from common.constants import MODEL_NAME_GEMINI_2_5_FLASH
from common.utils import dspy_configure, get_lm_for_model_name
from streaming_examples.json_string_stream_decoder import JsonStringStreamDecoder


class DecisionModel(BaseModel):
    response_type: Literal["answer", "unsupported"] = Field(...)
    answer: str = Field(default="")
    reason: str = Field(default="")
    used_evidence_ids: list[str] = Field(default_factory=list)


class NestedDecisionSig(dspy.Signature):
    """Nested Pydantic output.

    Streaming this tends to yield JSON *object fragments* which are hard to show cleanly.
    """

    question: str = dspy.InputField()
    decision: DecisionModel = dspy.OutputField()


class FlatDecisionSig(dspy.Signature):
    """Flat output fields.

    This makes it much easier to stream only `answer` for UX.
    """

    question: str = dspy.InputField()

    response_type: str = dspy.OutputField(desc="answer | unsupported")
    answer: str = dspy.OutputField(desc="User-facing answer")
    reason: str = dspy.OutputField(desc="Explanation if unsupported")
    used_evidence_ids: list[str] = dspy.OutputField(desc="Evidence IDs")


def _configure() -> None:
    # Uses the repo's standard LM+adapter config helper.
    lm = get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, reasoning_effort="disable", max_tokens=1200)
    dspy_configure(lm)


async def run_flat_stream_answer() -> None:
    predictor = dspy.Predict(FlatDecisionSig)

    stream_predictor = dspy.streamify(
        predictor,
        stream_listeners=[streaming.StreamListener(signature_field_name="answer")],
    )

    decoder = JsonStringStreamDecoder()
    final = None

    print("\n--- STREAM: answer (decoded) ---\n")
    async for item in stream_predictor(question="Write a short markdown list with 3 bullet points."):
        if isinstance(item, streaming.StreamResponse):
            raw = item.chunk or ""
            decoded = decoder.feed(raw)
            if decoded:
                print(decoded, end="", flush=True)
        else:
            final = item

    print("\n\n--- FINAL (flat) ---")
    if final is not None:
        print("response_type:", getattr(final, "response_type", None))
        print("answer:\n", getattr(final, "answer", ""))


async def run_nested_stream_decision() -> None:
    predictor = dspy.Predict(NestedDecisionSig)

    stream_predictor = dspy.streamify(
        predictor,
        stream_listeners=[streaming.StreamListener(signature_field_name="decision")],
    )

    final = None

    print("\n--- STREAM: decision (raw fragments; usually not UX-friendly) ---\n")
    async for item in stream_predictor(question="Answer with response_type=answer and used_evidence_ids=['e1']"):
        if isinstance(item, streaming.StreamResponse):
            if item.chunk:
                print(item.chunk, end="", flush=True)
        else:
            final = item

    print("\n\n--- FINAL (nested) ---")
    if final is not None:
        print(final)
        try:
            print("decision.answer:", final.decision.answer)
        except Exception:  # noqa: BLE001
            pass


async def main_async() -> None:
    _configure()
    await run_flat_stream_answer()
    await run_nested_stream_decision()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
