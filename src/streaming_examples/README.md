# Streaming structured outputs with DSPy (`streamify`)

This folder contains a small, runnable example for **streaming** while still returning a **structured final result**.

## Why this is tricky

DSPy streaming is **field-aware**, not a raw token passthrough.

- You wrap a program with `dspy.streamify(...)`.
- You attach one or more `dspy.streaming.StreamListener(signature_field_name=...)`.
- The generator yields:
  - `dspy.streaming.StreamResponse` (field deltas)
  - possibly status messages
  - finally a full `dspy.Prediction` (the final structured object)

This is a great pattern for chat UIs: you can stream the user-facing `answer` while waiting for the final structured decision.

## Key best practices (distilled from asks/webs research + implementation experience)

### 1) Flatten your output signature for clean UX streaming

If your signature outputs a nested Pydantic model (e.g. `decision: DecisionModel`), streaming tends to produce **JSON object fragments** that are not pleasant to render in a chat bubble.

Prefer a **flat signature** where `answer` is a top-level `OutputField`:

- ✅ `answer: str` as top-level output field → stream `answer`
- ✅ still return full structured fields in the final `Prediction`
- ❌ nested `decision: PydanticModel` → streamed chunks are JSON fragments

### 2) Expect cache hits to skip streaming chunks

When DSPy cache hits, you may get **no `StreamResponse` chunks** and only the final `Prediction`.

Your UI must handle both cases:
- streaming path (many chunks)
- instant path (final result immediately)

### 3) If you stream inside loops/ReAct, set `allow_reuse=True`

`StreamListener` is not reused by default (performance optimization). For iterative modules (ReAct, loops), pass:

```python
StreamListener("answer", allow_reuse=True)
```

### 4) Beware ambiguous field names in multi-module programs

If multiple predictors have the same output field name (e.g. two modules both output `answer`), you may need to specify which predictor to listen to (via `predict`/`predict_name`), otherwise you can stream from the wrong place.

### 5) JSONAdapter streaming often yields **JSON-string-literal chunks**

With `JSONAdapter`, a streamed string field can come through as raw JSON string literal fragments, e.g.:

- starts with a quote: `"Here are the...`
- contains escaped newlines: `\\n`

For a chat UI, you typically want the decoded text.

This example includes a small incremental decoder (`JsonStringStreamDecoder`) that turns:

- `\\n` → real newline
- `\\"` → `"`
- strips the leading/trailing JSON quotes

## Run it

From the repo root (`dspy-intro`):

```bash
uv run python -m streaming_examples.streamify_structured_outputs
```

You need model credentials configured for `common.utils.get_model_access_prefix_or_fail()` (Vertex or Gemini). If credentials are missing, the script will fail early.

## Files

- `streamify_structured_outputs.py` — demonstrates:
  - flat signature → stream `answer` (decoded)
  - nested signature → stream `decision` (raw fragments)
- `json_string_stream_decoder.py` — incremental JSON string literal decoder
