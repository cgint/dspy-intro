# Repro + results: cached token statistics via DSPy (Gemini API + Vertex AI)

## Summary
This repo includes a small DSPy probe script that makes sequential requests with a large shared prefix to observe **provider-side implicit context caching** via token usage statistics.

The key signal is available on the returned `dspy.Prediction`:

- `prediction.get_lm_usage()[<lm_name>]["prompt_tokens_details"]["cached_tokens"]`

When caching is used, `cached_tokens` becomes a positive integer (and `text_tokens` typically drops to the “uncached remainder”).

## References (repo files)
- Probe script (run this):
  - `src/simplest/cached_tokens_probe_gemini_vertex.py`
- Background findings and context:
  - `src/simplest/docs/cached_tokens_findings.md`
  - `src/simplest/docs/cached_tokens_findings_flow.d2`
  - `src/simplest/docs/cached_tokens_findings_flow.svg`

## Reproduce (commands)

### Run both providers (2 sequential requests each)
```bash
uv run python src/simplest/cached_tokens_probe_gemini_vertex.py --provider both --rounds 2
```

### Increase likelihood of observing implicit caching
Implicit caching is opportunistic, so you may need more rounds:
```bash
uv run python src/simplest/cached_tokens_probe_gemini_vertex.py --provider both --rounds 5
```

### Run only one provider
```bash
uv run python src/simplest/cached_tokens_probe_gemini_vertex.py --provider gemini --rounds 5
uv run python src/simplest/cached_tokens_probe_gemini_vertex.py --provider vertex --rounds 5
```

### Notes on credentials
- Gemini API requires: `GEMINI_API_KEY`
- Vertex AI requires: `VERTEXAI_PROJECT` and `VERTEXAI_LOCATION`

The probe uses explicit provider-prefixed model ids, so it can test both in one run (when both credential sets are available):
- `gemini/gemini-2.5-flash`
- `vertex_ai/gemini-2.5-flash`

## Observed token statistics (examples from runs in this environment)

### Example A — Gemini API cache hit on call #2
Command:
```bash
uv run python src/simplest/cached_tokens_probe_gemini_vertex.py --provider gemini --rounds 2
```
Observed output:
```text
Provider probe: gemini/gemini-2.5-flash
 1 | prompt_tokens=20010 cached_tokens=None  text_tokens=20010 completion_tokens=41 total_tokens=20051
 2 | prompt_tokens=20010 cached_tokens=19252 text_tokens=758   completion_tokens=41 total_tokens=20051
```
Interpreting the stats:
- Call #2 shows `cached_tokens=19252` indicating most of the prompt was served from the implicit cache.
- `text_tokens=758` indicates the remaining uncached portion of the prompt for that call.

### Example B — Vertex AI cache hit on call #2
Command:
```bash
uv run python src/simplest/cached_tokens_probe_gemini_vertex.py --provider vertex --rounds 2
```
Observed output:
```text
Provider probe: vertex_ai/gemini-2.5-flash
 1 | prompt_tokens=20023 cached_tokens=None  text_tokens=20023 completion_tokens=45 total_tokens=20068
 2 | prompt_tokens=20023 cached_tokens=19293 text_tokens=730   completion_tokens=45 total_tokens=20068
```
Interpreting the stats:
- Call #2 shows `cached_tokens=19293` for Vertex AI.
- The pattern is consistent: large cached prefix + smaller remainder in `text_tokens`.

### Example C — cache hit/miss can alternate (Vertex AI)
Command:
```bash
uv run python src/simplest/cached_tokens_probe_gemini_vertex.py --provider vertex --rounds 4
```
Observed output:
```text
Provider probe: vertex_ai/gemini-2.5-flash
 1 | prompt_tokens=20024 cached_tokens=None  text_tokens=20024 completion_tokens=46 total_tokens=20070
 2 | prompt_tokens=20024 cached_tokens=19293 text_tokens=731   completion_tokens=46 total_tokens=20070
 3 | prompt_tokens=20024 cached_tokens=None  text_tokens=20024 completion_tokens=46 total_tokens=20070
 4 | prompt_tokens=20024 cached_tokens=19293 text_tokens=731   completion_tokens=46 total_tokens=20070
```
This shows why the probe defaults to “at least 2” calls but also supports `--rounds N`.

## External documentation
- Gemini API — context caching (implicit + explicit):
  - https://ai.google.dev/gemini-api/docs/caching
- Gemini API response usage metadata fields (includes cached token count):
  - https://ai.google.dev/api/rest/v1beta/GenerateContentResponse

## Notes / caveats
- **Implicit caching is opportunistic** (Google docs: “no cost saving guarantee”). Even with identical prefixes, you may not see `cached_tokens` on exactly the 2nd request every run.
- The probe intentionally creates a **large shared prefix** (well above the minimum threshold) and sends calls close in time (default `--sleep 0.25`).
