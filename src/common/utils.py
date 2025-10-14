import os
import dspy
from typing import Literal

def get_model_access_prefix_or_fail() -> str:
    if os.getenv("VERTEXAI_PROJECT") and os.getenv("VERTEXAI_LOCATION"):
        return 'vertex_ai/'
    elif os.getenv("GEMINI_API_KEY"):
        return 'gemini/'
    else:
        raise ValueError("Either (VERTEXAI_PROJECT and VERTEXAI_LOCATION) or (GEMINI_API_KEY) must be set as environment variables.")
    
def dspy_configure(lm: dspy.LM, track_usage: bool = True):
    dspy.settings.configure(lm=lm, track_usage=track_usage)
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

def get_lm_for_model_name(model_name: str, reasoning_effort: Literal["low", "medium", "high", "disable"] = "disable", max_tokens: int = 8192, temperature: float = 0.3) -> dspy.LM:
    model_access_prefix: str = get_model_access_prefix_or_fail()
    return dspy.LM(
        model=f'{model_access_prefix}{model_name}',
        max_tokens=max_tokens, temperature=temperature,
        reasoning_effort=reasoning_effort # other options are Literal["low", "medium", "high"]
        # thinking={"type": "enabled", "budget_tokens": 512}
    )