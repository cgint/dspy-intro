import os
import dspy
from typing import Literal
GOOGLE_PROVIDER_GEMINI = "gemini"
GOOGLE_PROVIDER_VERTEX_AI = "vertex_ai"
GOOGLE_PROVIDER_LIST = [GOOGLE_PROVIDER_GEMINI, GOOGLE_PROVIDER_VERTEX_AI]

def get_model_access_prefix_or_fail(model_name: str) -> str:
    """
    Whenever the model name already contains the access prefix then look up if the env variables are set and return the access prefix.
    In case the model name does not contain the access prefix then decide from the env variables what access prefix to return.
    When no prefix is specified, vertex_ai/ is prioritized if available.

    Returns e.g. "" or "gemini/" or "vertex_ai/"
    """
    provider_from_model_name: str | None = model_name.split("/")[0] if "/" in model_name else None
    if provider_from_model_name is not None and provider_from_model_name not in GOOGLE_PROVIDER_LIST:
        return "" # In this case the model name already contains the provider and it is one we use as is

    # Collect state: what credentials are available
    has_vertex_ai_env_vars = os.getenv("VERTEXAI_PROJECT") and os.getenv("VERTEXAI_LOCATION")
    has_gemini_env_vars = os.getenv("GEMINI_API_KEY")
    
    selected_provider: str | None = None
    # Evaluate based on model name and available credentials
    if provider_from_model_name is not None and provider_from_model_name == GOOGLE_PROVIDER_VERTEX_AI:
        if not has_vertex_ai_env_vars:
            raise ValueError(f"Both VERTEXAI_PROJECT and VERTEXAI_LOCATION must be set as environment variables for {GOOGLE_PROVIDER_VERTEX_AI} prefix")
        selected_provider = provider_from_model_name
    elif provider_from_model_name is not None and provider_from_model_name == GOOGLE_PROVIDER_GEMINI:
        if not has_gemini_env_vars:
            raise ValueError(f"GEMINI_API_KEY must be set as environment variable for {GOOGLE_PROVIDER_GEMINI} prefix")
        selected_provider = provider_from_model_name
    
    # No explicit prefix - decide based on available credentials
    if not selected_provider:
        if has_vertex_ai_env_vars:
            selected_provider = GOOGLE_PROVIDER_VERTEX_AI
        elif has_gemini_env_vars:
            selected_provider = GOOGLE_PROVIDER_GEMINI
        else:
            raise ValueError("Either (VERTEXAI_PROJECT and VERTEXAI_LOCATION) or (GEMINI_API_KEY) must be set as environment variables.")
    
    # Unset the other env vars to have clarity
    if selected_provider == GOOGLE_PROVIDER_VERTEX_AI:
        os.unsetenv("GEMINI_API_KEY")
    elif selected_provider == GOOGLE_PROVIDER_GEMINI:
        os.unsetenv("VERTEXAI_PROJECT")
        os.unsetenv("VERTEXAI_LOCATION")

    # finished preparing
    print(f"Using model access prefix: {selected_provider}")
    return f"{selected_provider}/"

def dspy_configure(lm: dspy.LM, track_usage: bool = True, adapter: dspy.Adapter = dspy.JSONAdapter()):
    """
    Using JSONAdapter as it is the most reliable adapter from tests.
    XMLAdapter and ChatAdapter force retries using JSONAdapter as fallback anyways.
    """
    dspy.settings.configure(lm=lm, track_usage=track_usage, adapter=adapter)
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

def get_lm_for_model_name(model_name: str, reasoning_effort: Literal["low", "medium", "high", "disable"] | None = "disable", max_tokens: int = 8192, temperature: float = 0.3) -> dspy.LM:
    model_access_prefix: str = get_model_access_prefix_or_fail(model_name)
    return dspy.LM(
        model=f'{model_access_prefix}{model_name}',
        max_tokens=max_tokens, temperature=temperature,
        reasoning_effort=reasoning_effort if reasoning_effort is not None else None,
        # thinking={"type": "enabled", "budget_tokens": 512}
    )