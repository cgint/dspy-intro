import os

import dspy
from dotenv import load_dotenv

# Load environment variables from .env file if it exists, overriding existing shell env vars
load_dotenv(override=True)

from typing import Literal

from common.constants import (
    OLLAMA_MODEL,
    OLLAMA_OPENAI_API_KEY,
    OLLAMA_OPENAI_BASE_URL,
)

GOOGLE_PROVIDER_GEMINI = "gemini"
GOOGLE_PROVIDER_VERTEX_AI = "vertex_ai"
GOOGLE_PROVIDER_LIST = [GOOGLE_PROVIDER_GEMINI, GOOGLE_PROVIDER_VERTEX_AI]
VERTEX_AI_FALLBACK_MODEL = "gemini-3.5-flash-lite"

def _provider_from_model_name(model_name: str) -> str | None:
    """Extract provider prefix from model name if present, e.g. 'vertex_ai/gemini-3.5' -> 'vertex_ai'.
    Returns None if no prefix or prefix not in GOOGLE_PROVIDER_LIST.
    """
    prefix: str | None = model_name.split("/")[0] if "/" in model_name else None
    if prefix is not None and prefix in GOOGLE_PROVIDER_LIST:
        return prefix
    return None


def _resolve_provider(prefix: str | None, has_vertex_ai: bool, has_gemini: bool) -> str:
    """Resolve which provider to use based on explicit prefix and available credentials."""
    if prefix == GOOGLE_PROVIDER_VERTEX_AI:
        if not has_vertex_ai:
            raise ValueError(
                f"Both VERTEXAI_PROJECT and VERTEXAI_LOCATION must be set as "
                f"environment variables for {GOOGLE_PROVIDER_VERTEX_AI} prefix"
            )
        return GOOGLE_PROVIDER_VERTEX_AI
    if prefix == GOOGLE_PROVIDER_GEMINI:
        return _require_gemini_env(has_gemini)
    
    # No explicit prefix — auto-detect
    if has_vertex_ai:
        return GOOGLE_PROVIDER_VERTEX_AI
    if has_gemini:
        return GOOGLE_PROVIDER_GEMINI
    raise ValueError(
        "Either (VERTEXAI_PROJECT and VERTEXAI_LOCATION) or (GEMINI_API_KEY) "
        "must be set as environment variables."
    )


def _require_gemini_env(has_gemini_env: bool) -> str:
    if not has_gemini_env:
        raise ValueError("GEMINI_API_KEY must be set as environment variable for gemini prefix")
    return GOOGLE_PROVIDER_GEMINI


def _cleanup_other_env_vars(selected_provider: str) -> None:
    if selected_provider == GOOGLE_PROVIDER_VERTEX_AI:
        os.unsetenv("GEMINI_API_KEY")
    elif selected_provider == GOOGLE_PROVIDER_GEMINI:
        os.unsetenv("VERTEXAI_PROJECT")
        os.unsetenv("VERTEXAI_LOCATION")


def get_model_access_prefix_or_fail(model_name: str) -> str:
    """
    Determine which model access prefix to use ("", "gemini/", or "vertex_ai/").

    If the model name already contains a valid provider prefix (e.g. "gemini/..."),
    the prefix is returned as-is after validating env vars.
    Otherwise the function auto-selects vertex_ai (preferred) or gemini
    based on available environment variables.
    """
    prefix = _provider_from_model_name(model_name)
    if prefix is not None:
        return ""  # model name already has the prefix; caller uses it as-is

    has_vertex_ai = bool(os.getenv("VERTEXAI_PROJECT") and os.getenv("VERTEXAI_LOCATION"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))

    selected = _resolve_provider(prefix, has_vertex_ai, has_gemini)
    _cleanup_other_env_vars(selected)

    print(f"Using model access prefix: {selected}")
    return f"{selected}/"

def dspy_configure(lm: dspy.LM, track_usage: bool = True, adapter: dspy.Adapter = dspy.JSONAdapter()):
    """
    Using JSONAdapter as it is the most reliable adapter from tests.
    XMLAdapter and ChatAdapter force retries using JSONAdapter as fallback anyways.
    """
    dspy.settings.configure(lm=lm, track_usage=track_usage, adapter=adapter)
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

def _model_for_provider(model_name: str, model_access_prefix: str) -> str:
    if model_access_prefix == f"{GOOGLE_PROVIDER_VERTEX_AI}/" and model_name.startswith("gemini-3"):
        return VERTEX_AI_FALLBACK_MODEL
    return model_name

def get_lm_for_model_name(model_name: str, reasoning_effort: Literal["low", "medium", "high", "disable"] | None = "disable", max_tokens: int = 8192, temperature: float = 0.3) -> dspy.LM:
    model_access_prefix: str = get_model_access_prefix_or_fail(model_name)
    effective_model_name = _model_for_provider(model_name, model_access_prefix)
    return dspy.LM(
        model=f'{model_access_prefix}{effective_model_name}',
        max_tokens=max_tokens, temperature=temperature,
        reasoning_effort=reasoning_effort if reasoning_effort is not None else None,
        # thinking={"type": "enabled", "budget_tokens": 512}
    )

def get_lm_for_ollama(
    model_name: str | None = None,
    api_base_url: str | None = None,
    api_key: str | None = None,
    reasoning_effort: Literal["low", "medium", "high", "disable"] | None = "disable",
    max_tokens: int = 8192,
    temperature: float = 0.3
) -> dspy.LM:
    """
    Get DSPy LM for Ollama using OpenAI-compatible API.
    
    Args:
        model_name: Ollama model name (defaults to OLLAMA_MODEL from constants)
        api_base_url: Base URL for OpenAI-compatible API (defaults to OLLAMA_OPENAI_BASE_URL)
        api_key: API key placeholder (defaults to OLLAMA_OPENAI_API_KEY)
        reasoning_effort: Reasoning effort level
        max_tokens: Maximum tokens
        temperature: Temperature setting
    """
    selected_model = model_name if model_name else OLLAMA_MODEL
    selected_base_url = api_base_url if api_base_url else OLLAMA_OPENAI_BASE_URL
    selected_api_key = api_key if api_key else OLLAMA_OPENAI_API_KEY
    
    # Set environment variables for OpenAI-compatible API
    os.environ["OPENAI_API_KEY"] = selected_api_key
    os.environ["OPENAI_BASE_URL"] = selected_base_url
    # Use the root base URL (without /v1) for Ollama provider
    if selected_base_url.endswith("/v1"):
        os.environ["OLLAMA_BASE_URL"] = selected_base_url[:-3]
    else:
        os.environ["OLLAMA_BASE_URL"] = selected_base_url
    
    return dspy.LM(
        model=f"ollama/{selected_model}",
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort if reasoning_effort is not None else None,
    )