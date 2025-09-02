# DSPy Intro - Credentials Classifier Examples

This repository contains runnable DSPy examples:

- Credentials/passwords classifier
- Optimizer that improves the classifier with GEPA/MIPROv2
- A minimal DSPy demo

## Requirements

- Python >=3.11
- uv (dependency and environment management)
  - Install using `brew install uv`

## Setup

```bash
uv sync --all-groups --all-extras 
```

Configure model access (pick one):

```bash
# Option A: Vertex AI
export VERTEXAI_PROJECT="<your_gcp_project>"
export VERTEXAI_LOCATION="<region>"   # e.g., europe-west1

# Option B: Gemini API (Google AI Studio)
# Key can be taken from 1-password called 'Gemini API Key dev (Google AI Studio)'
export GEMINI_API_KEY="<your_gemini_api_key>"
```

## Running

Console scripts are defined in pyproject.toml:

```bash
# Minimal DSPy example
uv run simplest

# Basic credentials/passwords classifier
uv run password

# Credentials classifier optimization (GEPA / MIPROv2)
uv run optimizer
```

## Project Structure

```
├── pyproject.toml
├── README.md
└── src
    ├── simplest
    │   ├── simplest_dspy.py  # minimal DSPy example + main()
    ├── common
    │   ├── constants.py      # model names
    │   ├── mlflow_utils.py   # MLflow helpers
    │   └── utils.py          # LM factory and dspy_configure helpers
    ├── classifier_credentials
    │   ├── dspy_agent_classifier_credentials_passwords_examples.py   # data prep & example sets
    │   ├── dspy_agent_classifier_credentials_passwords.py            # basic classifier + main()
    │   └── dspy_agent_classifier_credentials_passwords_optimized.py  # optimizer (GEPA/MIPROv2) + main()
```
