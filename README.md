# DSPy Intro - Credentials Classifier Examples + Simplest Example

[https://github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)

This repository contains runnable DSPy examples:

- Credentials/passwords classifier
- Optimizer that improves the classifier with GEPA/MIPROv2
- A minimal DSPy demo
- A minimal DSPy demo processing a PDF

## Output of: "A minimal DSPy demo"

### Full code
```python
import dspy
from common.utils import get_lm_for_model_name, dspy_configure
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH

def joke_for_john() -> str:
    joker = dspy.Predict("name -> joke")
    the_joke_prediction = joker(name="John")
    return the_joke_prediction.joke

def joke_funnyness_factor_0_to_10(joke: str) -> int:
    funnyness_evaluator = dspy.Predict("joke -> funnyness_0_to_10: int")
    funnyness_prediction = funnyness_evaluator(joke=joke)
    return funnyness_prediction.funnyness_0_to_10

def main():
    dspy_configure(get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, "disable"))

    the_joke: str = joke_for_john()
    print(f"\n\n{the_joke}")
    
    funnyness: int = joke_funnyness_factor_0_to_10(the_joke)
    print(f" -> How funny is the joke on a scale of 0 to 10? {funnyness}\n")

if __name__ == "__main__":
    main()

``` 

### Output
```
Why did John bring a ladder to the bar? Because he heard the drinks were on the house!
 -> How funny is the joke on a scale of 0 to 10? 6
```

## Output of: "Credentials/passwords classifier"

```
Input text: My username is john and password is secret123
  -> Classification: unsafe


Input text: My login is admin and my password is --REDACTED--
  -> Classification: safe
```

## Output of: "A minimal DSPy demo processing a PDF"

```
Context: src/simplest/docs/simplest_dspy_with_attachments_2507.11299.pdf
 -> Processing ...

Answer to the question 'What is the main idea of the paper?':
=============================================================

The paper introduces Dr.Copilot, a multi-agent LLM system designed to improve the quality of doctor-patient communication in Romanian text-based telemedicine. It focuses on enhancing the presentation of medical advice rather than its clinical accuracy, providing feedback along 17 interpretable axes. The system uses automatically optimized prompts via DSPy and has shown measurable improvements in user reviews and response quality in a real-world deployment with 41 doctors.


Answer to the question 'What are the key takeaways of the paper?':
==================================================================

The key takeaways of the paper are:

1.  **Introduction of Dr.Copilot**: The paper introduces Dr.Copilot, a multi-agent LLM system designed to improve the presentation quality of written medical responses by Romanian-speaking doctors in telemedicine. It focuses on communication quality across 17 interpretable dimensions rather than medical accuracy.
2.  **Automatic Prompt Optimization with DSPy**: Dr.Copilot utilizes an automatic prompt optimization approach using DSPy, which allows for effective performance with limited labeled data (100 annotated examples). This method also ensures privacy-preserving deployment by using open-weight models.
3.  **Real-World Deployment and Impact**: The system has been deployed in a live environment with 41 doctors, demonstrating measurable improvements in response quality and patient satisfaction. Specifically, there was a 70.22% increase in positive patient reviews for responses that incorporated Dr.Copilot's suggestions. This marks one of the first real-world deployments of LLMs in Romanian medical settings, addressing challenges associated with a low-resource language.
4.  **Multi-Agent Framework**: Dr.Copilot consists of three main components: a Scoring Agent (evaluates responses based on quality metrics), a Recommender Agent (generates tailored suggestions), and a Reconciliation Agent (for self-evaluation of recommendations).
5.  **Ethical Considerations**: The system is designed as a supportive tool for physicians, not a replacement for professional judgment or a direct medical advice provider. It uses on-premise, open-weight models to minimize data privacy risks, ensuring patient data remains within the institution's infrastructure.


Summary of the pdf:
===================

This paper introduces Dr.Copilot, a multi-agent LLM system designed to improve patient-doctor communication in Romanian text-based telemedicine. Unlike systems that provide medical advice, Dr.Copilot focuses on enhancing the presentation quality of doctors' written responses across 17 interpretable dimensions, without interfering with medical content. The system uses three LLM agents (Scorer, Recommender, and Reconciliation) with prompts optimized via DSPy, enabling effective performance with limited labeled data and privacy-preserving deployment using open-weight models like MedGemma-27B. Live deployment with 41 doctors showed measurable improvements in user reviews and response quality, with a 70.22% increase in positive patient reviews for responses incorporating Dr.Copilot's suggestions. The study highlights the practical application of LLMs in healthcare for underrepresented languages and emphasizes ethical considerations by ensuring doctor control over medical content and local deployment to protect patient data.


Covered topics and their importance (from 0 low to 10 high):
============================================================

 - (Importance: 9) Dr.Copilot System Overview
 - (Importance: 9) Multi-Agent LLM System for Telemedicine
 - (Importance: 8) Improving Patient-Doctor Communication
 - (Importance: 8) Romanian Language in Medical AI
 - (Importance: 8) Evaluation and Live Deployment Results
 - (Importance: 7) Prompt Optimization with DSPy
 - (Importance: 7) Ethical Considerations in Medical AI
 - (Importance: 7) Scoring and Recommendation Agents
 - (Importance: 6) Limitations of the Study
 - (Importance: 6) Pretrained Models Used
 ```

# FunctAI Intro - Simplest Example

FunctAI is based on DSPy and makes python functions become typed LLM-Calls

[https://github.com/MaximeRivest/functai](https://github.com/MaximeRivest/functai)

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
uv run simplestdspy

# Minimal DSPy example processing a PDF
uv run simplestdspyattach

# Minimal FunctAI example
uv run simplestfunctai


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
    │   ├── simplest_dspy.py     # minimal DSPy example
    │   ├── simplest_dspy_with_attachments.py # minimal DSPy example processing a PDF
    │   └── simplest_functai.py  # minimal FunctAI example
    ├── common
    │   ├── constants.py      # model names
    │   ├── mlflow_utils.py   # MLflow helpers
    │   └── utils.py          # LM factory and dspy_configure helpers
    ├── classifier_credentials
    │   ├── dspy_agent_classifier_credentials_passwords.py            # basic classifier
    │   ├── dspy_agent_classifier_credentials_passwords_optimized.py  # optimizer (GEPA/MIPROv2)
    │   └── dspy_agent_classifier_credentials_passwords_examples.py   # data prep & example sets
```
