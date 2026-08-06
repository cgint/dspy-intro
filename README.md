# Getting Started with DSPy: A Beginner-Friendly Playground

Welcome! If you have no prior experience with DSPy or FunctAI, this repository is designed for you. It contains simple, runnable examples that show how to turn LLMs into structured, typed, and optimizable software components.

Unlike traditional prompt engineering (which relies on manually editing long strings of text), **DSPy** lets you define your inputs and outputs programmatically, and then automatically compiles/optimizes the prompts for you.

---

## 🚀 3-Minute Quick Start

Follow these steps to run your very first example.

### 1. Prerequisites
Make sure you have Python 3.13 installed. We use **`uv`** for lightning-fast dependency and environment management.
*   **Mac (via Homebrew):** `brew install uv`
*   **Windows/Linux/Other:** See [uv installation guide](https://github.com/astral-sh/uv).

### 2. Setup & Environment Keys
Clone this repository, navigate to its folder, and sync the environment:
```bash
uv sync --all-groups --all-extras
```

Next, configure model access by choosing **one** of the following options:

*   **Option A: Gemini API (Simplest)**
    Get a key from [Google AI Studio](https://aistudio.google.com/) (free tier available):
    ```bash
    export GEMINI_API_KEY="your-api-key-here"
    ```

*   **Option B: Google Cloud Vertex AI**
    If you are running in GCP or have active gcloud credentials:
    ```bash
    export VERTEXAI_PROJECT="your-gcp-project-id"
    export VERTEXAI_LOCATION="us-central1" # or your preferred region
    ```

### 3. Run Your First Example
Now, run the simplest demo in the repository:
```bash
uv run simplestdspy
```

This will call Google Gemini, generate a joke for "John", and then evaluate how funny that joke is on a scale of 0 to 10.

---

## 💡 Core Concepts: What is DSPy?

In classic LLM development, you write a string prompt like `"Please write a joke about {name}."`
In DSPy, we define a **Signature** instead.

Look at this snippet from `src/simplest/simplest_dspy.py`:

```python
# "name -> joke" tells DSPy:
# - Input parameter: name
# - Expected output parameter: joke
joker = dspy.Predict("name -> joke")
the_joke_prediction = joker(name="John")
```

And to force structured outputs with types, we can do:
```python
# The ": int" suffix forces the LLM to output a clean integer
funnyness_evaluator = dspy.Predict("joke -> funnyness_0_to_10: int")
```
DSPy automatically constructs the underlying system prompts, parses the outputs, and formats them into python types for you behind the scenes.

---

## 📂 Runnable Examples in this Repo

We have several console scripts defined in `pyproject.toml`. You can run any of them using `uv run <script-name>`:

| Script Command | File Path | What it does |
| :--- | :--- | :--- |
| **`uv run simplestdspy`** | `src/simplest/simplest_dspy.py` | Generates a joke and rates its funnyness using simple DSPy signatures. |
| **`uv run simplestdspyattach`** | `src/simplest/simplest_dspy_with_attachments.py` | Extracts structured key takeaways and summaries directly from a PDF file. |
| **`uv run simplestdspyrlm`** | `src/simplest/simplest_dspy_rlm.py` | Runs a sandboxed agent with sandboxed Python REPL tools. |
| **`uv run password`** | `src/classifier_credentials/...` | Classifies whether text inputs contain exposed credentials (e.g., outputs "safe" or "unsafe"). |
| **`uv run optimizer`** | `src/classifier_credentials/...` | Uses DSPy Optimizers (GEPA / MIPROv2) to automatically refine the classification prompts using training data. |
| **`uv run simplestfunctai`** | `src/simplest/simplest_functai.py` | Demonstrates **FunctAI**, which wraps python functions into type-safe LLM calls. |
| **`uv run extractprompt`** | `src/text_component_extract/...` | Extracts Persona, Task, Context, and Format from any arbitrary text block. |
| **`uv run extractgrammatical`** | `src/text_component_extract/...` | Extracts Subject, Verb, Object, and Modifier grammatical components from sentences. |

---

## 📊 Visualizing Results with MLflow

DSPy can track its inputs, outputs, and intermediate states. To inspect what was sent to the LLM and how it responded:

1. Start the MLflow server:
   ```bash
   uv run mlflow server --host 127.0.0.1 --port 8182
   ```
2. Open your browser and navigate to: **[http://127.0.0.1:8182](http://127.0.0.1:8182)**

---

## 🏗️ Repository Layout

```
├── pyproject.toml                         # Project configuration, scripts, and dependencies
├── src
│   ├── simplest                           # Minimal, high-clarity entry points (start here!)
│   │   ├── simplest_dspy.py               # Basic jokes example
│   │   ├── simplest_dspy_with_attachments.py # PDF processing example
│   │   └── simplest_functai.py            # FunctAI implementation
│   ├── classifier_credentials              # More advanced password classifier & training loop
│   ├── text_component_extract             # Text parser/extractor examples
│   └── common                             # Shared helpers (LLM connections, configurations)
```

---

# 🔍 Reference: Output Examples & Deep Dive Code

Below are the details and expected outputs of the main examples, preserved for detailed reference.

## 1. "A minimal DSPy demo" (`simplestdspy`)

### Full code:
```python
import dspy
from common.utils import get_lm_for_model_name, dspy_configure
from common.constants import MODEL_NAME_GEMINI_3_5_FLASH

def joke_for_john() -> str:
    joker = dspy.Predict("name -> joke")
    the_joke_prediction = joker(name="John")
    return the_joke_prediction.joke

def joke_funnyness_factor_0_to_10(joke: str) -> int:
    funnyness_evaluator = dspy.Predict("joke -> funnyness_0_to_10: int")
    funnyness_prediction = funnyness_evaluator(joke=joke)
    return funnyness_prediction.funnyness_0_to_10

def main():
    dspy_configure(get_lm_for_model_name(MODEL_NAME_GEMINI_3_5_FLASH, "disable"))

    the_joke: str = joke_for_john()
    print(f"\n\n{the_joke}")
  
    funnyness: int = joke_funnyness_factor_0_to_10(the_joke)
    print(f" -> How funny is the joke on a scale of 0 to 10? {funnyness}\n")

if __name__ == "__main__":
    main()
```

### Expected Output:
```
Why did John bring a ladder to the bar? Because he heard the drinks were on the house!
 -> How funny is the joke on a scale of 0 to 10? 6
```

---

## 2. "Credentials/passwords classifier" (`password`)

### Expected Output:
```
Input text: My username is john and password is secret123
  -> Classification: unsafe


Input text: My login is admin and my password is --REDACTED--
  -> Classification: safe
```

---

## 3. "A minimal DSPy demo processing a PDF" (`simplestdspyattach`)

This script extracts key metrics and takeaways from `src/simplest/docs/simplest_dspy_with_attachments_2507.11299.pdf`.

### Expected Output:
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

---

## 4. FunctAI & Text Component Extraction (`simplestfunctai`, `extractprompt`, `extractgrammatical`)

FunctAI is based on DSPy and turns Python functions into typed LLM-Calls (learn more at [https://github.com/MaximeRivest/functai](https://github.com/MaximeRivest/functai)).

### Prompt Component Extraction (`extractprompt`)
Extracts the four main components of a prompt from a given instruction:
- **Persona:** Who the AI should act as.
- **Task:** What specific action needs to be performed.
- **Context:** Background information or details.
- **Format:** How the output should be structured.

### Grammatical Component Extraction (`extractgrammatical`)
Extracts fundamental grammatical parts from sentences:
- **Subject**
- **Verb**
- **Object**
- **Modifier**
