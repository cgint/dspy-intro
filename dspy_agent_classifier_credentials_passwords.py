#!/usr/bin/env python3
# /// script
# dependencies = [
#     "dspy-ai",
#     "mlflow"
# ]
# requires-python = ">=3.11"
# ///

import dspy
import os
from typing import Literal

import mlflow
from dspy_constants import MODEL_NAME_GEMINI_2_5_FLASH_LITE
# mlflow.set_experiment("dspy_agent_classifier_credentials_passwords")
# mlflow.autolog()


# --- Main DSPy Agent ---

class ClassifierCredentialsPasswordsSignature(dspy.Signature):
    """Classify text to detect if it contains exposed credentials or passwords."""
    classify_input: str = dspy.InputField(
        desc="Text input to analyze for potential credential or password exposure"
    )
    classification: Literal["safe", "unsafe"] = dspy.OutputField(
        desc="Classification result: 'unsafe' if credentials/passwords are exposed, 'safe' if no credentials/passwords or properly protected/redacted"
    )

# --- Example Usage ---
# Make sure that VERTEXAI_PROJECT=smec-whoop-dev and VERTEXAI_LOCATION=europe-west1 are set as environment variables
if not os.getenv("VERTEXAI_PROJECT") or not os.getenv("VERTEXAI_LOCATION"):
    raise ValueError("VERTEXAI_PROJECT and VERTEXAI_LOCATION must be set as environment variables")

classifier_lm_model_name = MODEL_NAME_GEMINI_2_5_FLASH_LITE
classifier_lm_reasoning_effort = "disable"
classifier_lm = dspy.LM(
    model=f'vertex_ai/{classifier_lm_model_name}',
    reasoning_effort=classifier_lm_reasoning_effort # other options are Literal["low", "medium", "high"]
    # thinking={"type": "enabled", "budget_tokens": 512}
)

class ClassifierCredentialsPasswords(dspy.Module):
    def __init__(self, lm: dspy.LM = classifier_lm):
        super().__init__()
        self.classifier = dspy.Predict(ClassifierCredentialsPasswordsSignature)
        self.classifier_lm = lm

    def forward(self, classify_input: str) -> dspy.Prediction:
        with dspy.context(lm=self.classifier_lm, track_usage=True):
            return self.classifier(classify_input=classify_input)


if __name__ == "__main__":
    try:
        dspy.settings.configure(
            lm=classifier_lm, track_usage=True
        )
        dspy.configure_cache(
            enable_disk_cache=False,
            enable_memory_cache=False
        )
        print(f"DSPy configured to use {dspy.settings.lm.model}.")
    except Exception as e:
        print(f"Error configuring DSPy: {e}")
        exit(1)


    with mlflow.start_run():
        classifier = ClassifierCredentialsPasswords()
        input_text = "My username is john and password is secret123"
        result = classifier(classify_input=input_text)
        print(f"Input text: {input_text}")
        print(f"Classification: {result.classification}")
        print(result)