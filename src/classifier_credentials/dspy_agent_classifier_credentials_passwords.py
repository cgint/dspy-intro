import dspy
from typing import Literal

import mlflow
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH_LITE
from common.utils import get_lm_for_model_name, dspy_configure
mlflow.set_experiment("dspy_agent_classifier_credentials_passwords")
mlflow.autolog()
# call 'uv run mlflow server --host 127.0.0.1 --port 8182' and head to http://127.0.0.1:8182

# --- Main DSPy Classifier ---

class ClassifierCredentialsPasswordsSignature(dspy.Signature):
    """Classify text to detect if it contains exposed credentials or passwords."""
    classify_input: str = dspy.InputField(
        desc="Text input to analyze for potential credential or password exposure"
    )
    classification: Literal["safe", "unsafe"] = dspy.OutputField(
        desc="Classification result: 'unsafe' if credentials/passwords are exposed, 'safe' if no credentials/passwords or properly protected/redacted"
    )

# --- Example Usage ---
classifier_lm_model_name = MODEL_NAME_GEMINI_2_5_FLASH_LITE
classifier_lm_reasoning_effort = "disable"
classifier_lm = get_lm_for_model_name(classifier_lm_model_name, classifier_lm_reasoning_effort)

class ClassifierCredentialsPasswords(dspy.Module):
    def __init__(self, lm: dspy.LM = classifier_lm):
        super().__init__()
        self.classifier = dspy.Predict(ClassifierCredentialsPasswordsSignature)
        self.classifier_lm = lm

    def forward(self, classify_input: str) -> dspy.Prediction:
        with dspy.context(lm=self.classifier_lm, track_usage=True):
            return self.classifier(classify_input=classify_input)


def main():
    dspy_configure(classifier_lm)

    with mlflow.start_run():
        classifier = ClassifierCredentialsPasswords()
        input_text_unsafe = "My username is john and password is secret123"
        result = classifier(classify_input=input_text_unsafe)
        print(f"\n\nInput text: {input_text_unsafe}")
        print(f"  -> Classification: {result.classification}")

        input_text_safe = "My login is admin and my password is --REDACTED--"
        result = classifier(classify_input=input_text_safe)
        print(f"\n\nInput text: {input_text_safe}")
        print(f"  -> Classification: {result.classification}")


if __name__ == "__main__":
    main()