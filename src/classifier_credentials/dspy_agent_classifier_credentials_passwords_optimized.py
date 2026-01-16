from typing import Any, Literal, Optional, Union, Dict
from datetime import datetime
import time
import dspy
from dspy.teleprompt.gepa.gepa import GEPAFeedbackMetric
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
import mlflow
from classifier_credentials.dspy_agent_classifier_credentials_passwords_examples import (
    prepare_training_data,
    prepare_test_data
)
from classifier_credentials.dspy_agent_classifier_credentials_passwords import ClassifierCredentialsPasswords, classifier_lm_model_name, classifier_lm_reasoning_effort
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH
from common.utils import dspy_configure, get_lm_for_model_name
from common.mlflow_utils import log_as_table

mlflow.set_experiment("dspy_agent_classifier_credentials_passwords_optimized")
mlflow.autolog()
# call 'uv run mlflow server --host 127.0.0.1 --port 8182' and head to http://127.0.0.1:8182

# --- Metric for Optimization ---

def classification_accuracy(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Metric function for MIPROv2 optimization.
    Returns 1.0 for correct classification, 0.0 for incorrect.
    """
    return float(example.classification == pred.classification)


class ClassificationAccuracyWithFeedbackMetric(GEPAFeedbackMetric):
	def __call__(
		self,
		gold: dspy.Example,
		pred: dspy.Prediction,
		trace: Any = None,
		pred_name: Optional[str] = None,
		pred_trace: Any = None,
	) -> Union[float, ScoreWithFeedback]:
		answer_is_same = classification_accuracy(gold, pred)
		total = answer_is_same
		if pred_name is None:
			return total
		if not answer_is_same:
			feedback = "The classifier is not working as expected. Please check the classifier module."
		else:
			feedback = "The classifier is working as expected."
		return ScoreWithFeedback(score=total, feedback=feedback)


# --- Optimization ---

def to_percent_int(input: Any) -> int:
    if isinstance(input, float):
        return int(input)
    else:
        raise ValueError(f"Cannot convert {input} to float. Type: {type(input)}")

def optimize_classifier(optimizer_type: Literal["MIPROv2", "GEPA"], trainer_lm: dspy.LM, auto: Literal["light", "medium", "heavy"], limit_trainset: int, limit_testset: int, randomize_sets: bool, reflection_minibatch_size: int):
    """
    Optimize the classifier using DSPy optimizer
    """
    
    print(f"🚀 Starting optimization on {classifier_lm_model_name} with reasoning effort {classifier_lm_reasoning_effort} with parameters: optimizer_type={optimizer_type}, trainer_lm={trainer_lm.model}, auto={auto}")
    
    # Prepare data
    trainset = prepare_training_data(limit=limit_trainset, randomize=randomize_sets)
    testset = prepare_test_data(limit=limit_testset, randomize=randomize_sets)
    
    print(f"Training set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")
    
    # Test baseline performance
    print("\n📊 Baseline Performance:")
    baseline_score = dspy.Evaluate(
        devset=testset, 
        metric=classification_accuracy,
        num_threads=4,
        display_progress=True
    )(ClassifierCredentialsPasswords())
    print(f"Baseline accuracy: {to_percent_int(baseline_score.score)}%")
    
    # Initialize optimizer
    # Using "light" for fast optimization, can try "medium" or "heavy" for better results
    print(f"\n🔧 Running {optimizer_type} optimization...")
    print("This may take a few minutes...")
    
    if optimizer_type == "MIPROv2":
        optimizer = dspy.MIPROv2(
            metric=classification_accuracy,
            auto=auto,  # Options: "light", "medium", "heavy"
            num_threads=4,
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
            prompt_model=trainer_lm
        )
    elif optimizer_type == "GEPA":
        optimizer = dspy.GEPA(
            metric=ClassificationAccuracyWithFeedbackMetric(),
            auto=auto,
            num_threads=4,
            track_stats=True,
            skip_perfect_score=True,
            add_format_failure_as_feedback=True,
            reflection_minibatch_size=reflection_minibatch_size,
            reflection_lm=trainer_lm
        )
    
    # Compile/optimize the classifier
    classifier = ClassifierCredentialsPasswords()
    optimized_classifier = optimizer.compile(
        classifier,
        trainset=trainset,
        valset=testset
    )
    
    # Test optimized performance
    print("\n🎯 Optimized Performance:")
    optimized_score = dspy.Evaluate(
        devset=testset,
        metric=classification_accuracy,
        num_threads=4,
        display_progress=True
    )(optimized_classifier)
    
    baseline_score_int = to_percent_int(baseline_score.score)
    optimized_score_int = to_percent_int(optimized_score.score)
    print("\n📈 Results:")
    print(f"Baseline accuracy:  {baseline_score_int}%")
    print(f"Optimized accuracy: {optimized_score_int}%")

    
    # Save a combined metadata + program JSON
    combined_save_path = f"optimized_credentials_classifier_{int(time.time())}_{baseline_score_int}_to_{optimized_score_int}_combined.json"
    combined_data = {
        "metadata": {
            "classifier_lm_model_name": classifier_lm_model_name,
            "classifier_lm_reasoning_effort": str(classifier_lm_reasoning_effort),
            "optimizer_type": optimizer_type,
            "auto": auto,
            "baseline_accuracy": baseline_score_int,
            "optimized_accuracy": optimized_score_int,
            "timestamp": int(time.time()),
            "trainset_size": limit_trainset,
            "testset_size": limit_testset,
        },
        "program": optimized_classifier.dump_state()
    }
    import json
    with open(combined_save_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    print(f"💾 Combined format saved to: {combined_save_path}")
    
    return optimized_classifier, combined_save_path, baseline_score_int, optimized_score_int


def test_classifier_examples(classifier, examples_desc="", question_prefix="") -> Dict[str, str]:
    """Test the classifier with some example inputs"""
    print(f"\n🧪 Testing {examples_desc}:")
    
    test_inputs = [
        "My username is john and password is secret123",
        "API token: sk-abc123def456",
        "Please enter your password: [REDACTED]",
        "The user needs to provide valid credentials",
        "Database password: ***hidden***"
    ]
    
    results = {}
    for test_input in test_inputs:
        result = classifier(classify_input=test_input)
        results[test_input] = result.classification
        print(f"Input: {test_input}")
        print(f"Classification: {result.classification}")
        if hasattr(result, 'reasoning'):
            print(f"Reasoning: {result.reasoning}")
        print("-" * 50)
    return results


def main():
    dspy_configure(get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, "disable"))
    
    
    formatter_date_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with mlflow.start_run(run_name=f"pwd_classifier_{formatter_date_now}"):
        
        # Optimize classifier with
        trainer_lm_model_name = MODEL_NAME_GEMINI_2_5_FLASH
        trainer_lm_reasoning_effort = "disable"
        optimizer_type = "GEPA" # "MIPROv2" # "GEPA"
        auto = "heavy"  # <-- We will use a "light" budget for this tutorial. However, we typically recommend using auto="heavy" for optimized performance!
        limit_trainset = 50
        limit_testset = 30
        randomize_sets = True
        reflection_minibatch_size = limit_testset # Used in GEPA only - when too low then it can loop on seemingly perfect proposed candidates

        mlflow.log_param("classifier_lm_model", classifier_lm_model_name)
        mlflow.log_param("classifier_lm_reasoning_effort", classifier_lm_reasoning_effort)
        mlflow.log_param("trainer_lm_model", trainer_lm_model_name)
        mlflow.log_param("trainer_lm_reasoning_effort", trainer_lm_reasoning_effort)
        mlflow.log_param("optimizer_type", optimizer_type)
        mlflow.log_param("auto", auto)
        mlflow.log_param("limit_trainset", limit_trainset)
        mlflow.log_param("limit_testset", limit_testset)
        mlflow.log_param("randomize_sets", randomize_sets)
        mlflow.log_param("reflection_minibatch_size", reflection_minibatch_size)

        # Test baseline classifier
        baseline_classifier = ClassifierCredentialsPasswords()
        baseline_results = test_classifier_examples(baseline_classifier, "Baseline Classifier")
        log_as_table(baseline_results, optimization_type="baseline")

        trainer_lm = get_lm_for_model_name(trainer_lm_model_name, trainer_lm_reasoning_effort)

        optimizer_start_time_sec = time.time()
        optimized_classifier, combined_save_path, baseline_score_int, optimized_score_int = optimize_classifier(optimizer_type, trainer_lm, auto, limit_trainset, limit_testset, randomize_sets, reflection_minibatch_size)
        optimizer_end_time_sec = time.time()
        optimizer_duration_sec = optimizer_end_time_sec - optimizer_start_time_sec
        mlflow.log_metric("optimizer_duration_seconds", optimizer_duration_sec)
        
        # Test optimized classifier
        optimized_results = test_classifier_examples(optimized_classifier, "Optimized Classifier")
        log_as_table(optimized_results, optimization_type="optimized")

        mlflow.log_metric("baseline_accuracy", baseline_score_int)
        mlflow.log_metric("optimized_accuracy", optimized_score_int)
    
        print(f"\n🎉 Optimization complete! Optimized model + metadata saved to: {combined_save_path}")
        print(f"Baseline accuracy: {baseline_score_int}%")
        print(f"Optimized accuracy: {optimized_score_int}%")


if __name__ == "__main__":
    main()