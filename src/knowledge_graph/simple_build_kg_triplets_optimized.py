from typing import Any, Literal, Optional, Union
import time
import json
import dspy
from dspy.teleprompt.gepa.gepa import GEPAFeedbackMetric
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from common.utils import dspy_configure, get_lm_for_ollama, get_lm_for_model_name
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH

# Import from examples file
from knowledge_graph.dspy_agent_triplet_extraction_examples import (
    prepare_training_data,
    prepare_test_data
)

# Import TripletExtractor from simple_build_kg_triplets.py
from knowledge_graph.simple_build_kg_triplets import TripletExtractor

# Import judge signature
from knowledge_graph.triplet_quality_judge import TripletQualityJudgeSignature

# call 'uv run mlflow server --host 127.0.0.1 --port 8182' and head to http://127.0.0.1:8182

# --- LLM Judge for Triplet Quality Evaluation ---


class TripletExtractionQualityWithFeedbackMetric(GEPAFeedbackMetric):  # type: ignore[misc]
    """
    GEPA Feedback Metric with LLM as judge for evaluating triplet extraction quality.
    Uses Gemini-2.5-Pro with low reasoning effort to evaluate:
    - Correctness: Are extracted triplets semantically correct?
    - Completeness: Did we extract all important triplets from the text?
    - Quality: Are subjects/objects proper entities (not modifiers/locations)?
    - Predicate quality: Are predicates concise and meaningful?
    """
    
    def __init__(self, judge_lm: dspy.LM):
        super().__init__()
        self.judge_lm = judge_lm
        self.judge = dspy.Predict(TripletQualityJudgeSignature)
    
    def __call__(
        self,
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace: Any = None,
        pred_name: Optional[str] = None,
        pred_trace: Any = None,
    ) -> Union[float, ScoreWithFeedback]:
        """
        Evaluate triplet extraction quality using LLM as judge.
        Returns score (0.0 to 1.0) and detailed feedback.
        """
        # Extract expected and predicted triplets
        expected_triplets = gold.result.triplets if hasattr(gold, 'result') and hasattr(gold.result, 'triplets') else []
        predicted_triplets = pred.result.triplets if hasattr(pred, 'result') and hasattr(pred.result, 'triplets') else []
        
        # Convert triplets to JSON strings for LLM evaluation
        expected_json = json.dumps([{"subject": t.subject, "predicate": t.predicate, "object": t.object} for t in expected_triplets], ensure_ascii=False, indent=2)
        predicted_json = json.dumps([{"subject": t.subject, "predicate": t.predicate, "object": t.object} for t in predicted_triplets], ensure_ascii=False, indent=2)
        
        # Use DSPy judge to evaluate
        with dspy.context(lm=self.judge_lm, track_usage=False):
            judge_result = self.judge(
                original_text=gold.text,
                expected_triplets_json=expected_json,
                predicted_triplets_json=predicted_json
            )
        
        # Extract score and feedback from structured output
        try:
            evaluation = judge_result.evaluation
            score = float(evaluation.score)
            feedback = evaluation.feedback
        except Exception as e:
            # Fallback scoring
            import traceback
            print(f"Error extracting judge evaluation: {e}. Traceback: {traceback.format_exc()}")
            score = 0.5
            feedback = f"Error extracting judge evaluation: {e}. Result: {judge_result}"
        
        # If pred_name is None, return just the score
        if pred_name is None:
            return score
        
        # Return score with feedback
        return ScoreWithFeedback(score=score, feedback=feedback)


# --- Optimization ---

def to_percent_int(input: Any) -> int:
    if isinstance(input, float):
        return int(input * 100)
    else:
        raise ValueError(f"Cannot convert {input} to float. Type: {type(input)}")

def optimize_triplet_extractor(
    trainer_lm: dspy.LM,
    judge_lm: dspy.LM,
    auto: Literal["light", "medium", "heavy"],
    limit_trainset: int,
    limit_testset: int,
    randomize_sets: bool,
    reflection_minibatch_size: int
):
    """
    Optimize the triplet extractor using GEPA optimizer with LLM judge.
    """
    
    print("🚀 Starting GEPA optimization with LLM judge...")
    print(f"Trainer LM: {trainer_lm.model}")
    print(f"Judge LM: {judge_lm.model}")
    print(f"Auto: {auto}")
    
    # Prepare data
    trainset = prepare_training_data(limit=limit_trainset, randomize=randomize_sets)
    testset = prepare_test_data(limit=limit_testset, randomize=randomize_sets)
    
    print(f"Training set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")
    
    # Test baseline performance
    print("\n📊 Baseline Performance:")
    metric = TripletExtractionQualityWithFeedbackMetric(judge_lm=judge_lm)
    baseline_score = dspy.Evaluate(
        devset=testset,
        metric=metric,
        num_threads=4,
        display_progress=True
    )(TripletExtractor())
    print(f"Baseline quality score: {baseline_score.score:.2f}")
    
    # Initialize GEPA optimizer
    print("\n🔧 Running GEPA optimization...")
    print("This may take a few minutes...")
    
    optimizer = dspy.GEPA(
        metric=TripletExtractionQualityWithFeedbackMetric(judge_lm=judge_lm),
        auto=auto,
        num_threads=4,
        track_stats=False,
        skip_perfect_score=True,
        add_format_failure_as_feedback=True,
        reflection_minibatch_size=reflection_minibatch_size,
        reflection_lm=trainer_lm
    )
    
    # Compile/optimize the extractor
    extractor = TripletExtractor()
    optimized_extractor = optimizer.compile(
        extractor,
        trainset=trainset,
        valset=testset
    )
    
    # Test optimized performance
    print("\n🎯 Optimized Performance:")
    optimized_score = dspy.Evaluate(
        devset=testset,
        metric=metric,
        num_threads=4,
        display_progress=True
    )(optimized_extractor)
    
    baseline_score_int = to_percent_int(baseline_score.score)
    optimized_score_int = to_percent_int(optimized_score.score)
    print("\n📈 Results:")
    print(f"Baseline quality score:  {baseline_score_int}%")
    print(f"Optimized quality score: {optimized_score_int}%")
    
    # Save a combined metadata + program JSON
    combined_save_path = f"optimized_triplet_extractor_{int(time.time())}_{baseline_score_int}_to_{optimized_score_int}_combined.json"
    combined_data = {
        "metadata": {
            "trainer_lm_model": trainer_lm.model,
            "judge_lm_model": judge_lm.model,
            "judge_lm_reasoning_effort": "low",
            "optimizer_type": "GEPA",
            "auto": auto,
            "baseline_quality_score": baseline_score_int,
            "optimized_quality_score": optimized_score_int,
            "timestamp": int(time.time()),
            "trainset_size": limit_trainset,
            "testset_size": limit_testset,
        },
        "program": optimized_extractor.dump_state()
    }
    with open(combined_save_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    print(f"💾 Combined format saved to: {combined_save_path}")
    
    return optimized_extractor, combined_save_path, baseline_score_int, optimized_score_int


def test_extractor_examples(extractor, examples_desc="", text_prefix=""):
    """Test the extractor with some example inputs"""
    print(f"\n🧪 Testing {examples_desc}:")
    
    test_inputs = [
        "Linear is a communication tool. Linear provides custom instructions.",
        "The product development cycle flows through discovery, planning, building, and shipping.",
        "AI enhances each step of the product development cycle.",
        "Linear's automatic triaging system finds duplicates and provides context.",
        "Humans maintain ownership while delegating tasks to AI agents."
    ]
    
    results = {}
    for test_input in test_inputs:
        result = extractor(text=test_input)
        triplets = result.result.triplets if hasattr(result, 'result') and hasattr(result.result, 'triplets') else []
        results[test_input] = [{"subject": t.subject, "predicate": t.predicate, "object": t.object} for t in triplets]
        print(f"\nInput: {test_input}")
        print(f"Extracted {len(triplets)} triplets:")
        for i, triplet in enumerate(triplets, 1):
            print(f"  {i}. ({triplet.subject}, {triplet.predicate}, {triplet.object})")
        print("-" * 50)
    return results


def main():
    # Configure DSPy with Ollama for the model being trained
    dspy_configure(get_lm_for_ollama())
    
    # Configuration
    trainer_lm_reasoning_effort = "disable"
    judge_lm_reasoning_effort = "disable"
    auto = "heavy"  # Options: "light", "medium", "heavy"
    limit_trainset = 1000
    limit_testset = 3
    randomize_sets = True
    reflection_minibatch_size = limit_testset  # Used in GEPA - when too low then it can loop on seemingly perfect proposed candidates
    
    # Trainer and judge use Gemini Flash
    trainer_lm = get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, trainer_lm_reasoning_effort)
    judge_lm = get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, judge_lm_reasoning_effort)
    
    _, combined_save_path, baseline_score_int, optimized_score_int = optimize_triplet_extractor(
        trainer_lm, judge_lm, auto, limit_trainset, limit_testset, randomize_sets, reflection_minibatch_size
    )
    
    # Test optimized extractor
    
    print(f"\n🎉 Optimization complete! Optimized model + metadata saved to: {combined_save_path}")
    print(f"Baseline quality score: {baseline_score_int}%")
    print(f"Optimized quality score: {optimized_score_int}%")


if __name__ == "__main__":
    main()

