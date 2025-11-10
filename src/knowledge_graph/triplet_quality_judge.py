"""DSPy Signature for triplet quality evaluation judge."""

import dspy
import pydantic


class TripletEvaluationResult(pydantic.BaseModel):
    """Structured output for triplet quality evaluation."""
    score: float = pydantic.Field(description="Quality score from 0.0 to 1.0 (where 1.0 is perfect)", ge=0.0, le=1.0)
    feedback: str = pydantic.Field(description="Detailed feedback explaining what should be adapted")


class TripletQualityJudgeSignature(dspy.Signature):
    """
    Evaluate the quality of extracted knowledge graph triplets.
    
    Evaluate the predicted triplets on:
    1. Correctness: Are extracted triplets semantically correct?
    2. Completeness: Did we extract all important triplets from the text?
    3. Quality: Are subjects/objects proper entities (not modifiers/locations)?
    4. Predicate quality: Are predicates concise and meaningful?
    
    Provide detailed feedback explaining what should be adapted, including:
    - Missing triplets
    - Incorrect structure (locations/modifiers as objects)
    - Verbose predicates that should be concise
    - Reversed or incorrect semantic relationships
    - Abstract concepts as subjects
    """
    
    original_text: str = dspy.InputField(desc="The original source text")
    expected_triplets_json: str = dspy.InputField(desc="JSON string of expected triplets")
    predicted_triplets_json: str = dspy.InputField(desc="JSON string of predicted triplets")
    evaluation: TripletEvaluationResult = dspy.OutputField(desc="Evaluation result with score (0.0-1.0) and detailed feedback")

