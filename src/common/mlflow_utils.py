from typing import Literal, Dict
import mlflow

def log_as_table(results: Dict[str, str], optimization_type: Literal["baseline", "optimized"]) -> None:
    """
    Log test results as a structured table to MLflow.
    
    Args:
        results: Dictionary mapping test inputs to classification results
        optimization_type: Type of model ("baseline" or "optimized")
    """
    # Parse the results dictionary to extract questions and answers
    table_data = {
        "test_number": [],
        "question": [],
        "answer": [],
        "optimization_type": []
    }
    
    for i, (question_with_prefix, answer) in enumerate(results.items(), 1):
        # Remove the model type prefix from the question
        if question_with_prefix.startswith(f"{optimization_type}_"):
            question = question_with_prefix[len(f"{optimization_type}_"):]
        else:
            question = question_with_prefix
            
        table_data["test_number"].append(i)
        table_data["question"].append(question)
        table_data["answer"].append(answer)
        table_data["optimization_type"].append(optimization_type)
    
    # Log the table to MLflow
    artifact_file = f"{optimization_type}_test_results.json"
    mlflow.log_table(data=table_data, artifact_file=artifact_file)
    print(f"✅ {optimization_type.title()} test results logged to MLflow as table: {artifact_file}")
