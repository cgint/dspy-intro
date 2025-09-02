
from functai import ai, _ai, configure
from common.utils import get_model_access_prefix_or_fail
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH
import mlflow
mlflow.set_experiment("simplest_dspy")
mlflow.autolog()
# call 'uv run mlflow server --host 127.0.0.1 --port 8182' and head to http://127.0.0.1:8182

@ai
def joke_for_john() -> str:
    # could add more context through """Tell a short, family-friendly joke specifically for John."""
    return _ai  # type: ignore[return-value]

@ai
def joke_funnyness_factor(joke: str) -> int:
    # could add more context through """Rate how funny the joke is on a 0-10 integer scale. Return only the integer."""
    return _ai  # type: ignore[return-value]

def main():
    model_access_prefix = get_model_access_prefix_or_fail()
    model_name = f"{model_access_prefix}{MODEL_NAME_GEMINI_2_5_FLASH}"
    configure(lm=model_name, temperature=0.2, adapter="json")

    with mlflow.start_run(run_name="simplest_dspy_joke_for_john"):
        the_joke=joke_for_john()
        print(f"\n\n{the_joke}")
    
    with mlflow.start_run(run_name="simplest_dspy_joke_for_john_funnyness_factor"):
        funnyness: int = joke_funnyness_factor(the_joke)
        print(f" -> How funny is the joke on a scale of 0 to 10? {funnyness}\n")

if __name__ == "__main__":
    main()
