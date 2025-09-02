
import dspy
from common.utils import get_lm_for_model_name, dspy_configure
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH
import mlflow
mlflow.set_experiment("simplest_dspy")
mlflow.autolog()
# call 'uv run mlflow server --host 127.0.0.1 --port 8182' and head to http://127.0.0.1:8182

def joke_for_john() -> str:
    joker = dspy.Predict("name -> joke")
    the_joke_prediction = joker(name="John")
    the_joke = the_joke_prediction.joke
    return the_joke

def joke_funnyness_factor(joke: str) -> int:
    funnyness_evaluator = dspy.Predict("joke -> funnyness_0_to_10: int")
    funnyness_prediction = funnyness_evaluator(joke=joke)
    return funnyness_prediction.funnyness_0_to_10

def main():
    dspy_configure(get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, "disable"))
    with mlflow.start_run(run_name="simplest_dspy_joke_for_john"):
        the_joke=joke_for_john()
        print(f"\n\n{the_joke}\n\n ->")
    with mlflow.start_run(run_name="simplest_dspy_joke_for_john_funnyness_factor"):
        funnyness: int = joke_funnyness_factor(the_joke)
        print(f" -> How funny is the joke on a scale of 0 to 10? {funnyness}\n\n")

if __name__ == "__main__":
    main()
