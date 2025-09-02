
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
    joker = dspy.Predict("joke -> funnyness_0_to_10: int")
    the_joke_prediction = joker(joke=joke)
    return the_joke_prediction.funnyness_0_to_10

def main():
    dspy_configure(get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, "disable"))
    with mlflow.start_run(run_name="simplest_dspy_joke_for_john"):
        the_joke=joke_for_john()
        print("\n\n" + the_joke + "\n")
    with mlflow.start_run(run_name="simplest_dspy_joke_for_mary_with_funnyness_factor"):
        print(" -> How funny is the joke on a scale of 0 to 10? " + str(joke_funnyness_factor(the_joke)) + "\n\n")

if __name__ == "__main__":
    main()
