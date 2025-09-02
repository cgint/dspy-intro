
import mlflow
import dspy
from common.utils import get_lm_for_model_name, dspy_configure
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH
mlflow.set_experiment("simplest_dspy")
mlflow.autolog()
# call 'uv run mlflow server --host 127.0.0.1 --port 8182' and head to http://127.0.0.1:8182

def main():
    dspy_configure(get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, "disable"))
    with mlflow.start_run(run_name="simplest_dspy"):
        joker = dspy.Predict("name -> joke")
        the_joke_prediction = joker(name="John")
        the_joke = the_joke_prediction.joke
        print("\n\n" + the_joke + "\n\n")

if __name__ == "__main__":
    main()
