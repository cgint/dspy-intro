
import dspy
from common.utils import get_lm_for_model_name, dspy_configure
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH

def joke_for_john() -> str:
    joker = dspy.Predict("name -> joke")
    the_joke_prediction = joker(name="John")
    return the_joke_prediction.joke

def joke_funnyness_factor_0_to_10(joke: str) -> int:
    funnyness_evaluator = dspy.Predict("joke -> funnyness_0_to_10: int")
    funnyness_prediction = funnyness_evaluator(joke=joke)
    return funnyness_prediction.funnyness_0_to_10

def main():
    dspy_configure(get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, "disable"))

    the_joke: str = joke_for_john()
    print(f"\n\n{the_joke}")
    
    funnyness: int = joke_funnyness_factor_0_to_10(the_joke)
    print(f" -> How funny is the joke on a scale of 0 to 10? {funnyness}\n")

if __name__ == "__main__":
    main()
