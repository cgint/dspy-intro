
from functai import ai, _ai, configure
from common.utils import get_model_access_prefix_or_fail
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH

@ai
def joke_for_john() -> str:
    return _ai  # type: ignore[return-value]

@ai
def joke_funnyness_factor_0_to_10(joke: str) -> int:
    return _ai  # type: ignore[return-value]

def main():
    model_access_prefix = get_model_access_prefix_or_fail()
    model_name = f"{model_access_prefix}{MODEL_NAME_GEMINI_2_5_FLASH}"
    configure(lm=model_name, temperature=0.2, adapter="json")

    the_joke=joke_for_john()
    print(f"\n\n{the_joke}")
    
    funnyness: int = joke_funnyness_factor_0_to_10(the_joke)
    print(f" -> How funny is the joke on a scale of 0 to 10? {funnyness}\n")

if __name__ == "__main__":
    main()
