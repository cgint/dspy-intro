import dspy
from common.utils import get_lm_for_model_name, dspy_configure
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH

def main():
    dspy_configure(get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, "disable"))
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    
    ralph_loop = dspy.Refine(
        dspy.Predict("question -> answer"), threshold=1.0, N=5,
        reward_fn=lambda args, pred: 1.0 if 'e' not in pred.answer.lower() else 0.0
    )
    response = ralph_loop(question="Name a common fruit.")
    
    print("=" * 60)
    print("History of LLM Invocations (showing all attempts)")
    print("=" * 60)
    dspy.inspect_history(n=50)  # Show up to 50 most recent calls
    
    print(f"Final Answer: {response.answer}\n")


if __name__ == "__main__":
    main()
