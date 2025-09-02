import dspy
from attachments.dspy import Attachments
from pydantic import BaseModel, Field
from common.utils import get_lm_for_model_name, dspy_configure
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH

def context_question_answer(ctx: Attachments, question: str) -> str:
    qa = dspy.Predict("context: Attachments, question: str -> answer: str")
    qa_response = qa(context=ctx, question=question)
    return qa_response.answer

def context_summarizer(ctx: Attachments) -> str:
    summarizer = dspy.Predict("context: Attachments -> summary: str")
    summary_response = summarizer(context=ctx)
    return summary_response.summary


class CategorizerCategory(BaseModel):
    topic_name: str = Field(description="The name of the topic in short headline format")
    topic_importance: int = Field(description="The importance of the topic within the context. Use a scale of 0 (low) to 10 (high)")

class CategorizerResultList(BaseModel):
    covered_topics: list[CategorizerCategory]

# A signature object can replace the string
class ContextCategorizerSignature(dspy.Signature):
    context: Attachments = dspy.InputField()
    covered_topics: CategorizerResultList = dspy.OutputField()
    
def context_categorizer(ctx: Attachments) -> CategorizerResultList:
    categorizer = dspy.Predict(ContextCategorizerSignature)
    category_response = categorizer(context=ctx)
    return category_response.covered_topics

def print_headline_and_answer(headline: str, answer: str):
    print(f"\n{headline}")
    print("=" * len(headline))
    print(f"\n{answer}\n")
    
def main():
    dspy_configure(get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, "disable"))

    pdf_url = "src/simplest/docs/simplest_dspy_with_attachments_2507.11299.pdf"
    ctx = Attachments(pdf_url)
    print(f"\n\nContext: {pdf_url}\n -> Processing ...")

    questions = [
        "What is the main idea of the paper?",
        "What are the key takeaways of the paper?"
    ]
    for question in questions:
        qa_answer: str = context_question_answer(ctx, question)
        headline = f"Answer to the question '{question}':"
        print_headline_and_answer(headline, qa_answer)

    summary: str = context_summarizer(ctx)
    print_headline_and_answer("Summary of the pdf:", summary)

    covered_topics: CategorizerResultList = context_categorizer(ctx)
    covered_topics.covered_topics.sort(key=lambda topic: topic.topic_importance, reverse=True)
    covered_topics_str = " - " + "\n - ".join([f"(Importance: {topic.topic_importance}) {topic.topic_name}" for topic in covered_topics.covered_topics])    
    print_headline_and_answer("Covered topics and their importance (from 0 low to 10 high):", covered_topics_str)

if __name__ == "__main__":
    main()
