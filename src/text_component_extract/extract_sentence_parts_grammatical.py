from typing import List, Literal

import sys
import dspy
import pydantic

from dspy.teleprompt import LabeledFewShot
from common.utils import get_lm_for_model_name
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH


# ANSI color codes for CLI output
class Colors:
    BLUE = '\033[94m'    # Subject
    RED = '\033[91m'     # Verb
    YELLOW = '\033[93m'  # Object
    GREEN = '\033[92m'   # Modifier
    RESET = '\033[0m'    # Reset to default

# --- DSPy Implementation Components ---

# 1. Define the structured output with Pydantic
class GrammaticalComponent(pydantic.BaseModel):
    component_type: Literal["subject", "verb", "object", "modifier"] = pydantic.Field()
    extracted_text: str = pydantic.Field("Text has to be exact substring from the original text.")

class GrammaticalComponentsResult(pydantic.BaseModel):
    components: List[GrammaticalComponent] = pydantic.Field()

# 2. Define the DSPy Signature
class GrammaticalComponentSignature(dspy.Signature):
    """
Analyze the given sentence and extract the grammatical components if present:
- Subject: the doer/agent of the sentence (who/what the sentence is about)
- Verb: the main action/predicate (base form preserved from the sentence)
- Object: the direct object (what receives the action). If there is an indirect object, include it together with the direct object span.
- Modifier: any descriptive words or phrases (adjectives, adverbs, time/place phrases, prepositional phrases, etc.)
"""

    text: str = dspy.InputField(desc="The source sentence to analyze for grammatical components.")
    extracted_components: GrammaticalComponentsResult = dspy.OutputField()

# 3. Create the DSPy Module
class DspyGrammaticalExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(GrammaticalComponentSignature)

    def forward(self, text):
        return self.predictor(text=text)


def run_extraction(input_text: str, compiled_extractor: dspy.Module):
  """Run extraction using the compiled DSPy extractor."""
  result = compiled_extractor(text=input_text)
  return result.extracted_components


def print_colored_results(input_text: str, result: GrammaticalComponentsResult):
  """Print extraction results with colored output, including inline sentence colorization."""
  
  color_map = {
      "subject": Colors.BLUE,
      "verb": Colors.RED,
      "object": Colors.YELLOW,
      "modifier": Colors.GREEN,
  }

  if not result or not result.components:
    print(f"\n📝 Analyzing: {input_text}")
    print("=" * 60)
    print("⚠️  No grammatical components found")
    print()
    return

  # 1. Create spans from components using substring matching
  spans = []
  for component in result.components:
      start_index = input_text.find(component.extracted_text)
      if start_index != -1:
          end_index = start_index + len(component.extracted_text)
          color = color_map.get(component.component_type, Colors.RESET)
          spans.append((start_index, end_index, color, component.component_type.upper()))

  # 2. Sort by length (desc) and then by start index to handle nested/overlapping spans
  spans.sort(key=lambda s: (s[1] - s[0], -s[0]), reverse=True)

  # 3. Filter out overlapping spans
  final_spans = []
  is_colored = [False] * len(input_text)
  for start, end, color, comp_type in spans:
      if not any(is_colored[i] for i in range(start, end)):
          final_spans.append((start, end, color, comp_type))
          for i in range(start, end):
              is_colored[i] = True
  
  # Sort by start index for reconstruction
  final_spans.sort(key=lambda s: s[0])

  # 4. Reconstruct the colored sentence
  colored_sentence = ""
  last_index = 0
  for start, end, color, comp_type in final_spans:
      colored_sentence += input_text[last_index:start]
      colored_sentence += f"{color}{input_text[start:end]}{Colors.RESET}"
      last_index = end
  colored_sentence += input_text[last_index:]

  print(f"\n📝 Analyzing: {colored_sentence}")
  print("=" * 60)
  
  # 5. Print the legend with text
  legend_lines = []
  # Sort components for consistent order: Subject, Verb, Object, Modifier
  component_order = {"subject": 0, "verb": 1, "object": 2, "modifier": 3}
  sorted_components = sorted(result.components, key=lambda c: component_order.get(c.component_type, 99))
  for component in sorted_components:
      color = color_map.get(component.component_type, Colors.RESET)
      line = f"{color}● {component.component_type.upper()}: {component.extracted_text}{Colors.RESET}"
      legend_lines.append(line)

  print("\n".join(legend_lines))
  print()


def get_trainset() -> list[dspy.Example] | None:
  trainset = [
      dspy.Example(
          text="The brilliant scientist quickly discovered a groundbreaking solution.",
          extracted_components=GrammaticalComponentsResult(components=[
              GrammaticalComponent(component_type="subject", extracted_text="The brilliant scientist"),
              GrammaticalComponent(component_type="verb", extracted_text="discovered"),
              GrammaticalComponent(component_type="object", extracted_text="a groundbreaking solution"),
              GrammaticalComponent(component_type="modifier", extracted_text="quickly"),
          ])
      ).with_inputs("text"),
      
      dspy.Example(
          text="My grandmother baked delicious cookies yesterday.",
          extracted_components=GrammaticalComponentsResult(components=[
              GrammaticalComponent(component_type="subject", extracted_text="My grandmother"),
              GrammaticalComponent(component_type="verb", extracted_text="baked"),
              GrammaticalComponent(component_type="object", extracted_text="delicious cookies"),
              GrammaticalComponent(component_type="modifier", extracted_text="yesterday"),
          ])
      ).with_inputs("text"),
      
      dspy.Example(
          text="The children played happily in the park.",
          extracted_components=GrammaticalComponentsResult(components=[
              GrammaticalComponent(component_type="subject", extracted_text="The children"),
              GrammaticalComponent(component_type="verb", extracted_text="played"),
              GrammaticalComponent(component_type="modifier", extracted_text="happily"),
              GrammaticalComponent(component_type="modifier", extracted_text="in the park"),
          ])
      ).with_inputs("text"),
      
      dspy.Example(
          text="A sudden storm interrupted the outdoor concert.",
          extracted_components=GrammaticalComponentsResult(components=[
              GrammaticalComponent(component_type="subject", extracted_text="A sudden storm"),
              GrammaticalComponent(component_type="verb", extracted_text="interrupted"),
              GrammaticalComponent(component_type="object", extracted_text="the outdoor concert"),
          ])
      ).with_inputs("text"),
      
      dspy.Example(
          text="The teacher gave the students clear instructions before the exam.",
          extracted_components=GrammaticalComponentsResult(components=[
              GrammaticalComponent(component_type="subject", extracted_text="The teacher"),
              GrammaticalComponent(component_type="verb", extracted_text="gave"),
              GrammaticalComponent(component_type="object", extracted_text="the students clear instructions"),
              GrammaticalComponent(component_type="modifier", extracted_text="before the exam"),
          ])
      ).with_inputs("text"),
      
      dspy.Example(
          text="Our team will present the final report tomorrow morning.",
          extracted_components=GrammaticalComponentsResult(components=[
              GrammaticalComponent(component_type="subject", extracted_text="Our team"),
              GrammaticalComponent(component_type="verb", extracted_text="will present"),
              GrammaticalComponent(component_type="object", extracted_text="the final report"),
              GrammaticalComponent(component_type="modifier", extracted_text="tomorrow morning"),
          ])
      ).with_inputs("text"),
      
      dspy.Example(
          text="The curious cat quietly watched the birds from the window.",
          extracted_components=GrammaticalComponentsResult(components=[
              GrammaticalComponent(component_type="subject", extracted_text="The curious cat"),
              GrammaticalComponent(component_type="verb", extracted_text="watched"),
              GrammaticalComponent(component_type="object", extracted_text="the birds"),
              GrammaticalComponent(component_type="modifier", extracted_text="quietly"),
              GrammaticalComponent(component_type="modifier", extracted_text="from the window"),
          ])
      ).with_inputs("text")
  ]

  # Validate that all extracted components are contained in the original text
  print("Validating training examples...")
  validation_errors = []
  
  for i, example in enumerate(trainset):
      text = example.text
      components = example.extracted_components.components
      
      for j, component in enumerate(components):
          extracted_text = component.extracted_text
          if extracted_text not in text:
              error_msg = (
                  f"Example {i+1}, Component {j+1}: "
                  f"Extracted text '{extracted_text}' not found in original text: '{text}'"
              )
              validation_errors.append(error_msg)
  
  if validation_errors:
      print("❌ Training data validation failed!")
      print("\nValidation errors found:")
      for error in validation_errors:
          print(f"   {error}")
      print("\n💡 All extracted_text values must be exact substrings of the original text.")
      return None
  
  print("✅ Training data validation passed - all extracted components found in original text.")
  return trainset

def main() -> bool:
  """Main function to run the grammatical component extraction examples."""
  model_id = MODEL_NAME_GEMINI_2_5_FLASH
  
  print(f"🚀 Testing Grammatical Component Extraction with DSPy and {model_id}...")
  print("=" * 60)

  try:
    trainset = get_trainset()
    if trainset is None:
      return False

    # 1. Configure the Language Model for DSPy with Google Gemini
    lm = get_lm_for_model_name(model_name=model_id)
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    dspy.settings.configure(lm=lm)
    print(f"DSPy configured to use {dspy.settings.lm.model}.")


    # 3. Use a Teleprompter to build the few-shot prompt
    print(f"Using {len(trainset)} examples for training of the extractor. Training starts now ...")
    teleprompter = LabeledFewShot(k=len(trainset))
    final_extractor = teleprompter.compile(DspyGrammaticalExtractor(), trainset=trainset)
    # final_extractor.set_lm(lm)
    # final_extractor.save("dspy_grammatical_extractor.json")

    print("Compiled extractor ready.")
    # final_extractor = dspy.Predict(PromptComponentSignature)
    
    # 4. Test with sample inputs (different wordings from training data)
    sample_inputs = [
      "The brilliant scientist quickly discovered a groundbreaking solution.",
      "My grandmother baked delicious cookies yesterday.",
      "The children played happily in the park.",
      "A sudden storm interrupted the outdoor concert.",
      "The teacher gave the students clear instructions before the exam.",
      "Our team will present the final report tomorrow morning.",
      "The curious cat quietly watched the birds from the window.",
    ]

    for input_text in sample_inputs:
      try:
        result = run_extraction(input_text, final_extractor)
        print_colored_results(input_text, result)
      except Exception as e:
        print(f"❌ Error processing: {input_text}")
        print(f"   {type(e).__name__}: {e}\n")

    print("✅ SUCCESS! Grammatical component extraction completed")
    print(f"   Model: {model_id}")
    
    print("   DSPy with Pydantic types: enabled")
    return True

  except Exception as e:
    # Broad exception to catch potential issues with API connection or model availability
    print(f"\n❌ An unexpected error occurred: {type(e).__name__}: {e}")
    print("\n💡 Make sure your Google Cloud credentials are set up correctly.")
    print("   e.g., run 'gcloud auth application-default login'")
    return False


if __name__ == "__main__":
  SUCCESS = main()
  sys.exit(0 if SUCCESS else 1)