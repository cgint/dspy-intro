from typing import List, Literal

import sys
import dspy
import pydantic

from dspy.teleprompt import LabeledFewShot
from common.utils import get_lm_for_model_name
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH


# ANSI color codes for CLI output
class Colors:
    BLUE = '\033[94m'    # Persona
    RED = '\033[91m'     # Task
    YELLOW = '\033[93m'  # Context
    GREEN = '\033[92m'   # Format
    RESET = '\033[0m'    # Reset to default

# --- DSPy Implementation Components ---

# 1. Define the structured output with Pydantic
class PromptComponent(pydantic.BaseModel):
    component_type: Literal["persona", "task", "context", "format"] = pydantic.Field()
    extracted_text: str = pydantic.Field("Text has to be exact substring from the original text.")

class PromptComponentsResult(pydantic.BaseModel):
    components: List[PromptComponent] = pydantic.Field()

# 2. Define the DSPy Signature
class PromptComponentSignature(dspy.Signature):
    """
Analyze the given text and extract the four main prompt engineering components if present:
- Persona: Who the AI should act as or what role it should take
- Task: What specific action or task needs to be performed
- Context: Background information, setting, or additional details
- Format: How the output should be structured or presented
"""

    text: str = dspy.InputField(desc="The source text to split into components.")
    extracted_components: PromptComponentsResult = dspy.OutputField()

# 3. Create the DSPy Module
class DspyComponentExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(PromptComponentSignature)

    def forward(self, text):
        return self.predictor(text=text)


def run_extraction(input_text: str, compiled_extractor: dspy.Module):
  """Run extraction using the compiled DSPy extractor."""
  result = compiled_extractor(text=input_text)
  return result.extracted_components


def print_colored_results(input_text: str, result: PromptComponentsResult):
  """Print extraction results with colored output, including inline sentence colorization."""
  
  color_map = {
      "persona": Colors.BLUE,
      "task": Colors.RED,
      "context": Colors.YELLOW,
      "format": Colors.GREEN,
  }

  if not result or not result.components:
    print(f"\n📝 Analyzing: {input_text}")
    print("=" * 60)
    print("⚠️  No prompt components found")
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
  # Sort components for consistent order: Persona, Task, Context, Format
  component_order = {"persona": 0, "task": 1, "context": 2, "format": 3}
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
          text="You are a program manager in tech industry. Draft an executive summary email to stakeholders based on quarterly results. Limit to bullet points.",
          extracted_components=PromptComponentsResult(components=[
              PromptComponent(component_type="persona", extracted_text="You are a program manager in tech industry"),
              PromptComponent(component_type="task", extracted_text="Draft an executive summary email to stakeholders"),
              PromptComponent(component_type="context", extracted_text="based on quarterly results"),
              PromptComponent(component_type="format", extracted_text="Limit to bullet points"),
          ])
      ).with_inputs("text"),
      
      dspy.Example(
          text="Act as a professional chef and create a recipe for chocolate cake using only vegan ingredients. Present as step-by-step instructions.",
          extracted_components=PromptComponentsResult(components=[
              PromptComponent(component_type="persona", extracted_text="Act as a professional chef"),
              PromptComponent(component_type="task", extracted_text="create a recipe for chocolate cake"),
              PromptComponent(component_type="context", extracted_text="using only vegan ingredients"),
              PromptComponent(component_type="format", extracted_text="Present as step-by-step instructions"),
          ])
      ).with_inputs("text"),
      
      dspy.Example(
          text="You are a data scientist. Analyze the customer churn data from last quarter and provide insights in a PowerPoint presentation format.",
          extracted_components=PromptComponentsResult(components=[
              PromptComponent(component_type="persona", extracted_text="You are a data scientist"),
              PromptComponent(component_type="task", extracted_text="Analyze the customer churn data"),
              PromptComponent(component_type="context", extracted_text="from last quarter"),
              PromptComponent(component_type="format", extracted_text="in a PowerPoint presentation format"),
          ])
      ).with_inputs("text"),
      
      dspy.Example(
          text="Write a blog post about sustainable living for millennials. Keep it under 500 words.",
          extracted_components=PromptComponentsResult(components=[
              PromptComponent(component_type="task", extracted_text="Write a blog post about sustainable living"),
              PromptComponent(component_type="context", extracted_text="for millennials"),
              PromptComponent(component_type="format", extracted_text="Keep it under 500 words"),
          ])
      ).with_inputs("text"),
      
      dspy.Example(
          text="As a financial advisor, explain cryptocurrency investment risks to a first-time investor. Use simple language and examples.",
          extracted_components=PromptComponentsResult(components=[
              PromptComponent(component_type="persona", extracted_text="As a financial advisor"),
              PromptComponent(component_type="task", extracted_text="explain cryptocurrency investment risks"),
              PromptComponent(component_type="context", extracted_text="to a first-time investor"),
              PromptComponent(component_type="format", extracted_text="Use simple language and examples"),
          ])
      ).with_inputs("text"),
      
      dspy.Example(
          text="Create a training manual for new employees about our company culture and values. Format as a PDF guide.",
          extracted_components=PromptComponentsResult(components=[
              PromptComponent(component_type="task", extracted_text="Create a training manual for new employees"),
              PromptComponent(component_type="context", extracted_text="about our company culture and values"),
              PromptComponent(component_type="format", extracted_text="Format as a PDF guide"),
          ])
      ).with_inputs("text"),
      
      dspy.Example(
          text="You are a marketing specialist. Develop a social media strategy for our new product launch targeting Gen Z customers.",
          extracted_components=PromptComponentsResult(components=[
              PromptComponent(component_type="persona", extracted_text="You are a marketing specialist"),
              PromptComponent(component_type="task", extracted_text="Develop a social media strategy"),
              PromptComponent(component_type="context", extracted_text="for our new product launch targeting Gen Z customers"),
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
  """Main function to run the prompt component extraction examples."""
  model_id = MODEL_NAME_GEMINI_2_5_FLASH
  
  print(f"🚀 Testing Prompt Component Extraction with DSPy and {model_id}...")
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
    final_extractor = teleprompter.compile(DspyComponentExtractor(), trainset=trainset)
    # final_extractor.set_lm(lm)
    # final_extractor.save("dspy_component_extractor.json")

    print("Compiled extractor ready.")
    # final_extractor = DspyComponentExtractor() # to showcase example without FewShot
    
    # 4. Test with sample inputs (different wordings from training data)
    sample_inputs = [
      "As a tech program director, prepare a stakeholder briefing memo summarizing monthly performance metrics. Structure it as key bullet points.",
      "Function as an expert culinary specialist and develop a dessert preparation guide for vanilla ice cream with exclusively plant-based components. Organize into sequential preparation steps.",
      "In your capacity as a data analyst, examine user retention patterns from the previous three months and deliver findings as a slide deck presentation.",
      "Compose an online article discussing eco-friendly lifestyle choices for young adults. Ensure the piece stays within 600 words.",
      "Serving as an investment consultant, describe the potential downsides of investing in digital currencies to novice market participants. Explain using straightforward terms and practical scenarios.",
      "Design an orientation handbook for recent hires covering our organizational principles and workplace standards. Present it as a digital document.",
      "Working as a brand strategist, formulate a digital marketing plan for introducing our latest innovation aimed at young adult consumers."
    ]

    for input_text in sample_inputs:
      try:
        result = run_extraction(input_text, final_extractor)
        print_colored_results(input_text, result)
      except Exception as e:
        print(f"❌ Error processing: {input_text}")
        print(f"   {type(e).__name__}: {e}\n")

    print("✅ SUCCESS! Prompt component extraction completed")
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