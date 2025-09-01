# DSPy MIPROv2 Optimization Guide

## Overview

This guide explains how to optimize your credentials/passwords classifier using DSPy's **MIPROv2** (Multi-agent Interactive Prompt Optimization Version 2), which is the state-of-the-art optimizer from Stanford's DSP research.

## What is MIPROv2?

MIPROv2 is a sophisticated prompt optimization algorithm that:

1. **Automatically generates demonstrations** - Creates few-shot examples from your data
2. **Optimizes instructions** - Proposes better natural-language instructions for prompts  
3. **Uses Bayesian Optimization** - Systematically searches through the space of prompts
4. **Works with complex programs** - Can optimize multi-module DSPy programs

### How MIPROv2 Works

MIPROv2 operates in three stages:

1. **Demonstrate**: Bootstraps high-quality input/output examples from your training data
2. **Search**: Generates various prompt instructions and demonstrations using data analysis
3. **Predict**: Uses Bayesian Optimization to find the best-performing prompt combinations

## Key Configuration Options

### Auto Parameter

The `auto` parameter controls optimization intensity:

- `auto="light"` - Fast optimization, good for prototyping (~$0.50, ~10 minutes)
- `auto="medium"` - Balanced approach, better results (~$2-5, ~20-30 minutes)  
- `auto="heavy"` - Most thorough optimization, best results (~$10-20, ~1-2 hours)

### Important Parameters

```python
optimizer = dspy.MIPROv2(
    metric=classification_accuracy,        # Your evaluation metric
    auto="light",                         # Optimization intensity
    num_threads=4,                        # Parallel processing
)

optimized_classifier = optimizer.compile(
    classifier,                           # Your DSPy program
    trainset=trainset,                    # Training examples
    max_bootstrapped_demos=3,             # Generated demonstrations
    max_labeled_demos=2,                  # Labeled examples to use
)
```

## Implementation Guide

### 1. Prepare Your Data

Convert your examples to DSPy format:

```python
trainset = []
for ex in your_examples:
    trainset.append(dspy.Example(
        classify_input=ex["input"],
        classification=ex["output"]
    ).with_inputs("classify_input"))
```

### 2. Define Your Metric

Create a function that scores predictions:

```python
def classification_accuracy(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    return float(example.classification == pred.classification)
```

### 3. Set Up Your Module

Use `ChainOfThought` instead of basic `Predict` for better reasoning:

```python
class ClassifierCredentialsPasswords(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(ClassifierCredentialsPasswordsSignature)
```

### 4. Optimize

```python
# Initialize optimizer
optimizer = dspy.MIPROv2(
    metric=classification_accuracy,
    auto="light"  # Start with light, can upgrade to medium/heavy
)

# Compile/optimize
optimized_classifier = optimizer.compile(
    classifier,
    trainset=trainset
)
```

## Best Practices

### Data Requirements

- **Minimum**: 10-20 examples for light optimization
- **Recommended**: 50+ examples for medium optimization  
- **Optimal**: 200+ examples for heavy optimization

### Training Data Quality

1. **Diverse examples** - Cover different types of credentials and formats
2. **Balanced classes** - Include unsafe, safe examples
3. **Clear labels** - Ensure consistent labeling criteria
4. **Edge cases** - Include ambiguous or tricky examples

### Optimization Strategy

1. **Start light** - Use `auto="light"` for initial testing
2. **Evaluate carefully** - Test on held-out data
3. **Iterate** - Try different configurations if needed
4. **Scale up** - Move to `auto="medium"` or `auto="heavy"` for production

### Cost Management

- `auto="light"`: ~$0.50, good for development
- `auto="medium"`: ~$2-5, good for production  
- `auto="heavy"`: ~$10-20, for critical applications

## Example Results from Literature

From the research papers and blog posts:

- **AI text detection**: Improved from 76% to 91% accuracy
- **ReAct agents**: Improved from 24% to 51% accuracy  
- **Banking classification**: Improved from 66% to 87% accuracy

## Running the Optimization

### Quick Start

```bash
# Run the optimized classifier
uv run python dspy_agent_classifier_credentials_passwords_optimized.py
```

### Expected Output

```
🚀 Starting MIPROv2 optimization...
Training set size: 45
Test set size: 10

📊 Baseline Performance:
Baseline accuracy: 70.00%

🔧 Running MIPROv2 optimization...
This may take a few minutes...

🎯 Optimized Performance:
Optimized accuracy: 90.00%

📈 Results:
Baseline accuracy:  70.00%
Optimized accuracy: 90.00%
Improvement: 20.00%

💾 Optimized classifier saved to: optimized_credentials_classifier_1641234567.json
```

## Advanced Configuration

### For Complex Tasks

```python
# More thorough optimization
optimizer = dspy.MIPROv2(
    metric=classification_accuracy,
    auto="medium",  # or "heavy"
    num_threads=8,
)

optimized_classifier = optimizer.compile(
    classifier,
    trainset=trainset,
    max_bootstrapped_demos=5,  # More demonstrations
    max_labeled_demos=3,       # More labeled examples
)
```

### For Production Use

```python
# Save and load optimized models
optimized_classifier.save("production_classifier.json")

# Later, load the optimized model
loaded_classifier = ClassifierCredentialsPasswords()
loaded_classifier.load("production_classifier.json")
```

## Troubleshooting

### Common Issues

1. **Low improvement**: Increase training data or try `auto="medium"`
2. **High cost**: Start with `auto="light"` and smaller datasets
3. **Slow optimization**: Reduce `num_threads` or use smaller trainset
4. **Poor generalization**: Add more diverse training examples

### Performance Tips

1. **Use ChainOfThought** instead of basic Predict for reasoning tasks
2. **Add field descriptions** to your Signature for better prompts
3. **Balance your dataset** across all classification categories
4. **Validate on held-out test data** to check generalization

## References

- [DSPy MIPROv2 Documentation](https://dspy.ai/learn/optimization/optimizers/)
- [The power of MIPROv2 - DEV Community](https://dev.to/draismaaaa/the-power-of-miprov2-using-dspy-optimizers-for-your-llm-pipelines-2m47)
- [Optimizing OpenAI's GPT-4o-mini to Detect AI-Generated Text Using DSPy](https://dev.to/b-d055/optimizing-openais-gpt-4o-mini-to-detect-ai-generated-text-using-dspy-2775)
- [DSPy Advanced Tool Use Tutorial](https://dspy.ai/tutorials/tool_use/)

## Next Steps

1. Run the optimization with `auto="light"` 
2. Evaluate results on your test set
3. If needed, scale up to `auto="medium"` for better performance
4. Deploy the optimized classifier in production
5. Monitor performance and re-optimize with new data as needed