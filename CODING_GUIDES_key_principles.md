# Key Principles & Coding Guidelines for DSPy Projects

You are an expert in well-crafted, maintainable, and typed Python software incorporating DSPy and LLM components.

For the higher-level engineering stance and architectural rationale, see [PHILOSOPHY.md](PHILOSOPHY.md).

---

## 1. Core Engineering Discipline

* **Deterministic by Default:** Keep flow control, business logic, validation, and storage deterministic in Python. Call LLMs only for fuzzy classification, extraction, or semantic tasks.
* **Separation of Concerns:** Keep DSPy signatures, modules, evaluation metrics, and runtime orchestration decoupled.
* **Type Safety First:** Use modern Python type hints (Python 3.12+ / 3.13) and Pydantic models for all signature inputs and outputs.
* **No Unbounded Agents for Narrow Tasks:** Prefer single-purpose DSPy predictors (`Predict`, `ChainOfThought`, `Refine`) over heavy open-ended agentic loops.
* **Test & Metric-Driven:** Every DSPy module should have an associated dataset/examples and an evaluation metric function (e.g. 0/1 accuracy, precision/recall, exact match).

---

## 2. DSPy Design Best Practices

* **Clean Signatures:** Define signatures using explicit `dspy.InputField` and `dspy.OutputField` or Pydantic models. Avoid cramming behavioral begging into signature descriptions.
* **Decouple Interfaces from Optimization:** Let signatures describe *what* is needed. Use optimizers (MIPROv2, BootstrapFewShot, etc.) to learn the *how*.
* **Validation & Retry Seams:** Leverage Pydantic and DSPy assertion / reward loops (`dspy.Suggest`, `dspy.Assert`, or reward refinement) to handle ill-formatted outputs deterministically.

---

## 3. Environment & Tooling Guidelines

* **Fast Dependency Management:** Standardize on `uv` for package and environment management.
* **Self-Contained CLI Scripts:** For standalone tools and visual/evaluation helpers, use single-file scripts with PEP 723 inline metadata executable via `uv run script.py`.
* **Linting & Formatting:** Adhere to `ruff` rules and strict type checking via `mypy`.
