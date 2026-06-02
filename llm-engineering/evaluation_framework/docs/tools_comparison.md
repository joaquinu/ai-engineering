# LLM Evaluation Tools Comparison

This document provides a comparative analysis of the five evaluation frameworks implemented in this codebase: **Promptfoo**, **Braintrust**, **LangSmith**, **DeepEval**, and **Arize Phoenix**.

---

## High-Level Comparison Matrix

| Feature | Promptfoo | Braintrust | LangSmith | DeepEval | Arize Phoenix |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Interface** | CLI & YAML | Python SDK / Web UI | Python SDK / Web UI | Python SDK / Pytest | Python SDK / pandas |
| **Ecosystem** | Independent (OSS) | Independent (SaaS/OSS) | LangChain ecosystem | Independent (OSS) | Arize observability |
| **Configuration** | Declarative YAML | Programmatic Python | Programmatic Python | Programmatic Python | Programmatic Python |
| **Tracing Support** | Minimal | High | Deeply integrated | Moderate | Comprehensive |
| **Offline Execution** | Native & Easy | Supported (`no_send_logs=True`) | Supported (`upload_results=False`) | Native | Native |
| **Ideal For** | Prompt prototyping, CLI-based testing | Production logging, enterprise analytics | LangChain apps, trace inspection | CI/CD regression gates, Pytest suites | DataFrame-driven pipelines, observability |

---

## 1. Promptfoo
* **Overview**: A CLI-focused, open-source tool that allows you to define assertions and test suites in simple YAML configuration files.
* **Key Strengths**:
  * Decouples evaluation logic from application code.
  * Easy to run via standard terminal command (`npx promptfoo eval`).
  * Built-in assertion types (e.g., semantic similarity, rouge, model responses).
* **Code Pattern**:
  We configured it using `promptfooconfig.yaml` and a Python script wrapper `provider.py` which interfaces with our models as a custom CLI command.

---

## 2. Braintrust
* **Overview**: A robust SDK-driven evaluation platform focused on tracking experiments, datasets, and logs over time with cloud visualizations.
* **Key Strengths**:
  * Powerful dashboard for tracking historical runs and regressions.
  * Flexible custom scorers (simple callable Python functions).
  * Highly asynchronous execution.
* **Code Pattern**:
  Evaluations are run using the `Eval` method. We run it locally/offline using `no_send_logs=True`:
  ```python
  from braintrust import Eval
  from autoevals import LevenshteinScorer

  await Eval(
      name="eval-run",
      data=dataset,
      task=lambda input: run_model(model, input),
      scores=[LevenshteinScorer, my_custom_llm_judge],
      no_send_logs=True
  )
  ```

---

## 3. LangSmith
* **Overview**: LangChain's native SaaS platform designed for tracing, debugging, and evaluating LLM pipelines.
* **Key Strengths**:
  * Industry-standard tracing interface showing full span-level call stacks.
  * Clean integration with LangChain chains, agents, and custom Runnables.
  * Direct support for standard LangChain string/distance evaluators.
* **Code Pattern**:
  Evaluations are executed using the `evaluate()` function, run offline via environment parameters:
  ```python
  import os
  os.environ["LANGSMITH_TRACING"] = "false"
  from langsmith import evaluate

  evaluate(
      target_function,
      data=examples,
      evaluators=[langchain_string_distance_evaluator],
      upload_results=False
  )
  ```

---

## 4. DeepEval
* **Overview**: A unit-testing-oriented Python library that integrates directly with standard testing tools like `pytest`.
* **Key Strengths**:
  * Unit-test friendly (you can write `assert` statements directly on metrics).
  * Native metric classes that can be subclassed to run locally and offline.
  * Pre-built metric types for RAG pipelines (faithfulness, hallucination, etc.).
* **Code Pattern**:
  Requires wrapping metrics in subclasses of `BaseMetric` implementing `measure()` and `a_measure()` methods:
  ```python
  from deepeval import evaluate
  from deepeval.metrics import BaseMetric
  from deepeval.test_case import LLMTestCase

  class CustomMetric(BaseMetric):
      def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
          self.score = compute_score(test_case.actual_output)
          self.success = self.score >= 0.5
          return self.score
  ```

---

## 5. Arize Phoenix
* **Overview**: An open-source observability and tracing backend that provides simple, dataframe-native evaluation helpers.
* **Key Strengths**:
  * Dataframe-centric workflow; processes standard pandas dataframes natively.
  * Perfect if the evaluation targets are already logged in databases or dataframes.
  * Simple `@create_evaluator` decorator converts any standard python function into a benchmark scorer.
* **Code Pattern**:
  Using decorators to map function signatures directly to matching dataframe columns:
  ```python
  import pandas as pd
  from phoenix.evals import evaluate_dataframe, create_evaluator

  @create_evaluator(name="My Metric")
  def custom_evaluator(output: str, expected: str) -> float:
      return compute_score(output, expected)

  res_df = evaluate_dataframe(
      dataframe=df,  # must contain 'output' and 'expected' columns
      evaluators=[custom_evaluator]
  )
  ```

---

## Summary Summary Recommendation

* **Use Promptfoo** if you want to quickly test prompt permutations or mock configs without writing any Python wrapper code.
* **Use Braintrust or LangSmith** if you are deploying to production and want an elegant, cloud-hosted dashboard to debug traces, version datasets, and analyze long-term quality trends.
* **Use DeepEval** if you want your evaluations to run inside standard PR regression checks (e.g., via Github Actions and Pytest).
* **Use Arize Phoenix** if you have a structured dataframe pipeline or need a self-hosted tracing system that logs evaluations as OTel metadata.
