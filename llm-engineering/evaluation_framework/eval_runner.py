"""
Evaluation Runner Orchestration Layer

Ties together models, LLM-as-judge simulation, statistics, and metrics 
to run full test suites.
"""

from .models import EvalResult
from .judge import score_with_llm_judge

# Mock/Simulated LLM generators to test evaluation pipelines offline
SIMULATED_MODELS = {
    "gpt-4o": lambda inp: f"Based on the question about {inp.split()[0:3]}, the answer involves careful analysis of the key factors. The primary consideration is relevance to the topic at hand, with supporting evidence from established sources.",
    "baseline-v1": lambda inp: f"The answer to your question about {' '.join(inp.split()[0:5])} is as follows: this topic requires understanding of multiple interconnected concepts.",
    "baseline-v2": lambda inp: f"Regarding {' '.join(inp.split()[0:4])}: the short answer is that it depends on context, but here are the key points you should consider for a complete understanding.",
}


def run_model(model_name, input_text):
    """
    Simulates querying a model.
    
    Args:
        model_name (str): Name of the model (must be in SIMULATED_MODELS).
        input_text (str): Prompt input.
        
    Returns:
        str: The simulated output response.
    """
    generator = SIMULATED_MODELS.get(model_name)
    if not generator:
        return f"[ERROR] Unknown model: {model_name}"
    return generator(input_text)


def run_eval_suite(test_suite, model_name, prompt_version, criteria=None):
    """
    Executes a test suite against a model and scores outputs with the LLM-as-judge.
    
    Args:
        test_suite (list[TestCase]): Test cases to evaluate.
        model_name (str): Model name to evaluate.
        prompt_version (str): Metadata indicating prompt version.
        criteria (list[str], optional): Evaluation criteria. Defaults to None.
        
    Returns:
        list[EvalResult]: Comprehensive results of the evaluation run.
    """
    results = []
    for tc in test_suite:
        output = run_model(model_name, tc.input_text)
        scores = score_with_llm_judge(tc.input_text, output, tc.reference_output, criteria)
        result = EvalResult(
            test_case_id=tc.id,
            model_output=output,
            scores=scores,
            model=model_name,
            prompt_version=prompt_version,
        )
        results.append(result)
    return results
