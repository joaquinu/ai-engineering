# assertion.py
import sys
import os

# Import the actual scoring modules from the sibling evaluation_framework package
# We go up three levels to the root of evaluation_framework's parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation_framework.metrics import rouge_l_score
from evaluation_framework.judge import score_with_llm_judge

def get_assert(output, context):
    """
    Promptfoo custom Python assertion.
    Computes both ROUGE-L and simulated LLM-as-judge scores.
    """
    # Extract vars passed from the test case
    prompt = context.get('prompt')
    vars_dict = context.get('vars', {})
    reference = vars_dict.get('reference_output', '')

    # 1. Compute ROUGE-L
    r_score = rouge_l_score(reference, output)

    # 2. Compute LLM Judge Scores
    judge_scores = score_with_llm_judge(prompt, output, reference)
    
    # Calculate average score (out of 5)
    total_score = sum(s.score for s in judge_scores)
    avg_score = total_score / len(judge_scores) if judge_scores else 0.0

    # Format detail reasoning output
    reasoning_parts = []
    for s in judge_scores:
        reasoning_parts.append(f"{s.criterion}: {s.score}/5 ({s.reasoning})")
    
    reason = (
        f"ROUGE-L: {r_score:.4f}\n"
        f"Judge Avg: {avg_score:.2f}/5\n"
        f"Details:\n" + "\n".join(reasoning_parts)
    )

    # Define passing condition (e.g. average score >= 3.5)
    # Scaled score out of 1.0 for promptfoo grading
    norm_score = avg_score / 5.0
    is_pass = avg_score >= 3.5

    return {
        'pass': is_pass,
        'score': norm_score,
        'reason': reason
    }
