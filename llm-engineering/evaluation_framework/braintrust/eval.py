# eval.py
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from braintrust import Eval
    from autoevals import LevenshteinScorer
except ImportError:
    print("[ERROR] braintrust or autoevals module not found. Please install it using: pip install braintrust autoevals")
    sys.exit(1)

from evaluation_framework.suite import build_test_suite
from evaluation_framework.judge import score_with_llm_judge, pairwise_compare_with_judge
from evaluation_framework.eval_runner import run_model
from evaluation_framework.metrics import bert_score
from evaluation_framework.stats import wilson_confidence_interval, bootstrap_confidence_interval, fleiss_kappa, krippendorff_alpha
from evaluation_framework.cost_tracker import tracker

def bert_score_scorer(input, output, expected=None, **kwargs):
    if expected is None:
        return {"name": "BERTScore F1", "score": 0.0}
    res = bert_score(expected, output)
    return {
        "name": "BERTScore F1",
        "score": res["f1"],
        "metadata": {
            "precision": res["precision"],
            "recall": res["recall"]
        }
    }

current_model_name = None
model_scores = {
    "baseline-v1": {},
    "baseline-v2": {}
}

def llm_judge_scorer(input, output, expected=None, **kwargs):
    """Replicates LLM-as-judge scoring for relevance, correctness, helpfulness, and safety."""
    judge_scores = score_with_llm_judge(input, output, expected)
    
    metadata = {}
    total_normalized_score = 0.0
    for s in judge_scores:
        # Scale 1-5 to a 0.0-1.0 range
        normalized = (s.score - 1.0) / 4.0
        metadata[s.criterion] = s.score
        metadata[f"{s.criterion}_reasoning"] = s.reasoning
        total_normalized_score += normalized
        
    avg_score = total_normalized_score / len(judge_scores) if judge_scores else 0.0
    
    raw_avg = sum(s.score for s in judge_scores) / len(judge_scores) if judge_scores else 1.0
    global current_model_name
    if current_model_name in model_scores:
        model_scores[current_model_name][input] = raw_avg

    return {
        "name": "LLM Judge Avg",
        "score": avg_score,
        "metadata": metadata
    }

async def main():
    tracker.reset()
    test_cases = build_test_suite()
    data = [
        {
            "input": tc.input_text,
            "expected": tc.reference_output,
            "category": tc.category
        }
        for tc in test_cases
    ]

    for model_name in ["gpt-4o", "baseline-v1", "baseline-v2"]:
        print(f"\n=======================================================")
        print(f" Running Braintrust Evaluation for Model: {model_name}")
        print(f"=======================================================")
        
        global current_model_name
        current_model_name = model_name

        def task(input_text):
            return run_model(model_name, input_text)

        await Eval(
            name=f"evaluation_framework_{model_name}",
            data=data,
            task=task,
            scores=[LevenshteinScorer, llm_judge_scorer, bert_score_scorer],
            no_send_logs=True  # Prevent logging to cloud, run entirely locally
        )

    print(f"\n--- Stratified Category Analysis (baseline-v1 vs baseline-v2) ---")
    categories = sorted(list(set(tc.category for tc in test_cases)))
    for cat in categories:
        cat_v1 = [
            score for input_text, score in model_scores["baseline-v1"].items()
            if any(tc.input_text == input_text and tc.category == cat for tc in test_cases)
        ]
        cat_v2 = [
            score for input_text, score in model_scores["baseline-v2"].items()
            if any(tc.input_text == input_text and tc.category == cat for tc in test_cases)
        ]
        
        v1_mean = sum(cat_v1) / len(cat_v1) if cat_v1 else 0.0
        v2_mean = sum(cat_v2) / len(cat_v2) if cat_v2 else 0.0
        delta = v2_mean - v1_mean
        
        v1_ci = bootstrap_confidence_interval(cat_v1)
        v2_ci = bootstrap_confidence_interval(cat_v2)
        
        if delta > 0.0:
            status = "IMPROVED"
        elif delta < 0.0:
            status = "REGRESSED"
        else:
            status = "STABLE"
            
        print(f"  {cat:<15} v1: {v1_mean:.3f} [{v1_ci[0]:.2f}, {v1_ci[2]:.2f}] | v2: {v2_mean:.3f} [{v2_ci[0]:.2f}, {v2_ci[2]:.2f}] | delta: {delta:+.3f} | {status}")

    print(f"\n--- Inter-Rater Reliability Verification (3 runs) ---")
    reliability_data = []
    criteria = ["relevance", "correctness", "helpfulness", "safety"]
    for tc in test_cases:
        out_v2 = run_model("baseline-v2", tc.input_text)
        for criterion in criteria:
            ratings = []
            for rater in [0, 1, 2]:
                scores_rater = score_with_llm_judge(tc.input_text, out_v2, tc.reference_output, criteria=[criterion], rater_idx=rater)
                ratings.append(scores_rater[0].score)
            reliability_data.append(ratings)
            
    alpha = krippendorff_alpha(reliability_data)
    kappa = fleiss_kappa(reliability_data)
    
    print(f"  Fleiss' Kappa: {kappa:.4f}")
    print(f"  Krippendorff's Alpha: {alpha:.4f}")
    
    if alpha >= 0.7:
        print(f"  Reliability: PASS (alpha={alpha:.4f} >= 0.7)")
    else:
        print(f"  [WARNING] Reliability: FAIL (alpha={alpha:.4f} < 0.7). Rubric might be too ambiguous!")

    print(f"\n=======================================================")
    print(f" Running Braintrust Pairwise Evaluation: baseline-v1 vs baseline-v2")
    print(f"=======================================================")

    pairwise_results = []
    id_map = {tc.input_text: tc.id for tc in test_cases}

    def pairwise_task(input_text):
        out_v1 = run_model("baseline-v1", input_text)
        out_v2 = run_model("baseline-v2", input_text)
        return {
            "output_v1": out_v1,
            "output_v2": out_v2
        }

    def pairwise_compare_scorer(input, output, expected=None, **kwargs):
        res = pairwise_compare_with_judge(input, output["output_v1"], output["output_v2"], expected)
        score = 1.0 if res["winner"] == "B" else 0.0
        pairwise_results.append({
            "input": input,
            "winner": res["winner"],
            "reason": res["reason"],
            "score": score
        })
        return {
            "name": "Pairwise Win (baseline-v2 over baseline-v1)",
            "score": score,
            "metadata": {
                "winner": res["winner"],
                "reason": res["reason"]
            }
        }

    await Eval(
        name="evaluation_framework_pairwise",
        data=data,
        task=pairwise_task,
        scores=[pairwise_compare_scorer],
        no_send_logs=True
    )

    print(f"\n--- Pairwise Comparison: baseline-v1 vs baseline-v2 ---")
    for res in pairwise_results:
        tc_id = id_map.get(res["input"], "unknown")
        print(f"  [{tc_id}] Winner: {res['winner']} | {res['reason']}")

    scores = [res["score"] for res in pairwise_results]
    wins = sum(1 for s in scores if s == 1.0)
    total = len(scores)
    win_rate = wins / total if total > 0 else 0.0

    wilson_ci = wilson_confidence_interval(wins, total)
    bootstrap_ci = bootstrap_confidence_interval(scores)

    print(f"\nPairwise Comparison Summary:")
    print(f"  baseline-v2 Win Rate: {wins}/{total} = {win_rate:.1%}")
    print(f"  Wilson Confidence Interval (95%): [{wilson_ci[0]:.4f}, {wilson_ci[1]:.4f}]")
    print(f"  Bootstrap Confidence Interval (95%): [{bootstrap_ci[0]:.4f}, {bootstrap_ci[1]:.4f}, {bootstrap_ci[2]:.4f}]")

    tracker.print_summary()

if __name__ == "__main__":
    asyncio.run(main())
