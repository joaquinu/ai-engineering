# eval.py
import sys
import os
import pandas as pd

# Ensure package imports work for evaluation_framework
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from phoenix.evals import evaluate_dataframe, create_evaluator
from evaluation_framework.suite import build_test_suite
from evaluation_framework.metrics import rouge_l_score, bert_score
from evaluation_framework.judge import score_with_llm_judge, pairwise_compare_with_judge
from evaluation_framework.eval_runner import run_model
from evaluation_framework.stats import wilson_confidence_interval, bootstrap_confidence_interval, fleiss_kappa, krippendorff_alpha
from evaluation_framework.cost_tracker import tracker

@create_evaluator(name="ROUGE-L")
def rouge_evaluator(expected: str, output: str) -> float:
    return rouge_l_score(expected, output)

current_model_name = None
model_scores = {
    "baseline-v1": {},
    "baseline-v2": {}
}

@create_evaluator(name="LLM Judge Avg")
def llm_judge_evaluator(input: str, output: str, expected: str) -> float:
    judge_scores = score_with_llm_judge(input, output, expected)
    total_normalized_score = 0.0
    for s in judge_scores:
        # Scale 1-5 to a 0.0-1.0 range
        normalized = (s.score - 1.0) / 4.0
        total_normalized_score += normalized
        
    raw_avg = sum(s.score for s in judge_scores) / len(judge_scores) if judge_scores else 1.0
    global current_model_name
    if current_model_name in model_scores:
        model_scores[current_model_name][input] = raw_avg
        
    return total_normalized_score / len(judge_scores) if judge_scores else 0.0

@create_evaluator(name="BERTScore")
def bert_score_evaluator(expected: str, output: str) -> float:
    res = bert_score(expected, output)
    return res["f1"]

pairwise_results = []

@create_evaluator(name="Pairwise Win")
def pairwise_evaluator(input: str, output_v1: str, output_v2: str, expected: str) -> float:
    res = pairwise_compare_with_judge(input, output_v1, output_v2, expected)
    score = 1.0 if res["winner"] == "B" else 0.0
    pairwise_results.append({
        "input": input,
        "winner": res["winner"],
        "reason": res["reason"],
        "score": score
    })
    return score

def main():
    tracker.reset()
    test_cases = build_test_suite()

    for model_name in ["gpt-4o", "baseline-v1", "baseline-v2"]:
        print(f"\n=======================================================")
        print(f" Running Arize Phoenix Evaluation for Model: {model_name}")
        print(f"=======================================================")

        global current_model_name
        current_model_name = model_name

        # Run the model to collect outputs and construct a DataFrame
        data = []
        for tc in test_cases:
            output = run_model(model_name, tc.input_text)
            data.append({
                "input": tc.input_text,
                "query": tc.input_text,
                "output": output,
                "expected": tc.reference_output
            })
        df = pd.DataFrame(data)

        # Run evaluation using local evaluate_dataframe
        res_df = evaluate_dataframe(
            dataframe=df,
            evaluators=[rouge_evaluator, llm_judge_evaluator, bert_score_evaluator]
        )

        # Print summary scores
        # The output columns contain Score objects represented as dictionaries
        avg_rouge = res_df["ROUGE-L_score"].apply(lambda x: x.get("score") if isinstance(x, dict) else x).mean()
        avg_judge = res_df["LLM Judge Avg_score"].apply(lambda x: x.get("score") if isinstance(x, dict) else x).mean()
        avg_bert = res_df["BERTScore_score"].apply(lambda x: x.get("score") if isinstance(x, dict) else x).mean()

        print(f"\n=========================SUMMARY FOR {model_name}=========================")
        print(f"{avg_rouge:.2%} 'ROUGE-L' score")
        print(f"{avg_judge:.2%} 'LLM Judge Avg' score")
        print(f"{avg_bert:.2%} 'BERTScore' score")
        print(f"========================================================================\n")

    print(f"\n--- Stratified Category Analysis (baseline-v1 vs baseline-v2) ---")
    categories = sorted(list(set(tc.category for tc in test_cases)))
    for cat in categories:
        cat_v1 = [
            score for prompt, score in model_scores["baseline-v1"].items()
            if any(tc.input_text == prompt and tc.category == cat for tc in test_cases)
        ]
        cat_v2 = [
            score for prompt, score in model_scores["baseline-v2"].items()
            if any(tc.input_text == prompt and tc.category == cat for tc in test_cases)
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
    print(f" Running Arize Phoenix Pairwise Evaluation: baseline-v1 vs baseline-v2")
    print(f"=======================================================")

    pairwise_data = []
    for tc in test_cases:
        out_v1 = run_model("baseline-v1", tc.input_text)
        out_v2 = run_model("baseline-v2", tc.input_text)
        pairwise_data.append({
            "input": tc.input_text,
            "output_v1": out_v1,
            "output_v2": out_v2,
            "expected": tc.reference_output
        })
    pairwise_df = pd.DataFrame(pairwise_data)

    evaluate_dataframe(
        dataframe=pairwise_df,
        evaluators=[pairwise_evaluator]
    )

    print(f"\n--- Pairwise Comparison: baseline-v1 vs baseline-v2 ---")
    id_map = {tc.input_text: tc.id for tc in test_cases}
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
    main()
