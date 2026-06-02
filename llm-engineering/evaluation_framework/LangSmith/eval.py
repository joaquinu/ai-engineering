# eval.py
import sys
import os
import uuid

# Ensure package imports work for evaluation_framework
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Disable LangSmith tracing to prevent API connection / authentication warnings offline
os.environ["LANGSMITH_TRACING"] = "false"

from langsmith import evaluate
from langsmith.schemas import Example
from langchain_classic.evaluation import load_evaluator  # Import LangChain Classic's evaluator loader

from evaluation_framework.suite import build_test_suite
from evaluation_framework.judge import score_with_llm_judge, pairwise_compare_with_judge
from evaluation_framework.eval_runner import run_model
from evaluation_framework.metrics import bert_score
from evaluation_framework.stats import wilson_confidence_interval, bootstrap_confidence_interval, fleiss_kappa, krippendorff_alpha
from evaluation_framework.cost_tracker import tracker

# Load LangChain Classic's default string_distance evaluator
langchain_string_distance = load_evaluator("string_distance")

# 1. Custom wrapper for LangChain's default string_distance evaluator
def default_string_distance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    reference = reference_outputs.get("reference_output", "")
    output = outputs.get("output", "")
    
    result = langchain_string_distance.evaluate_strings(
        prediction=output,
        reference=reference
    )
    return {
        "key": "string_distance",
        "score": result["score"]
    }

current_model_name = None
model_scores = {
    "baseline-v1": {},
    "baseline-v2": {}
}

# 2. Custom Evaluator replicating LLM Judge (relevance, correctness, helpfulness, safety)
# (Kept simulated/custom to allow offline execution without LLM API keys)
def llm_judge_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    prompt = inputs.get("input_text", "")
    reference = reference_outputs.get("reference_output", "")
    output = outputs.get("output", "")
    
    judge_scores = score_with_llm_judge(prompt, output, reference)
    
    total_normalized_score = 0.0
    for s in judge_scores:
        # Scale 1-5 to a 0.0-1.0 range
        normalized = (s.score - 1.0) / 4.0
        total_normalized_score += normalized
        
    avg_score = total_normalized_score / len(judge_scores) if judge_scores else 0.0
    
    raw_avg = sum(s.score for s in judge_scores) / len(judge_scores) if judge_scores else 1.0
    global current_model_name
    if current_model_name in model_scores:
        model_scores[current_model_name][prompt] = raw_avg

    return {
        "key": "LLM Judge Avg",
        "score": avg_score
    }

def bert_score_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    reference = reference_outputs.get("reference_output", "")
    output = outputs.get("output", "")
    res = bert_score(reference, output)
    return {
        "key": "BERTScore F1",
        "score": res["f1"]
    }

def main():
    tracker.reset()
    # Load evaluation test cases in LangSmith Example objects
    test_cases = build_test_suite()
    data = [
        Example(
            id=str(uuid.uuid4()),
            inputs={"input_text": tc.input_text},
            outputs={"reference_output": tc.reference_output}
        )
        for tc in test_cases
    ]

    # Run local LangSmith evaluations for each model in our pipeline
    for model_name in ["gpt-4o", "baseline-v1", "baseline-v2"]:
        print(f"\n=======================================================")
        print(f" Running LangSmith Evaluation for Model: {model_name}")
        print(f"=======================================================")
        
        global current_model_name
        current_model_name = model_name
        
        # LangSmith target function receiving input dict and returning output dict
        def target(inputs: dict) -> dict:
            return {"output": run_model(model_name, inputs["input_text"])}

        # Execute evaluation using default evaluators
        experiment_results = evaluate(
            target,
            data=data,
            evaluators=[default_string_distance_evaluator, llm_judge_evaluator, bert_score_evaluator],
            experiment_prefix=f"eval-framework-{model_name}",
            upload_results=False  # Run locally without requiring cloud upload
        )

        string_distances = []
        judge_scores = []
        bert_scores = []
        for row in experiment_results:
            results_list = row.get("evaluation_results", {}).get("results", [])
            for res in results_list:
                if res.key == "string_distance":
                    string_distances.append(res.score)
                elif res.key == "LLM Judge Avg":
                    judge_scores.append(res.score)
                elif res.key == "BERTScore F1":
                    bert_scores.append(res.score)

        avg_str = sum(string_distances) / len(string_distances) if string_distances else 0.0
        avg_jdg = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
        avg_bert = sum(bert_scores) / len(bert_scores) if bert_scores else 0.0

        print(f"\n=========================SUMMARY FOR {model_name}=========================")
        print(f"{avg_str:.2%} 'string_distance' score")
        print(f"{avg_jdg:.2%} 'LLM Judge Avg' score")
        print(f"{avg_bert:.2%} 'BERTScore F1' score")
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
    print(f" Running LangSmith Pairwise Evaluation: baseline-v1 vs baseline-v2")
    print(f"=======================================================")

    def pairwise_target(inputs: dict) -> dict:
        return {
            "output_v1": run_model("baseline-v1", inputs["input_text"]),
            "output_v2": run_model("baseline-v2", inputs["input_text"])
        }

    pairwise_results = []
    id_map = {tc.input_text: tc.id for tc in test_cases}

    def pairwise_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        prompt = inputs.get("input_text", "")
        reference = reference_outputs.get("reference_output", "")
        out_v1 = outputs.get("output_v1", "")
        out_v2 = outputs.get("output_v2", "")
        
        res = pairwise_compare_with_judge(prompt, out_v1, out_v2, reference)
        score = 1.0 if res["winner"] == "B" else 0.0
        
        pairwise_results.append({
            "input": prompt,
            "winner": res["winner"],
            "reason": res["reason"],
            "score": score
        })
        return {
            "key": "Pairwise Win",
            "score": score
        }

    evaluate(
        pairwise_target,
        data=data,
        evaluators=[pairwise_evaluator],
        experiment_prefix="eval-framework-pairwise",
        upload_results=False
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
    main()
