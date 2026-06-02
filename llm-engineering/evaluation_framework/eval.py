import sys
import os

# Ensure package imports work even when executed as a script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_framework.metrics import rouge_l_score, bert_score
from evaluation_framework.stats import wilson_confidence_interval, bootstrap_confidence_interval, fleiss_kappa, krippendorff_alpha
from evaluation_framework.judge import score_with_llm_judge, pairwise_compare_with_judge
from evaluation_framework.eval_runner import (
    run_model,
    run_eval_suite,
)
from evaluation_framework.suite import build_test_suite
from evaluation_framework.compare import compare_eval_runs, print_comparison_report
from evaluation_framework.cost_tracker import tracker


def run_demo():
    tracker.reset()
    print("=" * 70)
    print("  Evaluation & Testing LLM Applications")
    print("=" * 70)

    test_suite = build_test_suite()
    print(f"\n--- Test Suite: {len(test_suite)} cases ---")
    for tc in test_suite:
        print(f"  [{tc.id}] {tc.category}: {tc.input_text[:60]}...")

    print(f"\n--- ROUGE-L Scores ---")
    rouge_tests = [
        ("The capital of France is Paris.", "Paris is the capital of France."),
        ("Machine learning uses data to learn patterns.", "Deep learning is a subset of AI."),
        ("Python is a programming language.", "Python is a programming language."),
    ]
    for ref, hyp in rouge_tests:
        score = rouge_l_score(ref, hyp)
        print(f"  ROUGE-L: {score:.4f}")
        print(f"    ref: {ref[:50]}")
        print(f"    hyp: {hyp[:50]}")

    print(f"\n--- BERTScore (Simplified) ---")
    for ref, hyp in rouge_tests:
        scores = bert_score(ref, hyp)
        print(f"  BERTScore F1: {scores['f1']:.4f} (P: {scores['precision']:.4f}, R: {scores['recall']:.4f})")
        print(f"    ref: {ref[:50]}")
        print(f"    hyp: {hyp[:50]}")

    print(f"\n--- LLM-as-Judge Scoring ---")
    sample_case = test_suite[1]
    sample_output = run_model("gpt-4o", sample_case.input_text)
    scores = score_with_llm_judge(
        sample_case.input_text, sample_output, sample_case.reference_output
    )
    print(f"  Input: {sample_case.input_text[:60]}...")
    print(f"  Output: {sample_output[:60]}...")
    for s in scores:
        print(f"    {s.criterion}: {s.score}/5 -- {s.reasoning[:70]}...")

    print(f"\n--- Confidence Intervals ---")
    sample_scores = [4, 5, 3, 4, 4, 5, 3, 4, 5, 4, 3, 4, 4, 5, 4]
    ci = bootstrap_confidence_interval(sample_scores)
    print(f"  Scores: {sample_scores}")
    print(f"  Bootstrap CI: [{ci[0]:.4f}, {ci[1]:.4f}, {ci[2]:.4f}]")
    print(f"  (lower bound, mean, upper bound)")

    passing = sum(1 for s in sample_scores if s >= 4)
    wilson_ci = wilson_confidence_interval(passing, len(sample_scores))
    print(f"  Pass rate (>=4): {passing}/{len(sample_scores)} = {passing/len(sample_scores):.1%}")
    print(f"  Wilson CI: [{wilson_ci[0]:.4f}, {wilson_ci[1]:.4f}]")

    print(f"\n--- Full Eval Run: baseline-v1 ---")
    baseline_results = run_eval_suite(test_suite, "baseline-v1", "v1.0")
    for r in baseline_results:
        avg = r.average_score()
        print(f"  [{r.test_case_id}] avg={avg:.2f} | {', '.join(f'{s.criterion}={s.score}' for s in r.scores)}")

    print(f"\n--- Full Eval Run: baseline-v2 ---")
    new_results = run_eval_suite(test_suite, "baseline-v2", "v2.0")
    for r in new_results:
        avg = r.average_score()
        print(f"  [{r.test_case_id}] avg={avg:.2f} | {', '.join(f'{s.criterion}={s.score}' for s in r.scores)}")

    print(f"\n--- Comparison Report ---")
    report = compare_eval_runs(baseline_results, new_results)
    print_comparison_report(report)

    print(f"\n--- Stratified Category Analysis (baseline-v1 vs baseline-v2) ---")
    tc_category_map = {tc.id: tc.category for tc in test_suite}
    categories = sorted(list(set(tc.category for tc in test_suite)))
    for cat in categories:
        cat_v1 = [r.average_score() for r in baseline_results if tc_category_map[r.test_case_id] == cat]
        cat_v2 = [r.average_score() for r in new_results if tc_category_map[r.test_case_id] == cat]
        
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

    print(f"\n--- Sample Size Analysis ---")
    for n in [50, 100, 200, 500, 1000]:
        ci = wilson_confidence_interval(int(n * 0.9), n)
        width = ci[1] - ci[0]
        print(f"  n={n:>5}: 90% accuracy -> CI [{ci[0]:.3f}, {ci[1]:.3f}] (width: {width:.3f})")

    print(f"\n--- Inter-Rater Reliability Verification (3 runs) ---")
    reliability_data = []
    criteria = ["relevance", "correctness", "helpfulness", "safety"]
    for tc in test_suite:
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

    print(f"\n--- Pairwise Comparison: baseline-v1 vs baseline-v2 ---")
    pairwise_scores = []
    for tc in test_suite:
        out_v1 = run_model("baseline-v1", tc.input_text)
        out_v2 = run_model("baseline-v2", tc.input_text)
        res = pairwise_compare_with_judge(tc.input_text, out_v1, out_v2, tc.reference_output)
        
        score = 1.0 if res["winner"] == "B" else 0.0
        pairwise_scores.append(score)
        print(f"  [{tc.id}] Winner: {res['winner']} | {res['reason']}")

    wins = sum(1 for s in pairwise_scores if s == 1.0)
    total = len(pairwise_scores)
    win_rate = wins / total if total > 0 else 0.0
    
    wilson_ci = wilson_confidence_interval(wins, total)
    bootstrap_ci = bootstrap_confidence_interval(pairwise_scores)
    
    print(f"\nPairwise Comparison Summary:")
    print(f"  baseline-v2 Win Rate: {wins}/{total} = {win_rate:.1%}")
    print(f"  Wilson Confidence Interval (95%): [{wilson_ci[0]:.4f}, {wilson_ci[1]:.4f}]")
    print(f"  Bootstrap Confidence Interval (95%): [{bootstrap_ci[0]:.4f}, {bootstrap_ci[1]:.4f}, {bootstrap_ci[2]:.4f}]")

    # Print cost summary at the end
    tracker.print_summary()


if __name__ == "__main__":
    run_demo()