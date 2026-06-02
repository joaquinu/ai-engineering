# eval.py
import sys
import os

# Ensure package imports work for evaluation_framework
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepeval import evaluate
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evaluation_framework.suite import build_test_suite
from evaluation_framework.metrics import rouge_l_score, bert_score
from evaluation_framework.judge import score_with_llm_judge, pairwise_compare_with_judge
from evaluation_framework.eval_runner import run_model
from evaluation_framework.stats import wilson_confidence_interval, bootstrap_confidence_interval, fleiss_kappa, krippendorff_alpha
from evaluation_framework.cost_tracker import tracker

class RougeLMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.score = rouge_l_score(test_case.expected_output, test_case.actual_output)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "ROUGE-L"

current_model_name = None
model_scores = {
    "baseline-v1": {},
    "baseline-v2": {}
}

class LLMJudgeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.6):
        super().__init__()
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        judge_scores = score_with_llm_judge(test_case.input, test_case.actual_output, test_case.expected_output)
        
        total_normalized_score = 0.0
        for s in judge_scores:
            # Scale 1-5 to a 0.0-1.0 range
            normalized = (s.score - 1.0) / 4.0
            total_normalized_score += normalized
            
        self.score = total_normalized_score / len(judge_scores) if judge_scores else 0.0
        self.success = self.score >= self.threshold
        
        raw_avg = sum(s.score for s in judge_scores) / len(judge_scores) if judge_scores else 1.0
        global current_model_name
        if current_model_name in model_scores:
            model_scores[current_model_name][test_case.input] = raw_avg
            
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "LLM Judge Avg"

class BERTScoreMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        res = bert_score(test_case.expected_output, test_case.actual_output)
        self.score = res["f1"]
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "BERTScore"

class PairwiseLLMTestCase(LLMTestCase):
    output_v1: str = ""
    output_v2: str = ""
    reference_output: str = ""

class PairwiseCompareMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5, results_list: list = None):
        super().__init__()
        self.threshold = threshold
        self.results_list = results_list if results_list is not None else []

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        input_text = test_case.input
        out_v1 = getattr(test_case, "output_v1", "")
        out_v2 = getattr(test_case, "output_v2", "")
        ref_out = getattr(test_case, "reference_output", "")
        
        res = pairwise_compare_with_judge(input_text, out_v1, out_v2, ref_out)
        score = 1.0 if res["winner"] == "B" else 0.0
        
        self.results_list.append({
            "input": input_text,
            "winner": res["winner"],
            "reason": res["reason"],
            "score": score
        })
        
        self.score = score
        self.success = score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Pairwise Win"

def main():
    tracker.reset()
    test_cases = build_test_suite()

    rouge_metric = RougeLMetric()
    judge_metric = LLMJudgeMetric()
    bert_metric = BERTScoreMetric()

    for model_name in ["gpt-4o", "baseline-v1", "baseline-v2"]:
        print(f"\n=======================================================")
        print(f" Running DeepEval Evaluation for Model: {model_name}")
        print(f"=======================================================")

        global current_model_name
        current_model_name = model_name

        deepeval_test_cases = []
        for tc in test_cases:
            actual_output = run_model(model_name, tc.input_text)
            deepeval_test_cases.append(
                LLMTestCase(
                    input=tc.input_text,
                    actual_output=actual_output,
                    expected_output=tc.reference_output
                )
            )

        results = evaluate(
            test_cases=deepeval_test_cases,
            metrics=[rouge_metric, judge_metric, bert_metric]
        )

        rouge_scores = []
        judge_scores = []
        bert_scores = []
 
        for test_result in results.test_results:
            for metric_data in test_result.metrics_data:
                if metric_data.name == "ROUGE-L":
                    rouge_scores.append(metric_data.score)
                elif metric_data.name == "LLM Judge Avg":
                    judge_scores.append(metric_data.score)
                elif metric_data.name == "BERTScore":
                    bert_scores.append(metric_data.score)
 
        avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
        avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
        avg_bert = sum(bert_scores) / len(bert_scores) if bert_scores else 0.0
 
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
    print(f" Running DeepEval Pairwise Evaluation: baseline-v1 vs baseline-v2")
    print(f"=======================================================")

    pairwise_results = []
    pairwise_metric = PairwiseCompareMetric(results_list=pairwise_results)

    deepeval_pairwise_test_cases = []
    for tc in test_cases:
        out_v1 = run_model("baseline-v1", tc.input_text)
        out_v2 = run_model("baseline-v2", tc.input_text)
        
        test_case = PairwiseLLMTestCase(
            input=tc.input_text,
            actual_output=out_v2,
            expected_output=tc.reference_output,
            output_v1=out_v1,
            output_v2=out_v2,
            reference_output=tc.reference_output
        )
        deepeval_pairwise_test_cases.append(test_case)

    evaluate(
        test_cases=deepeval_pairwise_test_cases,
        metrics=[pairwise_metric]
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
