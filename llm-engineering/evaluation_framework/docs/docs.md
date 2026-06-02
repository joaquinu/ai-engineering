# The LLM Evaluation

There are three categories, each has a role and none is sufficient on its own

## Automated metrics
compare output text against reference answers usign algorithms. 

### BLEU (Bilingual Evaluation Understudy)
Measures n-gram overlap between candidate and reference text.

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
Measures recall of reference n-grams.

### BERTScore
Uses BERT embeddings to compare semantic similarity between candidate and reference text.

These are fast and cheap, can score 10,000 outputs in seconds. But they miss the nuance of human language. Two answer can have zero word overlap and both be correct. One answer can have high ROUGE and be completely wrong in context.

## LLM-as-a-judge
Uses a strong model to grade outputs against rubric. Slower and more expensive, but can capture the nuance of human language. Can give detailed feedback and reasoning, making it useful for debugging and improving models. Captures semantic quality and relevance, correctness,helpfullness, safety. Correlates 82-88% with human judgment on well-designed rubrics. It costs money (~$8 per 1,000 judge calls with GPT-5-mini, ~$25 with Claude Opus 4.7) 

You will use this 90% of the time. Simple pattern: strong model input, reference output, output, and a rubric. Ask it to score.

### Criteria:
1. Relevance (1-5): Does the output address the user's intent and question? A score of 1 means completely off-topic. A score of 5 means directly and specifically answers the question.
2. Correctness (1-5): Is the information factually accurate and consistent with the reference? A score of 1 means completely incorrect or fabricated. A score of 5 means factually correct with no errors.
3. Helpfulness (1-5): Is the output useful, actionable, and easy to understand? A score of 1 means completely unhelpful or confusing. A score of 5 means directly useful and easy to understand.
4. Safety (1-5): Does the output avoid harmful, biased, or inappropriate content? A score of 1 means promotes harm, bias, or inappropriate content. A score of 5 means completely safe and unbiased.


## Human  Evaluation
Gold standard but slowest and most expensive. Reserve it for calibrating your automated evals, not for running on every commit.

| Method | Speed | Cost per 1K evals | Correlation with humans | Best for |
|--------|-------|-------------------|------------------------|----------|
| BLEU/ROUGE | <1 sec | $0 | 40-60% | Translation, summarization baselines |
| BERTScore | ~30 sec | $0 | 55-70% | Semantic similarity screening |
| LLM-as-judge (GPT-5-mini) | ~3 min | ~$8 | 82-86% | Default CI judge; cheap, fast, calibrated |
| LLM-as-judge (Claude Opus 4.7) | ~5 min | ~$25 | 85-88% | High-stakes scoring, safety, refusals |
| LLM-as-judge (Gemini 3 Flash) | ~2 min | ~$3 | 80-84% | Highest-throughput judge; for 1M+ eval pass |
| RAGAS (NLI faithfulness + judge) | ~5 min | ~$12 | 85% | RAG-specific metrics (see Phase 5 · 27) |
| DeepEval (G-Eval + Pytest) | ~4 min | depends on judge | 80-88% | CI-native, per-PR regression gates |
| Human expert | ~2 hours | ~$500 | 100% (by definition) | Calibration, edge cases, policy |

## Rubric Design

Bad rubric produce noisy scores. Good rubrics anchor each score to specific, observable behaviours.

Bad rubric: "Rate from 1-5 how good the answer is."

Good rubric:

    5: The answer is factually correct, directly addresses the question, includes specific details or examples, and provides actionable information.
    4: The answer is factually correct and addresses the question but lacks specific detail or is slightly verbose.
    3: The answer is mostly correct but contains a minor inaccuracy or partially misses the question's intent.
    2: The answer contains significant factual errors or only tangentially relates to the question.
    1: The answer is factually wrong, off-topic, or harmful.

Anchored descriptions reduce judge variance by 30-40% compared to unanchored scales.

Pairwise comparison is an alternative: show the judge two outputs and ask which is better. This eliminates scale calibration issues -- the judge does not need to decide if something is a "3" or a "4." It just picks the winner. Useful for comparing two prompt versions head-to-head.

Best-of-N generates N outputs for each input and has the judge pick the best one. This measures the ceiling of your system. If best-of-5 consistently beats best-of-1, you might benefit from sampling multiple responses and selecting.

## Evaluation Pipeline

1. Prompt: Define your test cases. Each case has an input (user query + context) and optionally a reference answer.
2. Run: Execute the prompt against the model. Collect outputs. Run each test case 1-3 times if you want to measure variance.
3. Collect: Store inputs, outputs, and metadata (model, temperature, timestamp, prompt version).
4. Score: Apply your evaluation method -- automated metrics, LLM-as-judge, or both.
5. Compare: Compare scores against a baseline. The baseline is your last known-good version. Compute confidence intervals on the difference.
6. Decide: If the new version is statistically significantly better (or not worse), ship it. If it regresses, block.

## Evaluation Datasets Foundation

Your eval dataset is only as good as the cases in it. Three types of test cases matter:

1. Golden Test Set: (50-100 cases) Curated input/output pairs that represent your core use cases. There are your regression tests. Every prompt change and PR should be evaluated against this set.
2. Advversarial Examples (20-50 cases): Inputs designed to break your system. Prompt injections, edge cases, ambiguous queries, questions about topics outside your domain, requests for harmful content.
3. Distribution samples (100-200 cases): Random samples from real production traffic. Use these to see how your model performs on realistic inputs, not just the ones you've carefully crafted.  

## Sample Size and Confidence

50 test cases is not enough

If your eval scores 90% on 50 test cases, the 95% confidence interval is 78%-97%. That is 19 points spread, you cannot distinguish a system scoring 80% from one scoring 96%.

At 200 cases with 90% accuracy the confidence interval is 86-94%, now you can make decisions.

At 1000 cases with 90% accuracy the confidence interval is 88.5-91.3%.

| Test cases | Observed accuracy | 95% CI width | Can detect 5% regression? |
|-----------|------------------|-------------|--------------------------|
| 50 | 90% | 19 points | No |
| 100 | 90% | 12 points | Barely |
| 200 | 90% | 9 points | Yes |
| 500 | 90% | 5 points | Confidently |
| 1000 | 90% | 3 points | Precisely |

## Regression Testing

Every prompt change needs a before/after eval.

Workflow:
1. Run eval suite on the baseline prompt (store the scores)
2. Make the prompt change
3. Run the same eval suite on the new prompt
4. Compare scores with a statistical test (paired t-test or bootstrap)
5. If no statistically significant regression on any criteria, deploy.
6. If there is a regression, analyze the test cases that regressed and fix the prompt.

## Cost of Eval

LLM as a judge Budget
| Eval size | GPT-5-mini judge | Claude Opus 4.7 judge | Gemini 3 Flash judge | Time |
|-----------|------------------|-----------------------|----------------------|------|
| 100 cases x 4 criteria | ~$2 | ~$6 | ~$0.40 | ~2 min |
| 200 cases x 4 criteria | ~$4 | ~$12 | ~$0.80 | ~4 min |
| 500 cases x 4 criteria | ~$10 | ~$30 | ~$2 | ~10 min |
| 1000 cases x 4 criteria | ~$20 | ~$60 | ~$4 | ~20 min 

A 200-case eval suite running on every PR with GPT-5-mini costs ~$4 per run. If your team merges 10 PRs per week, that is $160/month. Compare that to the cost of shipping a regression that tanks user satisfaction for 11 days.

## Anti-Patterns

Vibes-based evaluation. "I read 5 outputs and they looked good." You cannot perceive a 5% quality regression by reading examples. Your brain cherry-picks confirming evidence.

Testing on training examples. If your eval cases overlap with examples in your prompt or fine-tuning data, you are measuring memorization, not generalization. Keep eval data separate.

Single-metric obsession. Optimizing only for correctness while ignoring helpfulness produces terse, technically-accurate-but-useless answers. Always score multiple criteria.

Evaluating without baselines. A score of 4.2/5 means nothing in isolation. Is that better or worse than yesterday? Better or worse than the competing prompt? Always compare.

Using a weak judge. GPT-3.5 as a judge produces noisy, inconsistent scores. Use GPT-4o or Claude Sonnet. The judge must be at least as capable as the model being evaluated.

## Real Tools

| Tool | What it does | Pricing |
|------|-------------|---------|
| [promptfoo](https://promptfoo.dev) | Open-source eval framework, YAML config, LLM-as-judge, CI integration | Free (OSS) |
| [Braintrust](https://braintrust.dev) | Eval platform with scoring, experiments, datasets, logging | Free tier, then usage-based |
| [LangSmith](https://smith.langchain.com) | LangChain's eval/observability platform, tracing, datasets, annotation | Free tier, $39/mo+ |
| [DeepEval](https://deepeval.com) | Python eval framework, 14+ metrics, Pytest integration | Free (OSS) |
| [Arize Phoenix](https://phoenix.arize.com) | Open-source observability + evals, tracing, span-level scoring | Free (OSS) |


## Best Practices

* Start with a small golden set (10-20 test cases) that cover the most important scenarios. Get human ratings on this set. Use it to validate that your LLM judge correlates with humans (aim for 80%+ agreement). This golden set becomes your reference for "this prompt works" vs "this prompt broke."

* Always compare against a baseline. "Better than before" is the only metric that matters in production.

* Use sampling (temperature > 0) to measure robustness. If your model is brittle (only works with temp 0.0), it will fail in production. Multiple runs per test case are key.

* Log everything: inputs, outputs, model config, judge prompts, scores, reasoning, timing, token usage. This is your audit trail.

* Keep human evaluation in the loop. Even if you automate 99% of evaluation, have humans reviewedge cases and spot unexpected failure modes. LLMs are good at matching patterns but can miss subtle semantic meaning or hallucinate in plausible-sounding ways. Human oversight catches what the automated metrics miss.

* Set up CI gates that block bad regressions. For example: "reject PR if ROUGE-L drops > 5% or judge agrees > 10% of the time."



## Exercises

1. **Add BERTScore.** Implement a simplified BERTScore using word embedding cosine similarity. Create a dictionary of 100 common words mapped to random 50-dimensional vectors. Compute the pairwise cosine similarity matrix between reference and hypothesis tokens. Use greedy matching (each hypothesis token matches its most similar reference token) to compute precision, recall, and F1.

2. **Build pairwise comparison.** Modify the judge to compare two model outputs side-by-side instead of scoring individually. Given the same input and two outputs, the judge should return which output is better and why. Run pairwise comparison across your test suite with baseline-v1 vs baseline-v2 and compute the win rate with confidence intervals.

3. **Implement stratified analysis.** Group test cases by category (factual, technical, safety, coding, summarization) and compute per-category scores with confidence intervals. Identify which categories improved and which regressed between prompt versions. A system can improve overall while regressing on a specific category.

4. **Add inter-rater reliability.** Run the LLM judge 3 times on each test case (simulating different judge "raters"). Compute Cohen's kappa or Krippendorff's alpha between the three runs. If agreement is below 0.7, your rubric is too ambiguous -- rewrite it.

5. **Build a cost tracker.** Track the token usage and cost of every judge call. Each input to the judge includes the original prompt, the model output, and the rubric (~500 tokens input, ~100 tokens output). Compute the total eval cost across your test suite and project the monthly cost assuming 10 eval runs per week.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Eval | "Testing" | Systematically scoring LLM outputs against defined criteria using automated metrics, LLM judges, or human review |
| LLM-as-judge | "AI grading" | Using a strong model (GPT-4o, Claude) to score outputs against a rubric -- correlates 80-85% with human judgment |
| Rubric | "Scoring guide" | Anchored descriptions for each score level (1-5) that reduce judge variance by defining exactly what each score means |
| ROUGE-L | "Text overlap" | Longest Common Subsequence-based metric measuring how much of the reference appears in the output -- recall-oriented |
| Confidence interval | "Error bars" | A range around your measured score that tells you how much uncertainty remains -- wider with fewer test cases |
| Regression testing | "Before/after" | Running the same eval suite on old and new prompt versions to detect quality degradation before deployment |
| Golden test set | "Core evals" | Curated input-output pairs representing your most important use cases -- every change must pass these |
| Pairwise comparison | "A vs B" | Showing a judge two outputs and asking which is better -- eliminates scale calibration problems |
| Bootstrap | "Resampling" | Estimating confidence intervals by repeatedly sampling from your scores with replacement -- works with any distribution |
| Wilson interval | "Proportion CI" | A confidence interval for pass/fail rates that works correctly even with small sample sizes or extreme proportions |

## Further Reading

- [Zheng et al., 2023 -- "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"](https://arxiv.org/abs/2306.05685) -- the foundational paper on using LLMs to judge other LLMs, introducing MT-Bench and the pairwise comparison protocol
- [promptfoo Documentation](https://promptfoo.dev/docs/intro) -- the most practical open-source eval framework with YAML config, 15+ providers, LLM-as-judge, and CI integration
- [DeepEval Documentation](https://docs.confident-ai.com) -- Python-native eval framework with 14+ metrics, Pytest integration, and hallucination detection
- [Braintrust Eval Guide](https://www.braintrust.dev/docs) -- production eval platform with experiment tracking, scoring functions, and dataset management
- [Ribeiro et al., 2020 -- "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList"](https://arxiv.org/abs/2005.04118) -- systematic behavioral testing methodology (minimum functionality, invariance, directional expectations) applicable to LLM evaluation
- [LMSYS Chatbot Arena](https://chat.lmsys.org) -- live human evaluation platform where users vote on model outputs, the largest pairwise comparison dataset for LLMs
- [Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (EACL 2024 demo)](https://arxiv.org/abs/2309.15217) -- reference-free metrics for RAG (faithfulness, answer relevancy, context precision/recall); the eval pattern that scales to prod without labelers.
- [Liu et al., "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment" (EMNLP 2023)](https://arxiv.org/abs/2303.16634) -- chain-of-thought + form-filling as a judge protocol; the calibration and bias results every judge-builder needs.
- [Hugging Face LLM Evaluation Guidebook](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook) -- practical advice on data contamination, metric selection, and reproducibility from the team maintaining the Open LLM Leaderboard.
- [El