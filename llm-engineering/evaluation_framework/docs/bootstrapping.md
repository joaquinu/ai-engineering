# Bootstrapping in LLM Evaluation

Bootstrapping is a powerful, non-parametric statistical method used to estimate the uncertainty (such as confidence intervals and standard error) of a sample statistic (like the mean score of an LLM evaluation) by repeatedly resampling the observed data with replacement.

---

## The Core Concept: Resampling with Replacement

Imagine you ran an evaluation on a new prompt version using **8 test cases**, and obtained the following quality scores:
$$\text{Original Scores} = [1, 2, 3, 4, 5, 6, 7, 8]$$

The average (mean) score of this sample is:
$$\text{Sample Mean} = \frac{1+2+3+4+5+6+7+8}{8} = 4.5$$

To understand how much this mean might vary if you ran the experiment with a different set of test cases, you perform **bootstrapping**:

1. **Draw a Resample**: Randomly select 8 scores from your original list, **with replacement**.
   * *With replacement* means that once a score is selected, it goes back into the pool and can be selected again.
   * **Example Resample**: $[1, 3, 3, 5, 5, 5, 7, 8]$ (notice that `3` appears twice, `5` appears three times, and `2`, `4`, and `6` do not appear at all).
2. **Calculate the Statistic**: Compute the mean of this new bootstrap sample:
   $$\text{Bootstrap Mean} = \frac{1+3+3+5+5+5+7+8}{8} = 4.25$$
3. **Repeat**: Repeat this process a large number of times (typically $N = 1000$ or $10000$ times) to generate a distribution of bootstrap means.

---

## Step-by-Step Bootstrap Workflow

```
Original Data [1, 2, 3, 4, 5, 6, 7, 8]  (Mean = 4.5)
  │
  ├──► Resample 1: [1, 3, 3, 5, 5, 5, 7, 8]  ──► Mean = 4.25
  ├──► Resample 2: [2, 2, 4, 4, 6, 7, 8, 8]  ──► Mean = 5.125
  ├──► Resample 3: [1, 1, 2, 3, 5, 6, 6, 7]  ──► Mean = 3.875
  │    ...
  └──► Resample N: [2, 3, 3, 4, 5, 5, 8, 8]  ──► Mean = 4.750
```

1. **Generate $N$ Bootstrap Datasets**: Draw random samples of size $n$ (matching the original dataset size) with replacement.
2. **Calculate the Metric**: Compute the target metric (e.g., mean, median, standard deviation) for each bootstrap dataset.
3. **Keep Track**: Store the calculated values to form a **bootstrap distribution** of the statistic.
4. **Analyze the Variation**: Use this distribution to calculate the Standard Error and Confidence Intervals.

---

## Key Statistical Outputs

### 1. Standard Error (SE)
The **Standard Error** is the standard deviation of the bootstrap distribution of the means. It measures how precise your sample mean is:
$$\text{SE} = \text{Standard Deviation of }(\text{Mean}_1, \text{Mean}_2, \dots, \text{Mean}_N)$$
* A lower SE indicates that your sample mean is highly stable and precise.

### 2. Confidence Intervals (CI)
To calculate a **95% Confidence Interval** using the percentile method:
1. Sort the list of $N$ bootstrap means in ascending order.
2. The **2.5th percentile** value becomes the lower bound (e.g., at index $25$ if $N = 1000$).
3. The **97.5th percentile** value becomes the upper bound (e.g., at index $975$ if $N = 1000$).

This range tells you that if you repeated the evaluation experiment, the true performance mean would fall within this interval $95\%$ of the time.

---

## Why Use Bootstrapping for LLM Evaluations?

* **No Distribution Assumptions (Non-parametric)**: Classical statistical tests assume data is normally distributed (a bell curve). LLM evaluation scores (e.g., 1–5 scale) are often highly skewed (mostly 4s and 5s, with occasional 1s for safety failures). Bootstrapping handles any arbitrary distribution perfectly.
* **Small Sample Robustness**: It allows you to obtain reliable bounds of error even when your evaluation suite has a moderate number of test cases (e.g., 50–100 samples).
* **Deterministic Behavior in CI**: In our implementation, we use a deterministic pseudorandom number generator seeded by the input data itself. This guarantees that running the exact same evaluation scores will always yield the exact same confidence boundaries.
