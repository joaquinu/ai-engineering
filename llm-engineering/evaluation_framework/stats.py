import math

def wilson_confidence_interval(successes, total, z=1.96):
    """
    Computes the Wilson score interval for a binomial proportion.
    
    The Wilson score interval is a confidence interval formula for binomial 
    proportions that performs well even with small sample sizes or extreme 
    success rates (near 0 or 1), unlike the standard Wald interval.
    
    Args:
        successes (int): Number of successful trials.
        total (int): Total number of trials.
        z (float, optional): Z-score corresponding to desired confidence level. 
            Default is 1.96 (for a 95% confidence interval).
            
    Returns:
        tuple[float, float]: A tuple (lower_bound, upper_bound) representing 
            the confidence interval, rounded to 4 decimal places.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return (round(lower, 4), round(upper, 4))


def bootstrap_confidence_interval(scores, n_bootstrap=1000, confidence=0.95):
    """
    Computes a bootstrap confidence interval for the mean of a list of scores.
    
    Bootstrapping is a non-parametric method that estimates the distribution of a 
    statistic (here, the mean) by repeatedly resampling the observed scores with 
    replacement. It generates a distribution of bootstrap means, sorts them, and 
    extracts the percentiles corresponding to the specified confidence level.
    
    Args:
        scores (list[float | int]): A list of numerical scores.
        n_bootstrap (int, optional): Number of bootstrap resamples to generate. 
            Default is 1000.
        confidence (float, optional): Desired confidence level (between 0 and 1). 
            Default is 0.95 (for a 95% confidence interval).
            
    Returns:
        tuple[float, float, float]: A tuple (lower_bound, mean, upper_bound) 
            representing the confidence bounds and the actual sample mean, 
            rounded to 4 decimal places.
    """
    if len(scores) == 1:
        val = round(float(scores[0]), 4)
        return (val, val, val)
    elif len(scores) == 0:
        return (0.0, 0.0, 0.0)
    n = len(scores)
    means = []
    seed_base = int(sum(scores) * 1000) % 2**31
    for i in range(n_bootstrap):
        seed = (seed_base + i * 7919) % 2**31
        sample = []
        for j in range(n):
            idx = (seed + j * 31) % n
            sample.append(scores[idx])
            seed = (seed * 1103515245 + 12345) % 2**31
        means.append(sum(sample) / len(sample))
    means.sort()
    alpha = (1 - confidence) / 2
    lower_idx = int(alpha * n_bootstrap)
    upper_idx = int((1 - alpha) * n_bootstrap) - 1
    mean = sum(scores) / len(scores)
    return (round(means[lower_idx], 4), round(mean, 4), round(means[upper_idx], 4))


def fleiss_kappa(reliability_data):
    """
    Computes Fleiss' Kappa for a list of ratings per subject.
    reliability_data is a list of lists: [[rater1, rater2, rater3], ...]
    """
    N = len(reliability_data)
    if N == 0:
        return 0.0
    n = len(reliability_data[0])
    if n < 2:
        return 1.0
        
    unique_values = sorted(list(set(val for row in reliability_data for val in row)))
    val_to_idx = {val: idx for idx, val in enumerate(unique_values)}
    k = len(unique_values)
    
    counts = []
    for row in reliability_data:
        row_counts = [0] * k
        for val in row:
            row_counts[val_to_idx[val]] += 1
        counts.append(row_counts)
        
    P_i = []
    for row_counts in counts:
        sum_sq = sum(c * c for c in row_counts)
        P_i.append((sum_sq - n) / (n * (n - 1)))
        
    P = sum(P_i) / N
    
    p_j = [0.0] * k
    for j in range(k):
        col_sum = sum(counts[i][j] for i in range(N))
        p_j[j] = col_sum / (N * n)
        
    P_e = sum(pj * pj for pj in p_j)
    
    if P_e == 1.0:
        return 1.0
        
    kappa = (P - P_e) / (1 - P_e)
    return round(kappa, 4)


def krippendorff_alpha(reliability_data):
    """
    Computes nominal Krippendorff's alpha for a list of ratings per subject.
    reliability_data is a list of lists: [[rater1, rater2, rater3], ...]
    """
    N = len(reliability_data)
    if N == 0:
        return 0.0
    n = len(reliability_data[0])
    if n < 2:
        return 1.0
        
    unique_values = sorted(list(set(val for row in reliability_data for val in row)))
    val_to_idx = {val: idx for idx, val in enumerate(unique_values)}
    V = len(unique_values)
    
    value_counts = [0] * V
    for row in reliability_data:
        for val in row:
            value_counts[val_to_idx[val]] += 1
            
    observed_diff = 0
    for row in reliability_data:
        for r1 in range(n):
            for r2 in range(n):
                if r1 != r2:
                    if row[r1] != row[r2]:
                        observed_diff += 1
                        
    D_o = observed_diff / (N * n * (n - 1)) if N * n * (n - 1) > 0 else 0.0
    
    total_ratings = N * n
    expected_diff = 0
    for v1 in range(V):
        for v2 in range(V):
            if v1 != v2:
                expected_diff += value_counts[v1] * value_counts[v2]
                
    D_e = expected_diff / (total_ratings * (total_ratings - 1)) if total_ratings > 1 else 1.0
    
    if D_e == 0.0:
        return 1.0
        
    alpha = 1.0 - (D_o / D_e)
    return round(alpha, 4)
