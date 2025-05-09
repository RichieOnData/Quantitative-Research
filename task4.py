import pandas as pd
import numpy as np

def compute_log_likelihood(defaults, total):
    """Compute log-likelihood for a bucket."""
    if total == 0:
        return 0
    p = defaults / total if total > 0 else 0
    if p == 0 or p == 1:
        # Avoid log(0)
        return 0
    return defaults * np.log(p) + (total - defaults) * np.log(1 - p)

def optimal_buckets(fico_scores, defaults, n_buckets, criterion='loglik'):
    """
    Find optimal bucket boundaries for FICO scores.
    criterion: 'loglik' for log-likelihood, 'mse' for mean squared error.
    """
    # Sort by FICO
    df = pd.DataFrame({'fico': fico_scores, 'default': defaults}).sort_values('fico').reset_index(drop=True)
    N = len(df)
    # Precompute cumulative sums for fast lookup
    cum_defaults = np.cumsum(df['default'].values)
    cum_total = np.arange(1, N+1)
    # DP tables
    dp = np.full((N+1, n_buckets+1), -np.inf)
    prev = np.zeros((N+1, n_buckets+1), dtype=int)
    dp[0, 0] = 0
    # DP
    for k in range(1, n_buckets+1):
        for i in range(k, N+1):
            for j in range(k-1, i):
                # Bucket: j to i-1
                total = i - j
                defaults_in_bucket = cum_defaults[i-1] - (cum_defaults[j-1] if j > 0 else 0)
                if criterion == 'loglik':
                    score = compute_log_likelihood(defaults_in_bucket, total)
                else:  # MSE
                    p = defaults_in_bucket / total if total > 0 else 0
                    mse = np.sum((df['default'].values[j:i] - p) ** 2)
                    score = -mse  # minimize MSE
                if dp[j, k-1] + score > dp[i, k]:
                    dp[i, k] = dp[j, k-1] + score
                    prev[i, k] = j
    # Recover boundaries
    boundaries = []
    i, k = N, n_buckets
    while k > 0:
        j = prev[i, k]
        boundaries.append(df['fico'].values[j])
        i, k = j, k-1
    boundaries = sorted(boundaries)
    return boundaries

def assign_rating(fico, boundaries):
    """Assign a rating based on FICO and bucket boundaries."""
    for i, b in enumerate(boundaries):
        if fico < b:
            return i + 1  # Lower rating = better
    return len(boundaries) + 1

# --- Usage Example ---

# Load data
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')
fico_scores = df['fico_score'].values
defaults = df['default'].values

# Find optimal boundaries for 5 buckets using log-likelihood
boundaries = optimal_buckets(fico_scores, defaults, n_buckets=5, criterion='loglik')
print('Optimal FICO boundaries:', boundaries)

# Map FICO scores to ratings
df['rating'] = df['fico_score'].apply(lambda x: assign_rating(x, boundaries))
print(df[['fico_score', 'rating']].head(10))