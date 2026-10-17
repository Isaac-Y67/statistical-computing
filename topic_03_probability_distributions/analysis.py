"""
Topic 3: Probability Distributions
Libraries: numpy, pandas, scipy.stats, matplotlib
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# 1. Dataset (synthetic, reproducible)
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
n = 500
N_TRIALS = 20  # items per batch for the binomial variable
df = pd.DataFrame({
    "height_cm": rng.normal(170, 8, n).round(1),
    "defects_per_batch": rng.binomial(N_TRIALS, 0.10, n),
    "arrivals_per_hour": rng.poisson(4, n),
    "wait_minutes": rng.exponential(5, n).round(2),
})
df.to_csv(DATA / "distribution_data.csv", index=False)

height = df["height_cm"].to_numpy()
defects = df["defects_per_batch"].to_numpy()
arrivals = df["arrivals_per_hour"].to_numpy()
wait = df["wait_minutes"].to_numpy()

rows = []  # (distribution, quantity, value)

# ---------------------------------------------------------------
# 2. Normal distribution (heights)
# ---------------------------------------------------------------
mu, sigma = stats.norm.fit(height)
print("=== Normal (height) ===")
print(f"Fitted mean = {mu:.3f}, sd = {sigma:.3f}")
p1 = stats.norm.cdf(165, mu, sigma)
p2 = stats.norm.sf(180, mu, sigma)
p3 = stats.norm.cdf(180, mu, sigma) - stats.norm.cdf(160, mu, sigma)
q95 = stats.norm.ppf(0.95, mu, sigma)
print(f"P(H < 165)        = {p1:.4f}")
print(f"P(H > 180)        = {p2:.4f}")
print(f"P(160 < H < 180)  = {p3:.4f}")
print(f"95th percentile   = {q95:.2f} cm")
rows += [
    ("Normal", "mean", mu), ("Normal", "sd", sigma),
    ("Normal", "P(H < 165)", p1), ("Normal", "P(H > 180)", p2),
    ("Normal", "P(160 < H < 180)", p3), ("Normal", "95th percentile", q95),
]

# ---------------------------------------------------------------
# 3. Binomial distribution (defects per batch of 20)
# ---------------------------------------------------------------
p_hat = defects.mean() / N_TRIALS
print("\n=== Binomial (defects per batch of 20) ===")
print(f"Fitted n = {N_TRIALS}, p = {p_hat:.4f}")
b0 = stats.binom.pmf(0, N_TRIALS, p_hat)
b1 = stats.binom.cdf(2, N_TRIALS, p_hat)
b2 = stats.binom.sf(3, N_TRIALS, p_hat)
print(f"P(X = 0)   = {b0:.4f}")
print(f"P(X <= 2)  = {b1:.4f}")
print(f"P(X >= 4)  = {b2:.4f}")
rows += [
    ("Binomial", "p", p_hat), ("Binomial", "P(X = 0)", b0),
    ("Binomial", "P(X <= 2)", b1), ("Binomial", "P(X >= 4)", b2),
]

# ---------------------------------------------------------------
# 4. Poisson distribution (arrivals per hour)
# ---------------------------------------------------------------
lam = arrivals.mean()
print("\n=== Poisson (arrivals per hour) ===")
print(f"Fitted lambda = {lam:.4f}  (sample variance = {arrivals.var(ddof=1):.4f})")
k0 = stats.poisson.pmf(0, lam)
k1 = stats.poisson.cdf(2, lam)
k2 = stats.poisson.sf(6, lam)
print(f"P(X = 0)   = {k0:.4f}")
print(f"P(X <= 2)  = {k1:.4f}")
print(f"P(X > 6)   = {k2:.4f}")
rows += [
    ("Poisson", "lambda", lam), ("Poisson", "P(X = 0)", k0),
    ("Poisson", "P(X <= 2)", k1), ("Poisson", "P(X > 6)", k2),
]

# ---------------------------------------------------------------
# 5. Exponential distribution (waiting times)
# ---------------------------------------------------------------
_, scale = stats.expon.fit(wait, floc=0)
rate = 1 / scale
print("\n=== Exponential (waiting time) ===")
print(f"Fitted scale (mean) = {scale:.4f} min, rate = {rate:.4f} per min")
e1 = stats.expon.sf(10, scale=scale)
e2 = stats.expon.cdf(3, scale=scale)
e3 = stats.expon.median(scale=scale)
print(f"P(T > 10)  = {e1:.4f}")
print(f"P(T <= 3)  = {e2:.4f}")
print(f"Median     = {e3:.3f} min")
rows += [
    ("Exponential", "mean (scale)", scale), ("Exponential", "rate", rate),
    ("Exponential", "P(T > 10)", e1), ("Exponential", "P(T <= 3)", e2),
    ("Exponential", "median", e3),
]

# ---------------------------------------------------------------
# 6. Goodness of fit (Kolmogorov-Smirnov)
# ---------------------------------------------------------------
ks_norm = stats.kstest(height, stats.norm(mu, sigma).cdf)
ks_exp = stats.kstest(wait, stats.expon(scale=scale).cdf)
print("\n=== Kolmogorov-Smirnov goodness of fit ===")
print(f"Normal (height):      D = {ks_norm.statistic:.4f}, p = {ks_norm.pvalue:.4f}")
print(f"Exponential (wait):   D = {ks_exp.statistic:.4f}, p = {ks_exp.pvalue:.4f}")
rows += [
    ("KS test", "Normal D", ks_norm.statistic), ("KS test", "Normal p", ks_norm.pvalue),
    ("KS test", "Exponential D", ks_exp.statistic),
    ("KS test", "Exponential p", ks_exp.pvalue),
]

# ---------------------------------------------------------------
# 7. Model comparison for waiting times (AIC, lower is better)
# ---------------------------------------------------------------
candidates = {
    "expon": stats.expon, "gamma": stats.gamma,
    "lognorm": stats.lognorm, "weibull_min": stats.weibull_min,
}
n_params = {"expon": 1, "gamma": 2, "lognorm": 2, "weibull_min": 2}
aic = {}
for name, dist in candidates.items():
    params = dist.fit(wait, floc=0)
    loglik = np.sum(dist.logpdf(wait, *params))
    aic[name] = 2 * n_params[name] - 2 * loglik

print("\n=== AIC comparison (waiting time) ===")
for name, val in sorted(aic.items(), key=lambda kv: kv[1]):
    print(f"{name:12s} AIC = {val:.2f}")
    rows.append(("AIC", name, val))

# ---------------------------------------------------------------
# 8. Save results table
# ---------------------------------------------------------------
results = pd.DataFrame(rows, columns=["distribution", "quantity", "value"]).round(4)
results.to_csv(OUT / "distribution_results.csv", index=False)

# ---------------------------------------------------------------
# 9. Plots: histogram / bar chart with fitted curve
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# Normal
ax = axes[0, 0]
ax.hist(height, bins=25, density=True, alpha=0.6, color="steelblue", label="Data")
x = np.linspace(height.min(), height.max(), 300)
ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=2,
        label=f"Normal({mu:.1f}, {sigma:.1f})")
ax.set_title("Height (Normal)")
ax.set_xlabel("cm")
ax.legend()

# Binomial
ax = axes[0, 1]
k = np.arange(0, defects.max() + 1)
obs = np.bincount(defects, minlength=k.size)[: k.size] / n
ax.bar(k - 0.2, obs, width=0.4, color="steelblue", alpha=0.7, label="Observed")
ax.bar(k + 0.2, stats.binom.pmf(k, N_TRIALS, p_hat), width=0.4, color="tomato",
       alpha=0.7, label=f"Binomial(20, {p_hat:.3f})")
ax.set_title("Defects per batch (Binomial)")
ax.set_xlabel("defects")
ax.legend()

# Poisson
ax = axes[1, 0]
k = np.arange(0, arrivals.max() + 1)
obs = np.bincount(arrivals, minlength=k.size)[: k.size] / n
ax.bar(k - 0.2, obs, width=0.4, color="steelblue", alpha=0.7, label="Observed")
ax.bar(k + 0.2, stats.poisson.pmf(k, lam), width=0.4, color="tomato", alpha=0.7,
       label=f"Poisson({lam:.2f})")
ax.set_title("Arrivals per hour (Poisson)")
ax.set_xlabel("arrivals")
ax.legend()

# Exponential
ax = axes[1, 1]
ax.hist(wait, bins=30, density=True, alpha=0.6, color="steelblue", label="Data")
x = np.linspace(0, wait.max(), 300)
ax.plot(x, stats.expon.pdf(x, scale=scale), "r-", lw=2,
        label=f"Exponential(mean={scale:.2f})")
ax.set_title("Waiting time (Exponential)")
ax.set_xlabel("minutes")
ax.legend()

fig.suptitle("Observed data with fitted probability distributions", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "fitted_distributions.png", dpi=150)

# Q-Q plot for the normal fit
fig, ax = plt.subplots(figsize=(6, 6))
stats.probplot(height, dist="norm", plot=ax)
ax.set_title("Q-Q plot: height vs Normal")
fig.tight_layout()
fig.savefig(OUT / "qq_plot_height.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)