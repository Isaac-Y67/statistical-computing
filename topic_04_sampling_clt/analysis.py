"""
Topic 4: Sampling Distributions and the Central Limit Theorem
Libraries: numpy, pandas, scipy, matplotlib
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
# 1. Population (synthetic, strongly right-skewed)
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
population = rng.exponential(scale=10, size=100_000)
pop_mean = population.mean()
pop_sd = population.std(ddof=0)
pop_skew = stats.skew(population)

print("=== Population ===")
print(f"Size = {population.size}, mean = {pop_mean:.3f}, sd = {pop_sd:.3f}, "
      f"skewness = {pop_skew:.3f}")

# ---------------------------------------------------------------
# 2. Sampling distribution of the mean for several sample sizes
# ---------------------------------------------------------------
N_SAMPLES = 5000
sample_sizes = [5, 30, 100]
means = {}
rows = []

print("\n=== Sampling distribution of the sample mean (5000 samples each) ===")
for n in sample_sizes:
    draws = rng.choice(population, size=(N_SAMPLES, n), replace=True)
    xbar = draws.mean(axis=1)
    means[n] = xbar
    se_theory = pop_sd / np.sqrt(n)
    se_emp = xbar.std(ddof=1)
    skew = stats.skew(xbar)
    sw = stats.shapiro(xbar)
    print(f"n = {n:3d}: mean of means = {xbar.mean():.3f}, "
          f"empirical SE = {se_emp:.3f}, theoretical SE = {se_theory:.3f}, "
          f"skewness = {skew:.3f}, Shapiro p = {sw.pvalue:.4f}")
    rows.append({
        "n": n,
        "mean_of_sample_means": xbar.mean(),
        "empirical_SE": se_emp,
        "theoretical_SE": se_theory,
        "skewness": skew,
        "shapiro_p": sw.pvalue,
    })

results = pd.DataFrame(rows).round(4)
results.to_csv(OUT / "clt_results.csv", index=False)
pd.DataFrame({f"n_{n}": means[n] for n in sample_sizes}).round(4).to_csv(
    DATA / "sample_means.csv", index=False)

# ---------------------------------------------------------------
# 3. Standard error as sample size grows
# ---------------------------------------------------------------
n_grid = [2, 5, 10, 30, 50, 100, 200, 500]
se_emp_grid = []
for n in n_grid:
    xbar = rng.choice(population, size=(2000, n), replace=True).mean(axis=1)
    se_emp_grid.append(xbar.std(ddof=1))
se_theory_grid = pop_sd / np.sqrt(n_grid)

print("\n=== Standard error vs sample size ===")
for n, e, t in zip(n_grid, se_emp_grid, se_theory_grid):
    print(f"n = {n:3d}: empirical SE = {e:.3f}, theoretical SE = {t:.3f}")

# ---------------------------------------------------------------
# 4. Using the CLT to calculate a probability (n = 30)
# ---------------------------------------------------------------
n = 30
threshold = 12
se30 = pop_sd / np.sqrt(n)
p_clt = stats.norm.sf(threshold, loc=pop_mean, scale=se30)
p_emp = np.mean(means[30] > threshold)
print(f"\n=== P(sample mean > {threshold}) for n = {n} ===")
print(f"CLT normal approximation = {p_clt:.4f}")
print(f"Simulated proportion     = {p_emp:.4f}")

# ---------------------------------------------------------------
# 5. One real-world style sample: standard error from a single sample
# ---------------------------------------------------------------
one_sample = rng.choice(population, size=50, replace=False)
se_one = one_sample.std(ddof=1) / np.sqrt(50)
print("\n=== One sample of size 50 ===")
print(f"Sample mean = {one_sample.mean():.3f}, estimated SE = {se_one:.3f} "
      f"(population mean = {pop_mean:.3f})")

# ---------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------
# Plot 1: population
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(population, bins=60, density=True, color="grey", alpha=0.7)
ax.axvline(pop_mean, color="red", linestyle="--", label=f"Mean = {pop_mean:.2f}")
ax.set_title("Population distribution (right-skewed)")
ax.set_xlabel("Value")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "population.png", dpi=150)

# Plot 2: sampling distributions with normal curve
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
for ax, n in zip(axes, sample_sizes):
    xbar = means[n]
    ax.hist(xbar, bins=40, density=True, color="steelblue", alpha=0.7)
    x = np.linspace(xbar.min(), xbar.max(), 300)
    ax.plot(x, stats.norm.pdf(x, pop_mean, pop_sd / np.sqrt(n)), "r-", lw=2,
            label="Normal (CLT)")
    ax.set_title(f"Sample means, n = {n}")
    ax.set_xlabel("Sample mean")
    ax.legend()
fig.suptitle("Central Limit Theorem: sampling distribution of the mean", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "clt_sampling_distributions.png", dpi=150)

# Plot 3: Q-Q plots of the sample means
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, n in zip(axes, sample_sizes):
    stats.probplot(means[n], dist="norm", plot=ax)
    ax.set_title(f"Q-Q plot of sample means, n = {n}")
fig.tight_layout()
fig.savefig(OUT / "clt_qq_plots.png", dpi=150)

# Plot 4: standard error vs n
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_grid, se_theory_grid, "r-", label="Theoretical: sigma / sqrt(n)")
ax.plot(n_grid, se_emp_grid, "bo--", label="Empirical (simulated)")
ax.set_xlabel("Sample size n")
ax.set_ylabel("Standard error of the mean")
ax.set_title("Standard error decreases as sample size increases")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "standard_error_vs_n.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)