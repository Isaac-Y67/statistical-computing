"""
Topic 5: Confidence Intervals
Libraries: numpy, pandas, scipy.stats, statsmodels, matplotlib
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# 1. Dataset (synthetic, reproducible)
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
n = 40
TRUE_MEAN_A, TRUE_SD_A = 32, 6
a = rng.normal(TRUE_MEAN_A, TRUE_SD_A, n).round(1)
b = rng.normal(35, 7, n).round(1)
pd.DataFrame({"courier_a_minutes": a, "courier_b_minutes": b}).to_csv(
    DATA / "delivery_times.csv", index=False)

n_survey = 200
satisfied = (rng.random(n_survey) < 0.68).astype(int)
pd.DataFrame({"respondent": range(1, n_survey + 1), "satisfied": satisfied}).to_csv(
    DATA / "survey.csv", index=False)

rows = []  # (parameter, method, estimate, lower, upper)

# ---------------------------------------------------------------
# 2. CI for a mean (t-interval), courier A
# ---------------------------------------------------------------
xbar = a.mean()
s = a.std(ddof=1)
se = s / np.sqrt(n)
print("=== Courier A delivery time ===")
print(f"n = {n}, mean = {xbar:.3f}, sd = {s:.3f}, SE = {se:.3f}")

for level in (0.90, 0.95, 0.99):
    t_crit = stats.t.ppf(1 - (1 - level) / 2, df=n - 1)
    lo, hi = xbar - t_crit * se, xbar + t_crit * se
    print(f"{int(level * 100)}% t-interval: ({lo:.3f}, {hi:.3f}), "
          f"t* = {t_crit:.3f}, width = {hi - lo:.3f}")
    rows.append(("Mean (courier A)", f"t, {int(level * 100)}%", xbar, lo, hi))

z_crit = stats.norm.ppf(0.975)
z_lo, z_hi = xbar - z_crit * se, xbar + z_crit * se
print(f"95% z-interval (for comparison): ({z_lo:.3f}, {z_hi:.3f})")
rows.append(("Mean (courier A)", "z, 95%", xbar, z_lo, z_hi))

# ---------------------------------------------------------------
# 3. CI for a proportion
# ---------------------------------------------------------------
count = int(satisfied.sum())
p_hat = count / n_survey
w_lo, w_hi = proportion_confint(count, n_survey, alpha=0.05, method="normal")
s_lo, s_hi = proportion_confint(count, n_survey, alpha=0.05, method="wilson")
print("\n=== Survey: proportion satisfied ===")
print(f"{count} of {n_survey} satisfied, p-hat = {p_hat:.4f}")
print(f"95% Wald interval:   ({w_lo:.4f}, {w_hi:.4f})")
print(f"95% Wilson interval: ({s_lo:.4f}, {s_hi:.4f})")
rows += [
    ("Proportion satisfied", "Wald, 95%", p_hat, w_lo, w_hi),
    ("Proportion satisfied", "Wilson, 95%", p_hat, s_lo, s_hi),
]

# ---------------------------------------------------------------
# 4. CI for the variance and standard deviation (chi-square)
# ---------------------------------------------------------------
var = s ** 2
v_lo = (n - 1) * var / stats.chi2.ppf(0.975, n - 1)
v_hi = (n - 1) * var / stats.chi2.ppf(0.025, n - 1)
print("\n=== Variability of courier A ===")
print(f"Sample variance = {var:.3f}, 95% CI = ({v_lo:.3f}, {v_hi:.3f})")
print(f"Sample SD       = {s:.3f}, 95% CI = ({np.sqrt(v_lo):.3f}, {np.sqrt(v_hi):.3f})")
rows += [
    ("Variance (courier A)", "chi-square, 95%", var, v_lo, v_hi),
    ("SD (courier A)", "chi-square, 95%", s, np.sqrt(v_lo), np.sqrt(v_hi)),
]

# ---------------------------------------------------------------
# 5. CI for a difference in means (Welch), A minus B
# ---------------------------------------------------------------
va, vb = a.var(ddof=1), b.var(ddof=1)
diff = a.mean() - b.mean()
se_diff = np.sqrt(va / n + vb / n)
df_w = (va / n + vb / n) ** 2 / ((va / n) ** 2 / (n - 1) + (vb / n) ** 2 / (n - 1))
t_w = stats.t.ppf(0.975, df_w)
d_lo, d_hi = diff - t_w * se_diff, diff + t_w * se_diff
print("\n=== Difference in means (A - B), Welch ===")
print(f"Mean A = {a.mean():.3f}, mean B = {b.mean():.3f}")
print(f"Difference = {diff:.3f}, SE = {se_diff:.3f}, df = {df_w:.1f}")
print(f"95% CI = ({d_lo:.3f}, {d_hi:.3f})")
print("Interval contains 0:", d_lo <= 0 <= d_hi)
rows.append(("Mean difference A - B", "Welch t, 95%", diff, d_lo, d_hi))

# ---------------------------------------------------------------
# 6. Coverage simulation: does a 95% CI capture the true mean 95% of the time?
# ---------------------------------------------------------------
N_SIM = 2000
sims = rng.normal(TRUE_MEAN_A, TRUE_SD_A, size=(N_SIM, n))
m = sims.mean(axis=1)
sd = sims.std(axis=1, ddof=1)
print(f"\n=== Coverage simulation ({N_SIM} samples of n = {n}) ===")
coverage = {}
for level in (0.90, 0.95, 0.99):
    t_c = stats.t.ppf(1 - (1 - level) / 2, df=n - 1)
    lo_s, hi_s = m - t_c * sd / np.sqrt(n), m + t_c * sd / np.sqrt(n)
    covered = (lo_s <= TRUE_MEAN_A) & (TRUE_MEAN_A <= hi_s)
    coverage[level] = covered.mean()
    print(f"{int(level * 100)}% intervals: {covered.mean():.4f} captured the true mean")
    rows.append(("Coverage simulation", f"{int(level * 100)}%", covered.mean(),
                 np.nan, np.nan))
    if level == 0.95:
        lo95, hi95, cov95 = lo_s, hi_s, covered

# ---------------------------------------------------------------
# 7. Save results
# ---------------------------------------------------------------
pd.DataFrame(rows, columns=["parameter", "method", "estimate", "lower", "upper"]
             ).round(4).to_csv(OUT / "confidence_intervals.csv", index=False)

# ---------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------
# Plot 1: 50 simulated 95% intervals
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(50):
    colour = "seagreen" if cov95[i] else "red"
    ax.plot([lo95[i], hi95[i]], [i, i], color=colour, lw=2)
    ax.plot(m[i], i, "o", color=colour, ms=3)
ax.axvline(TRUE_MEAN_A, color="black", linestyle="--", label=f"True mean = {TRUE_MEAN_A}")
ax.set_xlabel("Delivery time (minutes)")
ax.set_ylabel("Sample number")
ax.set_title("50 simulated 95% confidence intervals (red = missed the true mean)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "ci_simulation.png", dpi=150)

# Plot 2: interval width vs sample size
n_grid = np.arange(5, 301, 5)
width = 2 * stats.t.ppf(0.975, n_grid - 1) * TRUE_SD_A / np.sqrt(n_grid)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_grid, width, color="steelblue")
ax.set_xlabel("Sample size n")
ax.set_ylabel("Width of 95% interval (minutes)")
ax.set_title("Larger samples give narrower intervals")
fig.tight_layout()
fig.savefig(OUT / "ci_width_vs_n.png", dpi=150)

# Plot 3: interval for the mean at three confidence levels
fig, ax = plt.subplots(figsize=(8, 4))
for i, level in enumerate((0.90, 0.95, 0.99)):
    t_c = stats.t.ppf(1 - (1 - level) / 2, df=n - 1)
    ax.errorbar(xbar, i, xerr=t_c * se, fmt="o", capsize=6, color="steelblue")
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["90%", "95%", "99%"])
ax.axvline(TRUE_MEAN_A, color="black", linestyle="--", label="True mean")
ax.set_xlabel("Mean delivery time (minutes)")
ax.set_title("Higher confidence gives a wider interval")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "ci_levels.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)