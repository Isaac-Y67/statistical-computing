"""
Topic 6: Hypothesis Testing
Libraries: numpy, pandas, scipy.stats, statsmodels, matplotlib
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportions_ztest

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

ALPHA = 0.05


def decision(p):
    return "Reject H0" if p < ALPHA else "Fail to reject H0"


# ---------------------------------------------------------------
# 1. Dataset (synthetic, reproducible)
# ---------------------------------------------------------------
rng = np.random.default_rng(42)

bottles = rng.normal(497, 6, 35).round(1)

new_method = rng.normal(72, 10, 40).round(1)
old_method = rng.normal(66, 11, 40).round(1)

before = rng.normal(60, 9, 30).round(1)
after = (before + rng.normal(4, 5, 30)).round(1)

conversions = int((rng.random(500) < 0.12).sum())
visitors = 500
a_conv = int(rng.binomial(1000, 0.10))
b_conv = int(rng.binomial(1000, 0.13))

pd.DataFrame({"bottle_ml": bottles}).to_csv(DATA / "bottles.csv", index=False)
pd.DataFrame({"new_method": new_method, "old_method": old_method}).to_csv(
    DATA / "teaching_methods.csv", index=False)
pd.DataFrame({"before": before, "after": after}).to_csv(
    DATA / "training_scores.csv", index=False)

rows = []  # (test, statistic, p_value, decision)

# ---------------------------------------------------------------
# 2. One-sample t-test: H0 mu = 500
# ---------------------------------------------------------------
n1 = bottles.size
res = stats.ttest_1samp(bottles, popmean=500)
se1 = bottles.std(ddof=1) / np.sqrt(n1)
t_crit = stats.t.ppf(1 - ALPHA / 2, n1 - 1)
ci_lo, ci_hi = bottles.mean() - t_crit * se1, bottles.mean() + t_crit * se1
print("=== 1. One-sample t-test (bottle fill, H0: mean = 500 ml) ===")
print(f"n = {n1}, mean = {bottles.mean():.3f}, sd = {bottles.std(ddof=1):.3f}")
print(f"t = {res.statistic:.3f}, df = {n1 - 1}, p = {res.pvalue:.4f}, "
      f"critical t = +/-{t_crit:.3f}")
print(f"95% CI for the mean: ({ci_lo:.3f}, {ci_hi:.3f})")
print(f"Decision: {decision(res.pvalue)}")
rows.append(("One-sample t", res.statistic, res.pvalue, decision(res.pvalue)))

# ---------------------------------------------------------------
# 3. Two-sample Welch t-test: H0 mu_new = mu_old
# ---------------------------------------------------------------
sw_new = stats.shapiro(new_method)
sw_old = stats.shapiro(old_method)
lev = stats.levene(new_method, old_method)
res2 = stats.ttest_ind(new_method, old_method, equal_var=False)
sp = np.sqrt(((new_method.size - 1) * new_method.var(ddof=1)
              + (old_method.size - 1) * old_method.var(ddof=1))
             / (new_method.size + old_method.size - 2))
d2 = (new_method.mean() - old_method.mean()) / sp
print("\n=== 2. Two-sample Welch t-test (new vs old teaching method) ===")
print(f"New: mean = {new_method.mean():.3f}, sd = {new_method.std(ddof=1):.3f} | "
      f"Old: mean = {old_method.mean():.3f}, sd = {old_method.std(ddof=1):.3f}")
print(f"Shapiro p (new) = {sw_new.pvalue:.4f}, Shapiro p (old) = {sw_old.pvalue:.4f}, "
      f"Levene p = {lev.pvalue:.4f}")
print(f"t = {res2.statistic:.3f}, p = {res2.pvalue:.4f}, Cohen's d = {d2:.3f}")
print(f"Decision: {decision(res2.pvalue)}")
rows.append(("Two-sample Welch t", res2.statistic, res2.pvalue, decision(res2.pvalue)))

# Non-parametric check
mw = stats.mannwhitneyu(new_method, old_method, alternative="two-sided")
print(f"Mann-Whitney U = {mw.statistic:.1f}, p = {mw.pvalue:.4f} "
      f"(non-parametric check)")
rows.append(("Mann-Whitney U", mw.statistic, mw.pvalue, decision(mw.pvalue)))

# ---------------------------------------------------------------
# 4. Paired t-test: H0 mean difference = 0
# ---------------------------------------------------------------
diff = after - before
res3 = stats.ttest_rel(after, before)
d3 = diff.mean() / diff.std(ddof=1)
sw_diff = stats.shapiro(diff)
print("\n=== 3. Paired t-test (before vs after training) ===")
print(f"n = {diff.size}, mean before = {before.mean():.3f}, "
      f"mean after = {after.mean():.3f}")
print(f"Mean difference = {diff.mean():.3f}, sd of differences = "
      f"{diff.std(ddof=1):.3f}, Shapiro p (differences) = {sw_diff.pvalue:.4f}")
print(f"t = {res3.statistic:.3f}, df = {diff.size - 1}, p = {res3.pvalue:.4f}, "
      f"Cohen's d = {d3:.3f}")
print(f"Decision: {decision(res3.pvalue)}")
rows.append(("Paired t", res3.statistic, res3.pvalue, decision(res3.pvalue)))

# ---------------------------------------------------------------
# 5. One-proportion z-test: H0 p = 0.10
# ---------------------------------------------------------------
z4, p4 = proportions_ztest(conversions, visitors, value=0.10, prop_var=0.10)
print("\n=== 4. One-proportion z-test (conversion rate, H0: p = 0.10) ===")
print(f"{conversions} conversions out of {visitors}, "
      f"p-hat = {conversions / visitors:.4f}")
print(f"z = {z4:.3f}, p = {p4:.4f}")
print(f"Decision: {decision(p4)}")
rows.append(("One-proportion z", z4, p4, decision(p4)))

# ---------------------------------------------------------------
# 6. Two-proportion z-test: H0 p_A = p_B
# ---------------------------------------------------------------
z5, p5 = proportions_ztest([b_conv, a_conv], [1000, 1000])
print("\n=== 5. Two-proportion z-test (page B vs page A) ===")
print(f"Page A: {a_conv}/1000 = {a_conv / 1000:.3f} | "
      f"Page B: {b_conv}/1000 = {b_conv / 1000:.3f}")
print(f"z = {z5:.3f}, p = {p5:.4f}")
print(f"Decision: {decision(p5)}")
rows.append(("Two-proportion z", z5, p5, decision(p5)))

# ---------------------------------------------------------------
# 7. Power analysis
# ---------------------------------------------------------------
power_calc = TTestIndPower()
power_now = power_calc.power(effect_size=abs(d2), nobs1=new_method.size, alpha=ALPHA)
n_needed = power_calc.solve_power(effect_size=0.5, power=0.80, alpha=ALPHA)
print("\n=== Power analysis (two-sample t-test) ===")
print(f"Power with observed d = {abs(d2):.3f} and n = {new_method.size} per group: "
      f"{power_now:.3f}")
print(f"Sample size per group for 80% power at d = 0.5: {np.ceil(n_needed):.0f}")

# ---------------------------------------------------------------
# 8. Simulation: Type I error and power
# ---------------------------------------------------------------
N_SIM, N = 5000, 30
x0 = rng.normal(50, 10, (N_SIM, N))
y0 = rng.normal(50, 10, (N_SIM, N))
p_null = stats.ttest_ind(x0, y0, axis=1, equal_var=False).pvalue
x1 = rng.normal(55, 10, (N_SIM, N))
y1 = rng.normal(50, 10, (N_SIM, N))
p_alt = stats.ttest_ind(x1, y1, axis=1, equal_var=False).pvalue
type1 = np.mean(p_null < ALPHA)
power_sim = np.mean(p_alt < ALPHA)
power_theory = power_calc.power(effect_size=0.5, nobs1=N, alpha=ALPHA)
print(f"\n=== Simulation ({N_SIM} experiments, n = {N} per group) ===")
print(f"Type I error rate when H0 is true     = {type1:.4f} (nominal {ALPHA})")
print(f"Power when true difference = 5 (d=0.5) = {power_sim:.4f} "
      f"(theoretical {power_theory:.4f})")

# ---------------------------------------------------------------
# 9. Save results
# ---------------------------------------------------------------
pd.DataFrame(rows, columns=["test", "statistic", "p_value", "decision"]
             ).round(4).to_csv(OUT / "hypothesis_test_results.csv", index=False)

# ---------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------
# Plot 1: t distribution with rejection regions (one-sample test)
fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(-5, 5, 500)
ax.plot(x, stats.t.pdf(x, n1 - 1), "k-")
ax.fill_between(x, stats.t.pdf(x, n1 - 1), where=(x <= -t_crit) | (x >= t_crit),
                color="tomato", alpha=0.5, label="Rejection region (alpha = 0.05)")
ax.axvline(res.statistic, color="blue", linestyle="--",
           label=f"Observed t = {res.statistic:.2f}")
ax.set_xlabel("t")
ax.set_title("One-sample t-test: bottle fill (H0: mean = 500 ml)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "t_test_rejection_region.png", dpi=150)

# Plot 2: group comparison
fig, ax = plt.subplots(figsize=(7, 5))
ax.boxplot([new_method, old_method], tick_labels=["New method", "Old method"])
ax.set_ylabel("Score")
ax.set_title(f"Teaching methods (Welch t-test p = {res2.pvalue:.4f})")
fig.tight_layout()
fig.savefig(OUT / "group_comparison.png", dpi=150)

# Plot 3: paired differences
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(diff, bins=12, color="steelblue", alpha=0.7)
ax.axvline(0, color="red", linestyle="--", label="No change")
ax.axvline(diff.mean(), color="black", label=f"Mean = {diff.mean():.2f}")
ax.set_xlabel("After minus before")
ax.set_title("Paired differences in scores")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "paired_differences.png", dpi=150)

# Plot 4: p-value distributions
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
axes[0].hist(p_null, bins=20, color="grey")
axes[0].set_title(f"p-values when H0 is true\n(rejected {type1:.1%} of the time)")
axes[1].hist(p_alt, bins=20, color="seagreen")
axes[1].set_title(f"p-values when H0 is false\n(power = {power_sim:.1%})")
for ax in axes:
    ax.axvline(ALPHA, color="red", linestyle="--")
    ax.set_xlabel("p-value")
fig.tight_layout()
fig.savefig(OUT / "p_value_simulation.png", dpi=150)

# Plot 5: power curve
n_grid = np.arange(5, 151, 5)
fig, ax = plt.subplots(figsize=(8, 5))
for d_val in (0.2, 0.5, 0.8):
    ax.plot(n_grid, power_calc.power(effect_size=d_val, nobs1=n_grid, alpha=ALPHA),
            label=f"d = {d_val}")
ax.axhline(0.8, color="grey", linestyle=":")
ax.set_xlabel("Sample size per group")
ax.set_ylabel("Power")
ax.set_title("Power of the two-sample t-test")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "power_curve.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)