"""
Topic 20: Bootstrap and Resampling Methods
Libraries: numpy, scipy.stats, scikit-learn (resample), pandas, matplotlib, seaborn
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.utils import resample

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
sns.set_theme(style="whitegrid")
rng = np.random.default_rng(42)
ALPHA = 0.05
B = 10_000

# ---------------------------------------------------------------
# 1. Dataset: skewed incomes
# ---------------------------------------------------------------
MU, SIGMA = 10.5, 0.6
x = rng.lognormal(MU, SIGMA, 40).round(0)
n = x.size
pd.DataFrame({"income": x}).to_csv(DATA / "incomes.csv", index=False)
true_mean, true_median = np.exp(MU + SIGMA ** 2 / 2), np.exp(MU)

print("=== 1. Dataset: incomes of 40 customers ===")
print(f"Sample mean = {x.mean():.0f}, median = {np.median(x):.0f}, "
      f"sd = {x.std(ddof=1):.0f}, skewness = {stats.skew(x):.3f}")
print(f"True population mean = {true_mean:.0f}, true median = {true_median:.0f}")

# ---------------------------------------------------------------
# 2. Bootstrap for several statistics
# ---------------------------------------------------------------
idx = rng.integers(0, n, size=(B, n))
bs = x[idx]
funcs = {
    "Mean": (lambda a: a.mean(axis=1), x.mean()),
    "Median": (lambda a: np.median(a, axis=1), np.median(x)),
    "20% trimmed mean": (lambda a: stats.trim_mean(a, 0.2, axis=1),
                         stats.trim_mean(x, 0.2)),
    "Standard deviation": (lambda a: a.std(axis=1, ddof=1), x.std(ddof=1)),
    "Coefficient of variation": (lambda a: a.std(axis=1, ddof=1) / a.mean(axis=1),
                                 x.std(ddof=1) / x.mean()),
}
boot = {}
rows = []
for name, (f, est) in funcs.items():
    d = f(bs)
    boot[name] = d
    lo, hi = np.percentile(d, [2.5, 97.5])
    rows.append((name, est, d.std(ddof=1), d.mean() - est, lo, hi))
tab = pd.DataFrame(rows, columns=["statistic", "estimate", "boot_se", "boot_bias",
                                  "ci_lower", "ci_upper"])
print(f"\n=== 2. Bootstrap ({B:,d} resamples of size {n}) ===")
print(tab.round(3).to_string(index=False))
tab.round(5).to_csv(OUT / "bootstrap_estimates.csv", index=False)

se_formula = x.std(ddof=1) / np.sqrt(n)
print(f"\nStandard error of the mean: bootstrap = {boot['Mean'].std(ddof=1):.1f}, "
      f"formula s / sqrt(n) = {se_formula:.1f}")
print("Bootstrap standard error of the median has no simple formula: "
      f"{boot['Median'].std(ddof=1):.1f}")

med_sk = [np.median(resample(x, replace=True, n_samples=n, random_state=i))
          for i in range(2000)]
print(f"scikit-learn resample, 2,000 resamples: SE of the median = "
      f"{np.std(med_sk, ddof=1):.1f}")

# ---------------------------------------------------------------
# 3. Interval methods
# ---------------------------------------------------------------
ci_rows = []
for name, est in (("Mean", x.mean()), ("Median", np.median(x))):
    d = boot[name]
    q_lo, q_hi = np.percentile(d, [2.5, 97.5])
    se_b = d.std(ddof=1)
    methods = {
        "Percentile": (q_lo, q_hi),
        "Basic": (2 * est - q_hi, 2 * est - q_lo),
        "Normal with bootstrap SE": (est - 1.96 * se_b, est + 1.96 * se_b),
    }
    fn = np.mean if name == "Mean" else np.median
    bca = stats.bootstrap((x,), fn, n_resamples=9999, confidence_level=0.95,
                          method="BCa", random_state=42).confidence_interval
    methods["BCa (scipy)"] = (float(bca.low), float(bca.high))
    if name == "Mean":
        tc = stats.t.ppf(0.975, n - 1)
        methods["Classical t"] = (est - tc * se_formula, est + tc * se_formula)
    for m, (lo, hi) in methods.items():
        ci_rows.append((name, m, lo, hi, hi - lo))
ci_tab = pd.DataFrame(ci_rows, columns=["statistic", "method", "lower", "upper",
                                        "width"])
print("\n=== 3. 95% confidence intervals by method ===")
print(ci_tab.round(1).to_string(index=False))
print(f"(true mean = {true_mean:.0f}, true median = {true_median:.0f})")
ci_tab.round(3).to_csv(OUT / "ci_comparison.csv", index=False)

# ---------------------------------------------------------------
# 4. Number of resamples needed
# ---------------------------------------------------------------
print("\n=== 4. Bootstrap standard error of the median against the number of resamples ===")
b_grid = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
se_b = [boot["Median"][:b].std(ddof=1) for b in b_grid]
for b, s in zip(b_grid, se_b):
    print(f"B = {b:6,d}: SE = {s:.1f}")

# ---------------------------------------------------------------
# 5. Jackknife
# ---------------------------------------------------------------
def jackknife(a, func):
    m = len(a)
    th = np.array([func(np.delete(a, i)) for i in range(m)])
    se = np.sqrt((m - 1) / m * np.sum((th - th.mean()) ** 2))
    bias = (m - 1) * (th.mean() - func(a))
    return se, bias


print("\n=== 5. Jackknife (leave one out) ===")
jk_rows = []
for name, fn in (("Mean", np.mean), ("Median", np.median),
                 ("Standard deviation", lambda a: a.std(ddof=1))):
    se_j, bias_j = jackknife(x, fn)
    jk_rows.append((name, se_j, bias_j, boot[name].std(ddof=1)))
    print(f"{name:19s}: jackknife SE = {se_j:.2f}, jackknife bias = {bias_j:.3f}, "
          f"bootstrap SE = {boot[name].std(ddof=1):.2f}")
print("Jackknife SE of the mean equals s / sqrt(n) exactly:",
      np.isclose(jk_rows[0][1], se_formula))
pd.DataFrame(jk_rows, columns=["statistic", "jackknife_se", "jackknife_bias",
                               "bootstrap_se"]).round(4).to_csv(
    OUT / "jackknife.csv", index=False)

# ---------------------------------------------------------------
# 6. Two groups: difference in medians
# ---------------------------------------------------------------
ga = rng.lognormal(np.log(50), 0.5, 30).round(1)
gb = rng.lognormal(np.log(65), 0.5, 30).round(1)
pd.DataFrame({"group_a": ga, "group_b": gb}).to_csv(DATA / "two_groups.csv",
                                                    index=False)
obs_diff = np.median(gb) - np.median(ga)
ia = rng.integers(0, 30, (B, 30))
ib = rng.integers(0, 30, (B, 30))
d_boot = np.median(gb[ib], axis=1) - np.median(ga[ia], axis=1)
dlo, dhi = np.percentile(d_boot, [2.5, 97.5])


def med_diff(u, v, axis):
    return np.median(u, axis=axis) - np.median(v, axis=axis)


perm = stats.permutation_test((gb, ga), med_diff, vectorized=True,
                              n_resamples=9999, alternative="two-sided",
                              random_state=42)
mw = stats.mannwhitneyu(gb, ga)
print("\n=== 6. Two groups (30 each): difference in medians, B minus A ===")
print(f"Median A = {np.median(ga):.1f}, median B = {np.median(gb):.1f}, "
      f"observed difference = {obs_diff:.2f}")
print(f"Bootstrap SE = {d_boot.std(ddof=1):.2f}, 95% percentile interval = "
      f"({dlo:.2f}, {dhi:.2f}), contains 0: {dlo <= 0 <= dhi}")
print(f"Permutation test of 'no difference' (9,999 resamples): p = {perm.pvalue:.4f}")
print(f"Mann-Whitney test: p = {mw.pvalue:.4f}")

# ---------------------------------------------------------------
# 7. Correlation
# ---------------------------------------------------------------
n_c, rho = 50, 0.6
cx = rng.standard_normal(n_c)
cy = rho * cx + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_c)
pd.DataFrame({"x": cx, "y": cy}).round(4).to_csv(DATA / "correlation_pairs.csv",
                                                 index=False)
r_obs = np.corrcoef(cx, cy)[0, 1]
ic = rng.integers(0, n_c, (B, n_c))
xs, ys = cx[ic], cy[ic]
xm, ym = xs - xs.mean(axis=1, keepdims=True), ys - ys.mean(axis=1, keepdims=True)
r_boot = (xm * ym).sum(axis=1) / np.sqrt((xm ** 2).sum(axis=1) * (ym ** 2).sum(axis=1))
rlo, rhi = np.percentile(r_boot, [2.5, 97.5])
z, se_z = np.arctanh(r_obs), 1 / np.sqrt(n_c - 3)
flo, fhi = np.tanh(z - 1.96 * se_z), np.tanh(z + 1.96 * se_z)
print(f"\n=== 7. Correlation (n = {n_c}, true correlation {rho}) ===")
print(f"Sample r = {r_obs:.3f}, bootstrap SE = {r_boot.std(ddof=1):.3f}, "
      f"bootstrap skewness = {stats.skew(r_boot):.3f}")
print(f"Bootstrap percentile interval = ({rlo:.3f}, {rhi:.3f})")
print(f"Fisher z interval             = ({flo:.3f}, {fhi:.3f})")

# ---------------------------------------------------------------
# 8. Regression with unequal variance
# ---------------------------------------------------------------
n_r = 60
rx = rng.uniform(1, 10, n_r)
ry = 2 + 1.5 * rx + rng.standard_normal(n_r) * 0.5 * rx
pd.DataFrame({"x": rx.round(3), "y": ry.round(3)}).to_csv(DATA / "regression_data.csv",
                                                          index=False)
sxx = np.sum((rx - rx.mean()) ** 2)
slope = np.sum((rx - rx.mean()) * (ry - ry.mean())) / sxx
icpt = ry.mean() - slope * rx.mean()
fit = icpt + slope * rx
res = ry - fit
se_ols = np.sqrt(np.sum(res ** 2) / (n_r - 2) / sxx)
tc = stats.t.ppf(0.975, n_r - 2)
ir = rng.integers(0, n_r, (B, n_r))
xp, yp = rx[ir], ry[ir]
xpm, ypm = xp - xp.mean(axis=1, keepdims=True), yp - yp.mean(axis=1, keepdims=True)
slope_pairs = (xpm * ypm).sum(axis=1) / (xpm ** 2).sum(axis=1)
yres = fit + res[ir]
slope_resid = ((rx - rx.mean()) * (yres - yres.mean(axis=1, keepdims=True))
               ).sum(axis=1) / sxx
reg_rows = [
    ("Classical OLS (t interval)", se_ols, slope - tc * se_ols, slope + tc * se_ols),
    ("Residual bootstrap", slope_resid.std(ddof=1),
     *np.percentile(slope_resid, [2.5, 97.5])),
    ("Pairs bootstrap", slope_pairs.std(ddof=1), *np.percentile(slope_pairs, [2.5, 97.5])),
]
reg_tab = pd.DataFrame(reg_rows, columns=["method", "se_slope", "lower_95", "upper_95"])
print("\n=== 8. Regression slope when the noise grows with x (true slope 1.5) ===")
print(f"OLS slope = {slope:.4f}, intercept = {icpt:.4f}")
print(reg_tab.round(4).to_string(index=False))
reg_tab.round(5).to_csv(OUT / "regression_slope.csv", index=False)
print(f"Residual spread: sd of residuals for x < 5.5 = {res[rx < 5.5].std(ddof=1):.2f}, "
      f"for x >= 5.5 = {res[rx >= 5.5].std(ddof=1):.2f}")

# ---------------------------------------------------------------
# 9. Coverage simulation
# ---------------------------------------------------------------
S, B_S = 500, 500
true_m = np.exp(0.5)
cov_rows = []
print(f"\n=== 9. Coverage of 95% intervals for the mean of lognormal data "
      f"({S} samples, {B_S} resamples each) ===")
for n_s in (10, 20, 50):
    tcv = stats.t.ppf(0.975, n_s - 1)
    hit_t = hit_p = 0
    w_t, w_p = [], []
    for _ in range(S):
        s = rng.lognormal(0, 1, n_s)
        m_, se_ = s.mean(), s.std(ddof=1) / np.sqrt(n_s)
        lt, ut = m_ - tcv * se_, m_ + tcv * se_
        bm = s[rng.integers(0, n_s, (B_S, n_s))].mean(axis=1)
        lp, up = np.percentile(bm, [2.5, 97.5])
        hit_t += lt <= true_m <= ut
        hit_p += lp <= true_m <= up
        w_t.append(ut - lt)
        w_p.append(up - lp)
    cov_rows.append((n_s, hit_t / S, hit_p / S, np.mean(w_t), np.mean(w_p)))
    print(f"n = {n_s:2d}: t-interval covers {hit_t / S:.3f} (mean width "
          f"{np.mean(w_t):.2f}); percentile bootstrap covers {hit_p / S:.3f} "
          f"(mean width {np.mean(w_p):.2f})")
cov = pd.DataFrame(cov_rows, columns=["n", "coverage_t", "coverage_bootstrap",
                                      "width_t", "width_bootstrap"])
cov.round(4).to_csv(OUT / "coverage_simulation.csv", index=False)
print("Standard error of each coverage figure is about 0.010")

# ---------------------------------------------------------------
# 10. When the bootstrap fails: the sample maximum
# ---------------------------------------------------------------
n_m = 50
u = rng.uniform(0, 1, n_m)
bmax = u[rng.integers(0, n_m, (B, n_m))].max(axis=1)
same = np.mean(bmax == u.max())
print("\n=== 10. Bootstrap of the maximum of Uniform(0, 1), n = 50 ===")
print(f"Sample maximum = {u.max():.4f} (true upper limit = 1)")
print(f"Share of bootstrap maxima equal to the sample maximum = {same:.3f} "
      f"(theory 1 - (1 - 1/n)^n = {1 - (1 - 1 / n_m) ** n_m:.3f})")
print(f"Largest bootstrap maximum = {bmax.max():.4f}; the true limit 1 can never "
      f"appear in a bootstrap sample")

# ---------------------------------------------------------------
# 11. Plots
# ---------------------------------------------------------------
# Plot 1: sampling distribution of the mean
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(x, bins=15, color="grey", alpha=0.7)
axes[0].axvline(x.mean(), color="red", label="Mean")
axes[0].axvline(np.median(x), color="green", label="Median")
axes[0].set_title("The original sample (right-skewed)")
axes[0].set_xlabel("Income")
axes[0].legend()
d = boot["Mean"]
axes[1].hist(d, bins=50, density=True, color="steelblue", alpha=0.7)
g = np.linspace(d.min(), d.max(), 300)
axes[1].plot(g, stats.norm.pdf(g, x.mean(), se_formula), "r-", lw=2,
             label="Normal with SE = s / sqrt(n)")
lo, hi = np.percentile(d, [2.5, 97.5])
axes[1].axvline(lo, color="black", linestyle="--")
axes[1].axvline(hi, color="black", linestyle="--", label="Percentile 95% interval")
axes[1].axvline(true_mean, color="green", lw=2, label="True mean")
axes[1].set_title(f"Bootstrap distribution of the mean ({B:,d} resamples)")
axes[1].set_xlabel("Resampled mean")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "bootstrap_distribution_mean.png", dpi=150)

# Plot 2: mean against median
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, name, colour, truth in ((axes[0], "Mean", "steelblue", true_mean),
                                (axes[1], "Median", "seagreen", true_median)):
    ax.hist(boot[name], bins=50, color=colour, alpha=0.7)
    ax.axvline(truth, color="red", lw=2, label="True value")
    ax.set_title(f"Bootstrap {name.lower()} (SE = {boot[name].std(ddof=1):.0f}, "
                 f"skewness = {stats.skew(boot[name]):.2f})")
    ax.legend()
fig.tight_layout()
fig.savefig(OUT / "bootstrap_mean_vs_median.png", dpi=150)

# Plot 3: interval comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, name, truth in ((axes[0], "Mean", true_mean), (axes[1], "Median", true_median)):
    sub = ci_tab[ci_tab["statistic"] == name].reset_index(drop=True)
    for i, r in sub.iterrows():
        ax.plot([r["lower"], r["upper"]], [i, i], "o-", lw=3, color="steelblue")
    ax.axvline(truth, color="red", linestyle="--", label="True value")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["method"])
    ax.set_title(f"95% intervals for the {name.lower()}")
    ax.legend()
fig.tight_layout()
fig.savefig(OUT / "interval_comparison.png", dpi=150)

# Plot 4: number of resamples
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(b_grid, se_b, "o-", color="tomato")
ax.axhline(se_b[-1], color="grey", linestyle=":")
ax.set_xlabel("Number of bootstrap resamples B")
ax.set_ylabel("Bootstrap standard error of the median")
ax.set_title("The estimate settles down as B grows")
fig.tight_layout()
fig.savefig(OUT / "resamples_needed.png", dpi=150)

# Plot 5: two groups
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(d_boot, bins=50, color="steelblue", alpha=0.7)
axes[0].axvline(0, color="black", linestyle="--", label="No difference")
axes[0].axvline(dlo, color="red")
axes[0].axvline(dhi, color="red", label="95% interval")
axes[0].set_title("Bootstrap: difference in medians (B minus A)")
axes[0].legend()
axes[1].hist(perm.null_distribution, bins=50, color="grey", alpha=0.7)
axes[1].axvline(obs_diff, color="red", lw=2, label=f"Observed = {obs_diff:.1f}")
axes[1].axvline(-obs_diff, color="red", lw=2, linestyle="--")
axes[1].set_title(f"Permutation distribution (p = {perm.pvalue:.4f})")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / "two_group_medians.png", dpi=150)

# Plot 6: correlation
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(cx, cy, alpha=0.7, color="steelblue")
axes[0].set_title(f"The data: r = {r_obs:.3f}")
axes[1].hist(r_boot, bins=50, color="seagreen", alpha=0.7)
axes[1].axvline(rho, color="red", lw=2, label=f"True = {rho}")
axes[1].axvline(rlo, color="black", linestyle="--")
axes[1].axvline(rhi, color="black", linestyle="--", label="Bootstrap interval")
axes[1].set_title("Bootstrap distribution of r (note the left skew)")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / "correlation_bootstrap.png", dpi=150)

# Plot 7: regression
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(rx, ry, alpha=0.7, color="steelblue")
gx = np.linspace(1, 10, 50)
axes[0].plot(gx, icpt + slope * gx, "r-", lw=2, label="OLS line")
axes[0].set_title("The spread of y grows with x")
axes[0].legend()
axes[1].hist(slope_pairs, bins=50, alpha=0.6, color="tomato", label="Pairs bootstrap")
axes[1].hist(slope_resid, bins=50, alpha=0.6, color="steelblue",
             label="Residual bootstrap")
axes[1].axvline(1.5, color="black", lw=2, label="True slope")
axes[1].set_title("Bootstrap distributions of the slope")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / "regression_bootstrap.png", dpi=150)

# Plot 8: coverage
fig, ax = plt.subplots(figsize=(8.5, 5))
xs_ = np.arange(len(cov))
ax.bar(xs_ - 0.2, cov["coverage_t"], 0.4, color="tomato", label="Classical t")
ax.bar(xs_ + 0.2, cov["coverage_bootstrap"], 0.4, color="steelblue",
       label="Percentile bootstrap")
ax.axhline(0.95, color="black", linestyle="--", label="Nominal 0.95")
ax.set_xticks(xs_)
ax.set_xticklabels([f"n = {k}" for k in cov["n"]])
ax.set_ylim(0.8, 1.0)
ax.set_ylabel("Coverage of the true mean")
ax.set_title("Skewed data: how often does a '95%' interval capture the truth?")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "coverage_simulation.png", dpi=150)

# Plot 9: bootstrap failure
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(bmax, bins=40, color="grey", alpha=0.7)
ax.axvline(1.0, color="red", lw=2, label="True upper limit = 1")
ax.axvline(u.max(), color="blue", linestyle="--", label=f"Sample max = {u.max():.3f}")
ax.set_title("Bootstrap distribution of the maximum cannot reach the true limit")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "bootstrap_failure_maximum.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)