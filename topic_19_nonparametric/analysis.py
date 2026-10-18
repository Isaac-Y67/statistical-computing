"""
Topic 19: Non-parametric Statistical Methods
Libraries: scipy.stats, statsmodels, numpy, pandas, matplotlib, seaborn
"""
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=UserWarning)

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
summary_rows = []  # (test, statistic, p_value)


def add(test, stat, p):
    summary_rows.append((test, float(stat), float(p)))


# ---------------------------------------------------------------
# 1. Worked example: Mann-Whitney U by hand (two samples of 5)
# ---------------------------------------------------------------
a5 = np.array([12, 15, 9, 20, 17])
b5 = np.array([8, 14, 11, 7, 10])
ranks5 = stats.rankdata(np.concatenate([a5, b5]))
R_a, R_b = ranks5[:5].sum(), ranks5[5:].sum()
U_a = R_a - 5 * 6 / 2
U_b = 25 - U_a
mw5 = stats.mannwhitneyu(a5, b5, alternative="two-sided", method="exact")
print("=== 1. Worked example: Mann-Whitney U by hand ===")
print("Sample A:", a5.tolist(), " Sample B:", b5.tolist())
print("Ranks of A:", ranks5[:5].tolist(), " ranks of B:", ranks5[5:].tolist())
print(f"Rank sums: R_A = {R_a:.0f}, R_B = {R_b:.0f} (total = {R_a + R_b:.0f} = "
      f"{10 * 11 // 2})")
print(f"U_A = R_A - n(n+1)/2 = {R_a:.0f} - 15 = {U_a:.0f}; U_B = 25 - U_A = {U_b:.0f}")
print(f"scipy U = {mw5.statistic:.0f}, exact two-sided p = {mw5.pvalue:.4f}; "
      f"matches hand calculation: {mw5.statistic == U_a}")

# ---------------------------------------------------------------
# 2. Two independent samples: reaction times (skewed)
# ---------------------------------------------------------------
n_g = 25
grp_a = rng.lognormal(np.log(300), 0.45, n_g).round(0)
grp_b = rng.lognormal(np.log(380), 0.45, n_g).round(0)
pd.DataFrame({"group_a_ms": grp_a, "group_b_ms": grp_b}).to_csv(
    DATA / "reaction_times.csv", index=False)

print("\n=== 2. Two independent samples: reaction times in ms (n = 25 each) ===")
desc = pd.DataFrame({
    "mean": [grp_a.mean(), grp_b.mean()], "median": [np.median(grp_a), np.median(grp_b)],
    "sd": [grp_a.std(ddof=1), grp_b.std(ddof=1)],
    "skewness": [stats.skew(grp_a), stats.skew(grp_b)],
    "shapiro_p": [stats.shapiro(grp_a).pvalue, stats.shapiro(grp_b).pvalue],
}, index=["Group A", "Group B"])
print(desc.round(3).to_string())
lev = stats.levene(grp_a, grp_b)
print(f"Levene (equal variances): p = {lev.pvalue:.4f}")

welch = stats.ttest_ind(grp_b, grp_a, equal_var=False)
mw = stats.mannwhitneyu(grp_b, grp_a, alternative="two-sided")
ks = stats.ks_2samp(grp_b, grp_a)
sp = np.sqrt((grp_a.var(ddof=1) + grp_b.var(ddof=1)) / 2)
cohen_d = (grp_b.mean() - grp_a.mean()) / sp
prob_sup = mw.statistic / (n_g * n_g)
rank_bis = 2 * prob_sup - 1


def perm_stat(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


perm = stats.permutation_test((grp_b, grp_a), perm_stat, vectorized=True,
                              n_resamples=9999, alternative="two-sided",
                              random_state=42)
diffs = np.subtract.outer(grp_b, grp_a).ravel()
hl = np.median(diffs)
hl_boot = [np.median(np.subtract.outer(rng.choice(grp_b, n_g), rng.choice(grp_a, n_g)))
           for _ in range(2000)]
hl_lo, hl_hi = np.percentile(hl_boot, [2.5, 97.5])

print(f"\nWelch t-test:   t = {welch.statistic:.3f}, p = {welch.pvalue:.4f}")
print(f"Mann-Whitney U: U = {mw.statistic:.1f}, p = {mw.pvalue:.4f}")
print(f"Permutation test (difference in means, 9999 resamples): "
      f"p = {perm.pvalue:.4f}")
print(f"Kolmogorov-Smirnov two-sample: D = {ks.statistic:.3f}, p = {ks.pvalue:.4f}")
print(f"Effect sizes: Cohen's d = {cohen_d:.3f}; probability that a random Group B "
      f"value exceeds a Group A value = {prob_sup:.3f}; rank-biserial = {rank_bis:.3f}")
print(f"Difference in means = {grp_b.mean() - grp_a.mean():.1f} ms; "
      f"difference in medians = {np.median(grp_b) - np.median(grp_a):.1f} ms")
print(f"Hodges-Lehmann shift estimate = {hl:.1f} ms "
      f"(bootstrap 95% interval {hl_lo:.1f} to {hl_hi:.1f})")
add("Two-sample Welch t", welch.statistic, welch.pvalue)
add("Mann-Whitney U", mw.statistic, mw.pvalue)
add("Permutation test (means)", perm.statistic, perm.pvalue)
add("Kolmogorov-Smirnov 2-sample", ks.statistic, ks.pvalue)

t_log = stats.ttest_ind(np.log(grp_b), np.log(grp_a), equal_var=False)
mw_log = stats.mannwhitneyu(np.log(grp_b), np.log(grp_a), alternative="two-sided")
print("\nEffect of a log transform (ranks do not change, so Mann-Whitney does not):")
print(f"Welch t-test, raw scale: p = {welch.pvalue:.4f}; log scale: p = "
      f"{t_log.pvalue:.4f}")
print(f"Mann-Whitney, raw scale: U = {mw.statistic:.1f}, p = {mw.pvalue:.4f}; "
      f"log scale: U = {mw_log.statistic:.1f}, p = {mw_log.pvalue:.4f}")

# ---------------------------------------------------------------
# 3. Paired data: pain scores before and after treatment
# ---------------------------------------------------------------
n_p = 20
before = np.clip(np.round(rng.normal(6.5, 1.5, n_p)), 0, 10).astype(int)
after = np.clip(before - np.round(rng.normal(1.5, 1.5, n_p)), 0, 10).astype(int)
pd.DataFrame({"before": before, "after": after}).to_csv(DATA / "pain_scores.csv",
                                                       index=False)
d = after - before
nz = d[d != 0]
r_abs = stats.rankdata(np.abs(nz))
w_plus, w_minus = r_abs[nz > 0].sum(), r_abs[nz < 0].sum()
rank_bis_p = (w_plus - w_minus) / (w_plus + w_minus)
wil = stats.wilcoxon(after, before)
tp = stats.ttest_rel(after, before)
n_neg = int((nz < 0).sum())
sign = stats.binomtest(n_neg, len(nz), 0.5)
print("\n=== 3. Paired data: pain scores (0 to 10) before and after treatment ===")
print(f"n = {n_p}; mean before = {before.mean():.2f}, mean after = {after.mean():.2f}; "
      f"median before = {np.median(before):.1f}, median after = {np.median(after):.1f}")
print(f"Changes: {int((d < 0).sum())} improved, {int((d == 0).sum())} unchanged, "
      f"{int((d > 0).sum())} worsened; median change = {np.median(d):.1f}")
print(f"Shapiro-Wilk on the differences: p = {stats.shapiro(d).pvalue:.4f}")
print(f"Wilcoxon signed-rank: W+ = {w_plus:.1f}, W- = {w_minus:.1f}, "
      f"p = {wil.pvalue:.4f} (zero differences dropped)")
print(f"Matched-pairs rank-biserial correlation = {rank_bis_p:.3f}")
print(f"Sign test: {n_neg} of {len(nz)} non-zero changes were improvements, "
      f"p = {sign.pvalue:.4f}")
print(f"Paired t-test: t = {tp.statistic:.3f}, p = {tp.pvalue:.4f}")
add("Wilcoxon signed-rank", min(w_plus, w_minus), wil.pvalue)
add("Sign test", n_neg, sign.pvalue)
add("Paired t", tp.statistic, tp.pvalue)

# ---------------------------------------------------------------
# 4. Several independent groups: Kruskal-Wallis
# ---------------------------------------------------------------
levels = ["Low", "Medium", "High"]
n_k = 20
dfk = pd.DataFrame({
    "dose": np.repeat(levels, n_k),
    "response": np.concatenate([rng.lognormal(m, 0.6, n_k)
                                for m in (3.0, 3.3, 3.6)]).round(1),
})
dfk.to_csv(DATA / "dose_response.csv", index=False)
samples = [dfk.loc[dfk["dose"] == g, "response"].to_numpy() for g in levels]
fa = stats.f_oneway(*samples)
kw = stats.kruskal(*samples)
N = len(dfk)
eps2 = kw.statistic / (N - 1)
resid = dfk["response"] - dfk.groupby("dose")["response"].transform("mean")
print("\n=== 4. Three independent groups: dose and response (20 each) ===")
print(dfk.groupby("dose")["response"].agg(["mean", "median", "std"]).loc[levels]
      .round(2).to_string())
print(f"Shapiro-Wilk on ANOVA residuals: p = {stats.shapiro(resid).pvalue:.4f}; "
      f"skewness of residuals = {stats.skew(resid):.3f}")
print(f"Levene: p = {stats.levene(*samples).pvalue:.4f}")
print(f"One-way ANOVA: F = {fa.statistic:.3f}, p = {fa.pvalue:.4g}")
print(f"Kruskal-Wallis: H = {kw.statistic:.3f} on 2 df, p = {kw.pvalue:.4g}; "
      f"epsilon squared = {eps2:.3f}")
add("One-way ANOVA", fa.statistic, fa.pvalue)
add("Kruskal-Wallis", kw.statistic, kw.pvalue)

pairs = list(combinations(range(3), 2))
raw_p = [stats.mannwhitneyu(samples[j], samples[i]).pvalue for i, j in pairs]
bonf = multipletests(raw_p, method="bonferroni")[1]
post = pd.DataFrame({
    "comparison": [f"{levels[j]} vs {levels[i]}" for i, j in pairs],
    "median_difference": [np.median(samples[j]) - np.median(samples[i])
                          for i, j in pairs],
    "unadjusted_p": raw_p, "bonferroni_p": bonf,
})
post["significant"] = post["bonferroni_p"] < ALPHA
print("\nPairwise Mann-Whitney tests with Bonferroni correction:")
print(post.round(4).to_string(index=False))
post.round(6).to_csv(OUT / "kruskal_posthoc.csv", index=False)

# ---------------------------------------------------------------
# 5. Several related groups: Friedman test
# ---------------------------------------------------------------
n_s = 12
methods = ["Method 1", "Method 2", "Method 3"]
subj_eff = rng.normal(0, 10, n_s)
scores = 60 + subj_eff[:, None] + np.array([0, 3, 6]) + rng.normal(0, 4, (n_s, 3))
scores = scores.round(1)
wide = pd.DataFrame(scores, columns=methods)
wide.insert(0, "subject", [f"S{i + 1}" for i in range(n_s)])
wide.to_csv(DATA / "repeated_measures.csv", index=False)
fr = stats.friedmanchisquare(*[scores[:, j] for j in range(3)])
kendall_w = fr.statistic / (n_s * (3 - 1))
long = wide.melt(id_vars="subject", var_name="method", value_name="score")
rm = AnovaRM(long, "score", "subject", within=["method"]).fit().anova_table
F_rm, p_rm = rm.loc["method", "F Value"], rm.loc["method", "Pr > F"]
print("\n=== 5. Related groups: 12 subjects, 3 methods ===")
print(f"Method means = {scores.mean(axis=0).round(2).tolist()}, "
      f"medians = {np.median(scores, axis=0).round(2).tolist()}")
print(f"Spread between subjects (sd of subject means) = "
      f"{scores.mean(axis=1).std(ddof=1):.2f}")
print(f"Friedman: chi-square = {fr.statistic:.3f} on 2 df, p = {fr.pvalue:.4g}; "
      f"Kendall's W = {kendall_w:.3f}")
print(f"Repeated-measures ANOVA: F = {F_rm:.3f}, p = {p_rm:.4g}")
ordinary = stats.f_oneway(*[scores[:, j] for j in range(3)])
print(f"Ordinary one-way ANOVA ignoring the subjects: F = {ordinary.statistic:.3f}, "
      f"p = {ordinary.pvalue:.4g}")
add("Friedman", fr.statistic, fr.pvalue)
add("Repeated-measures ANOVA", F_rm, p_rm)

fp = [stats.wilcoxon(scores[:, j], scores[:, i]).pvalue for i, j in pairs]
fb = multipletests(fp, method="bonferroni")[1]
fpost = pd.DataFrame({"comparison": [f"{methods[j]} vs {methods[i]}" for i, j in pairs],
                      "median_difference": [np.median(scores[:, j] - scores[:, i])
                                            for i, j in pairs],
                      "unadjusted_p": fp, "bonferroni_p": fb})
print("\nPairwise Wilcoxon signed-rank tests with Bonferroni correction:")
print(fpost.round(4).to_string(index=False))
fpost.round(6).to_csv(OUT / "friedman_posthoc.csv", index=False)

# ---------------------------------------------------------------
# 6. Rank correlation
# ---------------------------------------------------------------
n_c = 60
x = rng.uniform(0, 10, n_c)
y = np.exp(0.4 * x) + rng.normal(0, 1.0, n_c)
x_o, y_o = np.append(x, 1.0), np.append(y, 60.0)
cor_rows = []
print("\n=== 6. Correlation: curved, increasing relationship (n = 60) ===")
for label, xx, yy in (("Without outlier", x, y), ("With one outlier", x_o, y_o)):
    pr, sr, kt = stats.pearsonr(xx, yy), stats.spearmanr(xx, yy), stats.kendalltau(xx, yy)
    cor_rows.append((label, pr.statistic, pr.pvalue, sr.statistic, sr.pvalue,
                     kt.statistic, kt.pvalue))
    print(f"{label:17s}: Pearson r = {pr.statistic:.3f} (p = {pr.pvalue:.4g}), "
          f"Spearman rho = {sr.statistic:.3f} (p = {sr.pvalue:.4g}), "
          f"Kendall tau = {kt.statistic:.3f} (p = {kt.pvalue:.4g})")
pd.DataFrame(cor_rows, columns=["data", "pearson_r", "pearson_p", "spearman_rho",
                                "spearman_p", "kendall_tau", "kendall_p"]
             ).round(5).to_csv(OUT / "correlations.csv", index=False)
rx, ry = stats.rankdata(x), stats.rankdata(y)
print(f"Spearman = Pearson correlation of the ranks: "
      f"{np.isclose(stats.pearsonr(rx, ry).statistic, stats.spearmanr(x, y).statistic)}")

# ---------------------------------------------------------------
# 7. Simulation: t-test against Mann-Whitney
# ---------------------------------------------------------------
S, n = 3000, 20
dists = {
    "Normal": (lambda size: rng.standard_normal(size), 0.8),
    "Heavy-tailed (t, 3 df)": (lambda size: rng.standard_t(3, size), 1.0),
    "Skewed (lognormal)": (lambda size: rng.lognormal(0, 1, size), 1.0),
}
sim_rows = []
print(f"\n=== 7. Simulation: {S:,d} experiments per cell, {n} per group ===")
for name, (gen, delta) in dists.items():
    x0, y0 = gen((S, n)), gen((S, n))
    x1, y1 = gen((S, n)), gen((S, n)) + delta
    r = {}
    for tag, (xx, yy) in (("null", (x0, y0)), ("alt", (x1, y1))):
        pt = stats.ttest_ind(xx, yy, axis=1, equal_var=False).pvalue
        pm = stats.mannwhitneyu(xx, yy, axis=1, method="asymptotic").pvalue
        r[tag] = ((pt < ALPHA).mean(), (pm < ALPHA).mean())
    sim_rows.append((name, delta, r["null"][0], r["null"][1], r["alt"][0], r["alt"][1]))
    print(f"{name:23s} (shift {delta}): false positives t = {r['null'][0]:.4f}, "
          f"Mann-Whitney = {r['null'][1]:.4f}; power t = {r['alt'][0]:.4f}, "
          f"Mann-Whitney = {r['alt'][1]:.4f}")
sim = pd.DataFrame(sim_rows, columns=["distribution", "shift", "type1_t", "type1_mw",
                                      "power_t", "power_mw"])
sim.round(5).to_csv(OUT / "simulation_results.csv", index=False)
print("Large-sample efficiency of Mann-Whitney relative to the t-test for normal "
      f"data: 3 / pi = {3 / np.pi:.3f}")

# ---------------------------------------------------------------
# 8. Save summary
# ---------------------------------------------------------------
pd.DataFrame(summary_rows, columns=["test", "statistic", "p_value"]).round(6).to_csv(
    OUT / "test_summary.csv", index=False)

# ---------------------------------------------------------------
# 9. Plots
# ---------------------------------------------------------------
# Plot 1: two groups
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
long2 = pd.DataFrame({"group": ["A"] * n_g + ["B"] * n_g,
                      "ms": np.concatenate([grp_a, grp_b])})
sns.boxplot(data=long2, x="group", y="ms", hue="group", legend=False, ax=axes[0],
            showfliers=False, boxprops={"alpha": 0.5})
sns.stripplot(data=long2, x="group", y="ms", color="black", alpha=0.7, ax=axes[0])
axes[0].set_title(f"Reaction times (Mann-Whitney p = {mw.pvalue:.4f})")
sns.histplot(long2, x="ms", hue="group", bins=15, kde=True, ax=axes[1])
axes[1].set_title("Right-skewed distributions")
axes[2].hist(perm.null_distribution, bins=50, color="grey", alpha=0.7)
axes[2].axvline(perm.statistic, color="red", lw=2, label="Observed difference")
axes[2].axvline(-perm.statistic, color="red", lw=2, linestyle="--")
axes[2].set_title(f"Permutation distribution (p = {perm.pvalue:.4f})")
axes[2].legend()
fig.tight_layout()
fig.savefig(OUT / "two_sample_comparison.png", dpi=150)

# Plot 2: ranks
allx = np.concatenate([grp_a, grp_b])
rk = stats.rankdata(allx)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(allx, np.repeat([0, 1], n_g) + rng.uniform(-0.08, 0.08, 2 * n_g),
                alpha=0.7, color="steelblue")
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(["A", "B"])
axes[0].set_xlabel("Reaction time (ms)")
axes[0].set_title("Original values")
axes[1].scatter(rk, np.repeat([0, 1], n_g) + rng.uniform(-0.08, 0.08, 2 * n_g),
                alpha=0.7, color="tomato")
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(["A", "B"])
axes[1].set_xlabel("Rank in the combined sample")
axes[1].set_title(f"Ranks: mean rank A = {rk[:n_g].mean():.1f}, B = {rk[n_g:].mean():.1f}")
fig.tight_layout()
fig.savefig(OUT / "ranks.png", dpi=150)

# Plot 3: paired data
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for b_, a_ in zip(before, after):
    axes[0].plot([0, 1], [b_, a_], "o-", color="steelblue", alpha=0.35)
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Before", "After"])
axes[0].set_ylabel("Pain score")
axes[0].set_title("Each line is one patient")
vals, cnts = np.unique(d, return_counts=True)
axes[1].bar(vals, cnts, color=["tomato" if v > 0 else "seagreen" if v < 0 else "grey"
                               for v in vals])
axes[1].set_xlabel("Change in pain score (after minus before)")
axes[1].set_ylabel("Patients")
axes[1].set_title(f"Changes (Wilcoxon p = {wil.pvalue:.4f}, paired t p = {tp.pvalue:.4f})")
fig.tight_layout()
fig.savefig(OUT / "paired_data.png", dpi=150)

# Plot 4: Kruskal-Wallis
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.boxplot(data=dfk, x="dose", y="response", order=levels, hue="dose", hue_order=levels,
            legend=False, showfliers=False, ax=ax, boxprops={"alpha": 0.5})
sns.stripplot(data=dfk, x="dose", y="response", order=levels, color="black", alpha=0.6,
              ax=ax)
ax.set_title(f"Kruskal-Wallis H = {kw.statistic:.2f}, p = {kw.pvalue:.4g}")
fig.tight_layout()
fig.savefig(OUT / "kruskal_wallis.png", dpi=150)

# Plot 5: repeated measures
fig, ax = plt.subplots(figsize=(8, 5.5))
for row in scores:
    ax.plot(range(3), row, "o-", color="steelblue", alpha=0.35)
ax.plot(range(3), np.median(scores, axis=0), "D-", color="red", lw=3, ms=9,
        label="Median")
ax.set_xticks(range(3))
ax.set_xticklabels(methods)
ax.set_ylabel("Score")
ax.set_title(f"Same subjects, three methods (Friedman p = {fr.pvalue:.4g})")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "friedman.png", dpi=150)

# Plot 6: correlation
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (label, xx, yy), row in zip(axes, (("Without outlier", x, y),
                                           ("With one outlier", x_o, y_o)), cor_rows):
    ax.scatter(xx, yy, alpha=0.7, color="steelblue")
    if label == "With one outlier":
        ax.scatter([1.0], [60.0], color="red", s=80, label="Outlier")
        ax.legend()
    ax.set_title(f"{label}: Pearson {row[1]:.2f}, Spearman {row[3]:.2f}, "
                 f"Kendall {row[5]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
fig.tight_layout()
fig.savefig(OUT / "correlation.png", dpi=150)

# Plot 7: simulation
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
xs = np.arange(len(sim))
axes[0].bar(xs - 0.2, sim["type1_t"], 0.4, color="tomato", label="Welch t-test")
axes[0].bar(xs + 0.2, sim["type1_mw"], 0.4, color="steelblue", label="Mann-Whitney")
axes[0].axhline(ALPHA, color="black", linestyle="--", label="Nominal 0.05")
axes[0].set_title("False positive rate (no real difference)")
axes[1].bar(xs - 0.2, sim["power_t"], 0.4, color="tomato", label="Welch t-test")
axes[1].bar(xs + 0.2, sim["power_mw"], 0.4, color="steelblue", label="Mann-Whitney")
axes[1].set_title("Power (true shift present)")
for ax in axes:
    ax.set_xticks(xs)
    ax.set_xticklabels(sim["distribution"], rotation=12)
    ax.legend()
fig.tight_layout()
fig.savefig(OUT / "simulation_t_vs_mannwhitney.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)