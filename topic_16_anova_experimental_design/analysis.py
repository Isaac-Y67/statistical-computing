"""
Topic 16: ANOVA and Experimental Design
Libraries: statsmodels, scipy.stats, numpy, pandas, matplotlib, seaborn
"""
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.power import FTestAnovaPower

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

# ---------------------------------------------------------------
# 1. Datasets (synthetic, reproducible)
# ---------------------------------------------------------------
groups = ["Control", "Fert A", "Fert B", "Fert C"]
true_means = [20, 23, 26, 25]
n_g = 15
df1 = pd.DataFrame({
    "treatment": np.repeat(groups, n_g),
    "growth_cm": np.concatenate([rng.normal(m, 4.5, n_g) for m in true_means]).round(1),
})
df1.to_csv(DATA / "plant_growth_oneway.csv", index=False)

fert = ["None", "Fert A", "Fert B"]
water = ["Low", "High"]
rows = []
for fi, f in enumerate(fert):
    for wi, w in enumerate(water):
        mu = 20 + [0, 3, 5][fi] + [0, 4][wi] + (5 if (f == "Fert B" and w == "High") else 0)
        for v in rng.normal(mu, 4, 10):
            rows.append((f, w, round(v, 1)))
df2 = pd.DataFrame(rows, columns=["fertiliser", "water", "growth_cm"])
df2.to_csv(DATA / "plant_growth_twoway.csv", index=False)

n_blocks = 5
varieties = ["V1", "V2", "V3", "V4"]
v_eff = [0, 2.5, 5, 3]
b_eff = rng.normal(0, 5, n_blocks)
rows = []
for b in range(n_blocks):
    for vi, v in enumerate(varieties):
        rows.append((f"Field {b + 1}", v,
                     round(50 + b_eff[b] + v_eff[vi] + rng.normal(0, 1.5), 1)))
df3 = pd.DataFrame(rows, columns=["block", "variety", "yield_t_ha"])
df3.to_csv(DATA / "crop_blocks.csv", index=False)

print("=== 1. Datasets ===")
print(f"One-way: {len(df1)} plants, 4 treatments, {n_g} per treatment")
print(f"Two-way: {len(df2)} plants, 3 fertilisers x 2 watering levels, 10 per cell")
print(f"Block design: {len(df3)} plots, 4 varieties x {n_blocks} fields")

# ---------------------------------------------------------------
# 2. One-way ANOVA
# ---------------------------------------------------------------
desc = df1.groupby("treatment")["growth_cm"].agg(["count", "mean", "std", "min", "max"])
desc = desc.loc[groups]
model1 = smf.ols("growth_cm ~ C(treatment)", data=df1).fit()
aov1 = anova_lm(model1, typ=2)
ss_b, ss_w = aov1.loc["C(treatment)", "sum_sq"], aov1.loc["Residual", "sum_sq"]
df_b, df_w = int(aov1.loc["C(treatment)", "df"]), int(aov1.loc["Residual", "df"])
ms_b, ms_w = ss_b / df_b, ss_w / df_w
F1 = ms_b / ms_w
p1 = stats.f.sf(F1, df_b, df_w)
ss_t = ss_b + ss_w
eta2 = ss_b / ss_t
omega2 = (ss_b - df_b * ms_w) / (ss_t + ms_w)
cohen_f = np.sqrt(eta2 / (1 - eta2))
f_check = stats.f_oneway(*[df1.loc[df1["treatment"] == g, "growth_cm"] for g in groups])

print("\n=== 2. One-way ANOVA: growth by treatment ===")
print("Group summaries:")
print(desc.round(2).to_string())
print(f"Grand mean = {df1['growth_cm'].mean():.3f}")
print("\nANOVA table:")
tab = pd.DataFrame({
    "df": [df_b, df_w, df_b + df_w],
    "sum_sq": [ss_b, ss_w, ss_t],
    "mean_sq": [ms_b, ms_w, np.nan],
    "F": [F1, np.nan, np.nan],
    "p_value": [p1, np.nan, np.nan],
}, index=["Between treatments", "Within (error)", "Total"])
print(tab.round(4).to_string())
tab.round(6).to_csv(OUT / "anova_oneway.csv")
print(f"\nF = {F1:.3f} on ({df_b}, {df_w}) df, p = {p1:.4g}; "
      f"critical F at 5% = {stats.f.ppf(0.95, df_b, df_w):.3f}")
print("Hand calculation matches statsmodels and scipy f_oneway:",
      np.isclose(F1, aov1.loc["C(treatment)", "F"]) and np.isclose(F1, f_check.statistic))
print(f"Effect sizes: eta squared = {eta2:.4f}, omega squared = {omega2:.4f}, "
      f"Cohen's f = {cohen_f:.3f}")
print(f"Pooled standard deviation = sqrt(MSE) = {np.sqrt(ms_w):.3f} (true sd = 4.5)")

# ---------------------------------------------------------------
# 3. Assumption checks and robust alternatives
# ---------------------------------------------------------------
resid1 = model1.resid
sw_res = stats.shapiro(resid1)
lev = stats.levene(*[df1.loc[df1["treatment"] == g, "growth_cm"] for g in groups])
print("\n=== 3. Assumption checks ===")
for g in groups:
    sw = stats.shapiro(df1.loc[df1["treatment"] == g, "growth_cm"])
    print(f"Shapiro-Wilk, {g:8s}: W = {sw.statistic:.4f}, p = {sw.pvalue:.4f}")
print(f"Shapiro-Wilk on residuals: W = {sw_res.statistic:.4f}, p = {sw_res.pvalue:.4f}")
print(f"Levene (equal variances): W = {lev.statistic:.3f}, p = {lev.pvalue:.4f}")
print(f"Ratio of largest to smallest group sd = {desc['std'].max() / desc['std'].min():.2f}")

welch = anova_oneway(df1["growth_cm"].to_numpy(), df1["treatment"].to_numpy(),
                     use_var="unequal")
kw = stats.kruskal(*[df1.loc[df1["treatment"] == g, "growth_cm"] for g in groups])
print("\nRobust alternatives:")
print(f"Welch ANOVA: F = {float(welch.statistic):.3f}, p = {float(welch.pvalue):.4g}")
print(f"Kruskal-Wallis: H = {kw.statistic:.3f}, p = {kw.pvalue:.4g}")

# ---------------------------------------------------------------
# 4. Post-hoc comparisons
# ---------------------------------------------------------------
tuk = pairwise_tukeyhsd(df1["growth_cm"], df1["treatment"], alpha=ALPHA)
tuk_df = pd.DataFrame(tuk._results_table.data[1:], columns=tuk._results_table.data[0])
for c in ["meandiff", "p-adj", "lower", "upper"]:
    tuk_df[c] = tuk_df[c].astype(float)
tuk_df["reject"] = tuk_df["reject"].astype(bool)

pairs = list(combinations(groups, 2))
means = desc["mean"]
raw_p, diffs = [], []
for a, b in pairs:
    d = means[b] - means[a]
    t = d / np.sqrt(ms_w * (1 / n_g + 1 / n_g))
    raw_p.append(2 * stats.t.sf(abs(t), df_w))
    diffs.append(d)
bonf = multipletests(raw_p, method="bonferroni")[1]
tuk_p = {(r["group1"], r["group2"]): r["p-adj"] for _, r in tuk_df.iterrows()}
post = pd.DataFrame({
    "comparison": [f"{b} - {a}" for a, b in pairs],
    "difference": diffs,
    "unadjusted_p": raw_p,
    "bonferroni_p": bonf,
    "tukey_p": [tuk_p[p] for p in pairs],
})
post["tukey_significant"] = post["tukey_p"] < ALPHA
print("\n=== 4. Post-hoc pairwise comparisons (6 pairs) ===")
print(post.round(4).to_string(index=False))
post.round(6).to_csv(OUT / "posthoc_comparisons.csv", index=False)
print("\nTukey HSD with 95% simultaneous confidence intervals:")
print(tuk_df.round(3).to_string(index=False))
tuk_df.round(5).to_csv(OUT / "tukey_hsd.csv", index=False)
q_crit = stats.studentized_range.ppf(0.95, len(groups), df_w)
hsd = q_crit * np.sqrt(ms_w / n_g)
print(f"Tukey critical q = {q_crit:.3f}; honestly significant difference "
      f"HSD = {hsd:.3f} (differences larger than this are significant)")

try:
    others = [df1.loc[df1["treatment"] == g, "growth_cm"].to_numpy() for g in groups[1:]]
    dun = stats.dunnett(*others, control=df1.loc[df1["treatment"] == "Control",
                                                 "growth_cm"].to_numpy())
    print("\nDunnett's test (each fertiliser against the control):")
    for g, p_ in zip(groups[1:], dun.pvalue):
        print(f"{g} against Control: p = {p_:.4g}")
except (AttributeError, TypeError) as err:
    print("\nDunnett's test not available in this SciPy version:", err)

# ---------------------------------------------------------------
# 5. Why not run many t-tests? Familywise error simulation
# ---------------------------------------------------------------
S = 10_000
sim = rng.normal(0, 4.5, size=(S, 4, n_g))
m_s = sim.mean(axis=2)
msw_s = sim.var(axis=2, ddof=1).mean(axis=1)
dfw_s = 4 * (n_g - 1)
span = m_s.max(axis=1) - m_s.min(axis=1)
t_crit = stats.t.ppf(0.975, dfw_s)
any_t = span > t_crit * np.sqrt(2 * msw_s / n_g)
any_tukey = span / np.sqrt(msw_s / n_g) > stats.studentized_range.ppf(0.95, 4, dfw_s)
any_anova = stats.f_oneway(*[sim[:, i, :] for i in range(4)], axis=1).pvalue < ALPHA
print(f"\n=== 5. Familywise error simulation ({S:,d} experiments, all means equal) ===")
fw = pd.DataFrame({
    "procedure": ["6 unadjusted t-tests (any significant)", "ANOVA F-test",
                  "Tukey HSD (any significant)"],
    "false_positive_rate": [any_t.mean(), any_anova.mean(), any_tukey.mean()],
})
print(fw.round(4).to_string(index=False))
print(f"Nominal rate = {ALPHA}; if the 6 tests were independent the rate would be "
      f"{1 - (1 - ALPHA) ** 6:.3f}")
fw.round(5).to_csv(OUT / "familywise_error.csv", index=False)

# ---------------------------------------------------------------
# 6. Power and sample size
# ---------------------------------------------------------------
pw = FTestAnovaPower()
obs_power = pw.power(effect_size=cohen_f, nobs=len(df1), alpha=ALPHA, k_groups=4)
print("\n=== 6. Power analysis (one-way ANOVA, 4 groups) ===")
print(f"Power with the observed Cohen's f = {cohen_f:.3f} and {len(df1)} plants: "
      f"{obs_power:.3f}")
pow_rows = []
for fe, label in ((0.10, "small"), (0.25, "medium"), (0.40, "large")):
    total = pw.solve_power(effect_size=fe, nobs=None, alpha=ALPHA, power=0.80,
                           k_groups=4)
    per = int(np.ceil(total / 4))
    pow_rows.append((label, fe, per, per * 4))
    print(f"Cohen's f = {fe:.2f} ({label}): {per} per group ({per * 4} in total) "
          f"for 80% power")
pd.DataFrame(pow_rows, columns=["effect", "cohen_f", "n_per_group", "n_total"]
             ).to_csv(OUT / "sample_size.csv", index=False)

# ---------------------------------------------------------------
# 7. Two-way factorial ANOVA
# ---------------------------------------------------------------
full = smf.ols("growth_cm ~ C(fertiliser) * C(water)", data=df2).fit()
additive = smf.ols("growth_cm ~ C(fertiliser) + C(water)", data=df2).fit()
aov2 = anova_lm(full, typ=2)
ss_res2 = aov2.loc["Residual", "sum_sq"]
aov2["partial_eta_sq"] = aov2["sum_sq"] / (aov2["sum_sq"] + ss_res2)
aov2.loc["Residual", "partial_eta_sq"] = np.nan
print("\n=== 7. Two-way ANOVA: growth by fertiliser and watering ===")
cell = df2.pivot_table(index="fertiliser", columns="water", values="growth_cm",
                       aggfunc="mean").loc[fert, water]
cell["row_mean"] = df2.groupby("fertiliser")["growth_cm"].mean()
print("Cell means:")
print(cell.round(2).to_string())
print("Column means:", df2.groupby("water")["growth_cm"].mean().round(2).to_dict())
print("\nANOVA table (Type II sums of squares):")
print(aov2.round(4).to_string())
aov2.round(6).to_csv(OUT / "anova_twoway.csv")
cmp = anova_lm(additive, full)
print(f"\nInteraction test, additive against full model: F = {cmp['F'].iloc[1]:.3f}, "
      f"p = {cmp['Pr(>F)'].iloc[1]:.4g}")
print(f"Shapiro-Wilk on residuals: p = {stats.shapiro(full.resid).pvalue:.4f}; "
      f"Levene across the 6 cells: p = "
      f"{stats.levene(*[g['growth_cm'] for _, g in df2.groupby(['fertiliser', 'water'])]).pvalue:.4f}")

print("\nSimple effects: High against Low watering within each fertiliser (Welch t-test):")
simple_rows = []
for f in fert:
    hi = df2[(df2["fertiliser"] == f) & (df2["water"] == "High")]["growth_cm"]
    lo = df2[(df2["fertiliser"] == f) & (df2["water"] == "Low")]["growth_cm"]
    tt = stats.ttest_ind(hi, lo, equal_var=False)
    simple_rows.append((f, hi.mean() - lo.mean(), tt.statistic, tt.pvalue))
    print(f"{f:7s}: difference = {hi.mean() - lo.mean():+6.2f}, t = {tt.statistic:6.3f}, "
          f"p = {tt.pvalue:.4g}")
pd.DataFrame(simple_rows, columns=["fertiliser", "high_minus_low", "t", "p_value"]
             ).round(5).to_csv(OUT / "simple_effects.csv", index=False)

# ---------------------------------------------------------------
# 8. Randomised block design
# ---------------------------------------------------------------
m_ign = smf.ols("yield_t_ha ~ C(variety)", data=df3).fit()
m_blk = smf.ols("yield_t_ha ~ C(variety) + C(block)", data=df3).fit()
a_ign = anova_lm(m_ign, typ=2)
a_blk = anova_lm(m_blk, typ=2)
mse_ign = a_ign.loc["Residual", "sum_sq"] / a_ign.loc["Residual", "df"]
mse_blk = a_blk.loc["Residual", "sum_sq"] / a_blk.loc["Residual", "df"]
print("\n=== 8. Randomised block design: 4 varieties in 5 fields ===")
print("Variety means:", df3.groupby("variety")["yield_t_ha"].mean().round(2).to_dict())
print("Field means:  ", df3.groupby("block")["yield_t_ha"].mean().round(2).to_dict())
print("\nBlock design ANOVA (variety + block):")
print(a_blk.round(4).to_string())
print("\nIgnoring the blocks (one-way ANOVA on variety):")
print(a_ign.round(4).to_string())
print(f"\nVariety F: {a_blk.loc['C(variety)', 'F']:.2f} (p = "
      f"{a_blk.loc['C(variety)', 'PR(>F)']:.4g}) with blocks, against "
      f"{a_ign.loc['C(variety)', 'F']:.2f} (p = {a_ign.loc['C(variety)', 'PR(>F)']:.4g}) "
      f"without")
print(f"Error mean square: {mse_blk:.3f} with blocks, {mse_ign:.3f} without "
      f"(ratio {mse_ign / mse_blk:.1f})")
pd.concat({"with_blocks": a_blk, "ignoring_blocks": a_ign}).round(6).to_csv(
    OUT / "block_design.csv")

# ---------------------------------------------------------------
# 9. Plots
# ---------------------------------------------------------------
# Plot 1: group comparison
fig, ax = plt.subplots(figsize=(8.5, 5.5))
sns.boxplot(data=df1, x="treatment", y="growth_cm", order=groups, hue="treatment",
            hue_order=groups, legend=False, ax=ax, showfliers=False, boxprops={"alpha": 0.5})
sns.stripplot(data=df1, x="treatment", y="growth_cm", order=groups, color="black",
              size=4, alpha=0.7, ax=ax)
ax.plot(range(4), desc["mean"].to_numpy(), "D", color="red", ms=8, label="Group mean")
ax.axhline(df1["growth_cm"].mean(), color="grey", linestyle="--", label="Grand mean")
ax.set_ylabel("Growth (cm)")
ax.set_title(f"One-way ANOVA: F = {F1:.2f}, p = {p1:.4g}")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "group_comparison.png", dpi=150)

# Plot 2: F distribution
fig, ax = plt.subplots(figsize=(8, 5))
xs = np.linspace(0, max(F1 * 1.3, 8), 400)
ax.plot(xs, stats.f.pdf(xs, df_b, df_w), "k-")
crit = stats.f.ppf(0.95, df_b, df_w)
ax.fill_between(xs, stats.f.pdf(xs, df_b, df_w), where=xs >= crit, color="tomato",
                alpha=0.5, label=f"Rejection region (F > {crit:.2f})")
ax.axvline(F1, color="blue", linestyle="--", label=f"Observed F = {F1:.2f}")
ax.set_xlabel("F")
ax.set_title(f"F distribution with ({df_b}, {df_w}) degrees of freedom")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "f_distribution.png", dpi=150)

# Plot 3: Tukey intervals
fig, ax = plt.subplots(figsize=(8, 5))
lab = [f"{r['group2']} - {r['group1']}" for _, r in tuk_df.iterrows()]
yy = np.arange(len(lab))[::-1]
for y_, (_, r) in zip(yy, tuk_df.iterrows()):
    colour = "tomato" if r["reject"] else "grey"
    ax.errorbar(r["meandiff"], y_, xerr=[[r["meandiff"] - r["lower"]],
                                         [r["upper"] - r["meandiff"]]],
                fmt="o", color=colour, capsize=5)
ax.axvline(0, color="black", linestyle="--")
ax.set_yticks(yy)
ax.set_yticklabels(lab)
ax.set_xlabel("Difference in mean growth (cm)")
ax.set_title("Tukey HSD 95% intervals (red = significant)")
fig.tight_layout()
fig.savefig(OUT / "tukey_intervals.png", dpi=150)

# Plot 4: residual diagnostics
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
stats.probplot(resid1, dist="norm", plot=axes[0])
axes[0].set_title("Normal Q-Q plot of residuals")
axes[1].scatter(model1.fittedvalues, resid1, alpha=0.7)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("Fitted values (group means)")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residuals against fitted values")
sns.boxplot(data=df1.assign(resid=resid1), x="treatment", y="resid", order=groups,
            ax=axes[2])
axes[2].set_title("Spread of residuals by group")
fig.tight_layout()
fig.savefig(OUT / "anova_residual_diagnostics.png", dpi=150)

# Plot 5: familywise error
fig, ax = plt.subplots(figsize=(8.5, 5))
bars = ax.bar(["6 unadjusted\nt-tests", "ANOVA F-test", "Tukey HSD"],
              fw["false_positive_rate"], color=["tomato", "steelblue", "seagreen"])
ax.bar_label(bars, fmt="%.3f")
ax.axhline(ALPHA, color="black", linestyle="--", label="Nominal 0.05")
ax.set_ylabel("False positive rate when all means are equal")
ax.set_title("Multiple comparisons inflate the error rate")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "familywise_error.png", dpi=150)

# Plot 6: power curves
fig, ax = plt.subplots(figsize=(8, 5))
n_grid = np.arange(3, 101, 2)
for fe in (0.10, 0.25, 0.40):
    ax.plot(n_grid, [pw.power(effect_size=fe, nobs=4 * k, alpha=ALPHA, k_groups=4)
                     for k in n_grid], label=f"Cohen's f = {fe}")
ax.axhline(0.8, color="grey", linestyle=":")
ax.set_xlabel("Sample size per group")
ax.set_ylabel("Power")
ax.set_title("Power of the one-way ANOVA (4 groups)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "power_curve.png", dpi=150)

# Plot 7: interaction plot
fig, ax = plt.subplots(figsize=(8, 5.5))
for w, colour in zip(water, ("tab:blue", "tab:red")):
    sub = df2[df2["water"] == w].groupby("fertiliser")["growth_cm"]
    m_ = sub.mean().loc[fert]
    se_ = (sub.std() / np.sqrt(sub.count())).loc[fert]
    ax.errorbar(range(3), m_, yerr=se_, fmt="o-", color=colour, capsize=5,
                label=f"{w} watering")
ax.set_xticks(range(3))
ax.set_xticklabels(fert)
ax.set_ylabel("Mean growth (cm), error bars = 1 SE")
ax.set_title("Interaction plot: lines that are not parallel indicate an interaction")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "interaction_plot.png", dpi=150)

# Plot 8: block design
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for b, g in df3.groupby("block"):
    axes[0].plot(g["variety"], g["yield_t_ha"], "o-", label=b)
axes[0].set_ylabel("Yield (t/ha)")
axes[0].set_title("Each field (block) has its own level; the varieties differ within it")
axes[0].legend(fontsize=8)
sns.boxplot(data=df3, x="variety", y="yield_t_ha", ax=axes[1], hue="variety",
            legend=False)
axes[1].set_title("Ignoring the blocks: huge spread hides the variety effect")
fig.tight_layout()
fig.savefig(OUT / "block_design.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)