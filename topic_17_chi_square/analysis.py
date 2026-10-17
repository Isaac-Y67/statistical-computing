"""
Topic 17: Chi-Square and Categorical Data Analysis
Libraries: scipy.stats, statsmodels, pandas, numpy, matplotlib, seaborn
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

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
regions = ["North", "South", "East", "West"]
products = ["A", "B", "C"]
pref = {"North": [0.50, 0.30, 0.20], "South": [0.30, 0.40, 0.30],
        "East": [0.25, 0.35, 0.40], "West": [0.40, 0.35, 0.25]}
n = 600
region = rng.choice(regions, size=n, p=[0.30, 0.25, 0.25, 0.20])
product = np.empty(n, dtype=object)
for r in regions:
    idx = region == r
    product[idx] = rng.choice(products, size=int(idx.sum()), p=pref[r])
survey = pd.DataFrame({"region": region, "product": product.astype(str)})
survey.to_csv(DATA / "survey.csv", index=False)

N_ARM = 200
rec_drug = int(rng.binomial(N_ARM, 0.70))
rec_plac = int(rng.binomial(N_ARM, 0.55))
trial = pd.DataFrame({
    "group": ["Drug"] * N_ARM + ["Placebo"] * N_ARM,
    "recovered": [1] * rec_drug + [0] * (N_ARM - rec_drug)
                 + [1] * rec_plac + [0] * (N_ARM - rec_plac),
})
trial.to_csv(DATA / "trial.csv", index=False)

die_p = np.array([0.14, 0.16, 0.16, 0.17, 0.17, 0.20])
rolls = rng.choice(np.arange(1, 7), size=300, p=die_p)
die_counts = np.bincount(rolls, minlength=7)[1:]

N_PEAS = 556
round_peas = int(rng.binomial(N_PEAS, 0.75))

N_PAIR = 150
pair_cells = rng.multinomial(N_PAIR, [0.40, 0.07, 0.20, 0.33])
pair_table = pair_cells.reshape(2, 2)  # rows: before yes/no, columns: after yes/no

print("=== 1. Datasets ===")
print(f"Survey: {n} customers, {len(regions)} regions, {len(products)} products")
print(f"Trial: {N_ARM} patients per arm; Die: 300 rolls; Peas: {N_PEAS} plants; "
      f"Paired opinions: {N_PAIR} people")

# ---------------------------------------------------------------
# 2. Goodness of fit
# ---------------------------------------------------------------
print("\n=== 2a. Goodness of fit: is the die fair? (300 rolls) ===")
gof = stats.chisquare(die_counts)
g_test = stats.power_divergence(die_counts, lambda_="log-likelihood")
print("Face:      ", list(range(1, 7)))
print("Observed:  ", die_counts.tolist())
print("Expected:  ", [50.0] * 6)
print(f"Chi-square = {gof.statistic:.3f} on 5 df, p = {gof.pvalue:.4f}; "
      f"critical value at 5% = {stats.chi2.ppf(0.95, 5):.3f}")
print(f"G-test (likelihood ratio) = {g_test.statistic:.3f}, p = {g_test.pvalue:.4f}")
print(f"Largest standardised residual = "
      f"{np.max(np.abs((die_counts - 50) / np.sqrt(50))):.3f}")

wrinkled = N_PEAS - round_peas
mend = stats.chisquare([round_peas, wrinkled],
                       f_exp=[N_PEAS * 0.75, N_PEAS * 0.25])
print("\n=== 2b. Goodness of fit: Mendel-style 3:1 cross ===")
print(f"Round = {round_peas}, wrinkled = {wrinkled} (expected "
      f"{N_PEAS * 0.75:.0f} and {N_PEAS * 0.25:.0f})")
print(f"Chi-square = {mend.statistic:.3f} on 1 df, p = {mend.pvalue:.4f}")

# ---------------------------------------------------------------
# 3. Test of independence: region by product
# ---------------------------------------------------------------
ct = pd.crosstab(survey["region"], survey["product"]).loc[regions, products]
obs = ct.to_numpy().astype(float)
chi2, p_ind, dof, exp = stats.chi2_contingency(obs, correction=False)
g_ind = stats.chi2_contingency(obs, correction=False, lambda_="log-likelihood")
exp_df = pd.DataFrame(exp, index=regions, columns=products)
manual = ((obs - exp) ** 2 / exp).sum()
row_p = obs.sum(axis=1, keepdims=True) / n
col_p = obs.sum(axis=0, keepdims=True) / n
pearson = (obs - exp) / np.sqrt(exp)
adj = (obs - exp) / np.sqrt(exp * (1 - row_p) * (1 - col_p))
v = np.sqrt(chi2 / (n * (min(obs.shape) - 1)))

print("\n=== 3. Test of independence: region and preferred product ===")
print("Observed counts:")
print(ct.to_string())
print("\nRow percentages:")
print((pd.crosstab(survey["region"], survey["product"], normalize="index")
       .loc[regions, products] * 100).round(1).to_string())
print("\nExpected counts under independence:")
print(exp_df.round(2).to_string())
print(f"Smallest expected count = {exp.min():.2f} (rule of thumb: at least 5)")
print(f"\nChi-square = {chi2:.3f} on {dof} df, p = {p_ind:.4g}; "
      f"critical value at 5% = {stats.chi2.ppf(0.95, dof):.3f}")
print(f"Hand calculation sum((O - E)^2 / E) = {manual:.3f}, matches scipy: "
      f"{np.isclose(manual, chi2)}")
print(f"G-test: G = {g_ind[0]:.3f}, p = {g_ind[1]:.4g}")
print(f"Cramer's V = {v:.3f}")
print("\nAdjusted standardised residuals (beyond +/-1.96 are notable cells):")
adj_df = pd.DataFrame(adj, index=regions, columns=products)
print(adj_df.round(2).to_string())
ct.to_csv(OUT / "observed_counts.csv")
exp_df.round(3).to_csv(OUT / "expected_counts.csv")
adj_df.round(3).to_csv(OUT / "standardised_residuals.csv")

# ---------------------------------------------------------------
# 4. 2x2 table: drug against placebo
# ---------------------------------------------------------------
a, b = rec_drug, N_ARM - rec_drug
c, d = rec_plac, N_ARM - rec_plac
t22 = np.array([[a, b], [c, d]])
risk1, risk2 = a / N_ARM, c / N_ARM
rd = risk1 - risk2
se_rd = np.sqrt(risk1 * (1 - risk1) / N_ARM + risk2 * (1 - risk2) / N_ARM)
rr = risk1 / risk2
se_rr = np.sqrt(1 / a - 1 / N_ARM + 1 / c - 1 / N_ARM)
orr = (a * d) / (b * c)
se_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
z = stats.norm.ppf(0.975)
chi_nc = stats.chi2_contingency(t22, correction=False)
chi_yc = stats.chi2_contingency(t22, correction=True)
fisher = stats.fisher_exact(t22)

print("\n=== 4. Clinical trial: drug against placebo ===")
print(pd.DataFrame(t22, index=["Drug", "Placebo"],
                   columns=["Recovered", "Not recovered"]).to_string())
print(f"Recovery rates: drug = {risk1:.3f}, placebo = {risk2:.3f}")
eff = pd.DataFrame([
    ("Risk difference", rd, rd - z * se_rd, rd + z * se_rd),
    ("Relative risk", rr, np.exp(np.log(rr) - z * se_rr), np.exp(np.log(rr) + z * se_rr)),
    ("Odds ratio", orr, np.exp(np.log(orr) - z * se_or), np.exp(np.log(orr) + z * se_or)),
], columns=["measure", "estimate", "lower_95", "upper_95"])
print(eff.round(3).to_string(index=False))
print(f"Number needed to treat = {1 / rd:.1f}")
print(f"Chi-square without correction: {chi_nc[0]:.3f}, p = {chi_nc[1]:.4f}")
print(f"Chi-square with Yates correction: {chi_yc[0]:.3f}, p = {chi_yc[1]:.4f}")
print(f"Fisher's exact test: odds ratio = {fisher[0]:.3f}, p = {fisher[1]:.4f}")
eff.round(5).to_csv(OUT / "effect_measures.csv", index=False)

# ---------------------------------------------------------------
# 5. Small sample: chi-square against Fisher
# ---------------------------------------------------------------
small = np.array([[8, 2], [3, 7]])
exp_small = stats.chi2_contingency(small, correction=False)[3]
print("\n=== 5. Small sample (n = 20) ===")
print(small.tolist(), "-> smallest expected count =", round(exp_small.min(), 2))
print(f"Chi-square, no correction: p = "
      f"{stats.chi2_contingency(small, correction=False)[1]:.4f}")
print(f"Chi-square, Yates correction: p = "
      f"{stats.chi2_contingency(small, correction=True)[1]:.4f}")
print(f"Fisher's exact test: p = {stats.fisher_exact(small)[1]:.4f}")

# ---------------------------------------------------------------
# 6. McNemar's test for paired data
# ---------------------------------------------------------------
yy, yn, ny, nn = pair_cells
mc = mcnemar(pair_table, exact=False, correction=True)
exact_p = stats.binomtest(int(min(yn, ny)), int(yn + ny), 0.5).pvalue
print("\n=== 6. McNemar's test: opinion before and after a campaign ===")
print(pd.DataFrame(pair_table, index=["Before: yes", "Before: no"],
                   columns=["After: yes", "After: no"]).to_string())
print(f"Changed from yes to no: {yn}; changed from no to yes: {ny}")
print(f"Support before = {(yy + yn) / N_PAIR:.3f}, after = {(yy + ny) / N_PAIR:.3f}")
print(f"McNemar chi-square = {mc.statistic:.3f}, p = {mc.pvalue:.4f}; "
      f"exact binomial p = {exact_p:.4f}")

# ---------------------------------------------------------------
# 7. Simulation: false positive rates when there is no association
# ---------------------------------------------------------------
S = 4000
cell_p = np.outer([0.5, 0.5], [0.2, 0.8]).ravel()
sim_rows = []
print(f"\n=== 7. Simulation: {S:,d} tables per sample size, rows and columns independent ===")
for n_s in (20, 60, 400):
    tabs = rng.multinomial(n_s, cell_p, size=S)
    ta, tb, tc, td = tabs.T.astype(float)
    r1, r2, c1, c2 = ta + tb, tc + td, ta + tc, tb + td
    ok = (r1 > 0) & (r2 > 0) & (c1 > 0) & (c2 > 0)
    chi = n_s * (ta * td - tb * tc) ** 2 / np.where(ok, r1 * r2 * c1 * c2, 1)
    rej_chi = (stats.chi2.sf(chi, 1) < ALPHA)[ok].mean()
    idx = np.flatnonzero(ok)
    rej_fis = np.mean([stats.fisher_exact(tabs[i].reshape(2, 2))[1] < ALPHA
                       for i in idx])
    min_exp = (np.minimum(r1, r2) * np.minimum(c1, c2) / n_s)[ok]
    sim_rows.append((n_s, float(np.mean(min_exp < 5)), rej_chi, rej_fis))
    print(f"n = {n_s:3d}: tables with a smallest expected count below 5 = "
          f"{np.mean(min_exp < 5):.1%}; chi-square rejects {rej_chi:.4f}, "
          f"Fisher rejects {rej_fis:.4f} (nominal {ALPHA})")
sim = pd.DataFrame(sim_rows, columns=["n", "share_min_expected_below_5",
                                      "chi_square_rejection", "fisher_rejection"])
sim.round(5).to_csv(OUT / "simulation_false_positives.csv", index=False)

# ---------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------
# Plot 1: goodness of fit
fig, ax = plt.subplots(figsize=(8, 5))
faces = np.arange(1, 7)
ax.bar(faces - 0.2, die_counts, 0.4, color="steelblue", label="Observed")
ax.bar(faces + 0.2, [50] * 6, 0.4, color="lightgrey", label="Expected if fair")
ax.set_xlabel("Face")
ax.set_ylabel("Count in 300 rolls")
ax.set_title(f"Die rolls: chi-square = {gof.statistic:.2f}, p = {gof.pvalue:.3f}")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "goodness_of_fit.png", dpi=150)

# Plot 2: product preference by region
props = pd.crosstab(survey["region"], survey["product"], normalize="index"
                    ).loc[regions, products]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
props.plot(kind="bar", stacked=True, ax=axes[0], colormap="Set2")
axes[0].set_ylabel("Proportion within region")
axes[0].set_title("Preferred product by region")
axes[0].tick_params(axis="x", rotation=0)
sns.heatmap(adj_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[1])
axes[1].set_title("Adjusted standardised residuals (beyond +/-1.96 = notable)")
fig.tight_layout()
fig.savefig(OUT / "independence_test.png", dpi=150)

# Plot 3: observed against expected
fig, ax = plt.subplots(figsize=(10, 5))
labels = [f"{r}-{p_}" for r in regions for p_ in products]
xs = np.arange(len(labels))
ax.bar(xs - 0.2, obs.ravel(), 0.4, color="steelblue", label="Observed")
ax.bar(xs + 0.2, exp.ravel(), 0.4, color="lightgrey", label="Expected")
ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("Customers")
ax.set_title("Observed and expected counts in each cell")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "observed_vs_expected.png", dpi=150)

# Plot 4: chi-square distribution
fig, ax = plt.subplots(figsize=(8, 5))
xs = np.linspace(0, max(chi2 * 1.3, 20), 400)
ax.plot(xs, stats.chi2.pdf(xs, dof), "k-")
crit = stats.chi2.ppf(0.95, dof)
ax.fill_between(xs, stats.chi2.pdf(xs, dof), where=xs >= crit, color="tomato",
                alpha=0.5, label=f"Rejection region (> {crit:.2f})")
ax.axvline(chi2, color="blue", linestyle="--", label=f"Observed = {chi2:.2f}")
ax.set_xlabel("Chi-square")
ax.set_title(f"Chi-square distribution with {dof} degrees of freedom")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "chi_square_distribution.png", dpi=150)

# Plot 5: effect measures
fig, ax = plt.subplots(figsize=(8, 4))
for i, (_, r) in enumerate(eff.iloc[1:].iterrows()):
    ax.errorbar(r["estimate"], i, xerr=[[r["estimate"] - r["lower_95"]],
                                        [r["upper_95"] - r["estimate"]]],
                fmt="o", capsize=6, color="steelblue")
ax.axvline(1, color="black", linestyle="--", label="No effect (1)")
ax.set_yticks([0, 1])
ax.set_yticklabels(["Relative risk", "Odds ratio"])
ax.set_xscale("log")
ax.set_xlabel("Ratio (log scale) with 95% confidence interval")
ax.set_title("Drug against placebo: recovery")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "effect_measures.png", dpi=150)

# Plot 6: simulation
fig, ax = plt.subplots(figsize=(8, 5))
xs = np.arange(len(sim))
ax.bar(xs - 0.2, sim["chi_square_rejection"], 0.4, color="tomato",
       label="Chi-square (no correction)")
ax.bar(xs + 0.2, sim["fisher_rejection"], 0.4, color="steelblue", label="Fisher's exact")
ax.axhline(ALPHA, color="black", linestyle="--", label="Nominal 0.05")
ax.set_xticks(xs)
ax.set_xticklabels([f"n = {k}" for k in sim["n"]])
ax.set_ylabel("False positive rate when there is no association")
ax.set_title("Error rates of two tests for a 2x2 table")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "simulation_false_positives.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)