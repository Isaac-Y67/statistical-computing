"""
Topic 22: Missing-Data Analysis and Imputation
Libraries: pandas, numpy, scipy, statsmodels, scikit-learn (impute), matplotlib, seaborn
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
sns.set_theme(style="whitegrid")

COLS = ["age", "education_years", "income_k", "health_score"]
MISSING_COLS = ["education_years", "income_k", "health_score"]
METHODS = ["Mean imputation", "Regression imputation", "Stochastic regression",
           "kNN (k = 5)"]
SHORT = {"Complete cases": "Complete\ncases", "Mean imputation": "Mean",
         "Regression imputation": "Regression", "Stochastic regression": "Stochastic",
         "kNN (k = 5)": "kNN"}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------
# 1. Data generation: complete data, then deletion by three mechanisms
# ---------------------------------------------------------------
def generate(rg, n=1000):
    age = rg.normal(45, 12, n).clip(20, 80)
    edu = rg.normal(14, 3, n).clip(6, 22)
    income = (20 + 2.5 * edu + 0.3 * age + rg.normal(0, 10, n)).clip(5, None)
    health = 60 - 0.3 * age + 0.2 * income + rg.normal(0, 8, n)
    full = pd.DataFrame({"age": age, "education_years": edu, "income_k": income,
                         "health_score": health}).round(2)
    z_h = (health - health.mean()) / health.std()
    m_edu = rg.random(n) < 0.15                                  # MCAR
    m_inc = rg.random(n) < sigmoid(-1.6 + 0.06 * (age - 45))     # MAR (depends on age)
    m_hea = rg.random(n) < sigmoid(-1.7 - 1.2 * z_h)             # MNAR (depends on health)
    miss = full.copy()
    miss.loc[m_edu, "education_years"] = np.nan
    miss.loc[m_inc, "income_k"] = np.nan
    miss.loc[m_hea, "health_score"] = np.nan
    return full, miss


def estimates(d):
    X = np.column_stack([np.ones(len(d)), d["age"], d["income_k"]])
    b = np.linalg.lstsq(X, d["health_score"].to_numpy(), rcond=None)[0]
    return {"mean_income": d["income_k"].mean(), "sd_income": d["income_k"].std(ddof=1),
            "mean_health": d["health_score"].mean(),
            "sd_health": d["health_score"].std(ddof=1),
            "corr_income_health": d["income_k"].corr(d["health_score"]),
            "slope_income": b[2], "slope_age": b[1]}


def impute(method, d, seed=0):
    X = d.to_numpy(dtype=float)
    if method == "Mean imputation":
        out = SimpleImputer(strategy="mean").fit_transform(X)
    elif method == "kNN (k = 5)":
        sc = StandardScaler().fit(X)
        out = sc.inverse_transform(KNNImputer(n_neighbors=5).fit_transform(sc.transform(X)))
    else:
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                               sample_posterior=(method == "Stochastic regression"),
                               min_value=np.nanmin(X, axis=0),
                               max_value=np.nanmax(X, axis=0), random_state=seed)
        out = imp.fit_transform(X)
    return pd.DataFrame(out, columns=d.columns)


rng = np.random.default_rng(42)
full, miss = generate(rng)
full.to_csv(DATA / "complete_data.csv", index=False)
miss.to_csv(DATA / "data_with_missing.csv", index=False)
truth = estimates(full)
n = len(full)

print("=== 1. Dataset ===")
print(f"{n} people, variables: {', '.join(COLS)}")
print("Complete data (before deletion): means = "
      f"{full.mean().round(2).to_dict()}")

# ---------------------------------------------------------------
# 2. Summary of the missing data
# ---------------------------------------------------------------
na = miss.isna()
summ = pd.DataFrame({"missing": na.sum(), "percent": na.mean() * 100})
print("\n=== 2. Missing-data summary ===")
print(summ.round(1).to_string())
labels = na.apply(lambda r: "+".join(c for c in COLS if r[c]) or "complete", axis=1)
pat = labels.value_counts().to_frame("rows")
pat["percent"] = pat["rows"] / n * 100
print("\nMissingness patterns:")
print(pat.round(1).to_string())
n_complete = int((~na.any(axis=1)).sum())
print(f"Complete cases = {n_complete} of {n} ({n_complete / n:.1%}); "
      f"{n - n_complete} rows would be lost by listwise deletion")
print(f"Share of all cells that are missing = {na.to_numpy().mean():.1%}")
summ.round(3).to_csv(OUT / "missing_summary.csv")
pat.round(3).to_csv(OUT / "patterns.csv")

# ---------------------------------------------------------------
# 3. What the observed data say about the mechanism
# ---------------------------------------------------------------
print("\n=== 3a. Does missingness in one variable relate to the others? "
      "(Welch t-tests on the observed values) ===")
mech_rows = []
for target in MISSING_COLS:
    ind = na[target]
    for other in COLS:
        if other == target:
            continue
        ok = miss[other].notna()
        a, b = miss.loc[ok & ind, other], miss.loc[ok & ~ind, other]
        t = stats.ttest_ind(a, b, equal_var=False)
        mech_rows.append((target, other, a.mean(), b.mean(), len(a), t.pvalue))
        print(f"{target:16s} missing vs observed -> {other:16s}: mean {a.mean():7.2f} vs "
              f"{b.mean():7.2f} (n = {len(a)}), p = {t.pvalue:.4g}")
pd.DataFrame(mech_rows, columns=["target", "other", "mean_when_missing",
                                 "mean_when_observed", "n_missing", "p_value"]
             ).round(5).to_csv(OUT / "mechanism_tests.csv", index=False)

print("\n=== 3b. Logistic regression of each missingness indicator on age ===")
for target in MISSING_COLS:
    lg = sm.Logit(na[target].astype(int), sm.add_constant(miss["age"])).fit(disp=False)
    print(f"{target:16s}: odds ratio per year of age = {np.exp(lg.params['age']):.3f}, "
          f"p = {lg.pvalues['age']:.4g}")

print("\n=== 3c. What we know because we deleted the values ourselves ===")
print("True (deleted) health scores: mean of those that went missing = "
      f"{full.loc[na['health_score'], 'health_score'].mean():.2f}, mean of those "
      f"observed = {full.loc[~na['health_score'], 'health_score'].mean():.2f}")
print("Mean age of people with missing income = "
      f"{full.loc[na['income_k'], 'age'].mean():.2f}, with observed income = "
      f"{full.loc[~na['income_k'], 'age'].mean():.2f}")

# ---------------------------------------------------------------
# 4. Single imputation methods against the full data
# ---------------------------------------------------------------
imputed = {m: impute(m, miss, seed=1) for m in METHODS}
for m, d in imputed.items():
    d.round(3).to_csv(DATA / f"imputed_{m.split()[0].lower()}.csv", index=False)
est = {"Full data (truth)": truth, "Complete cases": estimates(miss.dropna())}
for m, d in imputed.items():
    est[m] = estimates(d)
comp = pd.DataFrame(est).T
print("\n=== 4a. Estimates from each approach (one dataset) ===")
print(comp.round(3).to_string())
bias = comp.drop(index="Full data (truth)") - comp.loc["Full data (truth)"]
print("\nDifference from the full-data estimate:")
print(bias.round(3).to_string())
comp.round(5).to_csv(OUT / "method_comparison.csv")

rm = {}
for m, d in imputed.items():
    rm[m] = {c: float(np.sqrt(np.mean((d.loc[na[c], c] - full.loc[na[c], c]) ** 2)))
             for c in MISSING_COLS}
rm_tab = pd.DataFrame(rm).T
print("\n=== 4b. Root mean squared error of the imputed values against the true "
      "(deleted) values ===")
print(rm_tab.round(3).to_string())
sd_true = {c: float(full.loc[na[c], c].std(ddof=0)) for c in MISSING_COLS}
print("Standard deviation of the true deleted values:",
      {k: round(v, 3) for k, v in sd_true.items()})
rm_tab.round(5).to_csv(OUT / "imputation_rmse.csv")

# ---------------------------------------------------------------
# 5. Multiple imputation with Rubin's rules
# ---------------------------------------------------------------
def rubin(q, u):
    q, u = np.asarray(q, float), np.asarray(u, float)
    m_ = len(q)
    qbar, w, b = q.mean(), u.mean(), q.var(ddof=1)
    t_ = w + (1 + 1 / m_) * b
    r_ = (1 + 1 / m_) * b / w
    df_ = (m_ - 1) * (1 + 1 / r_) ** 2 if b > 0 else np.inf
    se_ = np.sqrt(t_)
    tc = stats.t.ppf(0.975, df_)
    return qbar, se_, qbar - tc * se_, qbar + tc * se_, (1 + 1 / m_) * b / t_, df_


def mean_ci(s):
    m_, se_ = s.mean(), s.std(ddof=1) / np.sqrt(len(s))
    tc = stats.t.ppf(0.975, len(s) - 1)
    return m_, m_ - tc * se_, m_ + tc * se_


def ols_fit(d):
    return sm.OLS(d["health_score"], sm.add_constant(d[["age", "income_k"]])).fit()


M = 20
params = ["mean_income", "mean_health", "slope_income", "slope_age"]
Q = {k: [] for k in params}
U = {k: [] for k in params}
for i in range(M):
    d = impute("Stochastic regression", miss, seed=100 + i)
    o = ols_fit(d)
    Q["mean_income"].append(d["income_k"].mean())
    U["mean_income"].append(d["income_k"].var(ddof=1) / n)
    Q["mean_health"].append(d["health_score"].mean())
    U["mean_health"].append(d["health_score"].var(ddof=1) / n)
    Q["slope_income"].append(o.params["income_k"])
    U["slope_income"].append(o.bse["income_k"] ** 2)
    Q["slope_age"].append(o.params["age"])
    U["slope_age"].append(o.bse["age"] ** 2)
    if i == 0:
        d.round(3).to_csv(DATA / "imputed_multiple_first.csv", index=False)

pooled = {k: rubin(Q[k], U[k]) for k in params}
of, occ = ols_fit(full), ols_fit(miss.dropna())
cif, cicc = of.conf_int(), occ.conf_int()
cc = miss.dropna()
population = {"mean_income": 68.5, "mean_health": 60.2, "slope_income": 0.2,
              "slope_age": -0.3}
mi_rows = []
for k in params:
    if k == "mean_income":
        f_, c_ = mean_ci(full["income_k"]), mean_ci(cc["income_k"])
    elif k == "mean_health":
        f_, c_ = mean_ci(full["health_score"]), mean_ci(cc["health_score"])
    else:
        nm = "income_k" if k == "slope_income" else "age"
        f_ = (of.params[nm], cif.loc[nm, 0], cif.loc[nm, 1])
        c_ = (occ.params[nm], cicc.loc[nm, 0], cicc.loc[nm, 1])
    p_ = pooled[k]
    mi_rows += [(k, "Full data", *f_), (k, "Complete cases", *c_),
                (k, f"Multiple imputation (m = {M})", p_[0], p_[2], p_[3])]
mi_tab = pd.DataFrame(mi_rows, columns=["parameter", "method", "estimate", "lower_95",
                                        "upper_95"])
mi_tab["population_value"] = mi_tab["parameter"].map(population)
mi_tab["interval_contains_population_value"] = (
    (mi_tab["lower_95"] <= mi_tab["population_value"])
    & (mi_tab["population_value"] <= mi_tab["upper_95"]))
print(f"\n=== 5. Multiple imputation ({M} datasets, Rubin's rules) ===")
print(mi_tab.round(3).to_string(index=False))
print("(population values are approximate for the means: 68.5 and 60.2)")
print("\nPooling details:")
for k in params:
    q_, se_, lo_, hi_, lam, df_ = pooled[k]
    print(f"{k:13s}: pooled estimate = {q_:8.4f}, pooled SE = {se_:.4f}, "
          f"fraction of missing information = {lam:.3f}, df = {df_:.0f}, "
          f"between-imputation sd = {np.std(Q[k], ddof=1):.4f}")
mi_tab.round(5).to_csv(OUT / "multiple_imputation.csv", index=False)

# ---------------------------------------------------------------
# 6. Simulation: bias of each approach over repeated datasets
# ---------------------------------------------------------------
N_SIM = 200
sim_methods = ["Complete cases"] + METHODS
keys = ["mean_income", "sd_income", "mean_health", "sd_health",
        "corr_income_health", "slope_income", "slope_age"]
errs = {m: {k: [] for k in keys} for m in sim_methods}
sr = np.random.default_rng(7)
for r in range(N_SIM):
    f_, ms_ = generate(sr)
    tgt = estimates(f_)
    for m in sim_methods:
        d = ms_.dropna() if m == "Complete cases" else impute(m, ms_, seed=r)
        e = estimates(d)
        for k in keys:
            errs[m][k].append(e[k] - tgt[k])
bias_tab = pd.DataFrame({m: {k: np.mean(v) for k, v in errs[m].items()}
                         for m in sim_methods}).T
rmse_sim = pd.DataFrame({m: {k: np.sqrt(np.mean(np.square(v))) for k, v in errs[m].items()}
                         for m in sim_methods}).T
print(f"\n=== 6. Simulation: {N_SIM} datasets of {n} people ===")
print("Average error of each estimate relative to the full-data estimate (bias):")
print(bias_tab.round(3).to_string())
print("\nRoot mean squared error:")
print(rmse_sim.round(3).to_string())
print("True full-data values in the simulation are about: mean income 68.5, "
      "mean health 60.2, sd income 13, slope of income 0.2")
bias_tab.round(5).to_csv(OUT / "simulation_bias.csv")
rmse_sim.round(5).to_csv(OUT / "simulation_rmse.csv")

# ---------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------
# Plot 1: missingness overview
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
axes[0].bar(COLS, summ["percent"], color="tomato")
axes[0].set_ylabel("Percent missing")
axes[0].set_title("Missing values per variable")
axes[0].tick_params(axis="x", rotation=20)
axes[1].imshow(na.iloc[:150].T.to_numpy(), aspect="auto", cmap="Greys",
               interpolation="nearest")
axes[1].set_yticks(range(len(COLS)))
axes[1].set_yticklabels(COLS)
axes[1].set_xlabel("Row (first 150)")
axes[1].set_title("Missingness map (black = missing)")
axes[2].barh(pat.index[::-1], pat["rows"][::-1], color="steelblue")
axes[2].set_xlabel("Rows")
axes[2].set_title("Missingness patterns")
fig.tight_layout()
fig.savefig(OUT / "missingness_overview.png", dpi=150)

# Plot 2: mechanisms
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
specs = [("age", "education_years", "MCAR: age by education missing"),
         ("age", "income_k", "MAR: age by income missing"),
         ("health_score", "health_score", "MNAR: true health by health missing")]
for ax, (xv, tv, title) in zip(axes, specs):
    dd = pd.DataFrame({"value": full[xv],
                       "status": np.where(na[tv], "missing", "observed")})
    sns.kdeplot(data=dd, x="value", hue="status", common_norm=False, fill=True,
                alpha=0.35, ax=ax)
    ax.set_xlabel(xv)
    ax.set_title(title)
fig.tight_layout()
fig.savefig(OUT / "missingness_mechanisms.png", dpi=150)

# Plot 3: distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, c, title in ((axes[0], "income_k", "Income"), (axes[1], "health_score", "Health score")):
    ax.hist(full[c], bins=40, density=True, histtype="step", lw=2.5, color="black",
            label="Full data (truth)")
    ax.hist(miss[c].dropna(), bins=40, density=True, histtype="step", lw=2,
            linestyle="--", color="grey", label="Observed values only")
    for m_, col in zip(METHODS, ["tab:red", "tab:orange", "tab:green", "tab:blue"]):
        ax.hist(imputed[m_][c], bins=40, density=True, histtype="step", lw=1.5,
                color=col, label=m_)
    ax.set_title(f"{title}: distribution after imputation")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "distributions_after_imputation.png", dpi=150)

# Plot 4: scatter of health against income
any_imp = (na["income_k"] | na["health_score"]).to_numpy()
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
for ax, m_ in zip(axes.ravel(), METHODS):
    d = imputed[m_]
    ax.scatter(d["income_k"][~any_imp], d["health_score"][~any_imp], s=10, alpha=0.4,
               color="steelblue", label="Observed")
    ax.scatter(d["income_k"][any_imp], d["health_score"][any_imp], s=12, alpha=0.6,
               color="red", label="Imputed")
    ax.set_title(f"{m_}: r = {d['income_k'].corr(d['health_score']):.3f} "
                 f"(truth {truth['corr_income_health']:.3f})")
    ax.legend(fontsize=8)
for ax in axes[1]:
    ax.set_xlabel("Income (thousands)")
for ax in axes[:, 0]:
    ax.set_ylabel("Health score")
fig.tight_layout()
fig.savefig(OUT / "scatter_imputed_points.png", dpi=150)

# Plot 5: simulation bias
plot_keys = ["mean_income", "mean_health", "sd_income", "corr_income_health",
             "slope_income"]
fig, axes = plt.subplots(1, 5, figsize=(20, 4.6))
for ax, k in zip(axes, plot_keys):
    vals = bias_tab[k]
    ax.bar([SHORT[m] for m in vals.index], vals.values,
           color=["grey", "tab:red", "tab:orange", "tab:green", "tab:blue"])
    ax.axhline(0, color="black", lw=1)
    ax.set_title(f"Bias in {k}")
    ax.tick_params(axis="x", rotation=30)
fig.suptitle(f"Average error relative to the full data ({N_SIM} simulated datasets)")
fig.tight_layout()
fig.savefig(OUT / "simulation_bias.png", dpi=150)

# Plot 6: multiple imputation
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, k in zip(axes, ["mean_health", "slope_income"]):
    ax.scatter(range(1, M + 1), Q[k], color="steelblue", label="Each imputed dataset")
    q_, _, lo_, hi_, _, _ = pooled[k]
    ax.axhline(q_, color="red", lw=2, label="Pooled estimate")
    ax.axhspan(lo_, hi_, color="red", alpha=0.12, label="Pooled 95% interval")
    fr = mi_tab[(mi_tab["parameter"] == k) & (mi_tab["method"] == "Full data")]["estimate"].iloc[0]
    cr = mi_tab[(mi_tab["parameter"] == k) & (mi_tab["method"] == "Complete cases")]["estimate"].iloc[0]
    ax.axhline(fr, color="black", linestyle="--", label="Full data")
    ax.axhline(cr, color="grey", linestyle=":", lw=2, label="Complete cases")
    ax.set_xlabel("Imputation number")
    ax.set_title(f"Multiple imputation: {k}")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "multiple_imputation.png", dpi=150)

# Plot 7: accuracy of imputed values
fig, ax = plt.subplots(figsize=(9, 5))
rm_tab.plot(kind="bar", ax=ax)
ax.set_ylabel("RMSE of imputed values")
ax.set_title("Error of the imputed values against the true deleted values")
ax.tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig(OUT / "imputation_accuracy.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)