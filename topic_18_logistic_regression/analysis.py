"""
Topic 18: Logistic Regression
Libraries: statsmodels, scikit-learn, numpy, pandas, scipy, matplotlib, seaborn
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
sns.set_theme(style="whitegrid")
rng = np.random.default_rng(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------
# 1. Dataset (synthetic customer churn from a known logistic model)
# ---------------------------------------------------------------
n = 1000
tenure = rng.gamma(3.0, 10.0, n).clip(1, 72).round(0)
charges = rng.normal(70, 20, n).clip(20, 130).round(1)
calls = rng.poisson(2, n)
mtm = (rng.random(n) < 0.6).astype(int)
true_logit = -0.8 - 0.04 * tenure + 0.02 * (charges - 70) + 0.30 * calls + 0.9 * mtm
churn = (rng.random(n) < sigmoid(true_logit)).astype(int)

df = pd.DataFrame({
    "tenure_months": tenure, "monthly_charges": charges,
    "support_calls": calls, "month_to_month": mtm, "churn": churn,
})
df.to_csv(DATA / "churn.csv", index=False)
feats = ["tenure_months", "monthly_charges", "support_calls", "month_to_month"]
formula = "churn ~ " + " + ".join(feats)
y_all = df["churn"].to_numpy()

print("=== 1. Dataset ===")
print(f"{n} customers; churn rate = {df['churn'].mean():.3f} "
      f"({int(df['churn'].sum())} churned, {int((1 - df['churn']).sum())} stayed)")
print("Mean of each variable by outcome:")
print(df.groupby("churn")[feats].mean().round(2).to_string())
print("Churn rate by contract: month-to-month = "
      f"{df.loc[df['month_to_month'] == 1, 'churn'].mean():.3f}, annual = "
      f"{df.loc[df['month_to_month'] == 0, 'churn'].mean():.3f}")

# ---------------------------------------------------------------
# 2. Logistic regression on all the data
# ---------------------------------------------------------------
model = smf.logit(formula, data=df).fit(disp=False)
ci = model.conf_int()
true_coef = pd.Series({"Intercept": -0.8 - 0.02 * 70, "tenure_months": -0.04,
                       "monthly_charges": 0.02, "support_calls": 0.30,
                       "month_to_month": 0.90})
coef = pd.DataFrame({
    "coef": model.params, "std_err": model.bse, "z": model.tvalues,
    "p_value": model.pvalues, "odds_ratio": np.exp(model.params),
    "or_lower_95": np.exp(ci[0]), "or_upper_95": np.exp(ci[1]),
    "true_coef": true_coef,
})
coef["true_in_95_ci"] = (ci[0] <= true_coef) & (true_coef <= ci[1])
print("\n=== 2. Logistic regression: log-odds scale and odds ratios ===")
print(f"Converged: {model.mle_retvals['converged']}")
print(coef.round(4).to_string())
coef.round(5).to_csv(OUT / "coefficients.csv")
terms = " ".join(f"{model.params[f]:+.4f}*{f}" for f in feats)
print(f"\nlog-odds of churn = {model.params['Intercept']:.4f} {terms}")

b_ten = model.params["tenure_months"]
lo12, hi12 = np.exp(12 * ci.loc["tenure_months"])
print(f"\nOdds ratio for 12 more months of tenure = {np.exp(12 * b_ten):.3f} "
      f"(95% CI {lo12:.3f} to {hi12:.3f})")

# ---------------------------------------------------------------
# 3. Overall fit
# ---------------------------------------------------------------
print("\n=== 3. Model fit ===")
print(f"Log-likelihood = {model.llf:.2f}, null log-likelihood = {model.llnull:.2f}")
print(f"Likelihood ratio test: chi-square = {model.llr:.2f} on {int(model.df_model)} df, "
      f"p = {model.llr_pvalue:.4g}")
print(f"McFadden pseudo R-squared = {model.prsquared:.4f}")
print(f"AIC = {model.aic:.2f}, BIC = {model.bic:.2f}")

pf = model.predict(df).to_numpy()
grp = pd.qcut(pd.Series(pf).rank(method="first"), 10, labels=False).to_numpy()
hl, hl_rows = 0.0, []
for g in range(10):
    m = grp == g
    o, e, k = y_all[m].sum(), pf[m].sum(), m.sum()
    hl += (o - e) ** 2 / (k * (e / k) * (1 - e / k))
    hl_rows.append((g + 1, k, o, e))
hl_p = stats.chi2.sf(hl, 8)
print(f"Hosmer-Lemeshow test (10 groups): chi-square = {hl:.3f} on 8 df, p = {hl_p:.4f}")
pd.DataFrame(hl_rows, columns=["decile", "n", "observed_churn", "expected_churn"]
             ).round(3).to_csv(OUT / "hosmer_lemeshow.csv", index=False)

X = model.model.exog
vif = pd.Series([variance_inflation_factor(X, i) for i in range(1, X.shape[1])],
                index=feats)
print("Variance inflation factors:", vif.round(3).to_dict())

# ---------------------------------------------------------------
# 4. Marginal effects and predicted probabilities
# ---------------------------------------------------------------
me = model.get_margeff(at="overall").summary_frame()
print("\n=== 4a. Average marginal effects (change in churn probability per unit) ===")
print(me[["dy/dx", "Pr(>|z|)"]].round(5).to_string())
me.round(6).to_csv(OUT / "marginal_effects.csv")

profiles = pd.DataFrame({
    "tenure_months": [3, 24, 48], "monthly_charges": [90.0, 70.0, 50.0],
    "support_calls": [4, 2, 0], "month_to_month": [1, 1, 0],
}, index=["High risk", "Average", "Low risk"])
prob_model = model.predict(profiles).to_numpy()
Xp = np.column_stack([np.ones(len(profiles)), profiles[feats].to_numpy()])
eta = Xp @ model.params.to_numpy()
prob_hand = sigmoid(eta)
prof = profiles.copy()
prof["log_odds"] = eta
prof["odds"] = np.exp(eta)
prof["probability"] = prob_model
print("\n=== 4b. Predicted churn probability for three example customers ===")
print(prof.round(4).to_string())
print("Hand calculation 1 / (1 + exp(-eta)) matches the model:",
      np.allclose(prob_hand, prob_model))
prof.round(5).to_csv(OUT / "example_predictions.csv")

# ---------------------------------------------------------------
# 5. Train / test split and classification performance
# ---------------------------------------------------------------
train, test = train_test_split(df, test_size=0.30, stratify=df["churn"],
                               random_state=42)
y_tr, y_te = train["churn"].to_numpy(), test["churn"].to_numpy()
m_tr = smf.logit(formula, data=train).fit(disp=False)
p_tr = m_tr.predict(train).to_numpy()
p_te = m_tr.predict(test).to_numpy()

sk = LogisticRegression(penalty=None, max_iter=10000).fit(train[feats], y_tr)
sk_params = np.r_[sk.intercept_, sk.coef_[0]]
diff = np.abs(sk_params - m_tr.params.to_numpy()).max()
print("\n=== 5a. Training and test sets (stratified 70% / 30%) ===")
print(f"Training: {len(train)} customers, churn rate {y_tr.mean():.3f}; "
      f"test: {len(test)} customers, churn rate {y_te.mean():.3f}")
print(f"Largest difference between statsmodels and scikit-learn coefficients "
      f"= {diff:.2e}")


def class_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp > 0 else np.nan
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    spec = tn / (tn + fp) if tn + fp > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (tp + fp > 0 and prec + rec > 0) else 0.0
    return {"threshold": thr, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "specificity": spec, "F1": f1, "balanced_accuracy": (rec + spec) / 2}


fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr, p_tr)
thr_youden = float(thr_tr[np.argmax(tpr_tr - fpr_tr)])
grid = np.linspace(0.05, 0.95, 91)
thr_f1 = float(grid[np.argmax([class_metrics(y_tr, p_tr, g)["F1"] for g in grid])])
print(f"Thresholds chosen on the training data: Youden's J = {thr_youden:.3f}, "
      f"best F1 = {thr_f1:.3f}")

thr_list = [0.2, 0.3, 0.4, 0.5, 0.6, round(thr_youden, 3)]
tab = pd.DataFrame([class_metrics(y_te, p_te, t_) for t_ in thr_list])
print("\n=== 5b. Test-set performance at different thresholds ===")
print(tab.round(3).to_string(index=False))
tab.round(5).to_csv(OUT / "classification_metrics.csv", index=False)

base_acc = max(y_te.mean(), 1 - y_te.mean())
print(f"\nBaseline accuracy (always predict the majority class) = {base_acc:.3f}")

cm5 = class_metrics(y_te, p_te, 0.5)
cmy = class_metrics(y_te, p_te, thr_youden)
print(f"\nConfusion matrix at threshold 0.5 (rows = actual, columns = predicted):")
print(pd.DataFrame([[cm5["TN"], cm5["FP"]], [cm5["FN"], cm5["TP"]]],
                   index=["Actual: stayed", "Actual: churned"],
                   columns=["Predicted: stayed", "Predicted: churned"]).to_string())
print(f"Confusion matrix at the Youden threshold ({thr_youden:.3f}):")
print(pd.DataFrame([[cmy["TN"], cmy["FP"]], [cmy["FN"], cmy["TP"]]],
                   index=["Actual: stayed", "Actual: churned"],
                   columns=["Predicted: stayed", "Predicted: churned"]).to_string())

# ---------------------------------------------------------------
# 6. Ranking quality, calibration and cross-validation
# ---------------------------------------------------------------
auc_te = roc_auc_score(y_te, p_te)
auc_tr = roc_auc_score(y_tr, p_tr)
ap = average_precision_score(y_te, p_te)
brier = brier_score_loss(y_te, p_te)
brier_base = y_te.mean() * (1 - y_te.mean())
boot = []
for _ in range(1000):
    idx = rng.integers(0, len(y_te), len(y_te))
    if len(np.unique(y_te[idx])) == 2:
        boot.append(roc_auc_score(y_te[idx], p_te[idx]))
auc_lo, auc_hi = np.percentile(boot, [2.5, 97.5])
cv = cross_val_score(LogisticRegression(penalty=None, max_iter=10000), df[feats], y_all,
                     cv=StratifiedKFold(5, shuffle=True, random_state=42),
                     scoring="roc_auc")
print("\n=== 6. Ranking quality and calibration ===")
print(f"ROC AUC: training = {auc_tr:.4f}, test = {auc_te:.4f} "
      f"(bootstrap 95% interval {auc_lo:.3f} to {auc_hi:.3f})")
print(f"5-fold cross-validated AUC: mean = {cv.mean():.4f}, sd = {cv.std(ddof=1):.4f}")
print(f"Average precision = {ap:.4f} (a model with no skill scores about "
      f"{y_te.mean():.3f})")
print(f"Brier score = {brier:.4f} (always predicting the churn rate scores "
      f"{brier_base:.4f})")

bins = pd.qcut(pd.Series(p_te).rank(method="first"), 5, labels=False).to_numpy()
cal = pd.DataFrame([(b + 1, int((bins == b).sum()), p_te[bins == b].mean(),
                     y_te[bins == b].mean()) for b in range(5)],
                   columns=["bin", "n", "mean_predicted", "observed_rate"])
print("\nCalibration on the test set (5 groups of customers ordered by risk):")
print(cal.round(3).to_string(index=False))
cal.round(4).to_csv(OUT / "calibration.csv", index=False)

# ---------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------
# Plot 1: predicted probability against tenure
ts = np.arange(1, 73)
mean_ch, mean_ca = df["monthly_charges"].mean(), df["support_calls"].mean()
fig, ax = plt.subplots(figsize=(9, 5.5))
for flag, colour, label in ((1, "tab:red", "Month-to-month"), (0, "tab:blue", "Annual")):
    grid_df = pd.DataFrame({"tenure_months": ts, "monthly_charges": mean_ch,
                            "support_calls": mean_ca, "month_to_month": flag})
    ax.plot(ts, model.predict(grid_df), color=colour, lw=2.5,
            label=f"{label} (other variables at their means)")
    sub = df[df["month_to_month"] == flag].copy()
    sub["bin"] = pd.cut(sub["tenure_months"], [0, 12, 24, 36, 48, 72])
    obs = sub.groupby("bin", observed=True).agg(t=("tenure_months", "mean"),
                                                r=("churn", "mean"))
    ax.scatter(obs["t"], obs["r"], color=colour, s=45, edgecolor="black", zorder=3)
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Probability of churn")
ax.set_title("The logistic curve (dots = observed churn rate in tenure bands)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "probability_curves.png", dpi=150)

# Plot 2: odds ratios
fig, ax = plt.subplots(figsize=(8, 4.5))
rows_ = coef.drop(index="Intercept")
for i, (name, r) in enumerate(rows_.iterrows()):
    ax.errorbar(r["odds_ratio"], i, xerr=[[r["odds_ratio"] - r["or_lower_95"]],
                                           [r["or_upper_95"] - r["odds_ratio"]]],
                fmt="o", capsize=5, color="steelblue")
ax.axvline(1, color="black", linestyle="--", label="No effect (odds ratio = 1)")
ax.set_yticks(range(len(rows_)))
ax.set_yticklabels(rows_.index)
ax.set_xscale("log")
ax.set_xlabel("Odds ratio per one-unit increase (log scale), 95% interval")
ax.set_title("Odds ratios of the logistic regression")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "odds_ratios.png", dpi=150)

# Plot 3: ROC and precision-recall
fpr, tpr, _ = roc_curve(y_te, p_te)
prec_c, rec_c, _ = precision_recall_curve(y_te, p_te)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
axes[0].plot(fpr, tpr, color="steelblue", lw=2.5, label=f"Model (AUC = {auc_te:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", label="No skill")
axes[0].set_xlabel("False positive rate (1 - specificity)")
axes[0].set_ylabel("True positive rate (recall)")
axes[0].set_title("ROC curve (test set)")
axes[0].legend()
axes[1].plot(rec_c, prec_c, color="tomato", lw=2.5, label=f"Model (AP = {ap:.3f})")
axes[1].axhline(y_te.mean(), color="black", linestyle="--", label="No skill")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-recall curve (test set)")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / "roc_pr_curves.png", dpi=150)

# Plot 4: confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, cmx, title in ((axes[0], cm5, "Threshold 0.5"),
                       (axes[1], cmy, f"Threshold {thr_youden:.3f} (Youden)")):
    mat = np.array([[cmx["TN"], cmx["FP"]], [cmx["FN"], cmx["TP"]]])
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred. stayed", "Pred. churned"],
                yticklabels=["Actual stayed", "Actual churned"])
    ax.set_title(f"{title}: accuracy {cmx['accuracy']:.3f}, recall {cmx['recall']:.3f}")
fig.tight_layout()
fig.savefig(OUT / "confusion_matrices.png", dpi=150)

# Plot 5: threshold trade-off on the test set
curve = pd.DataFrame([class_metrics(y_te, p_te, g) for g in grid])
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(curve["threshold"], curve["precision"], label="Precision")
ax.plot(curve["threshold"], curve["recall"], label="Recall")
ax.plot(curve["threshold"], curve["specificity"], label="Specificity")
ax.plot(curve["threshold"], curve["F1"], "k--", label="F1")
ax.axvline(thr_youden, color="grey", linestyle=":", label=f"Youden = {thr_youden:.2f}")
ax.set_xlabel("Classification threshold")
ax.set_ylabel("Value")
ax.set_title("Moving the threshold trades precision against recall")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "threshold_tradeoff.png", dpi=150)

# Plot 6: calibration and predicted probabilities
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
axes[0].plot(cal["mean_predicted"], cal["observed_rate"], "o-", color="tomato",
             label="Model (5 risk groups)")
axes[0].set_xlabel("Mean predicted probability")
axes[0].set_ylabel("Observed churn rate")
axes[0].set_title("Calibration (test set)")
axes[0].legend()
axes[1].hist(p_te[y_te == 0], bins=20, alpha=0.6, color="steelblue", label="Stayed")
axes[1].hist(p_te[y_te == 1], bins=20, alpha=0.6, color="tomato", label="Churned")
axes[1].set_xlabel("Predicted probability of churn")
axes[1].set_ylabel("Customers")
axes[1].set_title("Predicted probabilities by actual outcome")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / "calibration_and_distribution.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)