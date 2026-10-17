"""
Topic 7: Correlation and Covariance
Libraries: pandas, numpy, scipy.stats, matplotlib, seaborn
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# ---------------------------------------------------------------
# 1. Dataset (synthetic, reproducible)
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
n = 200
study = rng.normal(5, 1.5, n).clip(0.5, 10)
sleep = rng.normal(7, 1, n).clip(4, 10)
attendance = (70 + 4 * study + rng.normal(0, 8, n)).clip(30, 100)
screen = rng.normal(4, 1.5, n).clip(0.5, 10)
exam = (20 + 6 * study + 2 * sleep + 0.15 * attendance - 1.5 * screen
        + rng.normal(0, 6, n)).clip(0, 100)

df = pd.DataFrame({
    "study_hours": study,
    "sleep_hours": sleep,
    "attendance_pct": attendance,
    "screen_hours": screen,
    "exam_score": exam,
}).round(2)
df.to_csv(DATA / "student_data.csv", index=False)

print("=== Dataset ===")
print(f"{n} students, variables: {', '.join(df.columns)}")

# ---------------------------------------------------------------
# 2. Covariance matrix
# ---------------------------------------------------------------
cov = df.cov()
print("\n=== Covariance matrix ===")
print(cov.round(3))
cov.round(4).to_csv(OUT / "covariance_matrix.csv")

# ---------------------------------------------------------------
# 3. Correlation matrices
# ---------------------------------------------------------------
pearson = df.corr(method="pearson")
spearman = df.corr(method="spearman")
print("\n=== Pearson correlation matrix ===")
print(pearson.round(3))
print("\n=== Spearman correlation matrix ===")
print(spearman.round(3))
pearson.round(4).to_csv(OUT / "pearson_matrix.csv")
spearman.round(4).to_csv(OUT / "spearman_matrix.csv")

# Check: correlation = covariance / (sd_x * sd_y)
sd = df.std(ddof=1)
manual = cov / np.outer(sd, sd)
print("\nCorrelation = covariance / (sd_x * sd_y) matches pandas:",
      np.allclose(manual, pearson))

# ---------------------------------------------------------------
# 4. Correlations with exam score: r, p-value, 95% CI (Fisher z)
# ---------------------------------------------------------------
rows = []
print("\n=== Correlation with exam_score ===")
for col in ["study_hours", "sleep_hours", "attendance_pct", "screen_hours"]:
    res = stats.pearsonr(df[col], df["exam_score"])
    r = res.statistic
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    rho = stats.spearmanr(df[col], df["exam_score"]).statistic
    print(f"{col:15s} r = {r:6.3f}, 95% CI = ({lo:.3f}, {hi:.3f}), "
          f"p = {res.pvalue:.4f}, r^2 = {r ** 2:.3f}, Spearman = {rho:.3f}")
    rows.append({"variable": col, "pearson_r": r, "ci_lower": lo, "ci_upper": hi,
                 "p_value": res.pvalue, "r_squared": r ** 2, "spearman_rho": rho})
pd.DataFrame(rows).round(4).to_csv(OUT / "exam_score_correlations.csv", index=False)

# ---------------------------------------------------------------
# 5. Effect of one outlier
# ---------------------------------------------------------------
x_o = np.append(df["study_hours"].to_numpy(), 1.0)
y_o = np.append(df["exam_score"].to_numpy(), 100.0)
r_before = stats.pearsonr(df["study_hours"], df["exam_score"]).statistic
r_after = stats.pearsonr(x_o, y_o).statistic
rho_after = stats.spearmanr(x_o, y_o).statistic
print("\n=== Effect of one outlier (1 hour of study, score 100) ===")
print(f"Pearson before = {r_before:.3f}, after = {r_after:.3f}, "
      f"Spearman after = {rho_after:.3f}")

# ---------------------------------------------------------------
# 6. Pearson vs Spearman, and a relationship correlation misses
# ---------------------------------------------------------------
x = np.linspace(0.5, 10, 100)
y_exp = np.exp(0.5 * x) + rng.normal(0, 5, x.size)
xq = np.linspace(-5, 5, 100)
y_quad = xq ** 2 + rng.normal(0, 1.5, xq.size)
r_exp = stats.pearsonr(x, y_exp).statistic
rho_exp = stats.spearmanr(x, y_exp).statistic
r_quad = stats.pearsonr(xq, y_quad).statistic
rho_quad = stats.spearmanr(xq, y_quad).statistic
print("\n=== Non-linear relationships ===")
print(f"Exponential growth: Pearson = {r_exp:.3f}, Spearman = {rho_exp:.3f}")
print(f"U-shaped curve:     Pearson = {r_quad:.3f}, Spearman = {rho_quad:.3f}")

print("\nNote: correlation does not imply causation.")

# ---------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------
sns.set_theme(style="whitegrid")

# Heatmap of both matrices
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(pearson, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            ax=axes[0])
axes[0].set_title("Pearson correlation")
sns.heatmap(spearman, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            ax=axes[1])
axes[1].set_title("Spearman correlation")
fig.tight_layout()
fig.savefig(OUT / "correlation_heatmaps.png", dpi=150)

# Scatter plot with fitted line
slope, intercept = np.polyfit(df["study_hours"], df["exam_score"], 1)
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(df["study_hours"], df["exam_score"], alpha=0.6, color="steelblue")
xs = np.linspace(df["study_hours"].min(), df["study_hours"].max(), 100)
ax.plot(xs, intercept + slope * xs, "r-", lw=2,
        label=f"Fitted line, r = {r_before:.2f}")
ax.set_xlabel("Study hours per day")
ax.set_ylabel("Exam score")
ax.set_title("Exam score against study hours")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "scatter_study_exam.png", dpi=150)

# Scatterplot matrix
pair = sns.pairplot(df, corner=True, plot_kws={"alpha": 0.5, "s": 20})
pair.figure.suptitle("Scatterplot matrix", y=1.02)
pair.savefig(OUT / "scatterplot_matrix.png", dpi=120)

# Outlier effect
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axes[0].scatter(df["study_hours"], df["exam_score"], alpha=0.6, color="steelblue")
axes[0].set_title(f"Without the outlier (r = {r_before:.3f})")
axes[1].scatter(df["study_hours"], df["exam_score"], alpha=0.6, color="steelblue")
axes[1].scatter([1.0], [100.0], color="red", s=80, label="Outlier")
axes[1].set_title(f"With one outlier (r = {r_after:.3f})")
axes[1].legend()
for ax in axes:
    ax.set_xlabel("Study hours per day")
axes[0].set_ylabel("Exam score")
fig.tight_layout()
fig.savefig(OUT / "outlier_effect.png", dpi=150)

# Non-linear relationships
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(x, y_exp, color="steelblue", alpha=0.7)
axes[0].set_title(f"Exponential: Pearson {r_exp:.2f}, Spearman {rho_exp:.2f}")
axes[1].scatter(xq, y_quad, color="tomato", alpha=0.7)
axes[1].set_title(f"U-shaped: Pearson {r_quad:.2f}, Spearman {rho_quad:.2f}")
fig.tight_layout()
fig.savefig(OUT / "nonlinear_relationships.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)