"""
Topic 1: Measures of Central Tendency and Dispersion
Libraries: pandas, numpy, scipy, matplotlib, seaborn
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

# ---------------------------------------------------------------
# 1. Dataset (synthetic student exam scores, reproducible)
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
n = 200
df = pd.DataFrame({
    "student_id": range(1, n + 1),
    "math": np.clip(rng.normal(62, 14, n), 0, 100).round(1),
    "english": np.clip(rng.normal(68, 10, n), 0, 100).round(1),
    "science": np.clip(rng.normal(58, 18, n), 0, 100).round(1),
    "statistics": np.clip(rng.normal(65, 12, n), 0, 100).round(1),
})
# add a few outliers to make the boxplot interesting
df.loc[[5, 50, 120], "math"] = [8.0, 99.0, 12.0]
df.to_csv(DATA / "exam_scores.csv", index=False)

subjects = ["math", "english", "science", "statistics"]

# ---------------------------------------------------------------
# 2. Central tendency
# ---------------------------------------------------------------
summary = pd.DataFrame({
    "mean": df[subjects].mean(),
    "median": df[subjects].median(),
    "mode": [df[c].mode().iloc[0] for c in subjects],
})

# ---------------------------------------------------------------
# 3. Dispersion
# ---------------------------------------------------------------
summary["variance"] = df[subjects].var(ddof=1)
summary["std_dev"] = df[subjects].std(ddof=1)
summary["range"] = df[subjects].max() - df[subjects].min()
summary["Q1"] = df[subjects].quantile(0.25)
summary["Q3"] = df[subjects].quantile(0.75)
summary["IQR"] = summary["Q3"] - summary["Q1"]
summary["CV_%"] = summary["std_dev"] / summary["mean"] * 100
summary["skewness"] = df[subjects].apply(stats.skew)

summary = summary.round(2)
print("\n=== Summary statistics ===")
print(summary.T)
summary.to_csv(OUT / "summary_statistics.csv")

# ---------------------------------------------------------------
# 4. Outlier detection (1.5 x IQR rule)
# ---------------------------------------------------------------
print("\n=== Outliers (1.5*IQR rule) ===")
for c in subjects:
    lo = summary.loc[c, "Q1"] - 1.5 * summary.loc[c, "IQR"]
    hi = summary.loc[c, "Q3"] + 1.5 * summary.loc[c, "IQR"]
    out = df[(df[c] < lo) | (df[c] > hi)][c]
    print(f"{c}: {len(out)} outliers, bounds=({lo:.1f}, {hi:.1f})")

# ---------------------------------------------------------------
# 5. Visualisation
# ---------------------------------------------------------------
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df[subjects], ax=ax)
ax.set_title("Boxplot of exam scores by subject")
ax.set_ylabel("Score")
fig.tight_layout()
fig.savefig(OUT / "boxplot_scores.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["math"], kde=True, ax=ax)
ax.axvline(df["math"].mean(), color="red", linestyle="--", label="Mean")
ax.axvline(df["math"].median(), color="green", linestyle="-", label="Median")
ax.set_title("Distribution of Math scores")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "hist_math.png", dpi=150)

plt.show()
print("\nSaved outputs to:", OUT)