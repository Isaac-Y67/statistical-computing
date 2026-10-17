"""
Topic 15: Data Transformation and Normalization
Libraries: pandas, numpy, scipy, sklearn.preprocessing, matplotlib, seaborn
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, PowerTransformer,
                                   QuantileTransformer, RobustScaler,
                                   StandardScaler)

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------
# 1. Dataset (synthetic, reproducible)
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
n = 500
age = rng.normal(38, 11, n).clip(18, 80)
income = rng.lognormal(mean=10.8, sigma=0.55, size=n)
income[[10, 200, 350]] = [450000, 520000, 600000]  # three extreme incomes
spending = rng.beta(2, 2, n) * 100
visits = rng.poisson(4, n)
membership = rng.choice(["Basic", "Silver", "Gold"], size=n, p=[0.5, 0.3, 0.2])
city = rng.choice(["Nairobi", "Mombasa", "Kisumu", "Nakuru"], size=n,
                  p=[0.4, 0.25, 0.2, 0.15])

df = pd.DataFrame({
    "age": age.round(1),
    "income": np.round(income),
    "spending_score": spending.round(1),
    "visits_per_month": visits,
    "membership": membership,
    "city": city,
})
df.to_csv(DATA / "customers.csv", index=False)
num = ["age", "income", "spending_score", "visits_per_month"]

print("=== 1. Dataset ===")
print(f"{n} customers; numeric columns: {', '.join(num)}; "
      f"categorical columns: membership, city")
desc = df[num].describe().T[["mean", "std", "min", "50%", "max"]]
desc["skewness"] = [stats.skew(df[c]) for c in num]
print(desc.round(2))

# ---------------------------------------------------------------
# 2. Scaling methods
# ---------------------------------------------------------------
scalers = {
    "Standardisation (z-score)": StandardScaler(),
    "Min-max [0, 1]": MinMaxScaler(),
    "Robust (median, IQR)": RobustScaler(),
    "Max-abs": MaxAbsScaler(),
}
scaled, rows = {}, []
for c in num:
    s = df[c]
    rows.append(("Original", c, s.mean(), s.std(ddof=0), s.min(), s.max()))
for name, sc in scalers.items():
    scaled[name] = pd.DataFrame(sc.fit_transform(df[num]), columns=num)
    for c in num:
        s = scaled[name][c]
        rows.append((name, c, s.mean(), s.std(ddof=0), s.min(), s.max()))
scale_tab = pd.DataFrame(rows, columns=["method", "variable", "mean", "sd", "min", "max"])
print("\n=== 2. Scaled values: mean, sd, min and max of each variable ===")
print(scale_tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
scale_tab.round(4).to_csv(OUT / "scaling_summary.csv", index=False)

z_manual = (df[num] - df[num].mean()) / df[num].std(ddof=0)
mm_manual = (df[num] - df[num].min()) / (df[num].max() - df[num].min())
iqr = df[num].quantile(0.75) - df[num].quantile(0.25)
rb_manual = (df[num] - df[num].median()) / iqr
print("\nChecks against hand calculations:")
print("z = (x - mean) / sd matches StandardScaler:",
      np.allclose(z_manual, scaled["Standardisation (z-score)"]))
print("(x - min) / (max - min) matches MinMaxScaler:",
      np.allclose(mm_manual, scaled["Min-max [0, 1]"]))
print("(x - median) / IQR matches RobustScaler:",
      np.allclose(rb_manual, scaled["Robust (median, IQR)"]))
sc_chk = StandardScaler().fit(df[num])
print("inverse_transform recovers the original data:",
      np.allclose(sc_chk.inverse_transform(sc_chk.transform(df[num])), df[num]))

# ---------------------------------------------------------------
# 3. Sensitivity of the scalers to outliers (income)
# ---------------------------------------------------------------
inc = df["income"]
clean = inc.drop(index=[10, 200, 350])
z_inc = scaled["Standardisation (z-score)"]["income"]
mm_inc = scaled["Min-max [0, 1]"]["income"]
rb_inc = scaled["Robust (median, IQR)"]["income"]
mm_clean = pd.Series(MinMaxScaler().fit_transform(clean.to_frame()).ravel())


def spread_stats(x):
    return x.mean(), x.std(ddof=1), x.median(), x.quantile(0.75) - x.quantile(0.25)


print("\n=== 3. Effect of three extreme incomes ===")
st = pd.DataFrame({"with outliers": spread_stats(inc),
                   "without outliers": spread_stats(clean)},
                  index=["mean", "std dev", "median", "IQR"])
st["change_%"] = (st["with outliers"] / st["without outliers"] - 1) * 100
print(st.round(1))
q1, q3 = mm_inc.quantile([0.25, 0.75])
q1c, q3c = mm_clean.quantile([0.25, 0.75])
print(f"Min-max: the middle 50% of customers fill {q3 - q1:.3f} of the [0, 1] range "
      f"with the outliers, against {q3c - q1c:.3f} without them")
print(f"Largest income z-score = {z_inc.max():.2f}; customers with |z| > 3: "
      f"{int((z_inc.abs() > 3).sum())}")
print(f"Robust scaling: median of scaled income = {rb_inc.median():.3f}, "
      f"IQR = {rb_inc.quantile(0.75) - rb_inc.quantile(0.25):.3f}, "
      f"largest value = {rb_inc.max():.2f}")

# ---------------------------------------------------------------
# 4. Transformations for skewed data (income)
# ---------------------------------------------------------------
inc_arr = inc.to_numpy()
bc, lam_bc = stats.boxcox(inc_arr)
yj_t = PowerTransformer(method="yeo-johnson", standardize=False)
yj = yj_t.fit_transform(inc_arr.reshape(-1, 1)).ravel()
qt = QuantileTransformer(output_distribution="normal", n_quantiles=200,
                         random_state=42).fit_transform(inc_arr.reshape(-1, 1)).ravel()
trans = {
    "Original": inc_arr,
    "Square root": np.sqrt(inc_arr),
    "Log": np.log(inc_arr),
    "Box-Cox": bc,
    "Yeo-Johnson": yj,
    "Quantile to normal": qt,
}


def n_out(x):
    a, b = np.percentile(x, [25, 75])
    i = b - a
    return int(((x < a - 1.5 * i) | (x > b + 1.5 * i)).sum())


trows = []
for name, x in trans.items():
    sw = stats.shapiro(x)
    trows.append((name, stats.skew(x), stats.kurtosis(x), sw.statistic, sw.pvalue,
                  n_out(x)))
trans_tab = pd.DataFrame(trows, columns=["transformation", "skewness", "excess_kurtosis",
                                         "shapiro_W", "shapiro_p", "outliers_1.5IQR"])
print("\n=== 4. Transformations of income ===")
print(trans_tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"Box-Cox lambda = {lam_bc:.3f} (0 means a log transform); "
      f"Yeo-Johnson lambda = {yj_t.lambdas_[0]:.3f}")
trans_tab.round(5).to_csv(OUT / "transformation_comparison.csv", index=False)

lo, hi = np.percentile(inc_arr, [1, 99])
inc_w = np.clip(inc_arr, lo, hi)
print(f"\nWinsorising at the 1st and 99th percentiles ({lo:,.0f} and {hi:,.0f}): "
      f"skewness {stats.skew(inc_arr):.3f} -> {stats.skew(inc_w):.3f}, "
      f"maximum {inc_arr.max():,.0f} -> {inc_w.max():,.0f}")

vis = df["visits_per_month"].to_numpy()
print("\nCount variable (visits per month):")
print(f"Original: mean = {vis.mean():.2f}, variance = {vis.var(ddof=1):.2f}, "
      f"skewness = {stats.skew(vis):.3f}")
print(f"Square root: skewness = {stats.skew(np.sqrt(vis)):.3f}, "
      f"variance = {np.sqrt(vis).var(ddof=1):.3f}")
print(f"log(1 + x):  skewness = {stats.skew(np.log1p(vis)):.3f}")

# ---------------------------------------------------------------
# 5. Encoding categorical variables and binning
# ---------------------------------------------------------------
enc = OneHotEncoder(sparse_output=False)
city_oh = pd.DataFrame(enc.fit_transform(df[["city"]]),
                       columns=enc.get_feature_names_out(["city"]))
mem_ord = OrdinalEncoder(categories=[["Basic", "Silver", "Gold"]]
                         ).fit_transform(df[["membership"]]).ravel()
print("\n=== 5. Encoding ===")
print("City counts:", df["city"].value_counts().to_dict())
print("Membership counts:", df["membership"].value_counts().to_dict())
show = pd.concat([df[["city", "membership"]].head(5).reset_index(drop=True),
                  city_oh.head(5),
                  pd.Series(mem_ord[:5], name="membership_ordinal")], axis=1)
print(show.to_string(index=False))
print("Mean of each one-hot column equals the category proportion:",
      city_oh.mean().round(3).to_dict())

age_bins = pd.qcut(df["age"], 4)
inc_bins = pd.cut(df["income"], 4)
print("\nEqual-frequency bins of age (4 quantile bins):")
print(age_bins.value_counts().sort_index().to_string())
print("\nEqual-width bins of income (4 bins):")
print(inc_bins.value_counts().sort_index().to_string())

# ---------------------------------------------------------------
# 6. Why scaling matters for distance-based methods
# ---------------------------------------------------------------
shares = {"Raw (no scaling)": df[num].var(ddof=0) / df[num].var(ddof=0).sum()}
for name in scalers:
    v = scaled[name].var(ddof=0)
    shares[name] = v / v.sum()
shares = pd.DataFrame(shares).T
print("\n=== 6. Share of squared Euclidean distance contributed by each variable ===")
print((shares * 100).round(1).to_string())
shares.round(4).to_csv(OUT / "distance_shares.csv")


def nearest(X):
    d = cdist(X, X)
    np.fill_diagonal(d, np.inf)
    return d.argmin(axis=1)


nn_raw = nearest(df[num].to_numpy())
nn_z = nearest(scaled["Standardisation (z-score)"].to_numpy())
nn_income = nearest(df[["income"]].to_numpy())
print(f"\nNearest neighbour on raw data is the same as the nearest neighbour by "
      f"income alone for {np.mean(nn_raw == nn_income):.1%} of customers")
print(f"Nearest neighbour is unchanged after z-scoring for "
      f"{np.mean(nn_raw == nn_z):.1%} of customers")

# ---------------------------------------------------------------
# 7. Fit the scaler on training data only (avoid leakage)
# ---------------------------------------------------------------
tr_idx, te_idx = train_test_split(np.arange(n), test_size=0.25, random_state=42)
tr, te = df.iloc[tr_idx][num], df.iloc[te_idx][num]
proper = StandardScaler().fit(tr).transform(te)
leaky = StandardScaler().fit(df[num]).transform(te)
print("\n=== 7. Scaling train and test data (75% / 25% split) ===")
leak = pd.DataFrame({
    "train mean": tr.mean().to_numpy(),
    "test z mean (fit on train)": proper.mean(axis=0),
    "test z mean (fit on all data)": leaky.mean(axis=0),
    "test z sd (fit on train)": proper.std(axis=0),
}, index=num)
print(leak.round(3))
print("Test-set z-scores do not have mean exactly 0 and sd exactly 1, which is "
      "correct: the test data are treated as new, unseen data")

# ---------------------------------------------------------------
# 8. Model-ready dataset
# ---------------------------------------------------------------
ready = pd.DataFrame({
    "age_z": StandardScaler().fit_transform(df[["age"]]).ravel(),
    "log_income_z": StandardScaler().fit_transform(np.log(df[["income"]])).ravel(),
    "spending_minmax": MinMaxScaler().fit_transform(df[["spending_score"]]).ravel(),
    "sqrt_visits_z": StandardScaler().fit_transform(
        np.sqrt(df[["visits_per_month"]])).ravel(),
})
ready = pd.concat([ready, city_oh, pd.Series(mem_ord, name="membership_ordinal")],
                  axis=1)
ready.round(5).to_csv(DATA / "model_ready.csv", index=False)
print("\n=== 8. Model-ready dataset ===")
print(f"{ready.shape[0]} rows, {ready.shape[1]} columns: {', '.join(ready.columns)}")
print(ready.describe().T[["mean", "std", "min", "max"]].round(3))

# ---------------------------------------------------------------
# 9. Plots
# ---------------------------------------------------------------
# Plot 1: income under each scaler (the shape does not change)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
items = [("Original income", df["income"])] + [(k, v["income"]) for k, v in scaled.items()]
for ax, (title, vals) in zip(axes.ravel(), items):
    sns.histplot(vals, bins=30, color="steelblue", ax=ax)
    ax.set_title(f"{title}\nrange {vals.min():.2f} to {vals.max():.2f}", fontsize=10)
    ax.set_xlabel("")
axes.ravel()[-1].axis("off")
fig.suptitle("Scaling changes the scale, not the shape")
fig.tight_layout()
fig.savefig(OUT / "scaling_histograms.png", dpi=150)

# Plot 2: boxplots of all variables, before and after scaling
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(data=df[num], ax=axes[0])
axes[0].set_title("Original: income dominates")
sns.boxplot(data=scaled["Standardisation (z-score)"], ax=axes[1])
axes[1].set_title("Standardised (z-scores)")
sns.boxplot(data=scaled["Min-max [0, 1]"], ax=axes[2])
axes[2].set_title("Min-max scaled")
for ax in axes:
    ax.tick_params(axis="x", rotation=25)
fig.tight_layout()
fig.savefig(OUT / "scaling_boxplots.png", dpi=150)

# Plot 3: transformations of income
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, (name, x) in zip(axes.ravel(), trans.items()):
    sns.histplot(x, bins=30, kde=True, color="seagreen", ax=ax)
    ax.set_title(f"{name} (skewness {stats.skew(x):.2f})", fontsize=10)
    ax.set_xlabel("")
fig.suptitle("Transformations of income")
fig.tight_layout()
fig.savefig(OUT / "transformations.png", dpi=150)

# Plot 4: Q-Q plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, name in zip(axes, ["Original", "Log", "Yeo-Johnson"]):
    stats.probplot(trans[name], dist="norm", plot=ax)
    ax.set_title(f"Q-Q plot: {name}")
fig.tight_layout()
fig.savefig(OUT / "qq_plots.png", dpi=150)

# Plot 5: sensitivity to outliers
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
sns.histplot(mm_inc, bins=40, color="tomato", ax=axes[0])
axes[0].set_title("Min-max income, with the 3 outliers")
sns.histplot(mm_clean, bins=40, color="steelblue", ax=axes[1])
axes[1].set_title("Min-max income, outliers removed")
sns.histplot(rb_inc, bins=40, color="seagreen", ax=axes[2])
axes[2].set_title("Robust-scaled income, with the outliers")
for ax in axes:
    ax.set_xlabel("Scaled income")
fig.tight_layout()
fig.savefig(OUT / "outlier_sensitivity.png", dpi=150)

# Plot 6: binning
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
age_bins.value_counts().sort_index().plot(kind="bar", color="steelblue", ax=axes[0])
axes[0].set_title("Equal-frequency bins of age")
inc_bins.value_counts().sort_index().plot(kind="bar", color="orange", ax=axes[1])
axes[1].set_title("Equal-width bins of income")
for ax in axes:
    ax.set_xlabel("")
    ax.set_ylabel("Customers")
    ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(OUT / "binning.png", dpi=150)

# Plot 7: share of distance by variable
fig, ax = plt.subplots(figsize=(10, 5.5))
(shares * 100).plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
ax.set_ylabel("Share of squared distance (%)")
ax.set_title("Which variables drive Euclidean distance?")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(OUT / "distance_shares.png", dpi=150)

# Plot 8: age against income, raw against transformed
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
axes[0].scatter(df["age"], df["income"], s=12, alpha=0.5)
axes[0].set_title("Raw: age against income")
axes[1].scatter(z_manual["age"], z_manual["income"], s=12, alpha=0.5, color="orange")
axes[1].set_title("Both z-scored")
axes[2].scatter(ready["age_z"], ready["log_income_z"], s=12, alpha=0.5, color="seagreen")
axes[2].set_title("Age z-scored, log income z-scored")
fig.tight_layout()
fig.savefig(OUT / "raw_vs_transformed_scatter.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)