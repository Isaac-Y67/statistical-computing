"""
Topic 21: Multivariate Statistical Analysis
Libraries: numpy, pandas, scipy, statsmodels, scikit-learn, matplotlib, seaborn
"""
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (adjusted_rand_score, confusion_matrix,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score)
from sklearn.preprocessing import StandardScaler
from statsmodels.multivariate.manova import MANOVA

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
# 1. Dataset (three species, shared covariance, different means)
# ---------------------------------------------------------------
vars_ = ["leaf_length", "leaf_width", "stem_height", "petal_size", "seed_mass"]
species = ["Alpha", "Beta", "Gamma"]
p, g, n_per = len(vars_), len(species), 60
N = g * n_per
sd = np.array([4.0, 2.5, 10.0, 3.0, 1.5])
R_true = np.array([[1.0, 0.6, 0.5, 0.4, 0.1],
                   [0.6, 1.0, 0.4, 0.5, 0.1],
                   [0.5, 0.4, 1.0, 0.3, 0.2],
                   [0.4, 0.5, 0.3, 1.0, 0.2],
                   [0.1, 0.1, 0.2, 0.2, 1.0]])
cov_true = np.outer(sd, sd) * R_true
L = np.linalg.cholesky(cov_true)
means_true = {"Alpha": [20, 10.0, 50, 12, 5.0],
              "Beta": [24, 11.0, 58, 15, 5.5],
              "Gamma": [25, 13.5, 60, 14, 7.0]}
frames = []
for s in species:
    block = np.array(means_true[s]) + rng.standard_normal((n_per, p)) @ L.T
    frames.append(pd.DataFrame(block, columns=vars_).assign(species=s))
df = pd.concat(frames, ignore_index=True)
df[vars_] = df[vars_].round(2)
df.to_csv(DATA / "plant_measurements.csv", index=False)

X = df[vars_].to_numpy()
sp = df["species"].to_numpy()
print("=== 1. Dataset ===")
print(f"{N} plants, {p} measurements, {g} species with {n_per} plants each")
gm = df.groupby("species")[vars_].mean().loc[species]
print("Group mean vectors:")
print(gm.round(2).to_string())
gm.round(4).to_csv(OUT / "group_means.csv")

# ---------------------------------------------------------------
# 2. Multivariate summaries and covariance structure
# ---------------------------------------------------------------
S_all = np.cov(X, rowvar=False)
R_all = np.corrcoef(X, rowvar=False)
resid = X - df.groupby("species")[vars_].transform("mean").to_numpy()
W = resid.T @ resid
Sw = W / (N - g)
Rw = Sw / np.sqrt(np.outer(np.diag(Sw), np.diag(Sw)))
print("\n=== 2. Covariance structure ===")
print("Overall covariance matrix:")
print(pd.DataFrame(S_all, index=vars_, columns=vars_).round(2).to_string())
print("\nOverall correlation matrix:")
print(pd.DataFrame(R_all, index=vars_, columns=vars_).round(3).to_string())
print("\nPooled within-species correlation matrix:")
print(pd.DataFrame(Rw, index=vars_, columns=vars_).round(3).to_string())
print(f"\nTotal variance (trace): overall = {np.trace(S_all):.2f}, "
      f"pooled within = {np.trace(Sw):.2f}")
print(f"Generalised variance (determinant): overall = {np.linalg.det(S_all):.1f}, "
      f"pooled within = {np.linalg.det(Sw):.1f}")
ev = np.sort(np.linalg.eigvalsh(S_all))[::-1]
print("Eigenvalues of the overall covariance matrix:", ev.round(2).tolist())
pd.DataFrame(S_all, index=vars_, columns=vars_).round(4).to_csv(
    OUT / "covariance_matrix.csv")
pd.DataFrame(Sw, index=vars_, columns=vars_).round(4).to_csv(
    OUT / "pooled_within_covariance.csv")

# ---------------------------------------------------------------
# 3. Mahalanobis distances
# ---------------------------------------------------------------
Sw_inv = np.linalg.inv(Sw)
d2_within = np.einsum("ij,jk,ik->i", resid, Sw_inv, resid)
Xc = X - X.mean(axis=0)
d2_overall = np.einsum("ij,jk,ik->i", Xc, np.linalg.inv(S_all), Xc)
cut = stats.chi2.ppf(0.975, p)
print("\n=== 3. Mahalanobis distances ===")
print(f"Squared distance to the plant's own species centre: mean = "
      f"{d2_within.mean():.3f} (chi-square mean = {p}), largest = {d2_within.max():.2f}")
print(f"Cutoff chi-square 97.5% with {p} df = {cut:.3f}: "
      f"{int((d2_within > cut).sum())} of {N} plants above it "
      f"({(d2_within > cut).mean():.1%}, expected about 2.5%)")
print(f"Squared distance to the overall centre (ignoring species): "
      f"{int((d2_overall > cut).sum())} plants above the cutoff")
D2 = pd.DataFrame(0.0, index=species, columns=species)
for a_, b_ in combinations(species, 2):
    d_ = gm.loc[a_].to_numpy() - gm.loc[b_].to_numpy()
    D2.loc[a_, b_] = D2.loc[b_, a_] = d_ @ Sw_inv @ d_
print("Squared Mahalanobis distance between species centres (pooled covariance):")
print(D2.round(3).to_string())
D2.round(4).to_csv(OUT / "centroid_distances.csv")

# ---------------------------------------------------------------
# 4. Multivariate normality: Mardia's tests on the within-group residuals
# ---------------------------------------------------------------
S_mle = resid.T @ resid / N
G = resid @ np.linalg.inv(S_mle) @ resid.T
b1 = (G ** 3).sum() / N ** 2
b2 = (np.diag(G) ** 2).sum() / N
skew_stat = N * b1 / 6
skew_df = p * (p + 1) * (p + 2) / 6
skew_p = stats.chi2.sf(skew_stat, skew_df)
kurt_z = (b2 - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / N)
kurt_p = 2 * stats.norm.sf(abs(kurt_z))
print("\n=== 4. Mardia's tests of multivariate normality (within-species residuals) ===")
print(f"Skewness: b1 = {b1:.4f}, statistic = {skew_stat:.2f} on {skew_df:.0f} df, "
      f"p = {skew_p:.4f}")
print(f"Kurtosis: b2 = {b2:.3f} (expected {p * (p + 2)}), z = {kurt_z:.3f}, "
      f"p = {kurt_p:.4f}")
print("Univariate Shapiro-Wilk p-values of the residuals:",
      {v: round(float(stats.shapiro(resid[:, j]).pvalue), 3)
       for j, v in enumerate(vars_)})

# ---------------------------------------------------------------
# 5. Comparing groups: Hotelling's T-squared and MANOVA
# ---------------------------------------------------------------
print("\n=== 5a. Hotelling's T-squared for each pair of species ===")
rows = []
for a_, b_ in combinations(species, 2):
    A, B = X[sp == a_], X[sp == b_]
    n1, n2 = len(A), len(B)
    d_ = A.mean(axis=0) - B.mean(axis=0)
    Sp = ((n1 - 1) * np.cov(A, rowvar=False)
          + (n2 - 1) * np.cov(B, rowvar=False)) / (n1 + n2 - 2)
    T2 = n1 * n2 / (n1 + n2) * d_ @ np.linalg.solve(Sp, d_)
    F = (n1 + n2 - p - 1) / (p * (n1 + n2 - 2)) * T2
    pF = stats.f.sf(F, p, n1 + n2 - p - 1)
    t_p = [stats.ttest_ind(A[:, j], B[:, j]).pvalue for j in range(p)]
    n_sig_uni = int(np.sum(np.array(t_p) * p < ALPHA))
    rows.append((f"{a_} vs {b_}", T2, F, p, n1 + n2 - p - 1, pF, n_sig_uni))
hot = pd.DataFrame(rows, columns=["comparison", "T2", "F", "df1", "df2", "p_value",
                                  "variables_significant_alone"])
hot["bonferroni_p"] = (hot["p_value"] * len(hot)).clip(upper=1)
print(hot.round(5).to_string(index=False))
print("(variables_significant_alone = number of the 5 separate t-tests that are "
      "significant after a Bonferroni correction over the 5 variables)")
hot.round(8).to_csv(OUT / "hotelling_pairs.csv", index=False)

print("\n=== 5b. MANOVA: five measurements by species ===")
formula = " + ".join(vars_) + " ~ species"
mv = MANOVA.from_formula(formula, data=df).mv_test()
stat_tab = mv.results["species"]["stat"]
print(stat_tab.round(5).to_string())
stat_tab.round(8).to_csv(OUT / "manova.csv")
Xt = X - X.mean(axis=0)
T_ = Xt.T @ Xt
lam_hand = np.linalg.det(W) / np.linalg.det(T_)
lam_sm = float(stat_tab.iloc[0]["Value"])
print(f"Wilks' lambda by hand = det(W) / det(T) = {lam_hand:.5f}; statsmodels = "
      f"{lam_sm:.5f}; match: {np.isclose(lam_hand, lam_sm)}")
s_ = min(p, g - 1)
F_exact = (1 - np.sqrt(lam_hand)) / np.sqrt(lam_hand) * (N - p - 2) / p
p_exact = stats.f.sf(F_exact, 2 * p, 2 * (N - p - 2))
print(f"Exact F for three groups = {F_exact:.3f} on ({2 * p}, {2 * (N - p - 2)}) df, "
      f"p = {p_exact:.3g}")
print(f"Multivariate effect size 1 - Wilks^(1/s) = {1 - lam_hand ** (1 / s_):.3f}")
print("Univariate one-way ANOVA for each measurement:")
uni = []
for v in vars_:
    f_ = stats.f_oneway(*[df.loc[df["species"] == s, v] for s in species])
    uni.append((v, f_.statistic, f_.pvalue))
    print(f"  {v:12s}: F = {f_.statistic:7.2f}, p = {f_.pvalue:.3g}")

# ---------------------------------------------------------------
# 6. Linear discriminant analysis
# ---------------------------------------------------------------
sc = StandardScaler().fit(X)
Z = sc.transform(X)
lda = LinearDiscriminantAnalysis(solver="svd").fit(Z, sp)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
acc_cv = cross_val_score(LinearDiscriminantAnalysis(), Z, sp, cv=cv)
pred_cv = cross_val_predict(LinearDiscriminantAnalysis(), Z, sp, cv=cv)
cm = confusion_matrix(sp, pred_cv, labels=species)
print("\n=== 6. Linear discriminant analysis (standardised measurements) ===")
print(f"Resubstitution accuracy = {lda.score(Z, sp):.4f}; 5-fold cross-validated "
      f"accuracy = {acc_cv.mean():.4f} (sd {acc_cv.std(ddof=1):.4f}); guessing = "
      f"{1 / g:.3f}")
print("Share of between-group variance on each discriminant axis:",
      lda.explained_variance_ratio_.round(4).tolist())
print("Cross-validated confusion matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(cm, index=species, columns=species).to_string())
coef = pd.DataFrame(lda.scalings_[:, :2], index=vars_, columns=["LD1", "LD2"])
print("Discriminant coefficients on standardised variables (signs are arbitrary):")
print(coef.round(3).to_string())
coef.round(5).to_csv(OUT / "lda_coefficients.csv")
lda_scores = lda.transform(Z)

# ---------------------------------------------------------------
# 7. Principal components (standardised)
# ---------------------------------------------------------------
pca = PCA().fit(Z)
pcs = pca.transform(Z)
load = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_),
                    index=vars_, columns=[f"PC{i + 1}" for i in range(p)])
print("\n=== 7. Principal components of the standardised data ===")
print("Explained variance ratio:", pca.explained_variance_ratio_.round(4).tolist())
print("Cumulative:", np.cumsum(pca.explained_variance_ratio_).round(4).tolist())
print("Loadings (correlation with the component), first 3:")
print(load.iloc[:, :3].round(3).to_string())
load.round(5).to_csv(OUT / "pca_loadings.csv")
for k in range(2):
    print(f"Mean PC{k + 1} score by species:",
          {s: round(float(pcs[sp == s, k].mean()), 2) for s in species})

# ---------------------------------------------------------------
# 8. Clustering without using the species labels
# ---------------------------------------------------------------
print("\n=== 8a. k-means ===")
km_rows, km_models = [], {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    km_models[k] = km
    km_rows.append((k, km.inertia_, silhouette_score(Z, km.labels_)))
    print(f"k = {k}: within-cluster sum of squares = {km.inertia_:7.1f}, "
          f"silhouette = {km_rows[-1][2]:.3f}")
km_tab = pd.DataFrame(km_rows, columns=["k", "inertia", "silhouette"])
best_k = int(km_tab.loc[km_tab["silhouette"].idxmax(), "k"])
print(f"Best k by silhouette = {best_k}")
lab_km3 = km_models[3].labels_
print(f"k = 3: adjusted Rand index against the true species = "
      f"{adjusted_rand_score(sp, lab_km3):.3f}")
print("Species (rows) against k-means cluster (columns), k = 3:")
print(pd.crosstab(pd.Series(sp, name="species"),
                  pd.Series(lab_km3, name="cluster")).to_string())

print("\n=== 8b. Hierarchical clustering (Ward linkage) ===")
Lk = linkage(Z, method="ward")
coph, _ = cophenet(Lk, pdist(Z))
lab_h3 = fcluster(Lk, 3, criterion="maxclust")
print(f"Cophenetic correlation = {coph:.3f}")
print(f"Cut into 3 clusters: adjusted Rand index = "
      f"{adjusted_rand_score(sp, lab_h3):.3f}")
print("Last five merge heights:", Lk[-5:, 2].round(2).tolist())
print(pd.crosstab(pd.Series(sp, name="species"),
                  pd.Series(lab_h3, name="cluster")).to_string())

print("\n=== 8c. Gaussian mixture model ===")
gmm_rows, gmm_models = [], {}
for k in range(1, 7):
    gm_ = GaussianMixture(n_components=k, covariance_type="full", n_init=3,
                          random_state=42).fit(Z)
    gmm_models[k] = gm_
    gmm_rows.append((k, gm_.bic(Z), gm_.aic(Z)))
    print(f"k = {k}: BIC = {gm_.bic(Z):8.1f}, AIC = {gm_.aic(Z):8.1f}")
gmm_tab = pd.DataFrame(gmm_rows, columns=["k", "bic", "aic"])
best_g = int(gmm_tab.loc[gmm_tab["bic"].idxmin(), "k"])
lab_g3 = gmm_models[3].predict(Z)
print(f"Best k by BIC = {best_g}; with k = 3 the adjusted Rand index = "
      f"{adjusted_rand_score(sp, lab_g3):.3f}")
pd.DataFrame({
    "method": ["k-means (k=3)", "Ward hierarchical (3 clusters)", "Gaussian mixture (k=3)"],
    "adjusted_rand_index": [adjusted_rand_score(sp, lab_km3),
                            adjusted_rand_score(sp, lab_h3),
                            adjusted_rand_score(sp, lab_g3)],
}).round(4).to_csv(OUT / "clustering_summary.csv", index=False)
km_tab.round(4).to_csv(OUT / "kmeans_selection.csv", index=False)
gmm_tab.round(3).to_csv(OUT / "gmm_selection.csv", index=False)

# ---------------------------------------------------------------
# 9. Plots
# ---------------------------------------------------------------
palette = dict(zip(species, sns.color_palette("Set1", 3)))

pair = sns.pairplot(df, hue="species", hue_order=species, palette=palette, corner=True,
                    plot_kws={"alpha": 0.6, "s": 18})
pair.figure.suptitle("Scatterplot matrix by species", y=1.02)
pair.savefig(OUT / "pair_plot.png", dpi=110)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(pd.DataFrame(R_all, index=vars_, columns=vars_), annot=True, fmt=".2f",
            cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title("Overall correlation (includes species differences)")
sns.heatmap(pd.DataFrame(Rw, index=vars_, columns=vars_), annot=True, fmt=".2f",
            cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title("Pooled within-species correlation")
fig.tight_layout()
fig.savefig(OUT / "correlation_heatmaps.png", dpi=150)

zdf = pd.DataFrame(Z, columns=vars_).assign(species=sp).groupby("species").mean().loc[species]
fig, ax = plt.subplots(figsize=(9, 5))
for s in species:
    ax.plot(vars_, zdf.loc[s], "o-", lw=2.5, color=palette[s], label=s)
ax.axhline(0, color="grey", linestyle=":")
ax.set_ylabel("Mean of the standardised measurement")
ax.set_title("Profile of each species")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "group_profiles.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(d2_within, bins=25, density=True, color="steelblue", alpha=0.7,
        label="Squared Mahalanobis distance")
grid = np.linspace(0, d2_within.max(), 300)
ax.plot(grid, stats.chi2.pdf(grid, p), "r-", lw=2, label=f"Chi-square ({p} df)")
ax.axvline(cut, color="black", linestyle="--", label=f"97.5% cutoff = {cut:.2f}")
ax.set_xlabel("Squared distance to the plant's own species centre")
ax.set_title("Mahalanobis distances")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "mahalanobis.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
axes[0].plot(range(1, p + 1), pca.explained_variance_, "o-", color="steelblue")
axes[0].axhline(1, color="red", linestyle="--", label="Eigenvalue = 1")
axes[0].set_xlabel("Component")
axes[0].set_ylabel("Eigenvalue")
axes[0].set_title("Scree plot")
axes[0].legend()
for s in species:
    m_ = sp == s
    axes[1].scatter(pcs[m_, 0], pcs[m_, 1], s=22, alpha=0.7, color=palette[s], label=s)
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
axes[1].set_title("PCA scores")
axes[1].legend()
for s in species:
    m_ = sp == s
    axes[2].scatter(lda_scores[m_, 0], lda_scores[m_, 1], s=22, alpha=0.7,
                    color=palette[s], label=s)
axes[2].set_xlabel("LD1")
axes[2].set_ylabel("LD2")
axes[2].set_title("Discriminant scores (uses the labels)")
axes[2].legend()
fig.tight_layout()
fig.savefig(OUT / "pca_lda_scores.png", dpi=150)

fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(pd.DataFrame(cm, index=species, columns=species), annot=True, fmt="d",
            cmap="Blues", cbar=False, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"LDA cross-validated confusion matrix (accuracy {acc_cv.mean():.3f})")
fig.tight_layout()
fig.savefig(OUT / "lda_confusion_matrix.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
axes[0].plot(km_tab["k"], km_tab["inertia"], "o-", color="steelblue")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Within-cluster sum of squares")
axes[0].set_title("k-means: elbow plot")
axes[1].plot(km_tab["k"], km_tab["silhouette"], "o-", color="tomato")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Mean silhouette")
axes[1].set_title("k-means: silhouette")
axes[2].plot(gmm_tab["k"], gmm_tab["bic"], "o-", color="seagreen")
axes[2].set_xlabel("Number of components")
axes[2].set_ylabel("BIC (lower is better)")
axes[2].set_title("Gaussian mixture: BIC")
fig.tight_layout()
fig.savefig(OUT / "cluster_selection.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, lab, title in ((axes[0], sp, "True species"),
                       (axes[1], lab_km3, "k-means, k = 3"),
                       (axes[2], lab_h3, "Ward hierarchical, 3 clusters")):
    for u in np.unique(lab):
        m_ = lab == u
        ax.scatter(pcs[m_, 0], pcs[m_, 1], s=22, alpha=0.7, label=str(u))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "clusters_on_pca.png", dpi=150)

fig, ax = plt.subplots(figsize=(12, 5.5))
dendrogram(Lk, truncate_mode="lastp", p=25, ax=ax, color_threshold=Lk[-3, 2],
           leaf_rotation=90)
ax.set_title("Ward dendrogram (last 25 merges)")
ax.set_ylabel("Merge distance")
fig.tight_layout()
fig.savefig(OUT / "dendrogram.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)