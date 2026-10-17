"""
Topic 10: Eigenvalues, Eigenvectors and PCA
Libraries: numpy.linalg, sklearn.decomposition, pandas, matplotlib
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

np.set_printoptions(precision=4, suppress=True, linewidth=150)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def sorted_eigh(matrix):
    """Eigendecomposition of a symmetric matrix, largest eigenvalue first."""
    vals, vecs = np.linalg.eigh(matrix)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    for j in range(vecs.shape[1]):  # sign convention: largest entry positive
        if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
            vecs[:, j] *= -1
    return vals, vecs


# ---------------------------------------------------------------
# 1. Small worked example: eigenvalues of a 2x2 matrix
# ---------------------------------------------------------------
S2 = np.array([[4.0, 2.0], [2.0, 3.0]])
tr, det = np.trace(S2), np.linalg.det(S2)
lam_formula = np.array([(tr + np.sqrt(tr ** 2 - 4 * det)) / 2,
                        (tr - np.sqrt(tr ** 2 - 4 * det)) / 2])
vals2, vecs2 = sorted_eigh(S2)

print("=== Worked example: eigenvalues of S = [[4, 2], [2, 3]] ===")
print(f"trace = {tr:.1f}, determinant = {det:.1f}")
print(f"Characteristic equation: lambda^2 - {tr:.0f} lambda + {det:.0f} = 0")
print(f"Eigenvalues from the formula = {lam_formula}")
print(f"Eigenvalues from numpy       = {vals2}")
print("Eigenvectors (columns) =\n", vecs2)
for j in range(2):
    ok = np.allclose(S2 @ vecs2[:, j], vals2[j] * vecs2[:, j])
    print(f"S v{j + 1} = lambda{j + 1} v{j + 1}: {ok}")
print("Eigenvectors are orthogonal (dot product ~ 0):",
      np.isclose(vecs2[:, 0] @ vecs2[:, 1], 0))
print("Sum of eigenvalues = trace:", np.isclose(vals2.sum(), tr))
print("Product of eigenvalues = determinant:", np.isclose(vals2.prod(), det))

# ---------------------------------------------------------------
# 2. Dataset (synthetic, two hidden abilities)
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
n = 300
quant = rng.normal(0, 1, n)
verbal = rng.normal(0, 1, n)
df = pd.DataFrame({
    "maths": 65 + 12 * quant + rng.normal(0, 4, n),
    "physics": 60 + 13 * quant + rng.normal(0, 5, n),
    "statistics": 62 + 12 * quant + rng.normal(0, 5, n),
    "english": 68 + 8 * verbal + rng.normal(0, 8, n),
    "history": 64 + 9 * verbal + rng.normal(0, 9, n),
    "literature": 66 + 9 * verbal + rng.normal(0, 8, n),
    "reading_speed": 250 + 25 * verbal + rng.normal(0, 40, n),
}).round(1)
df.to_csv(DATA / "student_assessment.csv", index=False)
names = list(df.columns)
p = len(names)

print("\n=== Dataset ===")
print(f"{n} students, {p} variables")
print(df.describe().loc[["mean", "std"]].round(2))

# ---------------------------------------------------------------
# 3. PCA by eigendecomposition of the correlation matrix
# ---------------------------------------------------------------
X = df.to_numpy()
Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
R = Z.T @ Z / (n - 1)
vals, vecs = sorted_eigh(R)

prop = vals / vals.sum()
cum = np.cumsum(prop)
print("\n=== Correlation matrix ===")
print(pd.DataFrame(R, index=names, columns=names).round(2))

print("\n=== Eigenvalues and explained variance (correlation matrix) ===")
eig_table = pd.DataFrame({
    "component": [f"PC{i + 1}" for i in range(p)],
    "eigenvalue": vals, "proportion": prop, "cumulative": cum,
})
print(eig_table.round(4).to_string(index=False))
print(f"Sum of eigenvalues = {vals.sum():.4f} (equals trace of R = {np.trace(R):.0f})")
n_kaiser = int((vals > 1).sum())
n_80 = int(np.argmax(cum >= 0.80) + 1)
print(f"Kaiser rule (eigenvalue > 1): keep {n_kaiser} components")
print(f"Components needed for 80% of the variance: {n_80}")
eig_table.round(4).to_csv(OUT / "eigenvalues.csv", index=False)

print("\n=== Eigenvectors (first 3 components) ===")
ev_df = pd.DataFrame(vecs[:, :3], index=names, columns=["PC1", "PC2", "PC3"])
print(ev_df.round(3))

loadings = vecs * np.sqrt(vals)  # correlation of each variable with each component
load_df = pd.DataFrame(loadings, index=names, columns=eig_table["component"])
print("\n=== Loadings (correlation of variable with component), first 3 ===")
print(load_df.iloc[:, :3].round(3))
load_df.round(4).to_csv(OUT / "loadings.csv")

# ---------------------------------------------------------------
# 4. Checks: scores, scikit-learn and SVD
# ---------------------------------------------------------------
scores = Z @ vecs
score_cov = np.cov(scores, rowvar=False)
off_diag = np.abs(score_cov - np.diag(np.diag(score_cov))).max()
print("\n=== Checks ===")
print("Variance of each score column equals its eigenvalue:",
      np.allclose(np.diag(score_cov), vals))
print(f"Largest covariance between different components = {off_diag:.2e} "
      f"(components are uncorrelated)")

pca = PCA().fit(Z)
print("scikit-learn eigenvalues match:", np.allclose(pca.explained_variance_, vals))
print("scikit-learn explained variance ratio matches:",
      np.allclose(pca.explained_variance_ratio_, prop))
print("scikit-learn components match (up to sign):",
      np.allclose(np.abs(pca.components_.T), np.abs(vecs), atol=1e-6))
sing = np.linalg.svd(Z, compute_uv=False)
print("SVD: singular values squared / (n - 1) equal the eigenvalues:",
      np.allclose(sing ** 2 / (n - 1), vals))

pd.DataFrame(scores[:, :3], columns=["PC1", "PC2", "PC3"]).round(4).to_csv(
    OUT / "pc_scores.csv", index=False)

# ---------------------------------------------------------------
# 5. Covariance PCA (no standardising) for comparison
# ---------------------------------------------------------------
S = np.cov(X, rowvar=False)
vals_c, vecs_c = sorted_eigh(S)
prop_c = vals_c / vals_c.sum()
print("\n=== Covariance-matrix PCA (variables NOT standardised) ===")
print("Variances of the variables:")
print(pd.Series(np.diag(S), index=names).round(1))
print(f"Proportion of variance: PC1 = {prop_c[0]:.4f}, PC2 = {prop_c[1]:.4f}, "
      f"PC3 = {prop_c[2]:.4f}")
print("PC1 eigenvector:")
print(pd.Series(vecs_c[:, 0], index=names).round(3))

# ---------------------------------------------------------------
# 6. Reconstruction error
# ---------------------------------------------------------------
print("\n=== Reconstruction from the first k components (standardised data) ===")
recon_rows = []
for k in range(1, p + 1):
    Zhat = scores[:, :k] @ vecs[:, :k].T
    rmse = np.sqrt(np.mean((Z - Zhat) ** 2))
    recon_rows.append((k, cum[k - 1], rmse))
    print(f"k = {k}: variance retained = {cum[k - 1]:.4f}, RMSE = {rmse:.4f}")
pd.DataFrame(recon_rows, columns=["k", "variance_retained", "rmse"]).round(4).to_csv(
    OUT / "reconstruction_error.csv", index=False)

# ---------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------
# Plot 1: scree plot and cumulative variance
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
x_pc = np.arange(1, p + 1)
axes[0].plot(x_pc, vals, "o-", color="steelblue")
axes[0].axhline(1, color="red", linestyle="--", label="Kaiser line (eigenvalue = 1)")
axes[0].set_xlabel("Component")
axes[0].set_ylabel("Eigenvalue")
axes[0].set_title("Scree plot")
axes[0].legend()
axes[1].bar(x_pc, prop, color="steelblue", alpha=0.7, label="Individual")
axes[1].plot(x_pc, cum, "ro-", label="Cumulative")
axes[1].axhline(0.8, color="grey", linestyle=":")
axes[1].set_xlabel("Component")
axes[1].set_ylabel("Proportion of variance")
axes[1].set_title("Explained variance")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / "scree_plot.png", dpi=150)

# Plot 2: loadings heatmap
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(loadings[:, :3], cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(3))
ax.set_xticklabels(["PC1", "PC2", "PC3"])
ax.set_yticks(range(p))
ax.set_yticklabels(names)
for (i, j), v in np.ndenumerate(loadings[:, :3]):
    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
ax.set_title("Loadings (correlation with each component)")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(OUT / "loadings_heatmap.png", dpi=150)

# Plot 3: biplot (scores and loading arrows)
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(scores[:, 0], scores[:, 1], alpha=0.4, color="steelblue", s=18)
arrow_scale = 3.0
for i, name in enumerate(names):
    ax.annotate("", xy=(loadings[i, 0] * arrow_scale, loadings[i, 1] * arrow_scale),
                xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax.text(loadings[i, 0] * arrow_scale * 1.1, loadings[i, 1] * arrow_scale * 1.1,
            name, color="red", ha="center", fontsize=9)
ax.axhline(0, color="grey", lw=0.5)
ax.axvline(0, color="grey", lw=0.5)
ax.set_xlabel(f"PC1 ({prop[0]:.1%} of variance)")
ax.set_ylabel(f"PC2 ({prop[1]:.1%} of variance)")
ax.set_title("PCA biplot: student scores and variable loadings")
fig.tight_layout()
fig.savefig(OUT / "pca_biplot.png", dpi=150)

# Plot 4: geometry of eigenvectors for the 2x2 example
pts = np.random.default_rng(1).multivariate_normal([0, 0], S2, 300)
t = np.linspace(0, 2 * np.pi, 200)
circle = np.vstack([np.cos(t), np.sin(t)])
ellipse = vecs2 @ np.diag(2 * np.sqrt(vals2)) @ circle
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(pts[:, 0], pts[:, 1], alpha=0.4, color="steelblue", s=15)
ax.plot(ellipse[0], ellipse[1], "k-", lw=1.5, label="2 SD ellipse")
for j, colour in enumerate(["red", "green"]):
    vec = vecs2[:, j] * 2 * np.sqrt(vals2[j])
    ax.annotate("", xy=vec, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=colour, lw=2.5))
    ax.plot([], [], color=colour, lw=2.5,
            label=f"Eigenvector {j + 1} (lambda = {vals2[j]:.2f})")
ax.set_aspect("equal")
ax.set_title("Eigenvectors point along the axes of the covariance ellipse")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "eigenvector_geometry.png", dpi=150)

# Plot 5: reconstruction error
fig, ax = plt.subplots(figsize=(7, 4.5))
ks = [r[0] for r in recon_rows]
ax.plot(ks, [r[2] for r in recon_rows], "o-", color="tomato")
ax.set_xlabel("Number of components kept (k)")
ax.set_ylabel("Reconstruction RMSE (standardised units)")
ax.set_title("Error falls as more components are kept")
fig.tight_layout()
fig.savefig(OUT / "reconstruction_error.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)