"""
Topic 9: Matrix Algebra in Statistics
Libraries: numpy, scipy.linalg, pandas, scipy.stats, matplotlib
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg, stats

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

np.set_printoptions(precision=4, suppress=True, linewidth=150)

# ---------------------------------------------------------------
# 1. Basic matrix operations
# ---------------------------------------------------------------
A = np.array([[2, 1, 3],
              [0, 4, 1],
              [5, 2, 6]], dtype=float)
B = np.array([[1, 0, 2],
              [3, 1, 0],
              [2, 5, 1]], dtype=float)

print("=== Matrices ===")
print("A =\n", A)
print("B =\n", B)

print("\n=== Addition, subtraction, scalar multiplication ===")
print("A + B =\n", A + B)
print("A - B =\n", A - B)
print("3 * A =\n", 3 * A)

print("\n=== Multiplication ===")
AB = A @ B
BA = B @ A
print("A @ B (matrix product) =\n", AB)
print("B @ A =\n", BA)
print("A * B (element-wise, NOT matrix product) =\n", A * B)
print("Matrix multiplication is commutative (AB = BA):", np.allclose(AB, BA))

print("\n=== Transpose ===")
print("A' =\n", A.T)
print("(AB)' = B'A':", np.allclose(AB.T, B.T @ A.T))

print("\n=== Determinant, trace and rank ===")
det_A, det_B = np.linalg.det(A), np.linalg.det(B)
print(f"det(A) = {det_A:.4f}, det(B) = {det_B:.4f}, det(AB) = {np.linalg.det(AB):.4f}")
print("det(AB) = det(A) * det(B):", np.isclose(np.linalg.det(AB), det_A * det_B))
print(f"trace(A) = {np.trace(A):.1f}, rank(A) = {np.linalg.matrix_rank(A)}")

print("\n=== Inverse ===")
A_inv = np.linalg.inv(A)
print("A^-1 =\n", A_inv)
print("A @ A^-1 equals the identity matrix:", np.allclose(A @ A_inv, np.eye(3)))
print("(AB)^-1 = B^-1 A^-1:", np.allclose(np.linalg.inv(AB),
                                           np.linalg.inv(B) @ A_inv))

print("\n=== Worked example: inverse of a 2x2 matrix ===")
M = np.array([[4.0, 7.0], [2.0, 6.0]])
det_M = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
M_inv_hand = (1 / det_M) * np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]])
print(f"det(M) = 4*6 - 7*2 = {det_M:.1f}")
print("Inverse by the formula (1/det) * [[d, -b], [-c, a]] =\n", M_inv_hand)
print("Matches numpy:", np.allclose(M_inv_hand, np.linalg.inv(M)))

pd.DataFrame(AB).round(4).to_csv(OUT / "product_AB.csv", index=False, header=False)
pd.DataFrame(A_inv).round(4).to_csv(OUT / "inverse_A.csv", index=False, header=False)

# ---------------------------------------------------------------
# 2. Solving linear equations
# ---------------------------------------------------------------
print("\n=== Solving a 2x2 system: 2x + y = 5 and x - y = 1 ===")
M2 = np.array([[2.0, 1.0], [1.0, -1.0]])
rhs2 = np.array([5.0, 1.0])
sol2 = np.linalg.solve(M2, rhs2)
print(f"Solution: x = {sol2[0]:.4f}, y = {sol2[1]:.4f}")

print("\n=== Solving a 3x3 system A x = b ===")
b = np.array([1.0, 2.0, 3.0])
x_solve = np.linalg.solve(A, b)
x_inv = A_inv @ b
x_scipy = linalg.solve(A, b)
print("x via solve()      =", x_solve)
print("x via A^-1 b       =", x_inv)
print("x via scipy.linalg =", x_scipy)
print("All three agree:", np.allclose(x_solve, x_inv) and np.allclose(x_solve, x_scipy))
print(f"Largest residual |A x - b| = {np.abs(A @ x_solve - b).max():.2e}")

print("\n=== LU decomposition: A = P L U ===")
P, L, U = linalg.lu(A)
print("L =\n", L)
print("U =\n", U)
print("P @ L @ U reproduces A:", np.allclose(P @ L @ U, A))

# ---------------------------------------------------------------
# 3. Singular and near-singular matrices
# ---------------------------------------------------------------
C = np.array([[1, 2, 3],
              [2, 4, 6],
              [1, 0, 1]], dtype=float)
print("\n=== Singular matrix (row 2 = 2 x row 1) ===")
print(f"det(C) = {np.linalg.det(C):.4f}, rank(C) = {np.linalg.matrix_rank(C)} (of 3)")
try:
    np.linalg.inv(C)
    print("numpy returned an inverse, but it is numerically meaningless")
except np.linalg.LinAlgError as err:
    print("np.linalg.inv failed:", err)
print("Pseudo-inverse still exists; C @ pinv(C) @ C reproduces C:",
      np.allclose(C @ linalg.pinv(C) @ C, C))

nearly = np.array([[1.0, 2.0], [2.0, 4.0001]])
print("\n=== Condition numbers (large = unstable inverse) ===")
print(f"Condition number of A (well-conditioned):    {np.linalg.cond(A):.2f}")
print(f"Condition number of a near-singular matrix:  {np.linalg.cond(nearly):.0f}")

# ---------------------------------------------------------------
# 4. Matrix algebra in statistics (student data from Topics 7 and 8)
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
    "study_hours": study, "sleep_hours": sleep, "attendance_pct": attendance,
    "screen_hours": screen, "exam_score": exam,
}).round(2)
df.to_csv(DATA / "student_data.csv", index=False)
predictors = ["study_hours", "sleep_hours", "attendance_pct", "screen_hours"]

# 4a. Covariance and correlation matrices from matrix algebra
Z = df.to_numpy()
Zc = Z - Z.mean(axis=0)
S = Zc.T @ Zc / (n - 1)
d_inv = np.diag(1 / np.sqrt(np.diag(S)))
R = d_inv @ S @ d_inv
print("\n=== Covariance matrix by matrix algebra: S = Zc' Zc / (n - 1) ===")
print(pd.DataFrame(S, index=df.columns, columns=df.columns).round(3))
print("Matches pandas .cov():", np.allclose(S, df.cov().to_numpy()))
print("Correlation R = D^-1/2 S D^-1/2 matches pandas .corr():",
      np.allclose(R, df.corr().to_numpy()))
pd.DataFrame(S, index=df.columns, columns=df.columns).round(4).to_csv(
    OUT / "covariance_by_matrix.csv")

# 4b. Regression by the normal equations
X = np.column_stack([np.ones(n), df[predictors].to_numpy()])
yv = df["exam_score"].to_numpy()
XtX = X.T @ X
Xty = X.T @ yv
beta = np.linalg.solve(XtX, Xty)
beta_lstsq = np.linalg.lstsq(X, yv, rcond=None)[0]
print("\n=== Regression by the normal equations: (X'X) beta = X'y ===")
print(f"X is {X.shape[0]} x {X.shape[1]}, X'X is {XtX.shape[0]} x {XtX.shape[1]}, "
      f"det(X'X) = {np.linalg.det(XtX):.4e}")
for name, val in zip(["Intercept"] + predictors, beta):
    print(f"{name:15s} {val:9.4f}")
print("Matches numpy lstsq:", np.allclose(beta, beta_lstsq))
print(f"Condition number of X'X = {np.linalg.cond(XtX):.0f}")
pd.DataFrame({"term": ["Intercept"] + predictors, "coefficient": beta}).round(4).to_csv(
    OUT / "beta_normal_equations.csv", index=False)

# 4c. Hat matrix, fitted values, residuals, leverage
H = X @ np.linalg.solve(XtX, X.T)
fitted = H @ yv
resid = yv - fitted
leverage = np.diag(H)
p = X.shape[1]
print("\n=== Hat matrix H = X (X'X)^-1 X' ===")
print(f"H is {H.shape[0]} x {H.shape[1]}")
print("H is symmetric:", np.allclose(H, H.T))
print("H is idempotent (H H = H):", np.allclose(H @ H, H))
print(f"trace(H) = {np.trace(H):.4f} (equals the number of parameters, {p})")
print("Fitted values H y equal X beta:", np.allclose(fitted, X @ beta))
print(f"Residuals (I - H) y: mean = {resid.mean():.2e}, "
      f"X' residuals max = {np.abs(X.T @ resid).max():.2e} (orthogonal)")
cut = 2 * p / n
print(f"Largest leverage = {leverage.max():.4f}; {int((leverage > cut).sum())} points "
      f"above 2p/n = {cut:.3f}")

# 4d. Mahalanobis distance
Xp = df[predictors].to_numpy()
Xpc = Xp - Xp.mean(axis=0)
S_inv = np.linalg.inv(np.cov(Xp, rowvar=False))
d2 = np.einsum("ij,jk,ik->i", Xpc, S_inv, Xpc)
cutoff = stats.chi2.ppf(0.975, df=len(predictors))
print("\n=== Mahalanobis distance (squared) ===")
print(f"Mean = {d2.mean():.3f}, largest = {d2.max():.3f}, "
      f"cutoff (chi-square 97.5%, df = 4) = {cutoff:.3f}")
print(f"{int((d2 > cutoff).sum())} of {n} students above the cutoff "
      f"({(d2 > cutoff).mean():.1%}, expected about 2.5%)")

# ---------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------
# Plot 1: A, B and the product as heatmaps
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, mat, title in zip(axes, [A, B, AB, A_inv], ["A", "B", "A x B", "Inverse of A"]):
    im = ax.imshow(mat, cmap="coolwarm")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(OUT / "matrix_operations.png", dpi=150)

# Plot 2: geometry of a linear system
xs = np.linspace(-1, 5, 100)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot(xs, 5 - 2 * xs, label="2x + y = 5")
ax.plot(xs, xs - 1, label="x - y = 1")
ax.plot(*sol2, "ro", ms=9, label=f"Solution ({sol2[0]:.0f}, {sol2[1]:.0f})")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Solving a linear system: the lines intersect at the solution")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "linear_system_geometry.png", dpi=150)

# Plot 3: covariance and correlation computed by matrix algebra
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, mat, title, kw in [
    (axes[0], S, "Covariance matrix S", {"cmap": "viridis"}),
    (axes[1], R, "Correlation matrix R", {"cmap": "coolwarm", "vmin": -1, "vmax": 1}),
]:
    im = ax.imshow(mat, **kw)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticklabels(df.columns)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white"
                if kw["cmap"] == "viridis" else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(OUT / "covariance_correlation_matrices.png", dpi=150)

# Plot 4: leverage from the hat matrix
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.stem(leverage, markerfmt="o", basefmt=" ")
ax.axhline(cut, color="red", linestyle="--", label=f"2p/n = {cut:.3f}")
ax.set_xlabel("Student")
ax.set_ylabel("Leverage (diagonal of H)")
ax.set_title("Leverage of each student in the regression")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "leverage.png", dpi=150)

# Plot 5: Mahalanobis distance
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(d2, bins=25, density=True, color="steelblue", alpha=0.7,
        label="Squared Mahalanobis distance")
grid = np.linspace(0, d2.max(), 300)
ax.plot(grid, stats.chi2.pdf(grid, df=4), "r-", lw=2, label="Chi-square (df = 4)")
ax.axvline(cutoff, color="black", linestyle="--", label=f"97.5% cutoff = {cutoff:.2f}")
ax.set_xlabel("Squared distance")
ax.set_title("Mahalanobis distances follow a chi-square distribution")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "mahalanobis.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)