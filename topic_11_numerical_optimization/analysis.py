"""
Topic 11: Numerical Optimization
Libraries: scipy.optimize, scipy.stats, numpy, pandas, matplotlib
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# 1. One variable: f(x) = x^4 - 3x^3 + 2
# ---------------------------------------------------------------
def f1(x):
    return x ** 4 - 3 * x ** 3 + 2


def d1(x):
    return 4 * x ** 3 - 9 * x ** 2


def d2(x):
    return 12 * x ** 2 - 18 * x


x_star = 9 / 4
print("=== 1. One variable: f(x) = x^4 - 3x^3 + 2 ===")
print(f"Analytic: f'(x) = x^2 (4x - 9) = 0 gives x = 9/4 = {x_star:.4f}, "
      f"f(x*) = {f1(x_star):.4f}, f''(x*) = {d2(x_star):.3f} (> 0, a minimum)")

brent = optimize.minimize_scalar(f1, bracket=(1, 3), method="brent")
bounded = optimize.minimize_scalar(f1, bounds=(0, 5), method="bounded")
root = optimize.root_scalar(d1, bracket=[1, 4], method="brentq")
print(f"Brent:            x = {brent.x:.6f}, f = {brent.fun:.6f}, "
      f"iterations = {brent.nit}, evaluations = {brent.nfev}")
print(f"Bounded search:   x = {bounded.x:.6f}, f = {bounded.fun:.6f}, "
      f"iterations = {bounded.nit}, evaluations = {bounded.nfev}")
print(f"Root of f'(x):    x = {root.root:.6f} (brentq, {root.iterations} iterations)")

print("\nNewton's method on f'(x) = 0, starting at x = 3:")
xn, newton_err = 3.0, []
for i in range(1, 10):
    xn = xn - d1(xn) / d2(xn)
    newton_err.append(abs(xn - x_star))
    print(f"  iteration {i}: x = {xn:.10f}, error = {newton_err[-1]:.2e}")
    if abs(d1(xn)) < 1e-10:
        break
print(f"Converged after {i} iterations (the error roughly squares each step)")

# ---------------------------------------------------------------
# 2. Several variables: Rosenbrock function, seven methods
# ---------------------------------------------------------------
x0 = np.array([-1.2, 1.0])
configs = [
    ("Nelder-Mead", dict(method="Nelder-Mead")),
    ("Powell", dict(method="Powell")),
    ("CG", dict(method="CG", jac=optimize.rosen_der)),
    ("BFGS (numerical gradient)", dict(method="BFGS")),
    ("BFGS (analytic gradient)", dict(method="BFGS", jac=optimize.rosen_der)),
    ("L-BFGS-B", dict(method="L-BFGS-B", jac=optimize.rosen_der)),
    ("Newton-CG", dict(method="Newton-CG", jac=optimize.rosen_der,
                       hess=optimize.rosen_hess)),
]
paths, rows = {}, []
for label, kw in configs:
    path = [x0.copy()]
    res = optimize.minimize(optimize.rosen, x0, tol=1e-6,
                            callback=lambda xk: path.append(np.copy(xk)), **kw)
    paths[label] = np.array(path)
    rows.append({
        "method": label, "converged": bool(res.success), "iterations": res.nit,
        "f_evals": res.nfev, "grad_evals": getattr(res, "njev", np.nan),
        "final_f": res.fun, "distance_to_optimum": np.linalg.norm(res.x - 1.0),
    })
rosen_table = pd.DataFrame(rows)
print("\n=== 2. Rosenbrock function f(x, y) = 100 (y - x^2)^2 + (1 - x)^2 ===")
print(f"Start = {x0}, true minimum at (1, 1) with f = 0")
print(rosen_table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
rosen_table.to_csv(OUT / "rosenbrock_methods.csv", index=False)

# ---------------------------------------------------------------
# 3. Gradient descent by hand: f(x, y) = x^2 + 10 y^2
# ---------------------------------------------------------------
def quad(z):
    return z[0] ** 2 + 10 * z[1] ** 2


def quad_grad(z):
    return np.array([2 * z[0], 20 * z[1]])


print("\n=== 3. Gradient descent on f(x, y) = x^2 + 10 y^2, start (8, 5) ===")
print("Curvatures are 2 and 20, so the best fixed step is 2 / (2 + 20) = 0.0909 "
      "and steps above 2 / 20 = 0.1 diverge")
gd_hist, gd_rows = {}, []
for lr in (0.01, 0.05, 0.0909, 0.11):
    z = np.array([8.0, 5.0])
    hist = [quad(z)]
    status = "max iterations reached"
    for it in range(1, 2001):
        z = z - lr * quad_grad(z)
        hist.append(quad(z))
        if hist[-1] < 1e-10:
            status = "converged"
            break
        if hist[-1] > 1e10:
            status = "diverged"
            break
    gd_hist[lr] = np.array(hist)
    gd_rows.append((lr, it, hist[-1], status))
    print(f"step = {lr:6.4f}: {status:22s} after {it:4d} iterations, "
          f"final f = {hist[-1]:.3e}")

# ---------------------------------------------------------------
# 4. Maximum likelihood by optimization: gamma distribution
# ---------------------------------------------------------------
TRUE_SHAPE, TRUE_SCALE = 3.0, 2.0
sample = rng.gamma(TRUE_SHAPE, TRUE_SCALE, 500)
pd.DataFrame({"value": sample.round(4)}).to_csv(DATA / "gamma_sample.csv", index=False)
n = sample.size
sum_log, sum_x = np.log(sample).sum(), sample.sum()


def nll(params):  # parameters on the log scale so shape and scale stay positive
    a, s = np.exp(params)
    return -np.sum(stats.gamma.logpdf(sample, a, scale=s))


def nll_grid(a, s):  # closed-form negative log-likelihood, vectorised
    return -((a - 1) * sum_log - sum_x / s - n * gammaln(a) - n * a * np.log(s))


mle = optimize.minimize(nll, x0=[0.0, 0.0], method="BFGS")
a_hat, s_hat = np.exp(mle.x)
a_fit, _, s_fit = stats.gamma.fit(sample, floc=0)
mom_a = sample.mean() ** 2 / sample.var(ddof=1)
mom_s = sample.var(ddof=1) / sample.mean()
print("\n=== 4. Maximum likelihood for a gamma distribution (n = 500) ===")
print(f"True values: shape = {TRUE_SHAPE}, scale = {TRUE_SCALE}")
print("Optimizer start: shape = 1, scale = 1")
print(f"Optimizer (BFGS): shape = {a_hat:.4f}, scale = {s_hat:.4f}, "
      f"negative log-likelihood = {mle.fun:.3f}")
print(f"  converged = {mle.success}, iterations = {mle.nit}, "
      f"function evaluations = {mle.nfev}")
print(f"scipy gamma.fit:  shape = {a_fit:.4f}, scale = {s_fit:.4f}")
print(f"Method of moments: shape = {mom_a:.4f}, scale = {mom_s:.4f}")
print(f"Closed-form grid formula matches the optimizer's objective: "
      f"{np.isclose(nll_grid(a_hat, s_hat), mle.fun)}")
print(f"Fitted mean = shape x scale = {a_hat * s_hat:.4f} "
      f"(sample mean = {sample.mean():.4f})")
pd.DataFrame({
    "method": ["True", "BFGS on log-likelihood", "scipy gamma.fit", "Method of moments"],
    "shape": [TRUE_SHAPE, a_hat, a_fit, mom_a],
    "scale": [TRUE_SCALE, s_hat, s_fit, mom_s],
}).round(4).to_csv(OUT / "mle_gamma.csv", index=False)

# ---------------------------------------------------------------
# 5. Constrained optimization: portfolio selection (SLSQP)
# ---------------------------------------------------------------
mu = np.array([0.06, 0.10, 0.14])
Sigma = np.array([[0.040, 0.006, 0.012],
                  [0.006, 0.090, 0.018],
                  [0.012, 0.018, 0.160]])
assets = ["Bonds", "Shares", "Growth"]
RF = 0.02


def variance(w):
    return w @ Sigma @ w


budget = {"type": "eq", "fun": lambda w: w.sum() - 1}
bounds = [(0, 1)] * 3
w0 = np.ones(3) / 3

# Tight tolerance so SLSQP matches the exact analytic solution
minvar = optimize.minimize(variance, w0, method="SLSQP", bounds=bounds,
                           constraints=[budget], options={"ftol": 1e-12})
ones = np.ones(3)
w_analytic = np.linalg.solve(Sigma, ones) / (ones @ np.linalg.solve(Sigma, ones))
print("\n=== 5. Portfolio optimization (3 assets, weights >= 0, sum to 1) ===")
print("Expected returns:", dict(zip(assets, mu)))
print(f"Minimum-variance portfolio (SLSQP): weights = {minvar.x.round(4)}")
print(f"  return = {minvar.x @ mu:.4f}, volatility = {np.sqrt(minvar.fun):.4f}, "
      f"converged = {minvar.success}, iterations = {minvar.nit}")
print(f"Analytic solution S^-1 1 / (1' S^-1 1) = {w_analytic.round(4)}, "
      f"matches: {np.allclose(minvar.x, w_analytic, atol=1e-4)}")
print(f"Largest weight difference = {np.abs(minvar.x - w_analytic).max():.2e}")


def neg_sharpe(w):
    return -(w @ mu - RF) / np.sqrt(variance(w))


sharpe = optimize.minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                           constraints=[budget])
print(f"Maximum-Sharpe portfolio (risk-free rate {RF:.0%}): weights = "
      f"{sharpe.x.round(4)}")
print(f"  return = {sharpe.x @ mu:.4f}, volatility = "
      f"{np.sqrt(variance(sharpe.x)):.4f}, Sharpe ratio = {-sharpe.fun:.4f}")

targets = np.linspace(0.07, 0.13, 25)
frontier = []
for t in targets:
    cons = [budget, {"type": "eq", "fun": lambda w, t=t: w @ mu - t}]
    r = optimize.minimize(variance, w0, method="SLSQP", bounds=bounds,
                          constraints=cons)
    frontier.append((t, np.sqrt(r.fun), *r.x, r.success))
frontier = pd.DataFrame(frontier, columns=["target_return", "volatility"] + assets
                        + ["converged"])
print(f"Efficient frontier: {len(frontier)} target returns from {targets[0]:.2f} to "
      f"{targets[-1]:.2f}, all converged: {bool(frontier['converged'].all())}")
print(f"Volatility ranges from {frontier['volatility'].min():.4f} to "
      f"{frontier['volatility'].max():.4f}")
frontier.round(4).to_csv(OUT / "efficient_frontier.csv", index=False)

# ---------------------------------------------------------------
# 6. Global optimization: Rastrigin function
# ---------------------------------------------------------------
def rastrigin(z):
    return 20 + z[0] ** 2 - 10 * np.cos(2 * np.pi * z[0]) \
        + z[1] ** 2 - 10 * np.cos(2 * np.pi * z[1])


lim = 5.12
starts = rng.uniform(-lim, lim, size=(100, 2))
local = [optimize.minimize(rastrigin, s, method="BFGS") for s in starts]
local_f = np.array([r.fun for r in local])
local_x = np.array([r.x for r in local])
hit = local_f < 1e-4
de = optimize.differential_evolution(rastrigin, [(-lim, lim)] * 2, seed=42, tol=1e-8)
print("\n=== 6. Global optimization: Rastrigin function (global minimum 0 at (0, 0)) ===")
print(f"100 local BFGS searches from random starts: {int(hit.sum())} reached the "
      f"global minimum")
print(f"Distinct local minima found: {np.unique(local_f.round(2)).size}, "
      f"median final f = {np.median(local_f):.3f}")
print(f"Differential evolution: x = {de.x.round(6)}, f = {de.fun:.2e}, "
      f"function evaluations = {de.nfev}, converged = {de.success}")

# ---------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------
# Plot 1: one-variable function
xs = np.linspace(-0.8, 3.6, 400)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(xs, f1(xs), color="steelblue", lw=2, label="f(x) = x^4 - 3x^3 + 2")
ax.plot(brent.x, brent.fun, "ro", ms=9, label=f"Minimum x = {brent.x:.3f}")
ax.plot(0, f1(0), "s", color="grey", ms=8, label="Stationary point at 0 (not a minimum)")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("One-variable optimization")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "one_variable_optimum.png", dpi=150)

# Plot 2: Rosenbrock paths and convergence
gx, gy = np.meshgrid(np.linspace(-1.6, 1.6, 400), np.linspace(-0.6, 1.8, 400))
gz = optimize.rosen(np.array([gx, gy]))
colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
           "tab:brown", "tab:pink"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
ax.contour(gx, gy, gz, levels=np.logspace(-1, 3.5, 20), cmap="Greys", alpha=0.6)
for (label, _), colour in zip(configs, colours):
    if label == "BFGS (numerical gradient)":
        continue
    p_ = paths[label]
    ax.plot(p_[:, 0], p_[:, 1], "-o", ms=3, lw=1.2, color=colour, label=label)
ax.plot(*x0, "ks", ms=8, label="Start")
ax.plot(1, 1, "r*", ms=15, label="Optimum (1, 1)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Search paths on the Rosenbrock function")
ax.legend(fontsize=8)
ax = axes[1]
for (label, _), colour in zip(configs, colours):
    fvals = np.maximum([optimize.rosen(q) for q in paths[label]], 1e-16)
    ax.semilogy(fvals, color=colour, label=label)
ax.set_xlabel("Iteration")
ax.set_ylabel("f(x, y)")
ax.set_title("Convergence of the objective function")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "rosenbrock_paths_convergence.png", dpi=150)

# Plot 3: gradient descent step sizes
fig, ax = plt.subplots(figsize=(8, 5))
for lr, h in gd_hist.items():
    ax.semilogy(np.maximum(h, 1e-16), label=f"step = {lr}")
ax.axhline(1e-10, color="grey", linestyle=":", label="Tolerance")
ax.set_xlabel("Iteration")
ax.set_ylabel("f(x, y)")
ax.set_title("Gradient descent: effect of the step size")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "gradient_descent_steps.png", dpi=150)

# Plot 4: maximum likelihood
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ag, sg = np.meshgrid(np.linspace(2.0, 4.5, 150), np.linspace(1.2, 3.2, 150))
ax = axes[0]
cs = ax.contourf(ag, sg, nll_grid(ag, sg), levels=30, cmap="viridis")
fig.colorbar(cs, ax=ax, label="Negative log-likelihood")
ax.plot(a_hat, s_hat, "r*", ms=15, label=f"MLE ({a_hat:.2f}, {s_hat:.2f})")
ax.plot(TRUE_SHAPE, TRUE_SCALE, "wo", ms=8, label="True (3, 2)")
ax.plot(1, 1, "ks", ms=7, label="Optimizer start")
ax.set_xlabel("Shape")
ax.set_ylabel("Scale")
ax.set_title("Negative log-likelihood surface")
ax.legend(loc="upper right", fontsize=8)
ax = axes[1]
ax.hist(sample, bins=30, density=True, color="steelblue", alpha=0.6, label="Data")
xx = np.linspace(0, sample.max(), 300)
ax.plot(xx, stats.gamma.pdf(xx, a_hat, scale=s_hat), "r-", lw=2,
        label=f"Fitted gamma({a_hat:.2f}, {s_hat:.2f})")
ax.set_xlabel("Value")
ax.set_title("Fitted distribution")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "mle_gamma.png", dpi=150)

# Plot 5: efficient frontier
wr = rng.dirichlet(np.ones(3), 3000)
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(np.sqrt(np.einsum("ij,jk,ik->i", wr, Sigma, wr)), wr @ mu, s=6,
           color="lightgrey", label="Random portfolios")
ax.plot(frontier["volatility"], frontier["target_return"], "b-", lw=2.5,
        label="Efficient frontier (SLSQP)")
ax.plot(np.sqrt(minvar.fun), minvar.x @ mu, "go", ms=10, label="Minimum variance")
ax.plot(np.sqrt(variance(sharpe.x)), sharpe.x @ mu, "r*", ms=15,
        label="Maximum Sharpe ratio")
ax.set_xlabel("Volatility (standard deviation)")
ax.set_ylabel("Expected return")
ax.set_title("Constrained optimization: efficient frontier")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "efficient_frontier.png", dpi=150)

# Plot 6: Rastrigin local vs global
rx, ry = np.meshgrid(np.linspace(-lim, lim, 400), np.linspace(-lim, lim, 400))
rz = rastrigin(np.array([rx, ry]))
fig, ax = plt.subplots(figsize=(7.5, 6.5))
cs = ax.contourf(rx, ry, rz, levels=30, cmap="viridis")
fig.colorbar(cs, ax=ax, label="f(x, y)")
ax.scatter(local_x[~hit, 0], local_x[~hit, 1], color="orange", s=25,
           label="Local search stuck in a local minimum")
ax.scatter(local_x[hit, 0], local_x[hit, 1], color="red", s=25, marker="s",
           label="Local search reached the global minimum")
ax.plot(*de.x, "w*", ms=18, markeredgecolor="black", label="Differential evolution")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Multimodal function: local versus global search")
ax.legend(loc="upper right", fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "rastrigin_local_vs_global.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)