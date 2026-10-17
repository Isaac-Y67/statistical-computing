"""
Topic 12: Random Numbers and Monte Carlo Methods
Libraries: numpy.random, scipy.stats, scipy.special, pandas, matplotlib
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erf

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
rows = []  # (experiment, quantity, value)


def add(experiment, quantity, value):
    rows.append((experiment, quantity, float(value)))


# ---------------------------------------------------------------
# 1. Random number generation
# ---------------------------------------------------------------
def lcg(seed, n, a=1103515245, c=12345, m=2 ** 31):
    """Linear congruential generator: x_{k+1} = (a x_k + c) mod m."""
    out = np.empty(n)
    x = seed
    for i in range(n):
        x = (a * x + c) % m
        out[i] = x / m
    return out


def lcg_period(seed, a, c, m):
    seen, x, i = {}, seed, 0
    while x not in seen:
        seen[x] = i
        x = (a * x + c) % m
        i += 1
    return i - seen[x]


print("=== 1a. Linear congruential generator (LCG) ===")
print("Toy generator x -> (5x + 3) mod 16, seed 1: period = "
      f"{lcg_period(1, 5, 3, 16)} (full period is 16)")
print("Toy generator x -> (4x + 2) mod 16, seed 1: period = "
      f"{lcg_period(1, 4, 2, 16)} (a poor choice of constants)")

N_LCG = 10000
u_lcg = lcg(12345, N_LCG)
u_np = rng.random(N_LCG)
test_rows = []
for label, u in (("Own LCG", u_lcg), ("numpy generator", u_np)):
    counts, _ = np.histogram(u, bins=10, range=(0, 1))
    chi = stats.chisquare(counts)
    ks = stats.kstest(u, stats.uniform.cdf)
    ac = np.corrcoef(u[:-1], u[1:])[0, 1]
    test_rows.append((label, u.mean(), u.var(ddof=1), chi.pvalue, ks.pvalue, ac))
    print(f"{label:16s}: mean = {u.mean():.4f}, variance = {u.var(ddof=1):.4f}, "
          f"chi-square p = {chi.pvalue:.4f}, K-S p = {ks.pvalue:.4f}, "
          f"lag-1 autocorrelation = {ac:.4f}")
    add("LCG uniformity", f"{label} K-S p", ks.pvalue)
print("Target for Uniform(0,1): mean = 0.5000, variance = 0.0833")
pd.DataFrame(test_rows, columns=["generator", "mean", "variance", "chi_square_p",
                                 "ks_p", "lag1_autocorr"]
             ).round(4).to_csv(OUT / "lcg_tests.csv", index=False)

# Inverse transform: Exponential(rate 0.5)
RATE = 0.5
n_s = 5000
u = rng.random(n_s)
x_exp = -np.log(1 - u) / RATE
ks_e = stats.kstest(x_exp, stats.expon(scale=1 / RATE).cdf)
print("\n=== 1b. Inverse transform sampling: Exponential(rate 0.5) ===")
print(f"X = -ln(1 - U) / rate. n = {n_s}: mean = {x_exp.mean():.4f} (theory 2.0000), "
      f"variance = {x_exp.var(ddof=1):.4f} (theory 4.0000), K-S p = {ks_e.pvalue:.4f}")
add("Inverse transform", "mean", x_exp.mean())
add("Inverse transform", "K-S p", ks_e.pvalue)

# Box-Muller: standard normal
u1, u2 = rng.random(n_s // 2), rng.random(n_s // 2)
r = np.sqrt(-2 * np.log(1 - u1))
z = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])
ks_z = stats.kstest(z, stats.norm.cdf)
print("\n=== 1c. Box-Muller: standard normal ===")
print(f"n = {z.size}: mean = {z.mean():.4f}, sd = {z.std(ddof=1):.4f}, "
      f"skewness = {stats.skew(z):.4f}, K-S p = {ks_z.pvalue:.4f}")
print(f"Corr between the cosine and sine halves = "
      f"{np.corrcoef(z[:n_s // 2], z[n_s // 2:])[0, 1]:.4f} (independent, so near 0)")
add("Box-Muller", "mean", z.mean())
add("Box-Muller", "sd", z.std(ddof=1))
add("Box-Muller", "K-S p", ks_z.pvalue)

# Acceptance-rejection: Beta(2, 2), pdf 6x(1-x), bound M = 1.5, uniform proposal
M = 1.5
n_prop = 20000
cand = rng.random(n_prop)
vv = rng.random(n_prop)
accepted = cand[vv * M < 6 * cand * (1 - cand)]
ks_b = stats.kstest(accepted, stats.beta(2, 2).cdf)
print("\n=== 1d. Acceptance-rejection: Beta(2, 2) ===")
print(f"{n_prop} proposals, {accepted.size} accepted "
      f"(rate {accepted.size / n_prop:.4f}, theory 1/M = {1 / M:.4f})")
print(f"Accepted sample: mean = {accepted.mean():.4f} (theory 0.5000), "
      f"variance = {accepted.var(ddof=1):.4f} (theory 0.0500), "
      f"K-S p = {ks_b.pvalue:.4f}")
add("Acceptance-rejection", "acceptance rate", accepted.size / n_prop)
add("Acceptance-rejection", "K-S p", ks_b.pvalue)

k = 1000
pd.DataFrame({
    "lcg_uniform": u_lcg[:k],
    "inverse_transform_exp": x_exp[:k],
    "box_muller_normal": z[:k],
    "accept_reject_beta": accepted[:k],
}).round(5).to_csv(DATA / "random_samples.csv", index=False)

# ---------------------------------------------------------------
# 2. Estimating pi
# ---------------------------------------------------------------
N = 1_000_000
px, py = rng.random(N), rng.random(N)
inside = px ** 2 + py ** 2 < 1
running_pi = 4 * np.cumsum(inside) / np.arange(1, N + 1)

print("\n=== 2. Monte Carlo estimate of pi (points in the unit square) ===")
for n_pts in (100, 1000, 10_000, 100_000, 1_000_000):
    est = running_pi[n_pts - 1]
    p_ = est / 4
    se = 4 * np.sqrt(p_ * (1 - p_) / n_pts)
    print(f"N = {n_pts:>9,d}: estimate = {est:.5f}, error = {est - np.pi:+.5f}, "
          f"standard error = {se:.5f}")
    add("Pi", f"estimate N={n_pts}", est)
est = running_pi[-1]
se = 4 * np.sqrt((est / 4) * (1 - est / 4) / N)
lo, hi = est - 1.96 * se, est + 1.96 * se
print(f"95% confidence interval at N = {N:,d}: ({lo:.5f}, {hi:.5f}); "
      f"contains pi = {np.pi:.5f}: {lo <= np.pi <= hi}")

print("\nError scaling: standard deviation of the estimate over 200 repeated runs")
Ns = [100, 1000, 10_000, 100_000]
sd_emp, sd_theory = [], []
for n_pts in Ns:
    ests = [4 * np.mean(np.sum(rng.random((n_pts, 2)) ** 2, axis=1) < 1)
            for _ in range(200)]
    sd_emp.append(np.std(ests, ddof=1))
    sd_theory.append(np.sqrt(np.pi * (4 - np.pi) / n_pts))
    print(f"N = {n_pts:>7,d}: empirical sd = {sd_emp[-1]:.5f}, "
          f"theoretical sqrt(pi (4 - pi) / N) = {sd_theory[-1]:.5f}")
print("Each 100-fold increase in N cuts the error by about 10 (the 1/sqrt(N) rate)")

# ---------------------------------------------------------------
# 3. Monte Carlo integration and variance reduction
# ---------------------------------------------------------------
def f(t):
    return np.exp(-t ** 2)


truth = 0.5 * np.sqrt(np.pi) * erf(1)
print("\n=== 3a. Monte Carlo integration of exp(-x^2) on [0, 1] ===")
print(f"Exact value = (sqrt(pi) / 2) erf(1) = {truth:.6f}")
u = rng.random(1000)
y = f(u)
m_hat, se_hat = y.mean(), y.std(ddof=1) / np.sqrt(y.size)
print(f"One run, n = 1000: estimate = {m_hat:.5f}, standard error = {se_hat:.5f}, "
      f"95% CI = ({m_hat - 1.96 * se_hat:.5f}, {m_hat + 1.96 * se_hat:.5f}), "
      f"contains truth: {abs(m_hat - truth) <= 1.96 * se_hat}")
add("Integration", "single-run estimate", m_hat)

R, n = 5000, 1000
u = rng.random((R, n))
simple = f(u).mean(axis=1)
u2 = rng.random((R, n // 2))
anti = (0.5 * (f(u2) + f(1 - u2))).mean(axis=1)
pilot = rng.random(200_000)
beta = np.cov(f(pilot), 1 - pilot ** 2)[0, 1] / np.var(1 - pilot ** 2, ddof=1)
control = (f(u) - beta * ((1 - u ** 2) - 2 / 3)).mean(axis=1)

print(f"\n=== 3b. Variance reduction ({R} repeated estimates, n = {n} evaluations each) ===")
vr_rows = []
for name, est_arr in (("Simple Monte Carlo", simple), ("Antithetic variates", anti),
                      ("Control variate", control)):
    ratio = simple.var(ddof=1) / est_arr.var(ddof=1)
    vr_rows.append((name, est_arr.mean(), est_arr.mean() - truth,
                    est_arr.std(ddof=1), ratio))
    print(f"{name:20s}: mean = {est_arr.mean():.6f}, bias = {est_arr.mean() - truth:+.2e}, "
          f"sd = {est_arr.std(ddof=1):.6f}, variance reduction factor = {ratio:.1f}")
    add("Variance reduction", f"{name} sd", est_arr.std(ddof=1))
print(f"Control variate g(x) = 1 - x^2 (known mean 2/3), coefficient beta = {beta:.4f}")
pd.DataFrame(vr_rows, columns=["method", "mean", "bias", "sd", "variance_reduction"]
             ).round(8).to_csv(OUT / "variance_reduction.csv", index=False)

# ---------------------------------------------------------------
# 4. Importance sampling: P(Z > 3)
# ---------------------------------------------------------------
p_true = stats.norm.sf(3)
R, n = 1000, 10_000
zz = rng.standard_normal((R, n))
naive = (zz > 3).mean(axis=1)
xs_is = rng.standard_normal((R, n)) + 3
imp = ((xs_is > 3) * np.exp(-3 * xs_is + 4.5)).mean(axis=1)
print("\n=== 4. Importance sampling for a rare event, P(Z > 3) ===")
print(f"Exact value = {p_true:.6f}; {R} repeated estimates with n = {n:,d} draws each")
print(f"Naive Monte Carlo: mean = {naive.mean():.6f}, sd = {naive.std(ddof=1):.6f}, "
      f"relative sd = {naive.std(ddof=1) / p_true:.1%}")
print(f"Importance sampling (proposal N(3, 1)): mean = {imp.mean():.6f}, "
      f"sd = {imp.std(ddof=1):.6f}, relative sd = {imp.std(ddof=1) / p_true:.1%}")
print(f"Variance reduction factor = {naive.var(ddof=1) / imp.var(ddof=1):.0f}")
add("Importance sampling", "naive sd", naive.std(ddof=1))
add("Importance sampling", "IS sd", imp.std(ddof=1))

# ---------------------------------------------------------------
# 5. Simulating the properties of estimators
# ---------------------------------------------------------------
S, n = 100_000, 10
x = rng.normal(0, 2, (S, n))
v_n, v_n1 = x.var(axis=1, ddof=0), x.var(axis=1, ddof=1)
print(f"\n=== 5a. Bias of the variance estimator (true variance 4, n = {n}, "
      f"{S:,d} samples) ===")
print(f"Divisor n:     mean = {v_n.mean():.4f}, bias = {v_n.mean() - 4:+.4f} "
      f"(theory {-4 / n:+.4f})")
print(f"Divisor n - 1: mean = {v_n1.mean():.4f}, bias = {v_n1.mean() - 4:+.4f} "
      f"(theory +0.0000)")
add("Variance estimator", "bias, divisor n", v_n.mean() - 4)
add("Variance estimator", "bias, divisor n-1", v_n1.mean() - 4)

S, n = 20_000, 25
xn = rng.standard_normal((S, n))
xt = rng.standard_t(3, (S, n))
print(f"\n=== 5b. Mean versus median as estimators of the centre "
      f"(n = {n}, {S:,d} samples) ===")
eff_rows = []
for label, data in (("Normal", xn), ("t distribution, 3 df", xt)):
    var_mean = data.mean(axis=1).var(ddof=1)
    var_med = np.median(data, axis=1).var(ddof=1)
    eff_rows.append((label, var_mean, var_med, var_mean / var_med))
    print(f"{label:21s}: variance of mean = {var_mean:.4f}, variance of median = "
          f"{var_med:.4f}, efficiency of mean relative to median = {var_mean / var_med:.2f}")
    add("Mean vs median", f"{label} efficiency", var_mean / var_med)
pd.DataFrame(eff_rows, columns=["distribution", "var_mean", "var_median",
                                "var_mean_over_var_median"]
             ).round(4).to_csv(OUT / "mean_vs_median.csv", index=False)

# ---------------------------------------------------------------
# 6. Birthday problem
# ---------------------------------------------------------------
def birthday_exact(k):
    return 1 - np.prod((365 - np.arange(k)) / 365)


group_sizes = [10, 15, 20, 23, 25, 30, 40, 50, 60]
R = 20_000
print(f"\n=== 6. Birthday problem ({R:,d} simulated groups per size) ===")
bday = []
for g in group_sizes:
    days = np.sort(rng.integers(0, 365, size=(R, g)), axis=1)
    sim = np.mean((np.diff(days, axis=1) == 0).any(axis=1))
    ex = birthday_exact(g)
    bday.append((g, sim, ex))
    print(f"{g:2d} people: simulated P(shared birthday) = {sim:.4f}, exact = {ex:.4f}")
    add("Birthday", f"simulated k={g}", sim)
pd.DataFrame(bday, columns=["group_size", "simulated", "exact"]).round(4).to_csv(
    OUT / "birthday_problem.csv", index=False)

# ---------------------------------------------------------------
# 7. Save summary table
# ---------------------------------------------------------------
pd.DataFrame(rows, columns=["experiment", "quantity", "value"]).round(6).to_csv(
    OUT / "monte_carlo_results.csv", index=False)

# ---------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------
# Plot 1: four random number generators
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
ax = axes[0, 0]
ax.hist(u_lcg, bins=20, density=True, color="steelblue", alpha=0.7)
ax.axhline(1, color="red", lw=2)
ax.set_title("Own LCG: Uniform(0, 1)")
ax = axes[0, 1]
ax.hist(x_exp, bins=40, density=True, color="steelblue", alpha=0.7)
grid = np.linspace(0, x_exp.max(), 300)
ax.plot(grid, stats.expon.pdf(grid, scale=1 / RATE), "r-", lw=2)
ax.set_title("Inverse transform: Exponential(0.5)")
ax = axes[1, 0]
ax.hist(z, bins=40, density=True, color="steelblue", alpha=0.7)
grid = np.linspace(-4, 4, 300)
ax.plot(grid, stats.norm.pdf(grid), "r-", lw=2)
ax.set_title("Box-Muller: standard normal")
ax = axes[1, 1]
ax.hist(accepted, bins=30, density=True, color="steelblue", alpha=0.7)
grid = np.linspace(0, 1, 300)
ax.plot(grid, stats.beta.pdf(grid, 2, 2), "r-", lw=2)
ax.set_title("Acceptance-rejection: Beta(2, 2)")
fig.suptitle("Random samples (bars) against the target densities (red lines)")
fig.tight_layout()
fig.savefig(OUT / "random_generation.png", dpi=150)

# Plot 2: pi estimation and convergence
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
m_pts = 3000
ax.scatter(px[:m_pts][inside[:m_pts]], py[:m_pts][inside[:m_pts]], s=4, color="tomato")
ax.scatter(px[:m_pts][~inside[:m_pts]], py[:m_pts][~inside[:m_pts]], s=4,
           color="steelblue")
t = np.linspace(0, np.pi / 2, 200)
ax.plot(np.cos(t), np.sin(t), "k-", lw=1.5)
ax.set_aspect("equal")
ax.set_title(f"First {m_pts} random points (red = inside the circle)")
ax = axes[1]
ax.semilogx(np.arange(1, N + 1), running_pi, color="steelblue", lw=1)
ax.axhline(np.pi, color="red", linestyle="--", label="True value of pi")
ax.set_ylim(2.9, 3.4)
ax.set_xlabel("Number of points")
ax.set_ylabel("Running estimate of pi")
ax.set_title("Convergence of the Monte Carlo estimate")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "pi_estimation_convergence.png", dpi=150)

# Plot 3: 1/sqrt(N) error scaling
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(Ns, sd_emp, "bo-", label="Empirical sd (200 runs)")
ax.loglog(Ns, sd_theory, "r--", label="Theory: sqrt(pi (4 - pi)) / sqrt(N)")
ax.set_xlabel("Number of points N")
ax.set_ylabel("Standard deviation of the estimate")
ax.set_title("Monte Carlo error falls like 1 / sqrt(N)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "error_scaling.png", dpi=150)

# Plot 4: variance reduction
fig, ax = plt.subplots(figsize=(8.5, 5))
bins = np.linspace(min(simple.min(), anti.min(), control.min()),
                   max(simple.max(), anti.max(), control.max()), 60)
ax.hist(simple, bins=bins, alpha=0.5, label="Simple", color="grey")
ax.hist(anti, bins=bins, alpha=0.5, label="Antithetic", color="steelblue")
ax.hist(control, bins=bins, alpha=0.6, label="Control variate", color="seagreen")
ax.axvline(truth, color="red", linestyle="--", label=f"Exact = {truth:.4f}")
ax.set_xlabel("Estimate of the integral")
ax.set_title("Variance reduction: distribution of 5000 estimates")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "variance_reduction.png", dpi=150)

# Plot 5: importance sampling
fig, ax = plt.subplots(figsize=(7, 5))
ax.boxplot([naive, imp], tick_labels=["Naive Monte Carlo", "Importance sampling"])
ax.axhline(p_true, color="red", linestyle="--", label=f"Exact = {p_true:.5f}")
ax.set_ylabel("Estimate of P(Z > 3)")
ax.set_title("Importance sampling for a rare event")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "importance_sampling.png", dpi=150)

# Plot 6: birthday problem
fig, ax = plt.subplots(figsize=(7.5, 5))
kk = np.arange(1, 61)
ax.plot(kk, [birthday_exact(i) for i in kk], "r-", lw=2, label="Exact")
ax.plot([b[0] for b in bday], [b[1] for b in bday], "bo", label="Simulated")
ax.axhline(0.5, color="grey", linestyle=":")
ax.set_xlabel("Number of people")
ax.set_ylabel("P(at least two share a birthday)")
ax.set_title("Birthday problem")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "birthday_problem.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)