"""
Topic 2: Probability and Conditional Probability
Libraries: numpy, pandas, scipy, matplotlib
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# 1. Dataset (synthetic customers, reproducible)
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
n = 1000
plan = rng.choice(["Basic", "Premium"], size=n, p=[0.6, 0.4])
# customers who made a support call are more likely to churn
support_call = rng.random(n) < np.where(plan == "Basic", 0.35, 0.20)
p_churn = np.where(support_call, 0.40, 0.10)
churned = rng.random(n) < p_churn

df = pd.DataFrame({
    "customer_id": range(1, n + 1),
    "plan": plan,
    "support_call": support_call,
    "churned": churned,
})
df.to_csv(DATA / "customers.csv", index=False)

# ---------------------------------------------------------------
# 2. Basic probabilities
# ---------------------------------------------------------------
p_churn_all = df["churned"].mean()
p_call = df["support_call"].mean()
p_premium = (df["plan"] == "Premium").mean()
p_call_and_churn = (df["support_call"] & df["churned"]).mean()

print("=== Basic probabilities ===")
print(f"P(Churn)             = {p_churn_all:.4f}")
print(f"P(Support call)      = {p_call:.4f}")
print(f"P(Premium)           = {p_premium:.4f}")
print(f"P(Call AND Churn)    = {p_call_and_churn:.4f}")

# ---------------------------------------------------------------
# 3. Conditional probabilities
# ---------------------------------------------------------------
p_churn_given_call = p_call_and_churn / p_call
p_churn_given_nocall = (
    (~df["support_call"] & df["churned"]).mean() / (1 - p_call)
)
p_churn_given_premium = df[df["plan"] == "Premium"]["churned"].mean()
p_churn_given_basic = df[df["plan"] == "Basic"]["churned"].mean()

print("\n=== Conditional probabilities ===")
print(f"P(Churn | Call)      = {p_churn_given_call:.4f}")
print(f"P(Churn | No call)   = {p_churn_given_nocall:.4f}")
print(f"P(Churn | Premium)   = {p_churn_given_premium:.4f}")
print(f"P(Churn | Basic)     = {p_churn_given_basic:.4f}")

# ---------------------------------------------------------------
# 4. Independence check: is P(Churn|Call) == P(Churn)?
# ---------------------------------------------------------------
print("\n=== Independence check ===")
print(f"P(Churn) = {p_churn_all:.4f} vs P(Churn | Call) = {p_churn_given_call:.4f}")
print("Events are", "independent" if np.isclose(p_churn_all, p_churn_given_call, atol=0.01)
      else "NOT independent")

# ---------------------------------------------------------------
# 5. Bayes' theorem: P(Call | Churn)
# ---------------------------------------------------------------
p_call_given_churn_bayes = p_churn_given_call * p_call / p_churn_all
p_call_given_churn_direct = df[df["churned"]]["support_call"].mean()

print("\n=== Bayes' theorem ===")
print(f"P(Call | Churn) via Bayes  = {p_call_given_churn_bayes:.4f}")
print(f"P(Call | Churn) direct     = {p_call_given_churn_direct:.4f}")

# ---------------------------------------------------------------
# 6. Simulation: estimate P(Churn | Call) by repeated sampling
# ---------------------------------------------------------------
sim_rng = np.random.default_rng(1)
calls = df[df["support_call"]]["churned"].to_numpy()
n_sims = 5000
draws = sim_rng.choice(calls, size=n_sims, replace=True)
running_estimate = np.cumsum(draws) / np.arange(1, n_sims + 1)

print("\n=== Simulation ===")
print(f"Simulated P(Churn | Call) after {n_sims} draws = {running_estimate[-1]:.4f}")
print(f"Theoretical (from data)                       = {p_churn_given_call:.4f}")

# ---------------------------------------------------------------
# 7. Save table and plots
# ---------------------------------------------------------------
results = pd.DataFrame({
    "probability": [
        "P(Churn)", "P(Support call)", "P(Premium)", "P(Call AND Churn)",
        "P(Churn | Call)", "P(Churn | No call)", "P(Churn | Premium)",
        "P(Churn | Basic)", "P(Call | Churn) Bayes", "P(Call | Churn) direct",
        "Simulated P(Churn | Call)",
    ],
    "value": [
        p_churn_all, p_call, p_premium, p_call_and_churn,
        p_churn_given_call, p_churn_given_nocall, p_churn_given_premium,
        p_churn_given_basic, p_call_given_churn_bayes,
        p_call_given_churn_direct, running_estimate[-1],
    ],
}).round(4)
results.to_csv(OUT / "probabilities.csv", index=False)

# Plot 1: conditional churn probabilities
fig, ax = plt.subplots(figsize=(8, 5))
labels = ["Overall", "Call", "No call", "Premium", "Basic"]
vals = [p_churn_all, p_churn_given_call, p_churn_given_nocall,
        p_churn_given_premium, p_churn_given_basic]
bars = ax.bar(labels, vals, color=["grey", "tomato", "seagreen", "steelblue", "orange"])
ax.bar_label(bars, fmt="%.3f")
ax.set_ylabel("Probability of churn")
ax.set_title("Churn probability under different conditions")
fig.tight_layout()
fig.savefig(OUT / "conditional_probabilities.png", dpi=150)

# Plot 2: simulation convergence
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(running_estimate, label="Simulated estimate")
ax.axhline(p_churn_given_call, color="red", linestyle="--", label="Value from data")
ax.set_xlabel("Number of draws")
ax.set_ylabel("Estimated P(Churn | Call)")
ax.set_title("Simulation convergence")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "simulation_convergence.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)