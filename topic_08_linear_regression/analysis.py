"""
Topic 8: Linear Regression and Least Squares
Libraries: statsmodels, scikit-learn, numpy, pandas, scipy, matplotlib
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# ---------------------------------------------------------------
# 1. Dataset (same synthetic students as Topic 7)
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

predictors = ["study_hours", "sleep_hours", "attendance_pct", "screen_hours"]
y = df["exam_score"].to_numpy()

print("=== Dataset ===")
print(f"{n} students, response = exam_score, predictors = {', '.join(predictors)}")

# ---------------------------------------------------------------
# 2. Simple linear regression: least squares by hand
# ---------------------------------------------------------------
x = df["study_hours"].to_numpy()
b1 = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
b0 = y.mean() - b1 * x.mean()

simple = smf.ols("exam_score ~ study_hours", data=df).fit()
print("\n=== Simple regression: exam_score ~ study_hours ===")
print(f"By hand:     intercept = {b0:.4f}, slope = {b1:.4f}")
print(f"statsmodels: intercept = {simple.params['Intercept']:.4f}, "
      f"slope = {simple.params['study_hours']:.4f}")
print(f"Equation: exam_score = {b0:.3f} + {b1:.3f} * study_hours")
print(f"R-squared = {simple.rsquared:.4f}, slope p-value = "
      f"{simple.pvalues['study_hours']:.4g}, residual SE = "
      f"{np.sqrt(simple.mse_resid):.3f}")

# ---------------------------------------------------------------
# 3. Multiple linear regression
# ---------------------------------------------------------------
model = smf.ols("exam_score ~ " + " + ".join(predictors), data=df).fit()
ci = model.conf_int()
coef = pd.DataFrame({
    "coef": model.params, "std_err": model.bse, "t": model.tvalues,
    "p_value": model.pvalues, "ci_lower": ci[0], "ci_upper": ci[1],
})
print("\n=== Multiple regression coefficients ===")
print(coef.round(4))
coef.round(4).to_csv(OUT / "regression_coefficients.csv")

terms = " ".join(f"{model.params[p]:+.3f}*{p}" for p in predictors)
print(f"\nEquation: exam_score = {model.params['Intercept']:.3f} {terms}")

print("\n=== Model fit ===")
print(f"R-squared          = {model.rsquared:.4f}")
print(f"Adjusted R-squared = {model.rsquared_adj:.4f}")
print(f"Residual std error = {np.sqrt(model.mse_resid):.3f} on "
      f"{int(model.df_resid)} degrees of freedom")
print(f"F-statistic        = {model.fvalue:.2f}, p-value = {model.f_pvalue:.4g}")
print(f"AIC = {model.aic:.2f}, BIC = {model.bic:.2f}")

# Check: matrix formula beta = (X'X)^-1 X'y
X = model.model.exog
beta_matrix = np.linalg.solve(X.T @ X, X.T @ y)
print("\nMatrix formula (X'X)^-1 X'y matches statsmodels:",
      np.allclose(beta_matrix, model.params.to_numpy()))

# Does the extra predictors improve on the simple model? (partial F-test)
cmp = anova_lm(simple, model)
f_part, p_part = cmp["F"].iloc[1], cmp["Pr(>F)"].iloc[1]
print(f"\n=== Simple vs multiple model (partial F-test) ===")
print(f"Simple R-squared = {simple.rsquared:.4f}, multiple R-squared = "
      f"{model.rsquared:.4f}")
print(f"F = {f_part:.3f}, p-value = {p_part:.4g}")

# Standardised coefficients
sd_x = df[predictors].std(ddof=1)
sd_y = df["exam_score"].std(ddof=1)
std_beta = model.params[predictors] * sd_x / sd_y
print("\n=== Standardised coefficients ===")
print(std_beta.round(4))

# ---------------------------------------------------------------
# 4. Diagnostics
# ---------------------------------------------------------------
resid = model.resid.to_numpy()
fitted = model.fittedvalues.to_numpy()

vif = pd.Series(
    [variance_inflation_factor(X, i) for i in range(1, X.shape[1])],
    index=predictors,
)
sw = stats.shapiro(resid)
bp = het_breuschpagan(resid, X)
dw = durbin_watson(resid)
cooks = model.get_influence().cooks_distance[0]

print("\n=== Variance inflation factors ===")
print(vif.round(3))
print("\n=== Residual diagnostics ===")
print(f"Residual mean = {resid.mean():.6f}, residual sd = {resid.std(ddof=1):.3f}")
print(f"Shapiro-Wilk (normality):          W = {sw.statistic:.4f}, p = {sw.pvalue:.4f}")
print(f"Breusch-Pagan (constant variance): LM = {bp[0]:.3f}, p = {bp[1]:.4f}")
print(f"Durbin-Watson (autocorrelation):   {dw:.3f}")
print(f"Largest Cook's distance = {cooks.max():.4f} "
      f"({int((cooks > 4 / n).sum())} points above 4/n = {4 / n:.3f})")

# ---------------------------------------------------------------
# 5. Prediction performance (scikit-learn)
# ---------------------------------------------------------------
X_df, y_s = df[predictors], df["exam_score"]
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_s, test_size=0.25, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
rmse_train = np.sqrt(mean_squared_error(y_train, lr.predict(X_train)))
rmse_test = np.sqrt(mean_squared_error(y_test, lr.predict(X_test)))
r2_train = r2_score(y_train, lr.predict(X_train))
r2_test = r2_score(y_test, lr.predict(X_test))
cv = cross_val_score(LinearRegression(), X_df, y_s,
                     cv=KFold(5, shuffle=True, random_state=42), scoring="r2")

print("\n=== Prediction performance (75% train / 25% test) ===")
print(f"Train: R-squared = {r2_train:.4f}, RMSE = {rmse_train:.3f}")
print(f"Test:  R-squared = {r2_test:.4f}, RMSE = {rmse_test:.3f}")
print(f"5-fold cross-validated R-squared: mean = {cv.mean():.4f}, "
      f"sd = {cv.std(ddof=1):.4f}")

# ---------------------------------------------------------------
# 6. Prediction for a new student
# ---------------------------------------------------------------
new = pd.DataFrame({"study_hours": [6.0], "sleep_hours": [7.0],
                    "attendance_pct": [85.0], "screen_hours": [3.0]})
pf = model.get_prediction(new).summary_frame(alpha=0.05)
print("\n=== Prediction for a new student (6 h study, 7 h sleep, 85% attendance, "
      "3 h screen) ===")
print(f"Predicted score = {pf['mean'].iloc[0]:.2f}")
print(f"95% confidence interval for the mean score: "
      f"({pf['mean_ci_lower'].iloc[0]:.2f}, {pf['mean_ci_upper'].iloc[0]:.2f})")
print(f"95% prediction interval for one student:    "
      f"({pf['obs_ci_lower'].iloc[0]:.2f}, {pf['obs_ci_upper'].iloc[0]:.2f})")

# ---------------------------------------------------------------
# 7. Save summary metrics
# ---------------------------------------------------------------
pd.DataFrame({
    "metric": ["simple_R2", "multiple_R2", "multiple_adj_R2", "residual_SE", "F_stat",
               "F_p_value", "test_R2", "test_RMSE", "cv_R2_mean"],
    "value": [simple.rsquared, model.rsquared, model.rsquared_adj,
              np.sqrt(model.mse_resid), model.fvalue, model.f_pvalue,
              r2_test, rmse_test, cv.mean()],
}).round(4).to_csv(OUT / "model_metrics.csv", index=False)

# ---------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------
# Plot 1: fitted regression line with confidence and prediction bands
xs = np.linspace(x.min(), x.max(), 100)
band = simple.get_prediction(pd.DataFrame({"study_hours": xs})).summary_frame(0.05)
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(x, y, alpha=0.5, color="steelblue", label="Students")
ax.plot(xs, band["mean"], "r-", lw=2,
        label=f"Fitted: {b0:.1f} + {b1:.2f} x (R2 = {simple.rsquared:.2f})")
ax.fill_between(xs, band["mean_ci_lower"], band["mean_ci_upper"], color="red",
                alpha=0.3, label="95% confidence band")
ax.fill_between(xs, band["obs_ci_lower"], band["obs_ci_upper"], color="grey",
                alpha=0.15, label="95% prediction band")
ax.set_xlabel("Study hours per day")
ax.set_ylabel("Exam score")
ax.set_title("Simple linear regression: exam score on study hours")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fitted_regression_line.png", dpi=150)

# Plot 2: diagnostic panel
std_resid = model.get_influence().resid_studentized_internal
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
ax = axes[0, 0]
ax.scatter(fitted, resid, alpha=0.6, color="steelblue")
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs fitted")
stats.probplot(resid, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title("Normal Q-Q plot of residuals")
ax = axes[1, 0]
ax.scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.6, color="steelblue")
ax.set_xlabel("Fitted values")
ax.set_ylabel("sqrt(|standardised residual|)")
ax.set_title("Scale-location")
ax = axes[1, 1]
ax.hist(resid, bins=20, density=True, color="steelblue", alpha=0.7)
grid = np.linspace(resid.min(), resid.max(), 200)
ax.plot(grid, stats.norm.pdf(grid, 0, resid.std(ddof=1)), "r-", lw=2)
ax.set_xlabel("Residual")
ax.set_title("Histogram of residuals")
fig.suptitle("Residual diagnostics for the multiple regression", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "residual_diagnostics.png", dpi=150)

# Plot 3: actual vs predicted on the test set
pred_test = lr.predict(X_test)
lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(y_test, pred_test, alpha=0.7, color="steelblue")
ax.plot(lims, lims, "r--", label="Perfect prediction")
ax.set_xlabel("Actual exam score")
ax.set_ylabel("Predicted exam score")
ax.set_title(f"Test set: R2 = {r2_test:.3f}, RMSE = {rmse_test:.2f}")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "actual_vs_predicted.png", dpi=150)

# Plot 4: standardised coefficients
fig, ax = plt.subplots(figsize=(8, 4.5))
order = std_beta.sort_values()
ax.barh(order.index, order.values,
        color=["tomato" if v < 0 else "seagreen" for v in order.values])
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Standardised coefficient")
ax.set_title("Relative importance of the predictors")
fig.tight_layout()
fig.savefig(OUT / "standardised_coefficients.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)