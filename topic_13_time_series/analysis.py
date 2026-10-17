"""
Topic 13: Time-Series Analysis
Libraries: pandas, numpy, statsmodels, scipy, matplotlib
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# ---------------------------------------------------------------
# 1. Dataset (synthetic monthly sales: trend + seasonality + AR(1) noise)
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
n = 120
idx = pd.date_range("2015-01-01", periods=n, freq="MS")
t = np.arange(n)
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
season_pattern = np.array([-30, -35, -15, -5, 5, 10, 15, 10, -5, 5, 20, 25],
                          dtype=float)  # sums to zero
eps = rng.normal(0, 7, n)
noise = np.zeros(n)
noise[0] = eps[0]
for i in range(1, n):
    noise[i] = 0.5 * noise[i - 1] + eps[i]
y = pd.Series((200 + 2.0 * t + season_pattern[t % 12] + noise).round(1),
              index=idx, name="sales")
y.rename_axis("date").reset_index().to_csv(DATA / "monthly_sales.csv", index=False)

print("=== Dataset ===")
print(f"{n} monthly observations from {idx[0]:%b %Y} to {idx[-1]:%b %Y}")
print(f"Mean = {y.mean():.2f}, sd = {y.std():.2f}, min = {y.min():.1f}, "
      f"max = {y.max():.1f}")

# ---------------------------------------------------------------
# 2. Yearly summary and linear trend
# ---------------------------------------------------------------
yearly = y.groupby(y.index.year).agg(["mean", "std", "min", "max"])
yearly["growth_pct"] = yearly["mean"].pct_change() * 100
print("\n=== Yearly summary ===")
print(yearly.round(2))
yearly.round(3).to_csv(OUT / "yearly_summary.csv")

slope, intercept = np.polyfit(t, y.to_numpy(), 1)
r2_trend = np.corrcoef(t, y.to_numpy())[0, 1] ** 2
print("\n=== Linear trend: sales = a + b * month ===")
print(f"Intercept = {intercept:.2f}, slope = {slope:.3f} per month "
      f"({slope * 12:.1f} per year), R-squared = {r2_trend:.4f} "
      f"(true slope = 2.000)")

# ---------------------------------------------------------------
# 3. Classical additive decomposition
# ---------------------------------------------------------------
dec = seasonal_decompose(y, model="additive", period=12)
resid_d = dec.resid.dropna()
var_r = resid_d.var()
Ft = max(0, 1 - var_r / (dec.trend + dec.resid).dropna().var())
Fs = max(0, 1 - var_r / (dec.seasonal + dec.resid).dropna().var())
print("\n=== Additive decomposition: observed = trend + seasonal + residual ===")
print(f"Strength of trend = {Ft:.3f}, strength of seasonality = {Fs:.3f} "
      f"(1 = very strong)")
print(f"Residual: mean = {resid_d.mean():.3f}, sd = {resid_d.std():.3f}")
seas_tab = pd.DataFrame({
    "month": months, "true_effect": season_pattern,
    "estimated_effect": dec.seasonal.iloc[:12].to_numpy(),
}).round(2)
print("\nSeasonal effects (true against estimated):")
print(seas_tab.to_string(index=False))
seas_tab.to_csv(OUT / "seasonal_effects.csv", index=False)

# ---------------------------------------------------------------
# 4. Stationarity (Augmented Dickey-Fuller; H0: unit root, non-stationary)
# ---------------------------------------------------------------
series_versions = {
    "Original": y,
    "First difference": y.diff(),
    "Seasonal difference (lag 12)": y.diff(12),
    "Seasonal + first difference": y.diff(12).diff(),
}
adf_rows = []
print("\n=== Augmented Dickey-Fuller tests (reject H0 if p < 0.05) ===")
for name, s in series_versions.items():
    stat, p, lags, nobs, crit, _ = adfuller(s.dropna(), autolag="AIC")
    adf_rows.append((name, stat, p, lags, crit["5%"],
                     "Stationary" if p < 0.05 else "Non-stationary"))
    print(f"{name:30s}: ADF = {stat:7.3f}, p = {p:.4f}, lags used = {lags}, "
          f"5% critical = {crit['5%']:.3f} -> "
          f"{'stationary' if p < 0.05 else 'non-stationary'}")
pd.DataFrame(adf_rows, columns=["series", "adf_statistic", "p_value", "lags_used",
                                "critical_5pct", "conclusion"]
             ).round(4).to_csv(OUT / "adf_tests.csv", index=False)

# ---------------------------------------------------------------
# 5. Autocorrelation
# ---------------------------------------------------------------
lag_list = [1, 2, 3, 6, 12, 24]
acf_orig = acf(y, nlags=24)
acf_diff = acf(y.diff(12).diff().dropna(), nlags=24)
bound = 1.96 / np.sqrt(n)
print("\n=== Autocorrelation at selected lags ===")
print(f"Approximate 95% bound for white noise: +/-{bound:.3f}")
acf_tab = pd.DataFrame({"lag": lag_list,
                        "original": acf_orig[lag_list],
                        "seasonal+first difference": acf_diff[lag_list]}).round(3)
print(acf_tab.to_string(index=False))
acf_tab.to_csv(OUT / "acf_values.csv", index=False)

lb = acorr_ljungbox(resid_d, lags=[12])
print(f"\nLjung-Box test on the decomposition residuals (lag 12): "
      f"Q = {lb['lb_stat'].iloc[0]:.2f}, p = {lb['lb_pvalue'].iloc[0]:.4f}")

# ---------------------------------------------------------------
# 6. Forecasting: hold out the last 12 months
# ---------------------------------------------------------------
H = 12
train, test = y.iloc[:-H], y.iloc[-H:]
actual = test.to_numpy()
fc = {
    "Naive": np.repeat(train.iloc[-1], H),
    "Seasonal naive": train.iloc[-H:].to_numpy(),
}
hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12,
                          initialization_method="estimated").fit()
fc["Holt-Winters"] = hw.forecast(H).to_numpy()
sar = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
sar_fc = sar.get_forecast(H)
fc["SARIMA(1,1,1)(0,1,1)12"] = sar_fc.predicted_mean.to_numpy()
ci = sar_fc.conf_int(alpha=0.05).to_numpy()


def metrics(a, p):
    e = a - p
    return np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e / a)) * 100


print(f"\n=== Forecast accuracy on the last {H} months "
      f"(trained on {train.index[0]:%b %Y} to {train.index[-1]:%b %Y}) ===")
acc_rows = []
for name, p in fc.items():
    mae, rmse, mape = metrics(actual, p)
    acc_rows.append((name, mae, rmse, mape))
    print(f"{name:24s}: MAE = {mae:6.2f}, RMSE = {rmse:6.2f}, MAPE = {mape:5.2f}%")
acc = pd.DataFrame(acc_rows, columns=["method", "MAE", "RMSE", "MAPE_pct"])
acc.round(3).to_csv(OUT / "forecast_accuracy.csv", index=False)
best = acc.loc[acc["RMSE"].idxmin(), "method"]
print(f"Best method by RMSE: {best}")

print("\n=== Holt-Winters parameters ===")
print(f"Level smoothing (alpha) = {hw.params['smoothing_level']:.4f}, "
      f"trend smoothing (beta) = {hw.params['smoothing_trend']:.4f}, "
      f"seasonal smoothing (gamma) = {hw.params['smoothing_seasonal']:.4f}, "
      f"AIC = {hw.aic:.2f}")

print("\n=== SARIMA(1,1,1)(0,1,1)12 coefficients ===")
coef = pd.DataFrame({"coef": sar.params, "std_err": sar.bse, "p_value": sar.pvalues})
print(coef.round(4))
print(f"AIC = {sar.aic:.2f}, BIC = {sar.bic:.2f}")
coef.round(4).to_csv(OUT / "sarima_coefficients.csv")
covered = np.mean((actual >= ci[:, 0]) & (actual <= ci[:, 1]))
print(f"95% prediction interval covered {covered:.0%} of the {H} held-out months "
      f"(average width {np.mean(ci[:, 1] - ci[:, 0]):.1f})")

pd.DataFrame({"actual": actual, **{k: v for k, v in fc.items()},
              "sarima_lower": ci[:, 0], "sarima_upper": ci[:, 1]},
             index=test.index).round(2).rename_axis("date").to_csv(
    OUT / "holdout_forecasts.csv")

# ---------------------------------------------------------------
# 7. Residual diagnostics for the SARIMA model
# ---------------------------------------------------------------
res_s = sar.resid.iloc[13:]  # drop start-up residuals created by differencing
lb2 = acorr_ljungbox(res_s, lags=[12, 24])
sw = stats.shapiro(res_s)
print("\n=== SARIMA residual diagnostics ===")
print(f"Residual mean = {res_s.mean():.3f}, sd = {res_s.std():.3f}")
print(f"Ljung-Box lag 12: Q = {lb2['lb_stat'].iloc[0]:.2f}, "
      f"p = {lb2['lb_pvalue'].iloc[0]:.4f}")
print(f"Ljung-Box lag 24: Q = {lb2['lb_stat'].iloc[1]:.2f}, "
      f"p = {lb2['lb_pvalue'].iloc[1]:.4f}")
print(f"Shapiro-Wilk normality: W = {sw.statistic:.4f}, p = {sw.pvalue:.4f}")

# ---------------------------------------------------------------
# 8. Final forecast for the next 12 months (model refitted on all data)
# ---------------------------------------------------------------
final = SARIMAX(y, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
ff = final.get_forecast(H)
f_mean = ff.predicted_mean
f_ci = ff.conf_int(alpha=0.05)
future = pd.DataFrame({"forecast": f_mean.to_numpy(),
                       "lower_95": f_ci.iloc[:, 0].to_numpy(),
                       "upper_95": f_ci.iloc[:, 1].to_numpy()},
                      index=f_mean.index)
print("\n=== Forecast for the next 12 months (SARIMA fitted on all 120 months) ===")
print(future.round(1).rename_axis("month").to_string())
print(f"Forecast total for {future.index[0].year}: {future['forecast'].sum():.0f} "
      f"(actual total for {y.index[-1].year}: {y.iloc[-12:].sum():.0f})")
future.round(2).rename_axis("date").to_csv(OUT / "future_forecast.csv")

# ---------------------------------------------------------------
# 9. Plots
# ---------------------------------------------------------------
# Plot 1: series with moving average and trend
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(idx, y.to_numpy(), color="steelblue", label="Monthly sales")
ax.plot(idx, y.rolling(12, center=True).mean().to_numpy(), color="red", lw=2,
        label="12-month moving average")
ax.plot(idx, intercept + slope * t, "k--", label=f"Linear trend ({slope:.2f} per month)")
ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.set_title("Monthly sales: trend and seasonality")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "time_series_plot.png", dpi=150)

# Plot 2: decomposition
fig = dec.plot()
fig.set_size_inches(11, 8)
fig.tight_layout()
fig.savefig(OUT / "decomposition.png", dpi=150)

# Plot 3: seasonal plot (one line per year)
pivot = pd.DataFrame({"year": idx.year, "month": idx.month,
                      "sales": y.to_numpy()}).pivot(index="month", columns="year",
                                                    values="sales")
fig, ax = plt.subplots(figsize=(9, 5.5))
cmap = plt.get_cmap("viridis")
for k, year in enumerate(pivot.columns):
    ax.plot(pivot.index, pivot[year].to_numpy(), color=cmap(k / (len(pivot.columns) - 1)),
            label=str(year))
ax.set_xticks(range(1, 13))
ax.set_xticklabels(months)
ax.set_ylabel("Sales")
ax.set_title("Seasonal plot: each line is one year")
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "seasonal_plot.png", dpi=150)

# Plot 4: ACF and PACF before and after differencing
d12 = y.diff(12).diff().dropna()
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plot_acf(y, lags=36, ax=axes[0, 0], title="ACF: original series")
plot_pacf(y, lags=24, ax=axes[0, 1], method="ywm", title="PACF: original series")
plot_acf(d12, lags=36, ax=axes[1, 0], title="ACF: seasonal + first difference")
plot_pacf(d12, lags=24, ax=axes[1, 1], method="ywm",
          title="PACF: seasonal + first difference")
fig.tight_layout()
fig.savefig(OUT / "acf_pacf.png", dpi=150)

# Plot 5: hold-out forecast comparison
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(train.index[-36:], train.iloc[-36:].to_numpy(), color="grey", label="Training data")
ax.plot(test.index, actual, "k-o", ms=4, lw=2, label="Actual")
colours = {"Naive": "tab:orange", "Seasonal naive": "tab:green",
           "Holt-Winters": "tab:purple", "SARIMA(1,1,1)(0,1,1)12": "tab:red"}
for name, p in fc.items():
    ax.plot(test.index, p, "--", color=colours[name], label=name)
ax.fill_between(test.index, ci[:, 0], ci[:, 1], color="tab:red", alpha=0.12,
                label="SARIMA 95% interval")
ax.set_ylabel("Sales")
ax.set_title("Forecasts for the held-out final 12 months")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "forecast_comparison.png", dpi=150)

# Plot 6: final forecast
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(y.index[-48:], y.iloc[-48:].to_numpy(), color="steelblue", label="Observed")
ax.plot(future.index, future["forecast"].to_numpy(), "r-o", ms=4, label="Forecast")
ax.fill_between(future.index, future["lower_95"].to_numpy(),
                future["upper_95"].to_numpy(), color="red", alpha=0.2,
                label="95% prediction interval")
ax.set_ylabel("Sales")
ax.set_title("SARIMA forecast for the next 12 months")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "future_forecast.png", dpi=150)

# Plot 7: residual diagnostics
fig = sar.plot_diagnostics(figsize=(11, 8))
fig.tight_layout()
fig.savefig(OUT / "sarima_residual_diagnostics.png", dpi=150)

plt.close("all")
print("\nSaved outputs to:", OUT)