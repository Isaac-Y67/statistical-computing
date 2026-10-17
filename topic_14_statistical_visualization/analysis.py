"""
Topic 14: Statistical Data Visualization
Libraries: matplotlib, seaborn, plotly, pandas, numpy, scipy
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy import stats

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------
# 1. Datasets (synthetic, reproducible)
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
n = 400
depts = ["Sales", "Engineering", "Marketing", "Support"]
base_salary = {"Sales": 38000, "Engineering": 52000, "Marketing": 42000,
               "Support": 33000}
department = rng.choice(depts, size=n, p=[0.3, 0.3, 0.2, 0.2])
experience = rng.gamma(3.0, 2.5, n).clip(0, 30).round(1)
salary = (np.array([base_salary[d] for d in department])
          * np.exp(0.035 * experience + rng.normal(0, 0.12, n)))
performance = (55 + 1.2 * experience + rng.normal(0, 10, n)).clip(0, 100)
remote = rng.choice(["Yes", "No"], size=n, p=[0.4, 0.6])
levels = np.arange(1, 6)
sat_remote = rng.choice(levels, size=n, p=[0.03, 0.07, 0.20, 0.40, 0.30])
sat_office = rng.choice(levels, size=n, p=[0.07, 0.13, 0.30, 0.30, 0.20])
satisfaction = np.where(remote == "Yes", sat_remote, sat_office)

df = pd.DataFrame({
    "department": department,
    "experience_years": experience,
    "salary": (np.round(salary / 100) * 100),
    "performance_score": performance.round(1),
    "remote": remote,
    "satisfaction": satisfaction,
})
df["log_salary"] = np.log(df["salary"])
df.drop(columns="log_salary").to_csv(DATA / "employees.csv", index=False)

months = pd.date_range("2022-01-01", periods=36, freq="MS")
t = np.arange(36)
region_base = {"North": 120, "South": 100, "East": 90, "West": 110}
region_phase = {"North": 0.0, "South": 1.5, "East": 3.0, "West": 4.5}
sales_rows = []
for r in region_base:
    vals = (region_base[r] + 1.5 * t
            + 12 * np.sin(2 * np.pi * t / 12 + region_phase[r])
            + rng.normal(0, 4, 36))
    sales_rows.append(pd.DataFrame({"month": months, "region": r,
                                    "sales": vals.round(1)}))
sales = pd.concat(sales_rows, ignore_index=True)
sales.to_csv(DATA / "regional_sales.csv", index=False)

print("=== Datasets ===")
print(f"Employees: {n} rows, columns: {', '.join(df.columns[:-1])}")
print(f"Regional sales: {len(sales)} rows (4 regions x 36 months)")
print("\nCounts by department:")
print(df["department"].value_counts().to_string())

# ---------------------------------------------------------------
# 2. Distribution of salary
# ---------------------------------------------------------------
print("\n=== Salary distribution ===")
print(df[["experience_years", "salary", "performance_score"]].describe().round(2))
skew_raw = stats.skew(df["salary"])
skew_log = stats.skew(df["log_salary"])
sw_raw = stats.shapiro(df["salary"])
sw_log = stats.shapiro(df["log_salary"])
print(f"\nSalary: mean = {df['salary'].mean():.0f}, median = {df['salary'].median():.0f}, "
      f"skewness = {skew_raw:.3f}, Shapiro p = {sw_raw.pvalue:.4g}")
print(f"log(Salary): skewness = {skew_log:.3f}, Shapiro p = {sw_log.pvalue:.4f}")

# ---------------------------------------------------------------
# 3. Salary by department
# ---------------------------------------------------------------
by_dept = df.groupby("department")["salary"].agg(["count", "mean", "median", "std"])
by_dept = by_dept.loc[depts]
tcrit = stats.t.ppf(0.975, by_dept["count"] - 1)
by_dept["ci_half_width"] = tcrit * by_dept["std"] / np.sqrt(by_dept["count"])
print("\n=== Salary by department ===")
print(by_dept.round(0).to_string())
by_dept.round(1).to_csv(OUT / "salary_by_department.csv")
kw = stats.kruskal(*[df.loc[df["department"] == d, "salary"] for d in depts])
print(f"Kruskal-Wallis test of salary across departments: H = {kw.statistic:.2f}, "
      f"p = {kw.pvalue:.4g}")

# ---------------------------------------------------------------
# 4. Relationships
# ---------------------------------------------------------------
corr_cols = ["experience_years", "salary", "log_salary", "performance_score"]
corr = df[corr_cols].corr()
print("\n=== Correlation matrix ===")
print(corr.round(3))
corr.round(4).to_csv(OUT / "correlation_matrix.csv")
b, a = np.polyfit(df["experience_years"], df["log_salary"], 1)
pct_per_year = (np.exp(b) - 1) * 100
print(f"\nlog(salary) = {a:.3f} + {b:.4f} * experience  -> each extra year of "
      f"experience is associated with a {pct_per_year:.1f}% higher salary")

# ---------------------------------------------------------------
# 5. Line graph statistics
# ---------------------------------------------------------------
print("\n=== Regional sales: average of first 12 months against last 12 months ===")
line_rows = []
for r in region_base:
    s = sales.loc[sales["region"] == r, "sales"].to_numpy()
    first, last = s[:12].mean(), s[-12:].mean()
    line_rows.append((r, first, last, (last / first - 1) * 100))
    print(f"{r:6s}: first year = {first:6.1f}, last year = {last:6.1f}, "
          f"growth = {(last / first - 1) * 100:5.1f}%")
pd.DataFrame(line_rows, columns=["region", "first_year_mean", "last_year_mean",
                                 "growth_pct"]).round(2).to_csv(
    OUT / "regional_growth.csv", index=False)

# ---------------------------------------------------------------
# 6. Categorical data: satisfaction by work mode
# ---------------------------------------------------------------
prop = pd.crosstab(df["remote"], df["satisfaction"], normalize="index")
print("\n=== Satisfaction (rows sum to 1) by remote work ===")
print(prop.round(3))
means = df.groupby("remote")["satisfaction"].mean()
mw = stats.mannwhitneyu(df.loc[df["remote"] == "Yes", "satisfaction"],
                        df.loc[df["remote"] == "No", "satisfaction"])
print(f"Mean satisfaction: remote = {means['Yes']:.3f}, office = {means['No']:.3f}; "
      f"Mann-Whitney p = {mw.pvalue:.4f}")
prop.round(4).to_csv(OUT / "satisfaction_by_remote.csv")

# ---------------------------------------------------------------
# 7. How a truncated axis misleads
# ---------------------------------------------------------------
y0 = 30000
hi, lo = by_dept["mean"].max(), by_dept["mean"].min()
print("\n=== Misleading axis demonstration ===")
print(f"Highest mean salary = {hi:.0f}, lowest = {lo:.0f}; true ratio = {hi / lo:.2f}")
print(f"On a bar chart whose axis starts at {y0}, the bars would look "
      f"{(hi - y0) / (lo - y0):.2f} times as tall")

# ---------------------------------------------------------------
# 8. Matplotlib / seaborn plots
# ---------------------------------------------------------------
# Plot 1: histograms
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(df["salary"], bins=30, kde=True, color="steelblue", ax=axes[0])
axes[0].axvline(df["salary"].mean(), color="red", linestyle="--",
                label=f"Mean = {df['salary'].mean():,.0f}")
axes[0].axvline(df["salary"].median(), color="green",
                label=f"Median = {df['salary'].median():,.0f}")
axes[0].set_title(f"Salary (skewness {skew_raw:.2f})")
axes[0].legend()
sns.histplot(df["log_salary"], bins=30, kde=True, color="seagreen", ax=axes[1])
axes[1].set_title(f"log(Salary) (skewness {skew_log:.2f})")
fig.tight_layout()
fig.savefig(OUT / "histograms.png", dpi=150)

# Plot 2: boxplot and violin
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=df, x="department", y="salary", order=depts, hue="department",
            hue_order=depts, legend=False, ax=axes[0])
axes[0].set_title("Boxplot of salary by department")
sns.violinplot(data=df, x="department", y="salary", order=depts, hue="department",
               hue_order=depts, legend=False, inner="quartile", ax=axes[1])
axes[1].set_title("Violin plot of salary by department")
fig.tight_layout()
fig.savefig(OUT / "boxplot_violin.png", dpi=150)

# Plot 3: scatter with fitted curve
fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(data=df, x="experience_years", y="salary", hue="department",
                hue_order=depts, alpha=0.7, ax=ax)
xs = np.linspace(0, df["experience_years"].max(), 100)
ax.plot(xs, np.exp(a + b * xs), "k--", lw=2,
        label=f"Fitted curve (+{pct_per_year:.1f}% per year)")
ax.set_yscale("log")
ax.set_title("Salary against experience (log scale)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "scatter_experience_salary.png", dpi=150)

# Plot 4: line graph
fig, ax = plt.subplots(figsize=(11, 5.5))
for r, colour in zip(region_base, sns.color_palette("tab10", 4)):
    s = sales[sales["region"] == r].set_index("month")["sales"]
    ax.plot(s.index, s.to_numpy(), color=colour, alpha=0.35)
    ax.plot(s.index, s.rolling(12, center=True).mean().to_numpy(), color=colour,
            lw=2.5, label=f"{r} (12-month average)")
ax.set_ylabel("Sales")
ax.set_title("Monthly sales by region (thin = monthly, thick = 12-month average)")
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig(OUT / "line_graph_sales.png", dpi=150)

# Plot 5: bar charts, honest and misleading
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, truncated in zip(axes, (False, True)):
    ax.bar(depts, by_dept["mean"], yerr=by_dept["ci_half_width"], capsize=6,
           color=sns.color_palette("tab10", 4), alpha=0.85)
    if truncated:
        ax.set_ylim(y0, by_dept["mean"].max() * 1.1)
        ax.set_title(f"Misleading: axis starts at {y0:,}")
    else:
        ax.set_ylim(0, by_dept["mean"].max() * 1.15)
        ax.set_title("Honest: axis starts at 0 (bars show mean and 95% CI)")
    ax.set_ylabel("Mean salary")
fig.tight_layout()
fig.savefig(OUT / "bar_charts_misleading_axis.png", dpi=150)

# Plot 6: correlation heatmap
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlation heatmap")
fig.tight_layout()
fig.savefig(OUT / "correlation_heatmap.png", dpi=150)

# Plot 7: pair plot
pair = sns.pairplot(df, vars=["experience_years", "salary", "performance_score"],
                    hue="department", hue_order=depts, corner=True,
                    plot_kws={"alpha": 0.5, "s": 18})
pair.figure.suptitle("Pair plot by department", y=1.02)
pair.savefig(OUT / "pair_plot.png", dpi=120)

# Plot 8: Q-Q plots and ECDF
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
stats.probplot(df["salary"], dist="norm", plot=axes[0])
axes[0].set_title("Q-Q plot: salary")
stats.probplot(df["log_salary"], dist="norm", plot=axes[1])
axes[1].set_title("Q-Q plot: log(salary)")
sns.ecdfplot(data=df, x="salary", hue="department", hue_order=depts, log_scale=True,
             ax=axes[2])
axes[2].set_title("Empirical CDF of salary by department")
fig.tight_layout()
fig.savefig(OUT / "qq_and_ecdf.png", dpi=150)

# Plot 9: categorical data
fig, ax = plt.subplots(figsize=(8, 5))
width = 0.38
ax.bar(levels - width / 2, prop.loc["Yes"].to_numpy(), width, label="Remote",
       color="steelblue")
ax.bar(levels + width / 2, prop.loc["No"].to_numpy(), width, label="Office",
       color="orange")
ax.set_xlabel("Satisfaction rating (1 = low, 5 = high)")
ax.set_ylabel("Proportion of employees")
ax.set_title("Satisfaction by work mode")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "satisfaction_bars.png", dpi=150)

plt.close("all")

# ---------------------------------------------------------------
# 9. Interactive Plotly charts (open the HTML files in a browser)
# ---------------------------------------------------------------
fig_p = px.scatter(df, x="experience_years", y="salary", color="department",
                   hover_data=["performance_score", "remote"], log_y=True,
                   category_orders={"department": depts},
                   title="Salary against experience (interactive)")
fig_p.write_html(OUT / "interactive_scatter.html", include_plotlyjs="cdn")

fig_b = px.box(df, x="department", y="salary", color="department", points="all",
               category_orders={"department": depts},
               title="Salary by department (interactive)")
fig_b.write_html(OUT / "interactive_boxplot.html", include_plotlyjs="cdn")

fig_l = px.line(sales, x="month", y="sales", color="region",
                title="Monthly sales by region (interactive)")
fig_l.write_html(OUT / "interactive_line.html", include_plotlyjs="cdn")

print("\nSaved outputs to:", OUT)
