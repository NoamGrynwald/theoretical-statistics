import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

# ------------------------------
# Utilities
# ------------------------------

def ensure_dirs():
    Path("outputs/figs").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)

def first_existing(df, cols, fallback=None):
    for c in cols:
        if c in df.columns:
            return c
    return fallback

def map_remote_status(remote_ratio):
    # Expects 0, 50, 100 (int); handles floats safely.
    try:
        v = int(round(float(remote_ratio)))
    except Exception:
        return np.nan
    if v <= 10:
        return "On-site"
    elif 10 < v < 90:
        return "Hybrid"
    else:
        return "Remote"

def map_experience_level(x):
    # Common coding in the DS salaries dataset
    m = {
        "EN": "Entry",
        "MI": "Mid",
        "SE": "Senior",
        "EX": "Executive",
        # Add some common aliases if present
        "JR": "Entry",
        "SR": "Senior"
    }
    if pd.isna(x):
        return np.nan
    x = str(x).strip().upper()
    return m.get(x, x.title())

EXP_ORDER = ["Entry", "Mid", "Senior", "Executive"]
REMOTE_ORDER = ["On-site", "Hybrid", "Remote"]

def ci_mean(x, alpha=0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    m = np.mean(x)
    se = stats.sem(x, nan_policy="omit")
    if n > 1 and se == se:  # not nan
        df = n - 1
        tcrit = stats.t.ppf(1 - alpha/2, df)
        lo, hi = m - tcrit*se, m + tcrit*se
    else:
        lo, hi = np.nan, np.nan
    return (lo, hi)

# ------------------------------
# Data loading and preparation
# ------------------------------

def load_and_prepare(path):
    df = pd.read_csv(path)
    # Identify columns (be defensive with names)
    salary_col = first_existing(df, ["salary_in_usd", "Salary_USD", "salaryUSD", "salary"], None)
    remote_col = first_existing(df, ["remote_ratio", "remoteRatio", "remote", "remote_percent"], None)
    country_col = first_existing(df, ["employee_residence", "company_location", "company_country"], None)
    exp_col = first_existing(df, ["experience_level", "experienceLevel", "seniority"], None)
    size_col = first_existing(df, ["company_size", "companySize"], None)

    required = [salary_col, remote_col, country_col, exp_col]
    missing = [c for c in required if c is None]
    if missing:
        raise ValueError(f"Missing required columns. Could not locate: {missing}\n"
                         f"Found columns: {list(df.columns)}")

    # Basic cleaning
    df = df.copy()
    df = df.rename(columns={
        salary_col: "salary_in_usd",
        remote_col: "remote_ratio",
        country_col: "country",
        exp_col: "experience_level",
        **({size_col: "company_size"} if size_col else {})
    })

    # drop non-positive or missing salaries
    df = df[pd.to_numeric(df["salary_in_usd"], errors="coerce") > 0].copy()
    df["salary_in_usd"] = df["salary_in_usd"].astype(float)

    # remote status categories
    df["remote_status"] = df["remote_ratio"].map(map_remote_status)
    df["remote_status"] = pd.Categorical(df["remote_status"], categories=REMOTE_ORDER, ordered=True)

    # experience level categories
    df["experience_level"] = df["experience_level"].map(map_experience_level)
    df["experience_level"] = pd.Categorical(df["experience_level"], categories=EXP_ORDER, ordered=True)

    # optional company size as categorical (if present)
    if "company_size" in df.columns:
        # Already usually like S/M/L; make categorical ordered S<M<L if possible
        order_guess = ["S", "M", "L"]
        if df["company_size"].dropna().isin(order_guess).mean() > 0.8:
            df["company_size"] = pd.Categorical(df["company_size"], categories=order_guess, ordered=True)
        else:
            df["company_size"] = df["company_size"].astype("category")

    # log salary for regression
    df["log_salary"] = np.log(df["salary_in_usd"].astype(float))
    return df

# ------------------------------
# Part 1: Distribution of remote vs on-site across countries
# ------------------------------

def remote_distribution_by_country(df, min_n=50, topN=20, save_prefix="outputs"):
    # focus on Remote vs On-site (but report Hybrid too)
    sub = df.dropna(subset=["country", "remote_status"]).copy()

    # Compute counts by (country, remote_status)
    counts = (sub
              .groupby(["country", "remote_status"], observed=True)
              .size()
              .rename("count")
              .reset_index())

    # Total per country
    total = counts.groupby("country", observed=True)["count"].sum().rename("total").reset_index()
    counts = counts.merge(total, on="country", how="left")
    counts["pct_row"] = 100 * counts["count"] / counts["total"]

    # Filter by minimum sample size and choose TopN by total N
    big_countries = total[total["total"] >= min_n].sort_values("total", ascending=False)
    if topN is not None and topN > 0:
        big_countries = big_countries.head(topN)
    keep = set(big_countries["country"])
    counts_top = counts[counts["country"].isin(keep)].copy()

    # Pivot for easy viewing (counts and percents)
    pivot_counts = counts_top.pivot(index="country", columns="remote_status", values="count").fillna(0).astype(int)
    pivot_pct = counts_top.pivot(index="country", columns="remote_status", values="pct_row").fillna(0.0)

    # Save tables
    pivot_counts.sort_values(by=["Remote","Hybrid","On-site"], ascending=False, inplace=True, na_position="last")
    pivot_pct = pivot_pct.reindex(index=pivot_counts.index)
    pivot_counts.to_csv(f"{save_prefix}/tables/remote_distribution_by_country_counts.csv")
    pivot_pct.to_csv(f"{save_prefix}/tables/remote_distribution_by_country_pct.csv", float_format="%.2f")

    # Stacked bar (row-percent)
    fig, ax = plt.subplots(figsize=(12, 6))
    idx = np.arange(len(pivot_pct.index))
    bottom = np.zeros(len(idx))
    for status in REMOTE_ORDER:
        if status in pivot_pct.columns:
            vals = pivot_pct[status].values
            ax.bar(idx, vals, bottom=bottom, label=status)
            bottom += vals
    ax.set_xticks(idx)
    ax.set_xticklabels(pivot_pct.index, rotation=45, ha="right")
    ax.set_ylabel("Share of roles (%)")
    ax.set_title(f"Remote vs On-site Distribution by Country (Top {len(pivot_pct)} by N, row %)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(f"{save_prefix}/figs/remote_distribution_by_country_stacked_topN.pdf", bbox_inches="tight")
    plt.close(fig)

    return pivot_counts, pivot_pct

# ------------------------------
# Part 2: Interaction between remote work and experience level
# ------------------------------

def interaction_remote_experience(df, save_prefix="outputs"):
    sub = df.dropna(subset=["remote_status", "experience_level", "salary_in_usd", "log_salary"]).copy()

    # ----- A) Raw group statistics (unadjusted) -----
    grp = (sub
           .groupby(["experience_level", "remote_status"], observed=True)
           .agg(n=("salary_in_usd","size"),
                mean_salary=("salary_in_usd","mean"),
                std_salary=("salary_in_usd","std"))
           .reset_index())
    # 95% CI for means
    lo, hi = [], []
    for _, row in grp.iterrows():
        mask = ((sub["experience_level"] == row["experience_level"]) &
                (sub["remote_status"] == row["remote_status"]))
        ci = ci_mean(sub.loc[mask, "salary_in_usd"].values, alpha=0.05)
        lo.append(ci[0]); hi.append(ci[1])
    grp["ci_lo"] = lo; grp["ci_hi"] = hi
    grp.to_csv(f"{save_prefix}/tables/interaction_group_stats.csv", index=False, float_format="%.2f")

    # Plot raw means with CIs
    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = np.arange(len(EXP_ORDER))
    width = 0.23
    for i, status in enumerate(REMOTE_ORDER):
        data = grp[grp["remote_status"] == status]
        # align bars to category order
        means = []
        err_lo, err_hi = [], []
        for exp in EXP_ORDER:
            r = data[data["experience_level"] == exp]
            if len(r) == 1:
                means.append(float(r["mean_salary"]))
                err_lo.append(float(r["mean_salary"] - r["ci_lo"]))
                err_hi.append(float(r["ci_hi"] - r["mean_salary"]))
            else:
                means.append(np.nan); err_lo.append(0); err_hi.append(0)
        pos = x_positions + (i - 1) * width
        ax.bar(pos, means, width=width, label=status, yerr=[err_lo, err_hi], capsize=3)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(EXP_ORDER)
    ax.set_ylabel("Mean salary (USD)")
    ax.set_title("Remote × Experience (raw means with 95% CI)")
    ax.legend(frameon=False, ncol=3)
    plt.tight_layout()
    fig.savefig(f"{save_prefix}/figs/interaction_remote_x_experience_rawmeans.pdf", bbox_inches="tight")
    plt.close(fig)

    # ----- B) OLS with interaction (adjusted) -----
    # Model: log_salary ~ C(remote_status)*C(experience_level) + controls
    # Include country FE and company size if present
    has_size = "company_size" in sub.columns
    formula = "log_salary ~ C(remote_status)*C(experience_level) + C(country)"
    if has_size:
        formula += " + C(company_size)"

    model = smf.ols(formula, data=sub).fit(cov_type="HC3")  # robust HC3
    # Save coefficients
    coefs = (model.params.to_frame("coef")
             .join(model.bse.to_frame("se"))
             .join(model.tvalues.to_frame("t"))
             .join(model.pvalues.to_frame("pval")))
    coefs.index.name = "term"
    coefs.reset_index().to_csv(f"{save_prefix}/tables/ols_remote_x_experience_coef.csv",
                               index=False, float_format="%.6f")

    # Wald test for the joint significance of the interaction block
    # Build a list of interaction terms present in the design
    inter_terms = [name for name in model.params.index
                   if "C(remote_status)" in name and ":C(experience_level)" in name]
    if inter_terms:
        R = np.zeros((len(inter_terms), len(model.params)))
        for i, term in enumerate(inter_terms):
            j = list(model.params.index).index(term)
            R[i, j] = 1.0
        wald = model.wald_test(R)
        wald_df = pd.DataFrame({
            "stat": [wald.statistic.item() if np.ndim(wald.statistic) else float(wald.statistic)],
            "df": [int(wald.df_denom) if hasattr(wald, "df_denom") else np.nan],
            "pval": [float(wald.pvalue)]
        })
    else:
        wald_df = pd.DataFrame({"stat":[np.nan], "df":[np.nan], "pval":[np.nan]})
    wald_df.to_csv(f"{save_prefix}/tables/ols_remote_x_experience_wald.csv", index=False, float_format="%.6f")

    # Adjusted prediction grid (marginal means) over Remote×Experience
    # We average over observed country/size composition to keep realistic weights.
    grid = pd.MultiIndex.from_product([EXP_ORDER, REMOTE_ORDER], names=["experience_level","remote_status"]).to_frame(index=False)
    # Build a big design matrix by replicating each grid row over the empirical distribution of country (+ size)
    if has_size:
        comp = sub.groupby(["country","company_size"]).size().rename("w").reset_index()
    else:
        comp = sub.groupby(["country"]).size().rename("w").reset_index()

    comp["w"] = comp["w"] / comp["w"].sum()
    rows = []
    for _, g in grid.iterrows():
        tmp = comp.copy()
        tmp["experience_level"] = g["experience_level"]
        tmp["remote_status"] = g["remote_status"]
        rows.append(tmp)
    design = pd.concat(rows, ignore_index=True)
    # Predict in levels (salary): exp(E[log_salary])
    preds = model.get_prediction(design).summary_frame(alpha=0.05)
    design = pd.concat([design, preds], axis=1)
    # Weighted average on each grid cell
    grouped = (design
               .groupby(["experience_level","remote_status"], observed=True)
               .apply(lambda d: pd.Series({
                   "pred_log_mean": np.average(d["mean"], weights=d["w"]),
                   "pred_log_lo":   np.average(d["mean_ci_lower"], weights=d["w"]),
                   "pred_log_hi":   np.average(d["mean_ci_upper"], weights=d["w"])
               }))
               .reset_index())
    # Convert back to USD level using exp
    for c in ["pred_log_mean", "pred_log_lo", "pred_log_hi"]:
        grouped[c.replace("log_", "")] = np.exp(grouped[c])

    grouped.to_csv(f"{save_prefix}/tables/interaction_adjusted_predictions.csv",
                   index=False, float_format="%.2f")

    # Plot adjusted predictions
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(EXP_ORDER))
    for status in REMOTE_ORDER:
        d = grouped[grouped["remote_status"] == status].set_index("experience_level").reindex(EXP_ORDER)
        y = d["pred_mean"].values
        lo = d["pred_lo"].values
        hi = d["pred_hi"].values
        ax.plot(x, y, marker="o", label=status)
        ax.fill_between(x, lo, hi, alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels(EXP_ORDER)
    ax.set_ylabel("Predicted mean salary (USD)")
    ax.set_title("Remote × Experience (adjusted predictions from OLS, 95% CI)")
    ax.legend(frameon=False, ncol=3)
    plt.tight_layout()
    fig.savefig(f"{save_prefix}/figs/interaction_remote_x_experience_adjusted_pred.pdf", bbox_inches="tight")
    plt.close(fig)

    return grp, coefs, wald_df, grouped

# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="salaries.csv", help="Path to salaries CSV")
    parser.add_argument("--topN", type=int, default=20, help="Top N countries by sample size")
    parser.add_argument("--min_n", type=int, default=50, help="Minimum N per country to include")
    args = parser.parse_args()

    ensure_dirs()
    df = load_and_prepare(args.input)

    # Part 1: Distribution by country
    pivot_counts, pivot_pct = remote_distribution_by_country(df, min_n=args.min_n, topN=args.topN)

    # Part 2: Interaction analysis
    grp, coefs, wald_df, grouped = interaction_remote_experience(df)

    # Console summary
    print("\n=== Remote vs On-site Distribution by Country (row %) ===")
    print(pivot_pct.round(2).head(10))
    print("\nSaved: outputs/tables/remote_distribution_by_country_{counts,pct}.csv and stacked bar PDF.\n")

    print("=== Remote × Experience: raw group means (USD) ===")
    print(grp.pivot(index="experience_level", columns="remote_status", values="mean_salary").round(0))
    print("\nOLS with interaction (HC3 robust SE). Top lines:")
    print(coefs.head(12))
    print("\nWald test for interaction block:")
    print(wald_df)

    print("\nAdjusted predictions written to outputs/tables/interaction_adjusted_predictions.csv")
    print("Figures saved under outputs/figs/")

if __name__ == "__main__":
    main()
