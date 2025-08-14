import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

# --- Paths ---
DATA_PATH = "salaries.csv"
OUT_DIR = "data/"

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load data ---
df = pd.read_csv(DATA_PATH)
print("Data shape:", df.shape)

# --- Top 5 employee residences ---
top5 = df['employee_residence'].value_counts().head(5)
print("\nTop 5 employee residences:\n", top5)

# --- 1. t-test: Remote 100% vs On-site, Same Country Only ---
same_country = df[df['employee_residence'] == df['company_location']].copy()

remote_group = same_country[same_country['remote_ratio'] == 100]['salary_in_usd']
onsite_group = same_country[same_country['remote_ratio'] == 0]['salary_in_usd']

if len(remote_group) > 1 and len(onsite_group) > 1:
    stat_ttest, pval_ttest = ttest_ind(remote_group, onsite_group, nan_policy='omit', equal_var=False)
    print("=" * 60)
    print("t-test: Remote 100% vs On-site (Same Country Only)")
    print(f"Remote Size = {len(remote_group)}")
    print(f"On-site Size = {len(onsite_group)}")
    print(f"Test Statistic = {stat_ttest:.4f}")
    print(f"p-value = {pval_ttest:.4f}")
else:
    print("=" * 60)
    print("t-test: Not enough data after filtering for same-country remote/on-site comparison.")

# --- 2. Chi-Square Test: Salary Bracket vs Company Size ---
print("=" * 60)
print("Chi-Square Test: Salary Bracket vs Company Size")

# Create salary brackets (low = bottom 33%, mid = middle, high = top 33%)
# Handle potential duplicate edges by allowing duplicates to drop, then re-map to 3 bins if needed.
try:
    df['salary_bracket'] = pd.qcut(df['salary_in_usd'], q=3, labels=['Low', 'Medium', 'High'])
except ValueError:
    # Fallback: use rank-based approach if qcut fails due to many ties
    ranks = df['salary_in_usd'].rank(method='average', pct=True)
    labels = pd.cut(ranks, bins=[0, 1 / 3, 2 / 3, 1], labels=['Low', 'Medium', 'High'], include_lowest=True)
    df['salary_bracket'] = labels

chi_table = pd.crosstab(df['salary_bracket'], df['company_size'])
stat_chi, pval_chi, _, _ = chi2_contingency(chi_table)
print(f"Test Statistic = {stat_chi:.4f}")
print(f"p-value = {pval_chi:.4f}")

# --- 3. One-Way ANOVA: Salary ~ All Employee Residences ---
print("=" * 60)
print("One-Way ANOVA: Salary ~ All Employee Residences")

df_all_res = df.copy()
df_all_res['employee_residence'] = df_all_res['employee_residence'].astype('category')

model = smf.ols('salary_in_usd ~ C(employee_residence)', data=df_all_res).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)


# ----------------------
# Plotting (Matplotlib-only, no seaborn)
# ----------------------

# 1) Boxplot: Remote vs On-site salaries (same country)
def save_remote_onsite_boxplot():
    plot_df = same_country[same_country['remote_ratio'].isin([0, 100])].copy()
    plot_df['group'] = np.where(plot_df['remote_ratio'] == 0, 'On-site (0%)', 'Remote (100%)')
    groups = ['On-site (0%)', 'Remote (100%)']
    data_to_plot = [plot_df.loc[plot_df['group'] == g, 'salary_in_usd'].dropna().values for g in groups]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data_to_plot, labels=groups, showmeans=True)
    plt.title("Salaries: Remote (100%) vs On-site (0%) - Same Country")
    plt.ylabel("Salary (USD)")
    out_path = os.path.join(OUT_DIR, "remote_vs_onsite_boxplot.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


# 2) Heatmap: Salary Bracket vs Company Size (annotated counts)
def save_company_size_heatmap():
    # Convert contingency table to numpy array
    table = chi_table.copy()
    rows = list(table.index.astype(str))
    cols = list(table.columns.astype(str))
    Z = table.values.astype(float)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(Z, aspect='auto')
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title("Salary Bracket vs Company Size (Counts)")

    # Annotate with counts
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(j, i, f"{int(Z[i, j])}", ha='center', va='center')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "company_size_heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


# 3) Bar plot: Mean salary by country (top 10 by frequency) with 95% CI
def save_country_salary_means():
    counts = df['employee_residence'].value_counts()
    top_countries = counts.head(10).index.tolist()
    sub = df[df['employee_residence'].isin(top_countries)].copy()

    grp = sub.groupby('employee_residence')['salary_in_usd']
    means = grp.mean()
    ns = grp.size()
    stds = grp.std(ddof=1)
    ses = stds / np.sqrt(ns)
    cis = 1.96 * ses

    # Order by mean descending for nicer display
    order = means.sort_values(ascending=False).index.tolist()
    means = means.loc[order]
    cis = cis.loc[order]
    ns = ns.loc[order]

    x = np.arange(len(order))
    y = means.values
    yerr = cis.values

    plt.figure(figsize=(10, 5))
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=yerr, fmt='none', capsize=4)
    plt.xticks(x, order, rotation=45, ha='right')
    plt.ylabel("Mean Salary (USD)")
    plt.title("Mean Salary by Country (Top 10) with 95% CI")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "country_salary_means.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


# Generate figures
try:
    save_remote_onsite_boxplot()
except Exception as e:
    print("Failed to create remote/on-site boxplot:", e)

try:
    save_company_size_heatmap()
except Exception as e:
    print("Failed to create company size heatmap:", e)

try:
    save_country_salary_means()
except Exception as e:
    print("Failed to create country means bar plot:", e)

# Also save the ANOVA table to CSV for reference
anova_csv_path = os.path.join(OUT_DIR, "anova_table.csv")
anova_table.to_csv(anova_csv_path)
print(f"Saved ANOVA table to {anova_csv_path}")
# ============================
# Country-level detailed analysis
# ============================
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from math import sqrt
import itertools

OUT_DIR = OUT_DIR  # reuse from above

def cohens_d_from_groups(a, b):
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    # Welch version: pooled using group variances (not assuming equal var)
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    n1, n2 = len(a), len(b)
    sp = sqrt((s1 + s2) / 2.0)
    if sp == 0:
        return 0.0
    return (a.mean() - b.mean()) / sp

def welch_t_ci(a, b, alpha=0.05):
    # Return CI for difference in means (a - b) using Welch's approximation
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    n1, n2 = len(a), len(b)
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(v1/n1 + v2/n2)
    if n1 < 2 or n2 < 2 or se == 0:
        return (np.nan, np.nan)
    # Welch-Satterthwaite df
    df = (v1/n1 + v2/n2)**2 / ((v1**2)/((n1**2)*(n1-1)) + (v2**2)/((n2**2)*(n2-1)))
    from scipy.stats import t
    q = t.ppf(1 - alpha/2, df)
    diff = m1 - m2
    return (diff - q*se, diff + q*se)

# 1) Country vs Rest Of World (ROW) p-values
def country_vs_row_tests(df, alpha=0.05):
    rows = []
    countries = df['employee_residence'].dropna().unique().tolist()
    for c in countries:
        a = df.loc[df['employee_residence'] == c, 'salary_in_usd']
        b = df.loc[df['employee_residence'] != c, 'salary_in_usd']
        # Welch t-test
        if a.dropna().size >= 2 and b.dropna().size >= 2:
            tstat, pval = ttest_ind(a, b, equal_var=False, nan_policy='omit')
            mean_a, mean_b = a.mean(), b.mean()
            d = cohens_d_from_groups(a, b)
            lo, hi = welch_t_ci(a, b)
            rows.append({
                'country': c,
                'n_country': int(a.dropna().size),
                'n_row': int(b.dropna().size),
                'mean_country': float(mean_a),
                'mean_ROW': float(mean_b),
                'diff_country_minus_ROW': float(mean_a - mean_b),
                't_stat': float(tstat),
                'p_value_raw': float(pval),
                'ci95_low': float(lo),
                'ci95_high': float(hi),
                'cohens_d': float(d),
            })
    out = pd.DataFrame(rows).sort_values('p_value_raw')
    # BH-FDR correction
    if not out.empty:
        rej, p_adj, *_ = multipletests(out['p_value_raw'].values, alpha=alpha, method='fdr_bh')
        out['p_value_fdr_bh'] = p_adj
        out['reject_at_alpha_fdr_bh'] = rej
    return out

cvrow = country_vs_row_tests(df)
cvrow_path = os.path.join(OUT_DIR, "country_vs_row_welch.csv")
cvrow.to_csv(cvrow_path, index=False)
print(f"Saved per-country vs ROW table to {cvrow_path}")

# 2) Tukey HSD post-hoc (all pairs) after ANOVA
def tukey_all_pairs(df, min_per_group=5):
    sub = df[['employee_residence','salary_in_usd']].dropna().copy()
    # optional: drop tiny groups to avoid noisy CIs
    counts = sub['employee_residence'].value_counts()
    keep = counts[counts >= min_per_group].index
    sub = sub[sub['employee_residence'].isin(keep)]
    res = pairwise_tukeyhsd(endog=sub['salary_in_usd'],
                            groups=sub['employee_residence'],
                            alpha=0.05)
    # Convert to DataFrame
    tuk = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
    # columns: group1, group2, meandiff, p-adj, lower, upper, reject
    return tuk

tukey_df = tukey_all_pairs(df, min_per_group=20)
tuk_path = os.path.join(OUT_DIR, "tukey_country_pairs.csv")
tukey_df.to_csv(tuk_path, index=False)
print(f"Saved Tukey HSD pairwise table to {tuk_path}")

# 3) Remote penalty within each country (100% vs 0%)
def remote_penalty_by_country(df, alpha=0.05):
    rows = []
    groups = df.groupby('employee_residence')
    for c, g in groups:
        r = g.loc[g['remote_ratio'] == 100, 'salary_in_usd']
        o = g.loc[g['remote_ratio'] == 0, 'salary_in_usd']
        n_r, n_o = r.dropna().size, o.dropna().size
        if n_r >= 2 and n_o >= 2:
            tstat, pval = ttest_ind(r, o, equal_var=False, nan_policy='omit')
            mean_r, mean_o = r.mean(), o.mean()
            d = cohens_d_from_groups(r, o)
            lo, hi = welch_t_ci(r, o)
            rows.append({
                'country': c,
                'n_remote': int(n_r),
                'n_onsite': int(n_o),
                'mean_remote': float(mean_r),
                'mean_onsite': float(mean_o),
                'diff_remote_minus_onsite': float(mean_r - mean_o),
                't_stat': float(tstat),
                'p_value_raw': float(pval),
                'ci95_low': float(lo),
                'ci95_high': float(hi),
                'cohens_d': float(d),
            })
    out = pd.DataFrame(rows).sort_values('p_value_raw')
    if not out.empty:
        rej, p_adj, *_ = multipletests(out['p_value_raw'].values, alpha=alpha, method='fdr_bh')
        out['p_value_fdr_bh'] = p_adj
        out['reject_at_alpha_fdr_bh'] = rej
    return out

remote_country = remote_penalty_by_country(df)
remote_path = os.path.join(OUT_DIR, "remote_penalty_by_country.csv")
remote_country.to_csv(remote_path, index=False)
print(f"Saved per-country remote penalty table to {remote_path}")

# 4) (Optional) Company size × salary bracket chi-square per country
RUN_COMPANY_SIZE_BY_COUNTRY = False
if RUN_COMPANY_SIZE_BY_COUNTRY:
    chi_rows = []
    for c, g in df.groupby('employee_residence'):
        tab = pd.crosstab(g['salary_bracket'], g['company_size'])
        if tab.shape[0] >= 2 and tab.shape[1] >= 2 and (tab.values >= 5).all():
            stat_chi, pval_chi, _, _ = chi2_contingency(tab)
            chi_rows.append({'country': c, 'chi2': stat_chi, 'p_value': pval_chi})
    chi_df = pd.DataFrame(chi_rows).sort_values('p_value')
    chi_path = os.path.join(OUT_DIR, "country_company_size_chisq.csv")
    chi_df.to_csv(chi_path, index=False)
    print(f"Saved company size × salary bracket per-country chi-sq to {chi_path}")
