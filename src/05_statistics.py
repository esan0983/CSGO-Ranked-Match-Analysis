import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import os
import sys
from pandas.api.types import CategoricalDtype
from scipy.stats import chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df = pd.read_csv("../data/processed/csgo_cleaned_3.csv")

custom_order = CategoricalDtype(categories=['Silver', 'Gold Nova', 'Master Guardian', 'Top Four'], ordered=True)
df["att_tier"] = df["att_tier"].astype(custom_order)
df["vic_tier"] = df["vic_tier"].astype(custom_order)

# SOME CHI-SQUARE TESTS

# ct = pd.crosstab(df["is_bomb_planted"], df["att_tier"])
# chi2_stat, p_val, dof, expected_table = chi2_contingency(ct)
# print(f"Chi-Square Statistic: {chi2_stat:.4f}")
# print(f"P-Value:             {p_val:.8f}")
# print(f"Degrees of Freedom:  {dof}")
# print("\n--- Expected Frequencies Table ---")
# print(pd.DataFrame(expected_table, index=ct.index, columns=ct.columns))

# ct = pd.crosstab(df["round_type"], df["att_tier"])
# chi2_stat, p_val, dof, expected_table = chi2_contingency(ct)
# print(f"Chi-Square Statistic: {chi2_stat:.4f}")
# print(f"P-Value:             {p_val:.8f}")
# print(f"Degrees of Freedom:  {dof}")
# print("\n--- Expected Frequencies Table ---")
# print(pd.DataFrame(expected_table, index=ct.index, columns=ct.columns))

# ANOVA FOR TOTAL_DMG VS ATT_TIER

# df["dmg_normalized"], optimal_lambda = stats.boxcox(df["total_dmg"])

# dmg_normalized_1 = df.loc[df["att_tier"] == "Silver", "dmg_normalized"]
# dmg_normalized_2 = df.loc[df["att_tier"] == "Gold Nova", "dmg_normalized"]
# dmg_normalized_3 = df.loc[df["att_tier"] == "Master Guardian", "dmg_normalized"]
# dmg_normalized_4 = df.loc[df["att_tier"] == "Top Four", "dmg_normalized"]

# f_stat, p_val = stats.f_oneway(
#     dmg_normalized_1, dmg_normalized_2, dmg_normalized_3, dmg_normalized_4
# )
# print(f"F-Statistic: {f_stat:.4f}, p-value: {p_val:.4e}")

# tukey = pairwise_tukeyhsd(
#     endog=df["dmg_normalized"],  # The normalized dependent variable
#     groups=df["att_tier"],       # The categorical tier groups
#     alpha=0.05                            # Significance level
# )

# print(tukey)

# CA TEST FOR WP VS ATT_TIER

# ct = pd.crosstab(df["wp"], df["att_tier"])

# df['att_tier_score'] = df['att_tier'].cat.codes

# def cochran_armitage_trend(scores, n, r, alternative='greater'):
#     scores, n, r = map(np.asarray, (scores, n, r))
#     N = n.sum()
#     R = r.sum()
#     p_bar = R / N
#     x_bar = (n * scores).sum() / N
#     numerator = (n * (scores - x_bar) * (r / n - p_bar)).sum()
#     denom = np.sqrt(p_bar * (1 - p_bar) * (n * (scores - x_bar) ** 2).sum())
#     z = numerator / denom
#     if alternative == 'greater':
#         p_value = 1 - stats.norm.cdf(z)
#     elif alternative == 'less':
#         p_value = stats.norm.cdf(z)
#     else:
#         p_value = 2 * (1 - stats.norm.cdf(abs(z)))
#     return z, p_value

# def build_counts(df, group_col, outcome_col, outcome_value):
#     n_total = df.groupby(group_col).size().rename('n_total')
#     n_hit = (df[df[outcome_col] == outcome_value]
#              .groupby(group_col).size()
#              .rename('n_hit'))

#     counts = pd.concat([n_total, n_hit], axis=1).fillna({'n_hit': 0})
#     counts['n_hit'] = counts['n_hit'].astype(int)
#     counts = counts.reset_index()
#     counts['score'] = counts[group_col].cat.codes
#     counts = counts.sort_values('score')
#     return counts

# def run_ca(counts, alternative='greater'):
#     z, p = cochran_armitage_trend(counts['score'].values, counts['n_total'].values,
#                                    counts['n_hit'].values, alternative=alternative)
#     N = counts['n_total'].sum()
#     effect_r = z / np.sqrt(N) if N > 0 else np.nan

#     X = sm.add_constant(counts['score'].astype(float))
#     y_prop = counts['n_hit'] / counts['n_total']
#     try:
#         model = sm.GLM(y_prop, X, family=sm.families.Binomial(),
#                         freq_weights=counts['n_total'].values).fit()
#         odds_ratio = np.exp(model.params['score'])
#     except Exception:
#         odds_ratio = np.nan

#     return z, p, effect_r, odds_ratio

# counts = build_counts(df, 'att_tier', 'wp', 'AK47')
# z, p, effect_r, orr = run_ca(counts, alternative='greater')
# print(f"Z={z:.3f}, p={p:.4g}, effect_r={effect_r:.4f}, odds_ratio_per_tier={orr:.4f}")
# print(counts)

# weapons = df['wp'].unique()
# results = []

# for wp in weapons:
#     counts = build_counts(df, 'att_tier', 'wp', wp)
#     if counts['n_hit'].sum() == 0 or counts['n_hit'].sum() == counts['n_total'].sum():
#         continue  # skip degenerate cases: weapon never used, or used every single time
#     z, p, effect_r, orr = run_ca(counts, alternative='greater')
#     results.append({'wp': wp, 'z': z, 'p_raw': p, 'effect_r': effect_r,
#                      'odds_ratio_per_tier': orr, 'counts': counts})

# res_df = pd.DataFrame(results).dropna(subset=['p_raw'])
# res_df['p_adj'] = multipletests(res_df['p_raw'], method='fdr_bh')[1]
# res_df = res_df.sort_values('effect_r', ascending=False)
# print(res_df.drop(columns='counts'))  # drop the nested counts table for a clean printout

# sig_df = res_df[res_df['p_adj'] < 0.05]

# fig, ax = plt.subplots(figsize=(9, 6))

# for _, row in sig_df.iterrows():
#     c = row['counts'].copy()
#     c['proportion'] = c['n_hit'] / c['n_total']
#     ax.plot(c['att_tier'], c['proportion'], marker='o', label=row['wp'])

# ax.set_xlabel('Rank Tier')
# ax.set_ylabel('Proportion of Damage Entries')
# ax.set_title('Weapon Usage Rate by Rank Tier (Only Statistically Significant Trends)')
# ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig("../images/05_wp_rank.png")
# plt.show()

# ANOVA FOR INBETWEEN_DISTANCE VS ATT_TIER

# temp_df = df[df["inbetween_distance"] > 0]

# temp_df["dist_normalized"], optimal_lambda = stats.boxcox(temp_df["inbetween_distance"])

# dist_normalized_1 = temp_df.loc[temp_df["att_tier"] == "Silver", "dist_normalized"]
# dist_normalized_2 = temp_df.loc[temp_df["att_tier"] == "Gold Nova", "dist_normalized"]
# dist_normalized_3 = temp_df.loc[temp_df["att_tier"] == "Master Guardian", "dist_normalized"]
# dist_normalized_4 = temp_df.loc[temp_df["att_tier"] == "Top Four", "dist_normalized"]

# f_stat, p_val = stats.f_oneway(
#     dist_normalized_1, dist_normalized_2, dist_normalized_3, dist_normalized_4
# )
# print(f"F-Statistic: {f_stat:.4f}, p-value: {p_val:.4e}")

# tukey = pairwise_tukeyhsd(
#     endog=temp_df["dist_normalized"],  # The normalized dependent variable
#     groups=temp_df["att_tier"],       # The categorical tier groups
#     alpha=0.05                            # Significance level
# )

# print(tukey)

# ANOVA FOR ATT_DISTANCE_TO_BOMBSITE VS ATT_TIER

# temp_df = df[df["att_distance_to_bombsite"] > 0]

# temp_df["dist_normalized"], optimal_lambda = stats.boxcox(temp_df["att_distance_to_bombsite"])

# dist_normalized_1 = temp_df.loc[temp_df["att_tier"] == "Silver", "dist_normalized"]
# dist_normalized_2 = temp_df.loc[temp_df["att_tier"] == "Gold Nova", "dist_normalized"]
# dist_normalized_3 = temp_df.loc[temp_df["att_tier"] == "Master Guardian", "dist_normalized"]
# dist_normalized_4 = temp_df.loc[temp_df["att_tier"] == "Top Four", "dist_normalized"]

# f_stat, p_val = stats.f_oneway(
#     dist_normalized_1, dist_normalized_2, dist_normalized_3, dist_normalized_4
# )
# print(f"F-Statistic: {f_stat:.4f}, p-value: {p_val:.4e}")

# tukey = pairwise_tukeyhsd(
#     endog=temp_df["dist_normalized"],  # The normalized dependent variable
#     groups=temp_df["att_tier"],       # The categorical tier groups
#     alpha=0.05                            # Significance level
# )

# print(tukey)

# ANOVA FOR VIC_DISTANCE_TO_BOMBSITE VS VIC_TIER

# temp_df = df[df["vic_distance_to_bombsite"] > 0]

# temp_df["dist_normalized"], optimal_lambda = stats.boxcox(temp_df["vic_distance_to_bombsite"])

# dist_normalized_1 = temp_df.loc[temp_df["vic_tier"] == "Silver", "dist_normalized"]
# dist_normalized_2 = temp_df.loc[temp_df["vic_tier"] == "Gold Nova", "dist_normalized"]
# dist_normalized_3 = temp_df.loc[temp_df["vic_tier"] == "Master Guardian", "dist_normalized"]
# dist_normalized_4 = temp_df.loc[temp_df["vic_tier"] == "Top Four", "dist_normalized"]

# f_stat, p_val = stats.f_oneway(
#     dist_normalized_1, dist_normalized_2, dist_normalized_3, dist_normalized_4
# )
# print(f"F-Statistic: {f_stat:.4f}, p-value: {p_val:.4e}")

# tukey = pairwise_tukeyhsd(
#     endog=temp_df["dist_normalized"],  # The normalized dependent variable
#     groups=temp_df["att_tier"],       # The categorical tier groups
#     alpha=0.05                            # Significance level
# )

# print(tukey)