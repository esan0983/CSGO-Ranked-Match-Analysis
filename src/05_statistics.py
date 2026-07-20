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

df = pd.read_csv("../data/processed/csgo_cleaned_3.csv")

def cochran_armitage_trend(ranks, n, r, alternative='greater'):
    """
    ranks: numeric score per rank level (e.g. the att_rank values themselves)
    n: total damage entries at that rank
    r: damage entries at that rank using the weapon of interest
    """
    ranks, n, r = map(np.asarray, (ranks, n, r))
    N = n.sum()
    R = r.sum()
    p_bar = R / N
    x_bar = (n * ranks).sum() / N

    numerator = (n * (ranks - x_bar) * (r / n - p_bar)).sum()
    denom = np.sqrt(p_bar * (1 - p_bar) * (n * (ranks - x_bar) ** 2).sum())
    z = numerator / denom

    if alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z)
    elif alternative == 'less':
        p_value = stats.norm.cdf(z)
    else:
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p_value


# --- Build the (att_rank, wp) contingency counts ---
# total damage entries per rank (denominator) — across ALL weapons
total_by_rank = df.groupby('att_rank').size().rename('n_total')

# damage entries per (rank, weapon) — numerator
usage_by_rank_wp = df.groupby(['att_rank', 'wp']).size().rename('n_wp').reset_index()

# fill in zeros for weapon/rank combos that never occur
all_ranks = df['att_rank'].unique()
all_weapons = df['wp'].unique()
full_grid = pd.MultiIndex.from_product([all_ranks, all_weapons], names=['att_rank', 'wp']).to_frame(index=False)

counts = (full_grid
          .merge(usage_by_rank_wp, on=['att_rank', 'wp'], how='left')
          .fillna({'n_wp': 0}))
counts = counts.merge(total_by_rank, on='att_rank', how='left')
counts['n_wp'] = counts['n_wp'].astype(int)

def run_ca_for_weapon(counts, weapon, alternative='greater'):
    sub = counts[counts['wp'] == weapon].sort_values('att_rank')
    z, p = cochran_armitage_trend(sub['att_rank'].values, sub['n_total'].values,
                                   sub['n_wp'].values, alternative=alternative)
    N = sub['n_total'].sum()
    effect_r = z / np.sqrt(N) if N > 0 else np.nan

    # logistic regression slope -> odds ratio per rank unit
    X = sm.add_constant(sub['att_rank'].astype(float))
    y_prop = sub['n_wp'] / sub['n_total']
    try:
        model = sm.GLM(y_prop, X, family=sm.families.Binomial(),
                        freq_weights=sub['n_total'].values).fit()
        odds_ratio = np.exp(model.params['att_rank'])
    except Exception:
        odds_ratio = np.nan

    return z, p, effect_r, odds_ratio


results = []
for weapon in df['wp'].unique():
    z, p, r, orr = run_ca_for_weapon(counts, weapon, alternative='greater')
    results.append({'wp': weapon, 'z': z, 'p_raw': p, 'effect_r': r, 'odds_ratio_per_rank': orr})

res_df = pd.DataFrame(results).dropna(subset=['p_raw'])
res_df['p_adj'] = multipletests(res_df['p_raw'], method='fdr_bh')[1]
res_df = res_df.sort_values('effect_r', ascending=False)  # sort by effect size, not p
print(res_df)

sys.path.insert(0, "../src") 
from jt_test import jonckheere_terpstra
from jt_summary import summarize_jt, plot_trend

result = jonckheere_terpstra(df, "inbetween_distance", "att_rank",
                              group_order=sorted(df["att_rank"].unique()))
print(summarize_jt(result))

fig, stats_table = plot_trend(df, "inbetween_distance", "att_rank")
fig.savefig("../images/05_jt_trend.png")
print(stats_table)