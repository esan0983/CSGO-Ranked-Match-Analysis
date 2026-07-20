import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def summarize_jt(result):
    """Human-readable summary of a jonckheere_terpstra() result dict."""
    J = result["J_statistic"]
    z = result["z"]
    p_a = result["p_value_asymptotic"]
    p_perm = result.get("p_value_permutation")
    group_order = list(result["group_order"])
    n_per_group = result["n_per_group"]

    N = sum(n_per_group.values())

    lines = []
    lines.append("=" * 60)
    lines.append("Jonckheere-Terpstra Test Summary")
    lines.append("=" * 60)
    lines.append(f"Groups (in order):  {group_order}")
    lines.append(f"Total N:             {N:,}")
    lines.append("")
    lines.append("Sample size per group:")
    for g in group_order:
        n = n_per_group[g]
        lines.append(f"  {str(g):>6}: {n:>10,}  ({n/N:5.1%})")
    lines.append("")
    lines.append(f"J statistic:         {J:,.1f}")
    lines.append(f"z-score:             {z:,.2f}")
    lines.append(f"p-value (asymptotic):{'  < 0.0001' if p_a < 1e-4 else f'  {p_a:.4f}'}")
    if p_perm is not None:
        lines.append(f"p-value (permutation): {p_perm:.4f}")
    lines.append("")
    verb = "increase" if z > 0 else "decrease"
    lines.append(f"Interpretation: the value tends to {verb} as group order increases.")
    if N > 10000:
        lines.append(
            "Note: with N this large, p-values will look significant even for tiny,\n"
            "practically meaningless effects. Check the actual per-group means/medians\n"
            "below (or plot them) to judge whether the trend is large enough to matter."
        )
    lines.append("=" * 60)
    return "\n".join(lines)


def plot_trend(df, value_col, group_col, group_order=None, title=None):
    """Plot mean + median value per group, with group sample size as bar width/annotation,
    to show the actual magnitude of the trend (not just its statistical significance)."""
    if group_order is None:
        group_order = sorted(df[group_col].unique())

    stats = (
        df.groupby(group_col)[value_col]
        .agg(mean="mean", median="median", n="count")
        .reindex(group_order)
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(group_order))

    ax1.plot(x, stats["mean"], marker="o", label="mean", color="tab:blue")
    ax1.plot(x, stats["median"], marker="s", label="median", color="tab:orange", linestyle="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(group_order)
    ax1.set_xlabel(group_col)
    ax1.set_ylabel(value_col)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.bar(x, stats["n"], alpha=0.15, color="gray", label="sample size (n)")
    ax2.set_ylabel("n per group")

    ax1.set_title(title or f"{value_col} by {group_col}: mean/median trend with sample size")
    fig.tight_layout()
    return fig, stats


if __name__ == "__main__":
    # sanity check with synthetic data resembling the real result's group sizes
    rng = np.random.default_rng(0)
    n_per = {1:144,2:401,3:630,4:1882,5:3389,6:11556,7:41806,8:68966,9:110681,
              10:117883,11:101528,12:97492,13:101240,14:86027,15:58749,16:52160,17:18256,18:2322}
    rows = []
    for rank, n in n_per.items():
        vals = rng.normal(loc=250 + rank * 5, scale=100, size=n)
        rows.extend([(v, rank) for v in vals])
    df_test = pd.DataFrame(rows, columns=["inbetween_distance", "att_rank"])

    from jt_test import jonckheere_terpstra
    res = jonckheere_terpstra(df_test, "inbetween_distance", "att_rank",
                               group_order=sorted(n_per.keys()))
    print(summarize_jt(res))

    fig, stats = plot_trend(df_test, "inbetween_distance", "att_rank",
                             group_order=sorted(n_per.keys()))
    fig.savefig("/home/claude/trend_check.png", dpi=100)
    print(stats)
