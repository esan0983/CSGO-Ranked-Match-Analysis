import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats


def jonckheere_terpstra(df, value_col, group_col, group_order=None, n_permutations=None, seed=None):
    """
    Jonckheere-Terpstra test for an ordered alternative across k groups.

    Tests whether `value_col` tends to increase (or decrease) monotonically
    as `group_col` moves through its ordered categories.

    Parameters
    ----------
    df : pandas.DataFrame
    value_col : str
        Name of the continuous/scalar column (e.g. "inbetween_distance").
    group_col : str
        Name of the ordinal grouping column (e.g. "att_rank").
    group_order : list, optional
        Explicit ordering of the group labels from lowest to highest.
        If None, sorted() order of the unique values is used -- so pass
        this explicitly if your ranks aren't already sortable into the
        right order (e.g. "Silver" < "Gold" < "MG" won't sort correctly
        alphabetically).
    n_permutations : int, optional
        If provided, also compute an exact/permutation p-value instead of
        relying solely on the normal approximation (recommended for smaller
        samples or to double check the asymptotic result).
    seed : int, optional
        Random seed for the permutation test.

    Returns
    -------
    dict with JT statistic, z-score, asymptotic p-value, and (optionally)
    permutation p-value.
    """
    data = df[[value_col, group_col]].dropna()

    if group_order is None:
        group_order = sorted(data[group_col].unique())

    groups = [data.loc[data[group_col] == g, value_col].to_numpy() for g in group_order]
    k = len(groups)
    if k < 2:
        raise ValueError("Need at least 2 groups.")

    def _pairwise_u(gi, gj):
        # Count of (gi < x) and (gi == x) summed over x in gj, done via
        # sort + searchsorted -- O(n log n) time and O(n) memory, instead
        # of materializing an (len(gj) x len(gi)) boolean matrix.
        gi_sorted = np.sort(gi)
        less = np.searchsorted(gi_sorted, gj, side="left")
        less_or_equal = np.searchsorted(gi_sorted, gj, side="right")
        ties = less_or_equal - less
        return less.sum() + 0.5 * ties.sum()

    def _jt_statistic(groups):
        # Sum of Mann-Whitney U statistics for every pair of groups i < j
        # (i.e. how many times a value in the "higher" group exceeds a
        # value in the "lower" group, with ties counted as 0.5).
        J = 0
        for i, j in combinations(range(len(groups)), 2):
            J += _pairwise_u(groups[i], groups[j])
        return J

    J = _jt_statistic(groups)

    n = [len(g) for g in groups]
    N = sum(n)

    # Asymptotic mean and variance under H0 (no ties correction version;
    # fine as a first pass, ties reduce variance slightly in practice)
    mean_J = (N ** 2 - sum(ni ** 2 for ni in n)) / 4
    var_J = (N ** 2 * (2 * N + 3)
             - sum(ni ** 2 * (2 * ni + 3) for ni in n)) / 72

    z = (J - mean_J) / np.sqrt(var_J)
    p_asymp = 2 * (1 - stats.norm.cdf(abs(z)))  # two-sided

    result = {
        "J_statistic": J,
        "z": z,
        "p_value_asymptotic": p_asymp,
        "group_order": group_order,
        "n_per_group": dict(zip(group_order, n)),
    }

    if n_permutations:
        rng = np.random.default_rng(seed)
        pooled = data[value_col].to_numpy()
        sizes = n
        count = 0
        for _ in range(n_permutations):
            perm = rng.permutation(pooled)
            idx = np.cumsum([0] + sizes)
            perm_groups = [perm[idx[i]:idx[i + 1]] for i in range(k)]
            J_perm = _jt_statistic(perm_groups)
            if abs(J_perm - mean_J) >= abs(J - mean_J):
                count += 1
        result["p_value_permutation"] = count / n_permutations

    return result


if __name__ == "__main__":
    # Sanity check with a synthetic example: distance should trend up with rank
    rng = np.random.default_rng(0)
    ranks = ["Silver", "Gold", "MG", "DMG", "LE"]
    rows = []
    for i, r in enumerate(ranks):
        vals = rng.normal(loc=300 + i * 40, scale=80, size=200)
        rows.extend([(v, r) for v in vals])
    df_test = pd.DataFrame(rows, columns=["inbetween_distance", "att_rank"])

    res = jonckheere_terpstra(
        df_test,
        value_col="inbetween_distance",
        group_col="att_rank",
        group_order=ranks,
        n_permutations=2000,
        seed=0,
    )
    print(res)
