"""Similarity-graph construction and the two structural quantities used by the
merging score.

Everything here follows Section 3.2 and Equation (1) of the manuscript:

    w_ij = cos(z_i, z_j)   if cos(z_i, z_j) >= theta_g
           (no edge)       otherwise

    Q = (1 / 2m) * sum_ij [ w_ij - k_i k_j / (2m) ] delta(c_i, c_j)
        with 2m = sum_ij w_ij  and  k_i = sum_j w_ij

    kbar(C) = |C|^-1 * sum_{i in C} sum_{j in C} w_ij
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "l2_normalize",
    "build_similarity_graph",
    "modularity",
    "internal_avg_degree",
    "delta_internal_avg_degree",
    "delta_modularity",
]


def l2_normalize(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise l2 normalisation. Section 3.2: 'All embeddings are l2-normalized.'"""
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(norms, eps)


def build_similarity_graph(z: np.ndarray, theta_g: float = 0.5) -> np.ndarray:
    """Dense weighted adjacency for one set of samples.

    Parameters
    ----------
    z : (n, d) array, assumed already l2-normalised.
    theta_g : similarity threshold. Edges with cos < theta_g are removed.

    Returns
    -------
    (n, n) symmetric array with zero diagonal. Entry (i, j) is cos(z_i, z_j)
    when that value is >= theta_g, else 0.
    """
    if z.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    w = z @ z.T
    np.fill_diagonal(w, 0.0)
    w[w < theta_g] = 0.0
    # numerical symmetry
    w = 0.5 * (w + w.T)
    return w


def modularity(w: np.ndarray, labels: np.ndarray) -> float:
    """Modularity of a partition of a weighted undirected graph, Equation (1).

    ``labels[i]`` is the community index of node i. Nodes carrying label < 0 are
    treated as belonging to no community and contribute nothing to the sum.
    """
    if w.size == 0:
        return 0.0
    k = w.sum(axis=1)
    two_m = w.sum()
    if two_m <= 0:
        return 0.0
    same = (labels[:, None] == labels[None, :]) & (labels[:, None] >= 0)
    expected = np.outer(k, k) / two_m
    return float(((w - expected) * same).sum() / two_m)


def internal_avg_degree(w: np.ndarray, members: np.ndarray) -> float:
    """kbar(C) = |C|^-1 sum_{i in C} sum_{j in C} w_ij."""
    members = np.asarray(members, dtype=int)
    if members.size == 0:
        return 0.0
    sub = w[np.ix_(members, members)]
    return float(sub.sum() / members.size)


def delta_internal_avg_degree(
    w: np.ndarray, main: np.ndarray, isolated: np.ndarray
) -> float:
    """Connectivity change of Section 3.3:

        delta_kbar = kbar(M_k union S_l)
                     - ( |M_k| kbar(M_k) + |S_l| kbar(S_l) ) / ( |M_k| + |S_l| )
    """
    main = np.asarray(main, dtype=int)
    isolated = np.asarray(isolated, dtype=int)
    if main.size == 0 or isolated.size == 0:
        return 0.0
    union = np.concatenate([main, isolated])
    kbar_union = internal_avg_degree(w, union)
    kbar_main = internal_avg_degree(w, main)
    kbar_iso = internal_avg_degree(w, isolated)
    baseline = (main.size * kbar_main + isolated.size * kbar_iso) / (
        main.size + isolated.size
    )
    return float(kbar_union - baseline)


def delta_modularity(
    w: np.ndarray, labels: np.ndarray, main_label: int, isolated_label: int
) -> float:
    """delta_Q = Q(P_{k<-l}) - Q(P): modularity after relabelling every node of
    ``isolated_label`` to ``main_label``, minus modularity before.
    """
    before = modularity(w, labels)
    after_labels = labels.copy()
    after_labels[after_labels == isolated_label] = main_label
    after = modularity(w, after_labels)
    return float(after - before)
