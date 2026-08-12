"""Leiden community detection and the conservative initialisation of Section 3.2.

Algorithm 1, lines 2-6:

    for k = 1..K
        build the similarity graph on the samples with yhat0 == k
        run Leiden; the largest community becomes the main community M_k,
        every remaining community goes into the unlabeled set U
        apply rho-quantile distance-to-centroid risk screening to M_k;
        screened-out samples also go into U
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .graph import build_similarity_graph

__all__ = ["leiden_partition", "risk_screening", "conservative_initialisation"]


def leiden_partition(w: np.ndarray, seed: int | None = None) -> List[np.ndarray]:
    """Partition a dense weighted graph with the Leiden algorithm.

    Returns a list of index arrays, sorted by size (largest first). Isolated
    nodes come back as singleton communities.
    """
    n = w.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [np.array([0], dtype=int)]

    import igraph as ig
    import leidenalg as la

    iu = np.triu_indices(n, k=1)
    mask = w[iu] > 0
    edges = list(zip(iu[0][mask].tolist(), iu[1][mask].tolist()))
    weights = w[iu][mask].tolist()

    g = ig.Graph(n=n, edges=edges)
    if weights:
        g.es["weight"] = weights
        part = la.find_partition(
            g, la.ModularityVertexPartition, weights="weight", seed=seed
        )
    else:
        part = la.find_partition(g, la.ModularityVertexPartition, seed=seed)

    comms = [np.asarray(sorted(c), dtype=int) for c in part if len(c) > 0]
    comms.sort(key=len, reverse=True)
    return comms


def risk_screening(z: np.ndarray, members: np.ndarray, rho: float = 0.90):
    """Distance-based risk screening, Section 3.2.

    centroid  mu_k = |M_k|^-1 sum_{i in M_k} z_i
    distance  r_i  = || z_i - mu_k ||_2
    keep      r_i <= q_rho({ r_j : j in M_k })

    With rho = 0.90 the farthest 10 percent of each main community is returned
    to the unlabeled pool.

    Returns
    -------
    (kept, screened_out) : two index arrays into the original numbering.
    """
    members = np.asarray(members, dtype=int)
    if members.size == 0:
        return members, members
    centroid = z[members].mean(axis=0)
    r = np.linalg.norm(z[members] - centroid, axis=1)
    q = np.quantile(r, rho)
    keep_mask = r <= q
    return members[keep_mask], members[~keep_mask]


def conservative_initialisation(
    z: np.ndarray,
    initial_labels: np.ndarray,
    theta_g: float = 0.5,
    rho: float = 0.90,
    seed: int | None = None,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Algorithm 1, lines 1-6.

    Parameters
    ----------
    z : (n, d) l2-normalised embeddings for the whole dataset.
    initial_labels : (n,) cluster assignment produced by the initial method.

    Returns
    -------
    main_communities : {cluster_id: index array} the screened main communities.
    unlabeled : index array, the pool U.
    """
    main_communities: Dict[int, np.ndarray] = {}
    unlabeled: List[np.ndarray] = []

    for k in np.unique(initial_labels):
        idx = np.flatnonzero(initial_labels == k)
        if idx.size == 0:
            continue
        if idx.size == 1:
            main_communities[int(k)] = idx
            continue

        w = build_similarity_graph(z[idx], theta_g=theta_g)
        comms = leiden_partition(w, seed=seed)
        if not comms:
            unlabeled.append(idx)
            continue

        # largest detected community becomes the main community
        largest = idx[comms[0]]
        # every remaining detected community goes to the unlabeled pool
        for c in comms[1:]:
            unlabeled.append(idx[c])

        kept, screened_out = risk_screening(z, largest, rho=rho)
        main_communities[int(k)] = kept
        if screened_out.size:
            unlabeled.append(screened_out)

    pool = (
        np.concatenate(unlabeled).astype(int)
        if unlabeled
        else np.zeros(0, dtype=int)
    )
    return main_communities, np.unique(pool)
