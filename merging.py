"""The merging score of Equation (2) and the greedy non-conflicting matching of
Algorithm 1, lines 11-13.

    L_kl =        delta_Q_kl        +        delta_kbar_kl        -        t_kl
           ----------------------     ------------------------     ----------------
           max_S |delta_Q_kS| + eps   max_S |delta_kbar_kS| + eps  max_S t_kS + eps

with eps = 1e-12, and each maximum taken over A_k, the candidates still
available to main community k in the current matching step.

    t_kl = (1 / (|M_k| |S_l|)) * sum_{i in M_k} sum_{j in S_l} || z_i - z_j ||_2
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .graph import delta_internal_avg_degree, delta_modularity

__all__ = ["feature_distance", "merging_scores", "greedy_merge"]

EPS = 1e-12


def feature_distance(z: np.ndarray, main: np.ndarray, isolated: np.ndarray) -> float:
    """Mean pairwise Euclidean distance between two communities, Equation (3)."""
    main = np.asarray(main, dtype=int)
    isolated = np.asarray(isolated, dtype=int)
    if main.size == 0 or isolated.size == 0:
        return 0.0
    diff = z[main][:, None, :] - z[isolated][None, :, :]
    return float(np.linalg.norm(diff, axis=2).mean())


def _pair_terms(
    z: np.ndarray,
    w: np.ndarray,
    node_index: Dict[int, int],
    labels_local: np.ndarray,
    main_label: int,
    main_members: np.ndarray,
    isolated_label: int,
    isolated_members: np.ndarray,
) -> Tuple[float, float, float]:
    """Raw (unnormalised) delta_Q, delta_kbar and t for one candidate pair."""
    dq = delta_modularity(w, labels_local, main_label, isolated_label)

    main_local = np.array([node_index[i] for i in main_members], dtype=int)
    iso_local = np.array([node_index[i] for i in isolated_members], dtype=int)
    dk = delta_internal_avg_degree(w, main_local, iso_local)

    t = feature_distance(z, main_members, isolated_members)
    return dq, dk, t


def merging_scores(
    z: np.ndarray,
    w: np.ndarray,
    node_index: Dict[int, int],
    labels_local: np.ndarray,
    main_communities: Dict[int, np.ndarray],
    isolated_communities: Sequence[np.ndarray],
    main_label_of: Dict[int, int],
    isolated_label_of: Sequence[int],
    available: Dict[int, Sequence[int]] | None = None,
) -> Dict[Tuple[int, int], float]:
    """Compute L_kl for every (main community, isolated community) pair.

    ``available[k]`` lists the indices into ``isolated_communities`` that are
    still candidates for main community k; the normalising maxima are taken over
    that set, as in Equation (2). When omitted, all isolated communities are
    considered available to every main community.
    """
    if available is None:
        available = {k: list(range(len(isolated_communities))) for k in main_communities}

    scores: Dict[Tuple[int, int], float] = {}
    for k, main_members in main_communities.items():
        cand = list(available.get(k, []))
        if not cand or main_members.size == 0:
            continue

        raw = {}
        for li in cand:
            iso = isolated_communities[li]
            if iso.size == 0:
                continue
            raw[li] = _pair_terms(
                z, w, node_index, labels_local,
                main_label_of[k], main_members,
                isolated_label_of[li], iso,
            )
        if not raw:
            continue

        max_dq = max(abs(v[0]) for v in raw.values())
        max_dk = max(abs(v[1]) for v in raw.values())
        max_t = max(v[2] for v in raw.values())

        for li, (dq, dk, t) in raw.items():
            scores[(k, li)] = (
                dq / (max_dq + EPS)
                + dk / (max_dk + EPS)
                - t / (max_t + EPS)
            )
    return scores


def greedy_merge(
    scores: Dict[Tuple[int, int], float]
) -> List[Tuple[int, int]]:
    """Greedy non-conflicting matching, Algorithm 1 line 12.

    Candidate pairs are ranked by L_kl. A pair is accepted only if neither its
    main community nor its isolated community has already been used in this
    iteration, so each main community absorbs at most one isolated community per
    iteration and each isolated community is selected at most once.
    """
    used_main: set[int] = set()
    used_iso: set[int] = set()
    accepted: List[Tuple[int, int]] = []
    for (k, li) in sorted(scores, key=lambda p: scores[p], reverse=True):
        if k in used_main or li in used_iso:
            continue
        used_main.add(k)
        used_iso.add(li)
        accepted.append((k, li))
    return accepted


def assign_edgeless_singletons(
    z: np.ndarray,
    singletons: np.ndarray,
    main_communities: Dict[int, np.ndarray],
) -> Dict[int, int]:
    """Algorithm 1 line 13: an isolated singleton with no retained graph edge is
    assigned to the nearest main-community centroid.
    """
    if len(main_communities) == 0 or singletons.size == 0:
        return {}
    keys = list(main_communities.keys())
    centroids = np.stack([z[main_communities[k]].mean(axis=0) for k in keys])
    d = np.linalg.norm(z[singletons][:, None, :] - centroids[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    return {int(s): int(keys[j]) for s, j in zip(singletons, nearest)}
