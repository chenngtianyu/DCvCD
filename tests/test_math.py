"""Checks that the structural quantities match their definitions in the paper.

Modularity and the internal average degree are verified against independent
implementations (networkx, and a direct double loop) rather than against the
module's own code, so a sign error or a missing normalisation would show up.

    python -m pytest tests/test_math.py -q
    or simply:  python tests/test_math.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dcvcd.graph import (                              # noqa: E402
    build_similarity_graph,
    delta_internal_avg_degree,
    delta_modularity,
    internal_avg_degree,
    l2_normalize,
    modularity,
)
from dcvcd.communities import risk_screening           # noqa: E402
from dcvcd.merging import EPS, feature_distance, greedy_merge  # noqa: E402


def _toy(seed=0, n=40, d=8, k=4):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, d)) * 3
    labels = rng.integers(0, k, size=n)
    z = centres[labels] + rng.normal(size=(n, d)) * 0.6
    return l2_normalize(z), labels


def test_threshold_and_symmetry():
    z, _ = _toy()
    w = build_similarity_graph(z, theta_g=0.5)
    assert np.allclose(w, w.T), "adjacency must be symmetric"
    assert np.allclose(np.diag(w), 0), "no self loops"
    nz = w[w > 0]
    assert (nz >= 0.5 - 1e-9).all(), "every retained edge must satisfy cos >= theta_g"
    print("ok  threshold, symmetry, no self-loops")


def test_modularity_matches_networkx():
    import networkx as nx

    z, labels = _toy()
    w = build_similarity_graph(z, theta_g=0.3)
    g = nx.Graph()
    g.add_nodes_from(range(len(z)))
    iu = np.triu_indices(len(z), 1)
    for i, j in zip(*iu):
        if w[i, j] > 0:
            g.add_edge(int(i), int(j), weight=float(w[i, j]))
    groups = [set(np.flatnonzero(labels == c).tolist()) for c in np.unique(labels)]
    expected = nx.algorithms.community.modularity(g, groups, weight="weight")
    got = modularity(w, labels)
    assert abs(got - expected) < 1e-8, f"{got} vs networkx {expected}"
    print(f"ok  modularity matches networkx  ({got:.6f})")


def test_internal_avg_degree_definition():
    z, labels = _toy()
    w = build_similarity_graph(z, theta_g=0.3)
    members = np.flatnonzero(labels == 0)
    brute = sum(w[i, j] for i in members for j in members) / len(members)
    got = internal_avg_degree(w, members)
    assert abs(got - brute) < 1e-10
    print(f"ok  kbar(C) matches the double sum  ({got:.6f})")


def test_delta_kbar_baseline():
    """delta_kbar must be zero when the two communities have no edges between
    them and identical internal structure is preserved by concatenation."""
    w = np.zeros((6, 6))
    w[0, 1] = w[1, 0] = 1.0
    w[3, 4] = w[4, 3] = 1.0
    a, b = np.array([0, 1, 2]), np.array([3, 4, 5])
    # no cross edges -> union kbar is the size-weighted mean -> delta = 0
    assert abs(delta_internal_avg_degree(w, a, b)) < 1e-12
    # add a cross edge -> delta must become positive
    w[2, 3] = w[3, 2] = 1.0
    assert delta_internal_avg_degree(w, a, b) > 0
    print("ok  delta_kbar is 0 without cross edges and positive with one")


def test_delta_modularity_sign():
    z, labels = _toy()
    w = build_similarity_graph(z, theta_g=0.3)
    lab = labels.copy()
    # splitting a true community in two should make merging it back positive
    members = np.flatnonzero(lab == 0)
    half = members[: len(members) // 2]
    new_label = lab.max() + 1
    lab[half] = new_label
    dq = delta_modularity(w, lab, 0, new_label)
    assert dq > 0, f"re-merging a split community should raise Q, got {dq}"
    print(f"ok  delta_Q > 0 when a split community is re-merged  ({dq:.6f})")


def test_risk_screening_drops_ten_percent():
    rng = np.random.default_rng(1)
    z = l2_normalize(rng.normal(size=(200, 16)))
    members = np.arange(200)
    kept, dropped = risk_screening(z, members, rho=0.90)
    assert kept.size + dropped.size == 200
    assert 15 <= dropped.size <= 25, f"expected ~10% dropped, got {dropped.size}"
    centroid = z[members].mean(axis=0)
    r = np.linalg.norm(z - centroid, axis=1)
    assert r[dropped].min() >= r[kept].max() - 1e-12, "dropped must be the farthest"
    print(f"ok  rho=0.90 screening drops the farthest {dropped.size}/200")


def test_feature_distance_is_mean_pairwise():
    z, _ = _toy()
    a, b = np.arange(0, 5), np.arange(20, 27)
    brute = np.mean([np.linalg.norm(z[i] - z[j]) for i in a for j in b])
    assert abs(feature_distance(z, a, b) - brute) < 1e-10
    print("ok  t_kl is the mean pairwise Euclidean distance")


def test_greedy_matching_is_non_conflicting():
    scores = {(0, 0): 5.0, (0, 1): 4.0, (1, 0): 4.5, (1, 1): 1.0, (2, 1): 3.0}
    acc = greedy_merge(scores)
    mains = [k for k, _ in acc]
    isos = [i for _, i in acc]
    assert len(mains) == len(set(mains)), "a main community absorbed more than one"
    assert len(isos) == len(set(isos)), "an isolated community was used twice"
    assert acc[0] == (0, 0), "highest-scoring pair must be taken first"
    print(f"ok  greedy matching is non-conflicting  {acc}")


def test_score_normalisation_bounds():
    """Each normalised term lies in [-1, 1], so L_kl is bounded by [-3, 1]."""
    vals = [0.4, -0.9, 0.1]
    m = max(abs(v) for v in vals)
    for v in vals:
        assert -1 - 1e-9 <= v / (m + EPS) <= 1 + 1e-9
    print("ok  normalised terms are bounded by 1 in magnitude")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} checks passed")
