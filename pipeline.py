"""Algorithm 1: DCvCD training and gradual community refinement.

    Require: dataset X, number of clusters K, backbone f_theta, initial cluster
             assignments yhat0, theta_g = 0.5, rho = 0.90, tau_NCE = 0.5
    Ensure : final cluster assignments yhat

     1  extract l2-normalised embeddings, U <- {}
     2  for k = 1..K
     3      build the weighted similarity graph on samples with yhat0 == k
     4      Leiden; largest community -> M_k, the rest -> U
     5      rho-quantile risk screening on M_k; screened-out samples -> U
     6  end for
     7  while U is not empty
     8      fine-tune f_theta on {M_k} with the clustering-aware InfoNCE loss
     9      refresh embeddings, rebuild the graph over U with threshold theta_g
    10      Leiden on that graph -> isolated communities S = {S_l}
    11      compute L_kl for every valid pair
    12      greedily merge the highest-scoring non-conflicting pairs
    13      assign edgeless singletons to the nearest main-community centroid
    14  end while
    15  return the assignment induced by {M_k}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from .communities import conservative_initialisation, leiden_partition
from .graph import build_similarity_graph, l2_normalize
from .merging import assign_edgeless_singletons, greedy_merge, merging_scores

__all__ = ["DCvCDConfig", "run_dcvcd"]


@dataclass
class DCvCDConfig:
    theta_g: float = 0.5          # Section 3.2 / Appendix B
    rho: float = 0.90             # Section 3.2
    tau_nce: float = 0.5          # Section 3.4
    epochs_per_iteration: int = 100   # Section 4.2
    batch_size: int = 64
    lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 5e-4
    max_iterations: int = 100     # safety valve on the while loop
    seed: Optional[int] = None
    verbose: bool = True


def run_dcvcd(
    embeddings: np.ndarray,
    initial_labels: np.ndarray,
    cfg: DCvCDConfig = DCvCDConfig(),
    finetune_fn: Optional[Callable[[Dict[int, np.ndarray], int], np.ndarray]] = None,
) -> np.ndarray:
    """Run gradual community refinement.

    Parameters
    ----------
    embeddings : (n, d) embeddings from the initial method's backbone.
    initial_labels : (n,) assignments from the initial method.
    finetune_fn : callable(main_communities, iteration) -> new (n, d) embeddings.
        This is Algorithm 1 lines 8-9. Pass ``None`` to run the graph-side
        algorithm on frozen embeddings, which is useful for unit tests and for
        ablations that isolate the merging rule.

    Returns
    -------
    (n,) final cluster assignment.
    """
    z = l2_normalize(np.asarray(embeddings, dtype=np.float64))
    initial_labels = np.asarray(initial_labels, dtype=int)
    n = z.shape[0]

    # ---- lines 1-6 ---------------------------------------------------------
    main, pool = conservative_initialisation(
        z, initial_labels, theta_g=cfg.theta_g, rho=cfg.rho, seed=cfg.seed
    )
    if cfg.verbose:
        print(f"[init] {len(main)} main communities, "
              f"{sum(len(v) for v in main.values())} labelled, {pool.size} in U")

    # ---- lines 7-14 --------------------------------------------------------
    it = 0
    while pool.size > 0 and it < cfg.max_iterations:
        it += 1

        # line 8-9: fine-tune and refresh embeddings
        if finetune_fn is not None:
            z = l2_normalize(np.asarray(finetune_fn(main, it), dtype=np.float64))

        # line 9-10: rebuild the graph over U and run Leiden on it
        w_pool = build_similarity_graph(z[pool], theta_g=cfg.theta_g)
        comms_local = leiden_partition(w_pool, seed=cfg.seed)
        isolated = [pool[c] for c in comms_local]

        # line 13: singletons with no retained edge go to the nearest centroid
        degrees = w_pool.sum(axis=1)
        edgeless_local = np.flatnonzero(degrees == 0)
        edgeless = pool[edgeless_local]
        isolated = [c for c in isolated if not (c.size == 1 and c[0] in set(edgeless.tolist()))]

        if edgeless.size:
            direct = assign_edgeless_singletons(z, edgeless, main)
            for s, k in direct.items():
                main[k] = np.append(main[k], s)
            pool = np.setdiff1d(pool, edgeless)
            if cfg.verbose:
                print(f"[iter {it}] {edgeless.size} edgeless singletons -> nearest centroid")
            if not isolated:
                continue

        if not isolated:
            if cfg.verbose:
                print(f"[iter {it}] no isolated communities left; stopping")
            break

        # ---- line 11: merging scores on the joint graph --------------------
        active = np.concatenate([np.concatenate(list(main.values())), pool])
        active = np.unique(active)
        node_index = {int(g): i for i, g in enumerate(active)}
        w_joint = build_similarity_graph(z[active], theta_g=cfg.theta_g)

        labels_local = -np.ones(active.size, dtype=int)
        main_label_of: Dict[int, int] = {}
        for lbl, (k, members) in enumerate(main.items()):
            main_label_of[k] = lbl
            for g in members:
                labels_local[node_index[int(g)]] = lbl
        isolated_label_of = []
        next_label = len(main)
        for c in isolated:
            isolated_label_of.append(next_label)
            for g in c:
                labels_local[node_index[int(g)]] = next_label
            next_label += 1

        scores = merging_scores(
            z, w_joint, node_index, labels_local,
            main, isolated, main_label_of, isolated_label_of,
        )
        if not scores:
            if cfg.verbose:
                print(f"[iter {it}] no scorable pairs; assigning remainder by centroid")
            direct = assign_edgeless_singletons(z, pool, main)
            for s, k in direct.items():
                main[k] = np.append(main[k], s)
            pool = np.zeros(0, dtype=int)
            break

        # ---- line 12: greedy non-conflicting matching ----------------------
        accepted = greedy_merge(scores)
        merged_nodes = []
        for k, li in accepted:
            main[k] = np.concatenate([main[k], isolated[li]])
            merged_nodes.append(isolated[li])
        if merged_nodes:
            pool = np.setdiff1d(pool, np.concatenate(merged_nodes))

        if cfg.verbose:
            print(f"[iter {it}] merged {len(accepted)} communities, |U| = {pool.size}")

        if not accepted:
            # nothing could be merged this round: fall back to centroid assignment
            direct = assign_edgeless_singletons(z, pool, main)
            for s, k in direct.items():
                main[k] = np.append(main[k], s)
            pool = np.zeros(0, dtype=int)
            break

    # ---- line 15 -----------------------------------------------------------
    yhat = -np.ones(n, dtype=int)
    for k, members in main.items():
        yhat[members.astype(int)] = k
    if (yhat < 0).any():
        leftovers = np.flatnonzero(yhat < 0)
        direct = assign_edgeless_singletons(z, leftovers, main)
        for s, k in direct.items():
            yhat[s] = k
    return yhat
