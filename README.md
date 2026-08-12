# DCvCD — reference implementation

Reference implementation of the method described in *Deep Clustering via Gradual
Community Detection* (DCvCD).

## Please read this first

**This is a reference implementation written from the method description in the
paper. It is not the code that produced the results in Tables 1–5.** That code
was lost when the AutoDL instance was reclaimed. Every function here follows the
equations and the algorithm as published, and the structural quantities are unit
tested against independent implementations, but no number in the paper was
produced by this code.

Two consequences, and both matter:

1. **Do not describe this as "the implementation used in our experiments."** A
   truthful availability statement reads something like: *"A reference
   implementation of the method described in this paper is available at <URL>.
   It was written from the algorithm description; the original experimental code
   is no longer available."*

2. **Run it before you attach it to the paper.** If it reproduces the reported
   numbers within a reasonable margin, say so and the release is a genuine asset.
   If it does not, that is something you need to know before a reviewer finds
   out, not after.

## What is implemented

Everything maps onto the paper section by section.

| Paper | Module | Notes |
|---|---|---|
| §3.2, edge rule `w_ij = cos(z_i,z_j)` if `>= θ_g` | `dcvcd/graph.py: build_similarity_graph` | θ_g = 0.5 |
| Eq. (1), modularity | `dcvcd/graph.py: modularity` | verified against `networkx` |
| §3.2, Leiden per initial cluster | `dcvcd/communities.py: leiden_partition` | `leidenalg`, modularity objective |
| §3.2, ρ-quantile risk screening | `dcvcd/communities.py: risk_screening` | ρ = 0.90, distance to centroid |
| §3.3, `k̄(C)` and `Δk̄` | `dcvcd/graph.py` | size-weighted baseline as published |
| §3.3, `ΔQ = Q(P_{k←ℓ}) − Q(P)` | `dcvcd/graph.py: delta_modularity` | actual modularity difference |
| Eq. (3), `t_kℓ` | `dcvcd/merging.py: feature_distance` | mean pairwise Euclidean distance |
| Eq. (2), `L_kℓ` | `dcvcd/merging.py: merging_scores` | ε = 1e-12, maxima over the available candidates `A_k` |
| Alg. 1 line 12, greedy non-conflicting matching | `dcvcd/merging.py: greedy_merge` | one isolated community per main community per iteration |
| Alg. 1 line 13, edgeless singletons | `dcvcd/merging.py: assign_edgeless_singletons` | nearest main-community centroid |
| §3.4, Eq. (4), clustering-aware InfoNCE | `dcvcd/finetune.py: clustering_infonce` | τ = 0.5, anchors without a positive are skipped |
| §4.2, optimiser | `dcvcd/finetune.py: finetune_backbone` | SGD, lr 1e-4, momentum 0.9, wd 5e-4, batch 64, 100 epochs per iteration |
| Alg. 1 lines 7–14, the while loop | `dcvcd/pipeline.py: run_dcvcd` | refreshes embeddings and rebuilds the graph each iteration |

## Install

```bash
pip install numpy scipy torch python-igraph leidenalg
pip install networkx        # tests only
```

## Test

```bash
python tests/test_math.py       # 9 checks on the structural quantities
python tests/test_pipeline.py   # end-to-end on synthetic clusters
```

`test_math.py` checks modularity against `networkx`, `k̄(C)` against a direct
double sum, that ρ = 0.90 screening removes exactly the farthest decile, that
`Δk̄` is zero without cross edges, that `ΔQ` is positive when a split community is
re-merged, and that the greedy matching never lets one community be used twice.

## Run on your own data

The graph side runs on embeddings and an initial partition, so you can drive it
from whichever initial method you are refining:

```python
import numpy as np
from dcvcd.pipeline import DCvCDConfig, run_dcvcd

embeddings_path = ""
initial_labels_path = ""
true_labels_path = ""
output_path = ""

z = np.load(embeddings_path)       # (n, d) from CC / TCL / DivClust / ProPos / CoNR
y0 = np.load(initial_labels_path)  # (n,) assignments from the same method

y_hat = run_dcvcd(z, y0, DCvCDConfig(theta_g=0.5, rho=0.90, seed=0))
```

To run the full algorithm including backbone fine-tuning, pass a `finetune_fn`
that performs Algorithm 1 lines 8–9 and returns refreshed embeddings:

```python
from dcvcd.finetune import PseudoLabeledSubset, extract_embeddings, finetune_backbone

def finetune_fn(main_communities, iteration):
    idx = np.concatenate(list(main_communities.values()))
    lab = np.concatenate([[k] * len(v) for k, v in main_communities.items()])
    finetune_backbone(backbone, PseudoLabeledSubset(dataset, idx, lab),
                      epochs=100, batch_size=64, lr=1e-4,
                      momentum=0.9, weight_decay=5e-4, device="cuda")
    return extract_embeddings(backbone, dataset, device="cuda")

y_hat = run_dcvcd(z, y0, DCvCDConfig(), finetune_fn=finetune_fn)
```

`backbone` should be the initial method's encoder returning an embedding, not
logits. ResNet-34 for ProPos and ResNet-18 for CoNR, per §4.2.

## What is deliberately left out

Loading the five initial methods' checkpoints and datasets. Those are the
authors' own environment, and stubbing them would be guessing. Supply
`embeddings.npy` and `initial_labels.npy` from whichever method you are
refining.

## Reproduction checklist

Before attaching this repository to the paper:

- [ ] Runs end to end on at least one benchmark with one initial method
- [ ] The refined accuracy is close to the corresponding row of Table 2
- [ ] Ten seeds, and the reported standard deviations are of the same order
- [ ] Ablations reproduce the ordering in Table 3 (removing ΔQ hurts most)
- [ ] Runtime is consistent with Table 4
- [ ] Availability statement says this is a reference implementation
