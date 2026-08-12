"""End-to-end smoke test on synthetic clusters (frozen embeddings, no backbone).

Checks that Algorithm 1 terminates, empties U, assigns every sample, and does not
degrade a deliberately corrupted initial partition.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dcvcd.graph import l2_normalize
from dcvcd.pipeline import DCvCDConfig, run_dcvcd
from scipy.optimize import linear_sum_assignment

def acc(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    cm = np.zeros((D, D), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[p, t] += 1
    r, c = linear_sum_assignment(cm, maximize=True)
    return cm[r, c].sum() / len(y_true)

rng = np.random.default_rng(0)
K, n_per, d = 5, 60, 32
centres = rng.normal(size=(K, d)) * 4
y_true = np.repeat(np.arange(K), n_per)
z = l2_normalize(centres[y_true] + rng.normal(size=(K * n_per, d)) * 0.9)

# corrupt 18% of the initial assignment
y_init = y_true.copy()
flip = rng.choice(len(y_init), size=int(0.18 * len(y_init)), replace=False)
y_init[flip] = rng.integers(0, K, size=flip.size)

a0 = acc(y_true, y_init)
y_hat = run_dcvcd(z, y_init, DCvCDConfig(seed=0, verbose=True))
a1 = acc(y_true, y_hat)

assert (y_hat >= 0).all(), "every sample must receive a label"
print(f"\ninitial ACC {a0:.4f} -> refined ACC {a1:.4f}")
assert a1 >= a0 - 1e-9, "refinement must not degrade the initial partition"
print("ok  pipeline terminates, assigns everything, does not degrade")
