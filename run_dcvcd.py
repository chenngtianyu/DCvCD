"""Command-line entry point.

    python run_dcvcd.py --embeddings emb.npy --initial-labels y0.npy \
                        --out yhat.npy [--true-labels y.npy] [--seed 0]

Runs the graph side of Algorithm 1 on frozen embeddings. For the full loop with
backbone fine-tuning, drive dcvcd.pipeline.run_dcvcd from Python and pass a
finetune_fn (see README).
"""
import argparse
import numpy as np

from dcvcd.pipeline import DCvCDConfig, run_dcvcd


def evaluate(y_true, y_pred):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import (adjusted_rand_score,
                                 normalized_mutual_info_score)
    D = max(y_pred.max(), y_true.max()) + 1
    cm = np.zeros((D, D), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[p, t] += 1
    r, c = linear_sum_assignment(cm, maximize=True)
    acc = cm[r, c].sum() / len(y_true)
    return (normalized_mutual_info_score(y_true, y_pred), acc,
            adjusted_rand_score(y_true, y_pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--initial-labels", required=True)
    ap.add_argument("--true-labels", default=None)
    ap.add_argument("--out", default="dcvcd_predictions.npy")
    ap.add_argument("--theta-g", type=float, default=0.5)
    ap.add_argument("--rho", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    z = np.load(a.embeddings)
    y0 = np.load(a.initial_labels)
    cfg = DCvCDConfig(theta_g=a.theta_g, rho=a.rho, seed=a.seed,
                      verbose=not a.quiet)
    y_hat = run_dcvcd(z, y0, cfg)
    np.save(a.out, y_hat)
    print(f"saved {a.out}")

    if a.true_labels:
        y = np.load(a.true_labels)
        n0, a0, r0 = evaluate(y, y0)
        n1, a1, r1 = evaluate(y, y_hat)
        print(f"\n            NMI     ACC     ARI")
        print(f"initial   {n0*100:6.1f}  {a0*100:6.1f}  {r0*100:6.1f}")
        print(f"DCvCD     {n1*100:6.1f}  {a1*100:6.1f}  {r1*100:6.1f}")


if __name__ == "__main__":
    main()
