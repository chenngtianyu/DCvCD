"""Clustering-aware InfoNCE fine-tuning, Section 3.4 and Equation (4).

    L = - sum_i log [ exp(s_{i,p_i} / tau)
                      / ( exp(s_{i,p_i} / tau) + sum_{j in N_i} exp(s_ij / tau) ) ]

    s_ij   = cos(z_i, z_j)
    p_i    sampled from the same main community as anchor i
    N_i    mini-batch samples carrying a different pseudo-label
    tau    = 0.5

Anchors with no valid positive in the current mini-batch are omitted from the
loss, as stated in Section 3.4.

Optimiser settings are those reported in Section 4.2: SGD, learning rate 1e-4,
momentum 0.9, weight decay 5e-4, batch size 64, 100 epochs per refinement
iteration.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

__all__ = ["clustering_infonce", "PseudoLabeledSubset", "finetune_backbone"]

TAU_NCE = 0.5


def clustering_infonce(
    z: torch.Tensor, labels: torch.Tensor, tau: float = TAU_NCE
) -> torch.Tensor:
    """Equation (4) over one mini-batch.

    Parameters
    ----------
    z : (B, d) embeddings, l2-normalised inside.
    labels : (B,) pseudo-labels, i.e. main-community identifiers.
    """
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / tau

    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=z.device)
    pos_mask = same & ~eye          # candidate positives
    neg_mask = ~same                # different pseudo-label -> N_i

    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return z.sum() * 0.0        # keeps the graph connected, contributes nothing

    # sample one positive per anchor
    probs = pos_mask.float()
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    pos_idx = torch.multinomial(probs[has_pos], num_samples=1).squeeze(1)

    rows = torch.nonzero(has_pos, as_tuple=False).squeeze(1)
    pos_logit = sim[rows, pos_idx]                       # exp(s_{i,p_i} / tau)

    neg = sim[rows].masked_fill(~neg_mask[rows], float("-inf"))
    # denominator = positive term + sum over negatives
    denom = torch.cat([pos_logit[:, None], neg], dim=1)
    loss = -(pos_logit - torch.logsumexp(denom, dim=1))
    return loss.sum()


class PseudoLabeledSubset(Dataset):
    """Samples belonging to the current main communities, with their pseudo-label."""

    def __init__(self, base_dataset, indices: np.ndarray, labels: np.ndarray):
        self.base = base_dataset
        self.indices = np.asarray(indices, dtype=int)
        self.labels = np.asarray(labels, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        x = self.base[self.indices[i]]
        if isinstance(x, (tuple, list)):
            x = x[0]
        return x, int(self.labels[i])


def finetune_backbone(
    backbone: torch.nn.Module,
    dataset: Dataset,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    tau: float = TAU_NCE,
    device: str | torch.device = "cuda",
    num_workers: int = 4,
    log_every: int = 10,
) -> torch.nn.Module:
    """One refinement iteration's worth of backbone fine-tuning."""
    backbone = backbone.to(device).train()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    opt = torch.optim.SGD(
        backbone.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    for epoch in range(epochs):
        total, seen = 0.0, 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = clustering_infonce(backbone(x), y, tau=tau)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            seen += len(y)
        if log_every and (epoch + 1) % log_every == 0:
            print(f"    epoch {epoch + 1:3d}/{epochs}  InfoNCE {total / max(seen, 1):.4f}")
    return backbone


@torch.no_grad()
def extract_embeddings(
    backbone: torch.nn.Module,
    dataset: Dataset,
    batch_size: int = 256,
    device: str | torch.device = "cuda",
    num_workers: int = 4,
) -> np.ndarray:
    """Algorithm 1 line 9: refresh embeddings for all samples."""
    backbone = backbone.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    out = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        out.append(F.normalize(backbone(x.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)
