"""VICReg self-supervised loss (Bardes, Ponce, LeCun, 2022).

Three terms applied to the projector outputs ``z1, z2`` of shape ``(N, D)``:

- **Invariance** (``sim``): MSE between ``z1`` and ``z2`` — pulls augmented
  views of the same sample together.
- **Variance** (``std``): hinge on per-dimension std deviation over the batch,
  encouraging ``std >= 1`` per dim. Prevents representation collapse without
  needing negative samples.
- **Covariance** (``cov``): squared off-diagonal entries of the covariance
  matrix, decorrelating dimensions.

Default coefficients (25, 25, 1) are the values from the paper. Loss is
balanced numerically when the projector output dim is in the ~1k-8k range.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VICRegLoss(nn.Module):
    def __init__(
        self,
        sim_coef: float = 25.0,
        std_coef: float = 25.0,
        cov_coef: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.sim_coef = sim_coef
        self.std_coef = std_coef
        self.cov_coef = cov_coef
        self.eps = eps

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if z1.shape != z2.shape or z1.ndim != 2:
            raise ValueError(
                f"expected matching 2-D tensors, got {tuple(z1.shape)} vs {tuple(z2.shape)}"
            )
        if z1.shape[0] < 2:
            raise ValueError(f"batch must have >= 2 elements for VICReg, got {z1.shape[0]}")

        sim_loss = F.mse_loss(z1, z2)

        std_z1 = torch.sqrt(z1.var(dim=0) + self.eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.eps)
        std_loss = (F.relu(1.0 - std_z1).mean() + F.relu(1.0 - std_z2).mean()) * 0.5

        n, d = z1.shape
        z1_c = z1 - z1.mean(dim=0)
        z2_c = z2 - z2.mean(dim=0)
        cov_z1 = (z1_c.T @ z1_c) / (n - 1)
        cov_z2 = (z2_c.T @ z2_c) / (n - 1)
        # zero the diagonal in a fresh tensor (in-place would break autograd
        # on MPS in some torch versions).
        eye = torch.eye(d, device=z1.device, dtype=z1.dtype)
        off_z1 = cov_z1 - cov_z1 * eye
        off_z2 = cov_z2 - cov_z2 * eye
        cov_loss = (off_z1.pow(2).sum() / d) + (off_z2.pow(2).sum() / d)

        loss = (
            self.sim_coef * sim_loss
            + self.std_coef * std_loss
            + self.cov_coef * cov_loss
        )
        components = {
            "sim": float(sim_loss.detach().cpu()),
            "std": float(std_loss.detach().cpu()),
            "cov": float(cov_loss.detach().cpu()),
        }
        return loss, components
