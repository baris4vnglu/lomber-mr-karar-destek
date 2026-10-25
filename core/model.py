"""Three-branch 2.5D CNN: one backbone per MRI series, slices stacked as channels.

Each branch sees one series (sag T1, sag T2/STIR, ax T2). Branch features are
concatenated and mapped to 25 targets x 3 severities. Keeping branches
separate lets Grad-CAM show *where on which series* the model looked.
"""

from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn as nn

from .constants import SERIES_ROLES, SLICES_PER_SERIES, SEVERITIES, TARGETS

DEFAULT_BACKBONE = "efficientnet_b0"


class LumbarNet(nn.Module):
    def __init__(self, backbone: str = DEFAULT_BACKBONE, pretrained: bool = False, dropout: float = 0.2):
        super().__init__()
        self.branches = nn.ModuleDict(
            {
                role: timm.create_model(
                    backbone, pretrained=pretrained, in_chans=SLICES_PER_SERIES[role], num_classes=0, global_pool=""
                )
                for role in SERIES_ROLES
            }
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat = self.branches[SERIES_ROLES[0]].num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat * len(SERIES_ROLES), 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, len(TARGETS) * len(SEVERITIES)),
        )
        # Filled by forward() so Grad-CAM can read the last conv maps per branch.
        self.feature_maps: dict[str, torch.Tensor] = {}

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        pooled = []
        for role in SERIES_ROLES:
            fmap = self.branches[role](inputs[role])
            if fmap.requires_grad:
                fmap.retain_grad()
            self.feature_maps[role] = fmap
            pooled.append(self.pool(fmap).flatten(1))
        logits = self.head(torch.cat(pooled, dim=1))
        return logits.view(-1, len(TARGETS), len(SEVERITIES))


def stacks_to_tensors(stacks: dict[str, np.ndarray], device: str = "cpu") -> dict[str, torch.Tensor]:
    """uint8 (C,H,W) stacks -> float tensors (1,C,H,W) in [0,1]."""
    return {role: torch.from_numpy(stacks[role]).float().div(255.0).unsqueeze(0).to(device) for role in SERIES_ROLES}
