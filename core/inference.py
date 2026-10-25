"""Local inference: calibrated per-level probabilities + Grad-CAM heatmaps."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .constants import (
    CONDITION_LABELS_TR,
    CONDITION_TO_ROLE,
    CONDITIONS,
    IMAGE_SIZE,
    LEVEL_LABELS,
    LEVELS,
    SERIES_ROLE_LABELS,
    SEVERITY_LABELS_TR,
    TARGETS,
)
from .model import LumbarNet, stacks_to_tensors

WEIGHTS_FILE = "rsna_lumbar.pth"
CONFIG_FILE = "model_config.json"


def _jet(x: np.ndarray) -> np.ndarray:
    """x in [0,1] -> RGB uint8 using a jet-like colormap."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _png_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Predictor:
    def __init__(self, models_dir: Path, device: str = "cpu"):
        self.models_dir = Path(models_dir)
        self.device = device
        self.model: LumbarNet | None = None
        self.config: dict = {}
        self.load()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        weights, cfg = self.models_dir / WEIGHTS_FILE, self.models_dir / CONFIG_FILE
        if not (weights.exists() and cfg.exists()):
            self.model, self.config = None, {}
            return
        self.config = json.loads(cfg.read_text(encoding="utf-8"))
        model = LumbarNet(self.config.get("backbone", "efficientnet_b0"), pretrained=False)
        model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
        self.model = model.to(self.device).eval()

    @property
    def temperature(self) -> float:
        return float(self.config.get("temperature", 1.0))

    def model_info(self) -> dict:
        if not self.ready:
            return {"ready": False}
        m = self.config.get("metrics", {})
        return {
            "ready": True,
            "backbone": self.config.get("backbone"),
            "dataset": self.config.get("dataset"),
            "trained_at": self.config.get("trained_at"),
            "temperature": self.temperature,
            "n_train": m.get("n_train"),
            "n_val": m.get("n_val"),
            "val_weighted_logloss": m.get("val_weighted_logloss_calibrated"),
            "per_condition": m.get("per_condition", {}),
        }

    # --- prediction ----------------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, stacks: dict[str, np.ndarray]) -> list[dict]:
        assert self.model is not None
        logits = self.model(stacks_to_tensors(stacks, self.device))[0]
        probs = torch.softmax(logits / self.temperature, dim=-1).cpu().numpy()
        findings = []
        for ti, target in enumerate(TARGETS):
            cond = next(c for c in CONDITIONS if target.startswith(c + "_"))
            level = target[len(cond) + 1 :]
            p = probs[ti]
            findings.append(
                {
                    "target": target,
                    "condition": cond,
                    "condition_tr": CONDITION_LABELS_TR[cond],
                    "level": level,
                    "level_label": LEVEL_LABELS[level],
                    "p_normal_mild": round(float(p[0]), 4),
                    "p_moderate": round(float(p[1]), 4),
                    "p_severe": round(float(p[2]), 4),
                    "p_moderate_or_severe": round(float(p[1] + p[2]), 4),
                    "predicted_grade": SEVERITY_LABELS_TR[int(p.argmax())],
                }
            )
        findings.sort(key=lambda f: (CONDITIONS.index(f["condition"]), LEVELS.index(f["level"])))
        return findings

    def gradcam(self, stacks: dict[str, np.ndarray], target: str) -> dict:
        """Grad-CAM for P(moderate or severe) of one target on its most relevant series."""
        assert self.model is not None
        ti = TARGETS.index(target)
        cond = next(c for c in CONDITIONS if target.startswith(c + "_"))
        role = CONDITION_TO_ROLE[cond]

        inputs = stacks_to_tensors(stacks, self.device)
        for t in inputs.values():
            t.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(inputs)[0]
            p = torch.softmax(logits[ti] / self.temperature, dim=-1)
            (p[1] + p[2]).backward()

        fmap = self.model.feature_maps[role]
        grads = fmap.grad
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * fmap).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam / cam.max() if cam.max() > 0 else cam

        # Slice the model was most sensitive to (input-gradient magnitude per channel).
        slice_idx = int(inputs[role].grad[0].abs().mean(dim=(1, 2)).argmax().item())
        base = stacks[role][slice_idx].astype(np.float32) / 255.0
        base_rgb = np.repeat((base * 255).astype(np.uint8)[..., None], 3, axis=-1)
        alpha = (0.55 * cam)[..., None]
        overlay = (base_rgb * (1 - alpha) + _jet(cam) * alpha).astype(np.uint8)

        level = target[len(cond) + 1 :]
        return {
            "target": target,
            "title": f"{CONDITION_LABELS_TR[cond]} – {LEVEL_LABELS[level]}",
            "series": SERIES_ROLE_LABELS[role],
            "slice_index": slice_idx,
            "original_png": _png_b64(base_rgb),
            "overlay_png": _png_b64(overlay),
        }

    def series_previews(self, stacks: dict[str, np.ndarray]) -> list[dict]:
        out = []
        for role, stack in stacks.items():
            mid = stack[len(stack) // 2]
            out.append(
                {
                    "series": SERIES_ROLE_LABELS[role],
                    "empty": bool(stack.max() == 0),
                    "png": _png_b64(np.repeat(mid[..., None], 3, axis=-1)),
                }
            )
        return out
