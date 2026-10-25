"""Train LumbarNet on the RSNA 2024 Lumbar Spine Degenerative Classification data.

Designed to run in a Kaggle notebook (GPU T4/P100, Internet ON for pretrained
weights). `build_notebook.py` inlines the shared core code so the notebook is
self-contained. Outputs (in --out-dir):
  rsna_lumbar.pth     model weights
  model_config.json   backbone, calibration temperature and validation metrics
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --- CORE IMPORTS (build_notebook.py replaces this block with the inlined core code) ---
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.constants import (  # noqa: E402
    CONDITIONS,
    IMAGE_SIZE,
    LEVELS,
    RSNA_DESCRIPTION_TO_ROLE,
    SERIES_ROLES,
    SEVERITIES,
    SEVERITY_WEIGHTS,
    SLICES_PER_SERIES,
    TARGETS,
)
from core.dicom_io import load_study_stacks  # noqa: E402
from core.model import DEFAULT_BACKBONE, LumbarNet  # noqa: E402
# --- END CORE IMPORTS ---

KAGGLE_DATA = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=KAGGLE_DATA)
    p.add_argument("--out-dir", default="/kaggle/working")
    p.add_argument("--cache-dir", default="/kaggle/working/cache")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-studies", type=int, default=0, help="debug: limit number of studies")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args, _ = p.parse_known_args()  # ignore Jupyter's own argv
    return args


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- Data preparation --------------------------------------------------------------------------


def build_labels(train_csv: pd.DataFrame) -> np.ndarray:
    sev_to_idx = {s: i for i, s in enumerate(SEVERITIES)}
    labels = np.full((len(train_csv), len(TARGETS)), -100, dtype=np.int64)
    for j, t in enumerate(TARGETS):
        col = train_csv[t].map(sev_to_idx)
        labels[:, j] = col.fillna(-100).astype(np.int64).values
    return labels


def study_series_paths(data_dir: Path, study_id: int, desc: pd.DataFrame) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for _, row in desc[desc.study_id == study_id].iterrows():
        role = RSNA_DESCRIPTION_TO_ROLE.get(row.series_description)
        if role is None or role in out:  # keep the first series of each type
            continue
        series_dir = data_dir / "train_images" / str(study_id) / str(row.series_id)
        out[role] = sorted(series_dir.glob("*.dcm"))
    return out


def _cache_one(job):
    data_dir, cache_dir, study_id, desc = job
    target = cache_dir / f"{study_id}.npz"
    if target.exists():
        return study_id, True
    try:
        stacks = load_study_stacks(study_series_paths(data_dir, study_id, desc))
        np.savez(target, **stacks)
        return study_id, True
    except Exception as e:  # corrupt/odd studies are skipped, not fatal
        print(f"[cache] study {study_id} failed: {e}")
        return study_id, False


def build_cache(data_dir: Path, cache_dir: Path, study_ids: list[int], desc: pd.DataFrame, workers: int) -> set[int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(data_dir, cache_dir, sid, desc) for sid in study_ids]
    ok = set()
    t0 = time.time()
    with Pool(max(1, workers)) as pool:
        for i, (sid, success) in enumerate(pool.imap_unordered(_cache_one, jobs, chunksize=4), 1):
            if success:
                ok.add(sid)
            if i % 100 == 0 or i == len(jobs):
                print(f"[cache] {i}/{len(jobs)} studies ({time.time() - t0:.0f}s)")
    return ok


# --- Dataset & augmentation --------------------------------------------------------------------


def random_affine(x: torch.Tensor, max_rot=10, max_scale=0.1, max_shift=0.08) -> torch.Tensor:
    """Same random affine for all slices of one series. x: (C,H,W). No flips (left/right labels)."""
    ang = math.radians(random.uniform(-max_rot, max_rot))
    s = 1.0 + random.uniform(-max_scale, max_scale)
    tx, ty = random.uniform(-max_shift, max_shift), random.uniform(-max_shift, max_shift)
    theta = torch.tensor(
        [[math.cos(ang) / s, -math.sin(ang) / s, tx], [math.sin(ang) / s, math.cos(ang) / s, ty]], dtype=x.dtype
    ).unsqueeze(0)
    grid = F.affine_grid(theta, (1, *x.shape), align_corners=False)
    return F.grid_sample(x.unsqueeze(0), grid, align_corners=False, padding_mode="zeros").squeeze(0)


def random_intensity(x: torch.Tensor) -> torch.Tensor:
    gamma = random.uniform(0.8, 1.25)
    x = x.clamp(0, 1) ** gamma
    return (x * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)).clamp(0, 1)


class StudyDataset(Dataset):
    def __init__(self, study_ids, labels, cache_dir: Path, train: bool):
        self.study_ids, self.labels, self.cache_dir, self.train = list(study_ids), labels, cache_dir, train

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, i):
        data = np.load(self.cache_dir / f"{self.study_ids[i]}.npz")
        x = {}
        for role in SERIES_ROLES:
            t = torch.from_numpy(data[role]).float() / 255.0
            if self.train:
                if random.random() < 0.8:
                    t = random_affine(t)
                if random.random() < 0.8:
                    t = random_intensity(t)
            x[role] = t
        return x, torch.from_numpy(self.labels[i])


# --- Metrics -----------------------------------------------------------------------------------


def weighted_log_loss(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> float:
    w = torch.tensor(SEVERITY_WEIGHTS, device=logits.device)
    return F.cross_entropy(
        (logits / temperature).reshape(-1, 3), labels.reshape(-1), weight=w, ignore_index=-100
    ).item()


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Rank-based AUROC (Mann-Whitney U), no sklearn needed."""
    pos, neg = y_true == 1, y_true == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks for ties
    for v in np.unique(y_score):
        m = y_score == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Temperature scaling on the validation set so probabilities are better calibrated."""
    log_t = torch.zeros(1, requires_grad=True, device=logits.device)
    w = torch.tensor(SEVERITY_WEIGHTS, device=logits.device)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(
            (logits / log_t.exp()).reshape(-1, 3), labels.reshape(-1), weight=w, ignore_index=-100
        )
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def condition_metrics(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> dict:
    probs = torch.softmax(logits / temperature, dim=-1).cpu().numpy()
    y = labels.cpu().numpy()
    out = {}
    for ci, cond in enumerate(CONDITIONS):
        sl = slice(ci * len(LEVELS), (ci + 1) * len(LEVELS))
        p, t = probs[:, sl].reshape(-1, 3), y[:, sl].reshape(-1)
        valid = t >= 0
        p, t = p[valid], t[valid]
        out[cond] = {
            "auroc_moderate_or_severe": auroc((t >= 1).astype(int), p[:, 1] + p[:, 2]),
            "auroc_severe": auroc((t == 2).astype(int), p[:, 2]),
            "n": int(valid.sum()),
            "prevalence_moderate_or_severe": float((t >= 1).mean()) if len(t) else None,
        }
    return out


# --- Training loop -----------------------------------------------------------------------------


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = {k: v.to(device, non_blocking=True) for k, v in x.items()}
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        all_logits.append(logits.float())
        all_labels.append(y.to(device))
    return torch.cat(all_logits), torch.cat(all_labels)


def main():
    args = parse_args()
    seed_everything(args.seed)
    data_dir, out_dir, cache_dir = Path(args.data_dir), Path(args.out_dir), Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  backbone={args.backbone}")

    train_csv = pd.read_csv(data_dir / "train.csv")
    desc = pd.read_csv(data_dir / "train_series_descriptions.csv")
    if args.max_studies:
        train_csv = train_csv.head(args.max_studies)

    ok = build_cache(data_dir, cache_dir, train_csv.study_id.tolist(), desc, args.num_workers)
    train_csv = train_csv[train_csv.study_id.isin(ok)].reset_index(drop=True)
    labels = build_labels(train_csv)
    print(f"studies with cached images: {len(train_csv)}")

    idx = np.random.permutation(len(train_csv))
    n_val = max(1, int(len(idx) * args.val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    ids = train_csv.study_id.values
    dl_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_dl = DataLoader(StudyDataset(ids[tr_idx], labels[tr_idx], cache_dir, True), shuffle=True, drop_last=True, **dl_kw)
    val_dl = DataLoader(StudyDataset(ids[val_idx], labels[val_idx], cache_dir, False), shuffle=False, **dl_kw)

    pretrained = not args.no_pretrained
    try:
        model = LumbarNet(args.backbone, pretrained=pretrained)
    except Exception as e:
        print(f"!! pretrained weights could not be loaded ({e}); is Internet enabled? training from scratch")
        pretrained = False
        model = LumbarNet(args.backbone, pretrained=False)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = max(1, args.epochs * len(train_dl))
    warmup = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_w = torch.tensor(SEVERITY_WEIGHTS, device=device)

    best_loss, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, (x, y) in enumerate(train_dl, 1):
            x = {k: v.to(device, non_blocking=True) for k, v in x.items()}
            y = y.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, 3), y.reshape(-1), weight=loss_w, ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        val_logits, val_labels = predict(model, val_dl, device)
        val_loss = weighted_log_loss(val_logits, val_labels)
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={running / max(1, len(train_dl)):.4f}  "
            f"val_wlogloss={val_loss:.4f}  ({time.time() - t0:.0f}s)"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_logits, val_labels = predict(model, val_dl, device)
    temperature = fit_temperature(val_logits, val_labels)
    metrics = {
        "val_weighted_logloss": weighted_log_loss(val_logits, val_labels),
        "val_weighted_logloss_calibrated": weighted_log_loss(val_logits, val_labels, temperature),
        "per_condition": condition_metrics(val_logits, val_labels, temperature),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx)),
    }
    print(json.dumps(metrics, indent=2))

    torch.save(best_state, out_dir / "rsna_lumbar.pth")
    config = {
        "backbone": args.backbone,
        "pretrained_init": pretrained,
        "temperature": temperature,
        "image_size": IMAGE_SIZE,
        "slices_per_series": SLICES_PER_SERIES,
        "targets": TARGETS,
        "severities": SEVERITIES,
        "epochs": args.epochs,
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "dataset": "RSNA 2024 Lumbar Spine Degenerative Classification",
        "metrics": metrics,
    }
    (out_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"saved -> {out_dir / 'rsna_lumbar.pth'} and model_config.json")


if __name__ == "__main__":
    main()
