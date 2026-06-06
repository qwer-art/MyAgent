#!/usr/bin/env python3
"""LeWM latent-space planning inference CLI.

Inputs: initial observation + goal observation images
Output: planned action sequence (JSON) + metrics

Usage (conda activate lewm, PYTHONNOUSERSITE=1):
  python analysis/wam/lewm/run_inference.py \\
    --init_image raw_data/le-wm/demo/init.png \\
    --goal_image raw_data/le-wm/demo/goal.png \\
    --checkpoint_dir raw_data/le-wm/checkpoints/hf_tworooms \\
    --output_dir /tmp/lewm_out
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2 as transforms

LEWM_ROOT = Path(__file__).resolve().parents[3] / "raw_data" / "le-wm"
if str(LEWM_ROOT) not in sys.path:
    sys.path.insert(0, str(LEWM_ROOT))

import stable_pretraining as spt  # noqa: E402
from jepa import JEPA  # noqa: E402
from module import ARPredictor, Embedder, MLP  # noqa: E402

IMAGENET = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}


def build_model(cfg: dict) -> JEPA:
    enc_cfg = cfg["encoder"]
    encoder = spt.backbone.utils.vit_hf(
        enc_cfg["size"],
        patch_size=enc_cfg["patch_size"],
        image_size=enc_cfg["image_size"],
        pretrained=False,
        use_mask_token=False,
    )

    def make_mlp(key: str) -> MLP:
        c = cfg[key]
        norm = torch.nn.BatchNorm1d if "BatchNorm" in c["norm_fn"]["_target_"] else torch.nn.LayerNorm
        return MLP(
            input_dim=c["input_dim"],
            output_dim=c["output_dim"],
            hidden_dim=c["hidden_dim"],
            norm_fn=norm,
        )

    pred_cfg = cfg["predictor"]
    ae_cfg = cfg["action_encoder"]
    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**{k: v for k, v in pred_cfg.items() if not k.startswith("_")}),
        action_encoder=Embedder(
            input_dim=ae_cfg["input_dim"],
            emb_dim=ae_cfg["emb_dim"],
        ),
        projector=make_mlp("projector"),
        pred_proj=make_mlp("pred_proj"),
    )
    return model


def load_checkpoint(checkpoint_dir: Path, device: torch.device) -> JEPA:
    cfg_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "weights.pt"
    if not cfg_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Need config.json + weights.pt in {checkpoint_dir}")
    cfg = json.loads(cfg_path.read_text())
    model = build_model(cfg)
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    return model.eval().to(device)


def img_transform(size: int = 224):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**IMAGENET),
        transforms.Resize(size=size),
    ])


def load_image(path: Path, tfm) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)  # [1, 3, H, W]


def cem_plan(
    model: JEPA,
    init_px: torch.Tensor,
    goal_px: torch.Tensor,
    *,
    horizon: int = 5,
    history: int = 1,
    n_samples: int = 64,
    n_iters: int = 10,
    n_elites: int = 10,
    action_dim: int = 2,
) -> tuple[torch.Tensor, float]:
    """Simplified CEM in latent space (Two-Room: action_dim=2)."""
    device = init_px.device
    B, S = 1, n_samples

    info = {
        "pixels": init_px.view(B, 1, 1, 3, 224, 224).expand(B, S, history, 3, 224, 224),
        "goal": goal_px.view(B, 1, 1, 3, 224, 224).expand(B, S, history, 3, 224, 224),
    }

    mu = torch.zeros(B, horizon, action_dim, device=device)
    sigma = torch.ones(B, horizon, action_dim, device=device)

    best_cost = float("inf")
    best_actions = mu[0].clone()

    for _ in range(n_iters):
        noise = torch.randn(B, S, horizon, action_dim, device=device)
        candidates = (mu.unsqueeze(1) + sigma.unsqueeze(1) * noise).clamp(-1.0, 1.0)
        act_seq = torch.cat(
            [torch.zeros(B, S, history, action_dim, device=device), candidates], dim=2
        )
        with torch.no_grad():
            goal = {"pixels": goal_px, "goal": goal_px}
            goal_enc = model.encode(goal)
            info_roll = dict(info)
            info_roll["goal_emb"] = goal_enc["emb"].unsqueeze(1).expand(B, S, -1, -1)
            info_roll = model.rollout(info_roll, act_seq, history_size=history)
            cost = model.criterion(info_roll)[0]
        elite_idx = cost.topk(n_elites, largest=False).indices
        elite = candidates[0, elite_idx]
        mu = elite.mean(dim=0, keepdim=True)
        sigma = elite.std(dim=0, keepdim=True).clamp(min=0.05)
        min_c, min_i = cost.min(dim=0)
        if min_c.item() < best_cost:
            best_cost = min_c.item()
            best_actions = candidates[0, min_i].clone()

    return best_actions, best_cost


def main():
    parser = argparse.ArgumentParser(description="LeWM latent planning inference")
    parser.add_argument("--init_image", type=Path, required=True)
    parser.add_argument("--goal_image", type=Path, required=True)
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=LEWM_ROOT / "checkpoints/hf_tworooms",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("lewm_out"))
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--n_iters", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tfm = img_transform()
    init_px = load_image(args.init_image, tfm).to(device)
    goal_px = load_image(args.goal_image, tfm).to(device)

    model = load_checkpoint(args.checkpoint_dir, device)
    for p in model.parameters():
        p.requires_grad_(False)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    actions, latent_cost = cem_plan(
        model,
        init_px,
        goal_px,
        horizon=args.horizon,
        n_samples=args.n_samples,
        n_iters=args.n_iters,
    )
    elapsed = time.perf_counter() - t0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "action_sequence": actions.cpu().tolist(),
        "horizon": args.horizon,
        "action_dim": 2,
        "latent_mse_cost": latent_cost,
    }
    plan_path = args.output_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))

    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(device) / 1024**3

    metrics = {
        "model": "LeWM-tworooms",
        "params": "~15M",
        "latency_sec": round(elapsed, 3),
        "peak_vram_gb": round(peak_vram, 3),
        "latent_mse_cost": round(latent_cost, 6),
        "plan_path": str(plan_path.resolve()),
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
