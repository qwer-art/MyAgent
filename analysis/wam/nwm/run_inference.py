#!/usr/bin/env python3
"""NWM single-frame inference CLI.

Inputs: 4 context images + navigation action (dx, dy, dyaw)
Output: predicted future RGB frame (224x224 PNG)

Usage (conda activate nwm):
  python analysis/wam/nwm/run_inference.py \\
    --context_dir raw_data/nwm/demo/context \\
    --action "1,0,0" \\
    --checkpoint raw_data/nwm/code/logs/nwm_cdit_s/checkpoints/cdit_s_100000.pth.tar \\
    --output /tmp/nwm_pred.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

NWM_CODE = Path(__file__).resolve().parents[3] / "raw_data" / "nwm" / "code"
if str(NWM_CODE) not in sys.path:
    sys.path.insert(0, str(NWM_CODE))

from diffusers.models import AutoencoderKL  # noqa: E402
from diffusion import create_diffusion  # noqa: E402
from isolated_nwm_infer import model_forward_wrapper  # noqa: E402
from misc import transform  # noqa: E402
from models import CDiT_models  # noqa: E402

MODEL_CONFIGS = {
    "CDiT-S/2": {"depth": 12, "hidden_size": 384, "params": "50M"},
    "CDiT-B/2": {"depth": 12, "hidden_size": 768, "params": "200M"},
    "CDiT-L/2": {"depth": 24, "hidden_size": 1024, "params": "700M"},
    "CDiT-XL/2": {"depth": 28, "hidden_size": 1152, "params": "1B"},
}


def load_context_frames(context_dir: Path, num_cond: int = 4) -> torch.Tensor:
    paths = sorted(context_dir.glob("*.png")) + sorted(context_dir.glob("*.jpg"))
    if len(paths) < num_cond:
        raise FileNotFoundError(
            f"Need {num_cond} images in {context_dir}, found {len(paths)}"
        )
    frames = []
    for p in paths[:num_cond]:
        img = Image.open(p).convert("RGB")
        frames.append(transform(img))
    return torch.stack(frames).unsqueeze(0)  # [1, T, 3, 224, 224]


def parse_action(s: str) -> torch.Tensor:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--action must be 'dx,dy,dyaw' (3 floats)")
    return torch.tensor(parts, dtype=torch.float32).unsqueeze(0)


def save_pred(tensor: torch.Tensor, path: Path) -> None:
    from misc import unnormalize

    img = unnormalize(tensor[0].detach().cpu())
    img = (img * 127.5 + 127.5).clamp(0, 255).byte()
    arr = img.permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


def load_model(checkpoint: Path, model_name: str, device: torch.device):
    latent_size = 224 // 8
    num_cond = 4
    model = CDiT_models[model_name](context_size=num_cond, input_size=latent_size, in_channels=4)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "NWM weights are gated: request access at https://huggingface.co/facebook/nwm\n"
            "Then: hf download facebook/nwm cdit_s_100000.pth.tar "
            f"--local-dir {checkpoint.parent}"
        )
    ckp = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckp["ema"] if isinstance(ckp, dict) and "ema" in ckp else ckp
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    diffusion = create_diffusion("250")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    return model, diffusion, vae, num_cond, latent_size


def main():
    parser = argparse.ArgumentParser(description="NWM single-frame inference")
    parser.add_argument("--context_dir", type=Path, required=True)
    parser.add_argument("--action", type=str, default="1,0,0", help="dx,dy,dyaw normalized")
    parser.add_argument("--rel_t", type=float, default=0.0078125, help="time shift (default 1/128)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=NWM_CODE / "logs/nwm_cdit_s/checkpoints/cdit_s_100000.pth.tar",
    )
    parser.add_argument("--model", type=str, default="CDiT-S/2", choices=list(CDiT_models))
    parser.add_argument("--output", type=Path, default=Path("nwm_pred.png"))
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CUDA unavailable, inference will be very slow.")

    model, diffusion, vae, num_cond, latent_size = load_model(
        args.checkpoint, args.model, device
    )
    x = load_context_frames(args.context_dir, num_cond).to(device)
    y = parse_action(args.action).to(device)
    rel_t = torch.tensor([args.rel_t], device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model_forward_wrapper(
            (model, diffusion, vae),
            x,
            y,
            num_timesteps=None,
            latent_size=latent_size,
            device=device,
            num_cond=num_cond,
            num_goals=1,
            rel_t=rel_t,
            progress=True,
        )
    elapsed = time.perf_counter() - t0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_pred(pred, args.output)

    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(device) / 1024**3

    metrics = {
        "model": args.model,
        "params": MODEL_CONFIGS.get(args.model, {}).get("params", "?"),
        "latency_sec": round(elapsed, 3),
        "peak_vram_gb": round(peak_vram, 3),
        "output": str(args.output.resolve()),
        "action": args.action,
    }
    print(json.dumps(metrics, indent=2))
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
