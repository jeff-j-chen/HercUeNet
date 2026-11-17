"""Tiled inference of the released Studio model (soft-loss EMA weights) on a uint8 [16, H, W] stack.

Runs inside .venv-studio (torch, numpy, safetensors; built by stage_studio.py). The network and the tiling are those of
priyankaranke/vesuvius-studio-multiscroll-cpt (MIT), studio-multiscroll-cpt-20261023/hf/model.py and inference.py: 128 px windows, the
central 96 px written, stride 64 with a Hann weight (the tiling of the PHerc0211 submission), input clip(raw, 0, 200) / 255.

  .venv-studio/bin/python utils/predict_studio.py --input planes16.npy --output prob.npy [--checkpoint W] [--device cuda]
  .venv-studio/bin/python utils/predict_studio.py --selftest       # load the weights, run one window

planes16 must be the 16 surface planes in Studio's depth order (ascending along the render's layer index, surface at index 8; see
ensemble_eval.ipynb run_studio). Pixels whose 16 planes are all 0 (outside the sheet) are written as 0.
"""
import argparse
import math
import sys
from pathlib import Path

# utils/platform.py would shadow the stdlib module when this file is run as a script
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO / "models" / "studio" / "studio-softloss-20261023-ema.safetensors"


def norm(channels):
    return nn.GroupNorm(math.gcd(32, channels), channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()
        self.branch = nn.Sequential(norm(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1, bias=False),
                                    norm(channels), nn.SiLU(), nn.Dropout3d(dropout), nn.Conv3d(channels, channels, 3, padding=1, bias=False))

    def forward(self, x):
        return x + self.branch(x)


class DeepResidualUNet(nn.Module):
    """Studio's DeepResidualUNet(deeper=False), 223,355,921 parameters."""

    def __init__(self):
        super().__init__()
        widths, counts, drops, decoder_counts = [48, 96, 192, 384, 640], [2, 3, 4, 5, 6], [.02, .05, .10, .15, .20], [3, 3, 2, 2]
        self.stem = nn.Conv3d(1, widths[0], 3, padding=1, bias=False)
        self.enc = nn.ModuleList([nn.Sequential(*[ResidualBlock(c, p) for _ in range(n)]) for c, n, p in zip(widths, counts, drops)])
        self.down = nn.ModuleList([nn.Sequential(norm(widths[i]), nn.SiLU(),
                                                 nn.Conv3d(widths[i], widths[i + 1], 3, stride=(1 if i == 0 else 2, 2, 2), padding=1, bias=False))
                                   for i in range(4)])
        self.up = nn.ModuleList([nn.Conv3d(widths[i + 1], widths[i], 1, bias=False) for i in range(3, -1, -1)])
        self.fuse = nn.ModuleList([nn.Sequential(norm(widths[i] * 2), nn.SiLU(), nn.Conv3d(widths[i] * 2, widths[i], 1, bias=False))
                                   for i in range(3, -1, -1)])
        self.dec = nn.ModuleList([nn.Sequential(*[ResidualBlock(widths[i], drops[i]) for _ in range(n)])
                                  for i, n in zip(range(3, -1, -1), decoder_counts)])
        self.head = nn.Sequential(norm(widths[0]), nn.SiLU(), nn.Conv3d(widths[0], 1, 1))
        self.depth_logits = nn.Parameter(torch.zeros(16))

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for i, stage in enumerate(self.enc):
            x = stage(x)
            if i < 4:
                skips.append(x)
                x = self.down[i](x)
        for up, fuse, stage, skip in zip(self.up, self.fuse, self.dec, reversed(skips)):
            x = up(F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False))
            x = stage(fuse(torch.cat([x, skip], dim=1)))
        return (self.head(x) * self.depth_logits.softmax(0)[None, None, :, None, None]).sum(2)


def load_model(checkpoint, device):
    model = DeepResidualUNet()
    model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    return model.eval().to(device)


def crop(array, y, x):
    h, w = array.shape[-2:]
    tile = np.asarray(array[:, max(0, y):min(h, y + 128), max(0, x):min(w, x + 128)])
    padding = ((0, 0), (max(0, -y), max(0, y + 128 - h)), (max(0, -x), max(0, x + 128 - w)))
    return np.pad(tile, padding, mode="reflect") if any(a or b for a, b in padding) else tile


@torch.inference_mode()
def predict(model, volume, device="cuda", batch_size=8, stride=64):
    """float32 [H, W] probabilities of a uint8 [16, H, W] stack (Hann-blended windows every `stride` px)."""
    if volume.ndim != 3 or volume.shape[0] != 16 or volume.dtype != np.uint8:
        raise ValueError("expected a uint8 [16, H, W] stack")
    h, w = volume.shape[-2:]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(96) + 0.5) / 96)
    kernel = (window[:, None] * window[None, :]).astype(np.float32)
    result, weight = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    ys = sorted(set(list(range(0, h, stride)) + [max(h - 96, 0)]))
    xs = sorted(set(list(range(0, w, stride)) + [max(w - 96, 0)]))
    coords = [(y, x) for y in ys for x in xs]
    for start in range(0, len(coords), batch_size):
        batch = coords[start:start + batch_size]
        raw = np.stack([crop(volume, y - 16, x - 16) for y, x in batch])
        tensor = torch.from_numpy(raw[:, None]).to(device=device, dtype=torch.float32).clamp_(0, 200).div_(255)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            logits = model(tensor)
        probability = logits[:, 0, 16:112, 16:112].float().sigmoid().cpu().numpy()
        for (y, x), p in zip(batch, probability):
            hh, ww = min(96, h - y), min(96, w - x)
            result[y:y + hh, x:x + ww] += p[:hh, :ww] * kernel[:hh, :ww]
            weight[y:y + hh, x:x + ww] += kernel[:hh, :ww]
    result = np.where(weight > 0, result / np.maximum(weight, 1e-12), 0).astype(np.float32)
    result[(np.asarray(volume) == 0).all(0)] = 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--selftest", action="store_true", help="load the weights and run one window on random data")
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.device)
    if args.selftest:
        volume = np.random.default_rng(0).integers(1, 200, (16, 128, 128), dtype=np.uint8)
        probability = predict(model, volume, args.device, 1)
        assert probability.shape == (128, 128) and np.isfinite(probability).all() and 0 <= probability.min() and probability.max() <= 1
        print(f"studio selftest ok on {args.device}: mean probability {probability.mean():.3f}")
        return
    volume = np.load(args.input, mmap_mode="r", allow_pickle=False)
    np.save(args.output, predict(model, volume, args.device, args.batch_size))


if __name__ == "__main__":
    main()
