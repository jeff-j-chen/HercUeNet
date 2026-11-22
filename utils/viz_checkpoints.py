"""Run every saved checkpoint of one model on a sheet (default: the prize sheet), cache the maps, and make ONE image per model.

  python3 -m utils.viz_checkpoints reader|studio|v8in [--sheet PRIZESHEET_11_19_flatboi] [--run NAME] [--ckpt CKPT ...]

Checkpoints (the mix28x_new _ps7 runs, utils/mix6.py RUN_TAG):
  reader  reader-v3/runs/<run>/ckpt_<step>.pth                                  run: mix28x_new_ps7_s1
  studio  models/studio/<run>/step_<n>.safetensors                              run: mix28x_new_ps7
  v8in    nader-fork/runs/<run>/runs/*/checkpoints/trained-epoch<N>.ckpt        run: v8in_mix28x_new_4ep_ps7_s1   (epoch N + 1)
Maps are cached as float32 .npy in output/prize_<date>/cache/<model>/<label>.npy (re-used while newer than the checkpoint), panels as PNGs in
output/prize_<date>/<model>/, and the overview of all checkpoints of the model is output/prize_<date>/<model>_checkpoints.png (stretch as in
ensemble_eval.ipynb: reader-v3 and v8-in p50 -> 0 and p99.5 -> 1, Studio p1 -> p99.5; no depth maxing, depth 0).
Inference: reader-v3 / v8-in through utils/predict_models.py (their own virtualenvs), Studio through utils/predict_studio.py in .venv-studio
(planes 5..20 of the render, stride 64 + Hann, like ensemble_eval.ipynb run_studio, but read straight from the zarr: no /dev/shm copy).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from utils import mix6, predict_models  # noqa: E402
from utils.fast_figure import colorize, orient_display  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
RUNS = {"reader": "mix28x_new_ps7_s1", "studio": "mix28x_new_ps7", "v8in": "v8in_mix28x_new_4ep_ps7_s1"}
FLAT_DEPTH, STUDIO_FIRST_PLANE = 13, 5


CKPT, STEP, EPOCH = r'ckpt_(\d+)', r'step_(\d+)', r'trained-epoch(\d+)'


def number(pattern: str, text: str) -> int:
    return int(re.search(pattern, text).group(1))


def checkpoints(model: str, run: str) -> list[tuple[str, Path]]:
    """[(label, path)] sorted by step / epoch."""
    if model == "reader":
        found = [(f"step {number(CKPT, p)}", Path(p)) for p in glob.glob(str(REPO / "reader-v3/runs" / run / "ckpt_*.pth"))]
    elif model == "studio":
        found = [(f"step {number(STEP, p)}", Path(p)) for p in glob.glob(str(REPO / "models/studio" / run / "step_*.safetensors"))]
    else:
        found = [(f"epoch {number(EPOCH, p) + 1} (trained-epoch{number(EPOCH, p)})", Path(p))
                 for p in glob.glob(str(REPO / "nader-fork/runs" / run / "runs/*/checkpoints/trained-epoch*.ckpt"))]
    return sorted(found, key=lambda item: int(item[0].split()[1]))


def studio_map(zarr_path: Path, ckpt: Path, out: Path) -> np.ndarray:
    python = REPO / ".venv-studio" / "bin" / "python"
    volume = predict_models.read_volume(zarr_path, STUDIO_FIRST_PLANE, STUDIO_FIRST_PLANE + 16)
    with tempfile.TemporaryDirectory(dir=REPO / "_ves_tmp") as tmp:
        np.save(Path(tmp) / "planes16.npy", np.clip(volume, 0, 255).astype(np.uint8))
        del volume
        subprocess.run([str(python), str(REPO / "utils" / "predict_studio.py"), "--input", str(Path(tmp) / "planes16.npy"),
                        "--output", str(Path(tmp) / "prob.npy"), "--checkpoint", str(ckpt), "--batch-size", os.environ.get("VIZ_BATCH", "8")], check=True)
        return np.load(Path(tmp) / "prob.npy").astype(np.float32)


def predict(model: str, zarr_path: Path, ckpt: Path, scratch: Path) -> np.ndarray:
    if model == "reader":
        return predict_models.predict_reader(zarr_path, ckpt, scratch.with_suffix(".tif"), batch_size=int(os.environ.get("VIZ_BATCH", 4)))
    if model == "v8in":
        return predict_models.predict_v8in(zarr_path, ckpt, scratch, batch_size=int(os.environ.get("VIZ_BATCH", 8)))
    return studio_map(zarr_path, ckpt, scratch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(RUNS))
    parser.add_argument("--sheet", default=mix6.PRIZE_SHEET)
    parser.add_argument("--run", default=None)
    parser.add_argument("--ckpt", nargs="*", default=[], help="explicit checkpoints instead of the run's (label = file stem)")
    parser.add_argument("--out", default=None, help="output folder (default output/prize_<sheet>)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    run = args.run or RUNS[args.model]
    items = [(f"{Path(c).resolve().parent.name} {Path(c).stem}", Path(c).resolve()) for c in args.ckpt] or checkpoints(args.model, run)
    if not items:
        raise SystemExit(f"no checkpoints of {args.model} found for run {run}")
    out = Path(args.out).resolve() if args.out else REPO / "output" / f"prize_{args.sheet}"  # absolute: the reader CLI runs in its own folder
    cache, panels = out / "cache" / args.model, out / args.model
    for folder in (cache, panels):
        folder.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = open(out / f".viz_{args.model}.lock", "w")  # one visualization per model and output folder at a time (a watcher and a queue may both call this)
    fcntl.flock(lock, fcntl.LOCK_EX)
    zarr_path = REPO / "ves_zarrs2" / f"{args.sheet}.zarr"
    mask = cv2.imread(str(REPO / "masks" / "extras" / f"{args.sheet}.png"), cv2.IMREAD_GRAYSCALE) > 0
    tiles = []
    for label, ckpt in items:
        slug = re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")
        path = cache / f"{slug}.npy"
        if path.is_file() and path.stat().st_mtime > max(ckpt.stat().st_mtime, mix6.zarr_mtime(zarr_path)):
            raw = np.load(path)
            print(f"[viz] {args.model} {label}: cached {path}", flush=True)
        else:
            print(f"[viz] {args.model} {label}: predicting with {ckpt}", flush=True)
            raw = predict(args.model, zarr_path, ckpt, cache / f"_scratch_{slug}").astype(np.float32)
            np.save(path, raw)
        full = np.full(mask.shape, np.nan, np.float32)
        rows, cols = min(mask.shape[0], raw.shape[0]), min(mask.shape[1], raw.shape[1])
        full[:rows, :cols] = raw[:rows, :cols]
        valid = mask & np.isfinite(full)
        low, high = np.percentile(full[valid], (50, 99.5) if args.model != "studio" else (1, 99.5))
        plain = np.where(valid, np.clip((full - low) / max(float(high - low), 1e-6), 0, 1), np.nan)
        rgb = colorize(orient_display(plain, args.sheet), "gray_r", (0.78, 0.86, 0.95))
        cv2.imwrite(str(panels / f"{slug}.png"), rgb[..., ::-1])
        stats = f"mean {np.nanmean(full[valid]):.3f}, >0.5 {100 * np.mean(full[valid] > 0.5):.1f}%"
        tiles.append((Image.fromarray(rgb), f"{args.title or args.model} {label}   ({stats})"))
        for leftover in cache.glob(f"_scratch_{slug}*"):
            if leftover.is_file():
                leftover.unlink()
    width = 1500
    from utils.fast_figure import _font

    resized = []
    for tile, caption in tiles:
        tile = tile.resize((width, int(tile.height * width / tile.width)), Image.Resampling.LANCZOS)
        labelled = Image.new("RGB", (width, tile.height + 44), (255, 255, 255))  # a caption bar above the map, drawn at the final size
        labelled.paste(tile, (0, 44))
        ImageDraw.Draw(labelled).text((8, 6), caption, fill=(0, 0, 0), font=_font(28))
        resized.append(labelled)
    tiles = resized
    columns = 1 if len(tiles) <= 2 else 2
    height = max(t.height for t in tiles) + 10
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * (width + 10), rows * height), (255, 255, 255))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * (width + 10), (index // columns) * height))
    target = out / f"{args.model}_checkpoints.png"
    sheet.save(target)
    print(f"[viz] saved {target} ({len(tiles)} checkpoints)", flush=True)


if __name__ == "__main__":
    main()
