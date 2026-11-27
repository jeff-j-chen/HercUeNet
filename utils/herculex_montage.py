"""reader-v3 on every HerculeX sheet (corpus mix28x additions: 0846A marks + 0846B partials, trained or not) -> one labelled montage PNG.

  python3 -m utils.herculex_montage [--ckpt reader-v3/runs/mix28x_ps6_s1/ckpt_002000.pth] [--scale 0.2] [--width 14000]

Each map is stretched p50 -> 0, p99.5 -> 1 over its own papyrus mask (as ensemble_eval.ipynb does for reader-v3); sheets are shown at one common scale.
Maps are cached in output/herculex_reader/<ckpt label>/; output/herculex_reader/montage_<label>.png.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils import mix6, predict_models  # noqa: E402
from utils.fast_figure import colorize  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=str(REPO / "reader-v3/runs/mix28x_ps6_s1/ckpt_002000.pth"))
parser.add_argument("--scale", type=float, default=0.2)
parser.add_argument("--width", type=int, default=14000)
parser.add_argument("--ink", action="store_true", help="draw the export inklabels in red over the maps (cached maps are reused)")
args = parser.parse_args()
ckpt = Path(args.ckpt).resolve()
label = f"{ckpt.parent.name}_{ckpt.stem}"
out = REPO / "output" / "herculex_reader" / label
out.mkdir(parents=True, exist_ok=True)
names = [f"PHerc0846A_marks_{n:03d}" for n in range(1, 56) if n != 38] + [f"PHerc0846B_partial_{n:03d}" for n in range(1, 48)]
names = [n for n in names if (REPO / "ves_zarrs2" / f"{n}.zarr").is_dir() and (REPO / "masks/extras" / f"{n}.png").is_file()]
dropped = {mix6.HERCULEX_SHEETS[int(a)][0] for a in mix6.HERCULEX_DROPPED["all"]} | {mix6.HERCULEX_EXCLUDED_NAME}
print(len(names), "sheets", flush=True)
tiles = []
for i, name in enumerate(names):
    mask = cv2.imread(str(REPO / "masks/extras" / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 0
    pred = predict_models.predict_reader(REPO / "ves_zarrs2" / f"{name}.zarr", ckpt, out / f"{name}.tif", batch_size=4)
    full = np.full(mask.shape, np.nan, np.float32)
    r, c = min(mask.shape[0], pred.shape[0]), min(mask.shape[1], pred.shape[1])
    full[:r, :c] = pred[:r, :c]
    valid = mask & np.isfinite(full)
    lo, hi = np.percentile(full[valid], (50, 99.5))
    plain = np.where(valid, np.clip((full - lo) / max(hi - lo, 1e-6), 0, 1), np.nan)
    rgb = colorize(plain, "gray_r", (0.78, 0.86, 0.95))
    if args.ink:
        export = "HerculeX_marks_export_0846" + name.split("0846")[1][0]
        label_img = cv2.imread(str(REPO / "tifxyz" / export / "inklabels" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        ink = np.zeros(mask.shape, bool)
        rr, cc = min(mask.shape[0], label_img.shape[0]), min(mask.shape[1], label_img.shape[1])
        ink[:rr, :cc] = label_img[:rr, :cc] >= 208
        rgb = rgb.copy()
        rgb[ink] = (0.45 * rgb[ink] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
    rgb = cv2.resize(rgb, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
    text = name.replace("PHerc0846", "") + (" (held out)" if name in dropped else "")
    bar = np.full((34, rgb.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    tiles.append(np.vstack([bar, rgb]))
    print(f"[{i + 1}/{len(names)}] {name}", flush=True)
# shelf packing: rows of tiles up to --width wide, each row as tall as its tallest tile
rows, row, used = [], [], 0
for tile in tiles:
    if row and used + tile.shape[1] + 10 > args.width:
        rows.append(row)
        row, used = [], 0
    row.append(tile)
    used += tile.shape[1] + 10
rows.append(row)
height = sum(max(t.shape[0] for t in r) + 10 for r in rows)
canvas = np.full((height, args.width, 3), 255, np.uint8)
y = 0
for r in rows:
    x = 0
    for tile in r:
        canvas[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
        x += tile.shape[1] + 10
    y += max(t.shape[0] for t in r) + 10
path = out.parent / f"montage_{label}{'_ink' if args.ink else ''}.png"
cv2.imwrite(str(path), canvas[..., ::-1])  # colorize gives RGB, cv2 writes BGR
print("saved", path, canvas.shape, flush=True)
