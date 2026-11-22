"""plain v8-in comparison on the prize sheet: epoch 2 and epoch 3 at depth -2, then epoch 2 at depth 0 (p50 -> 0, p99.5 -> 1 like the ensemble).
Maps are cached in output/ensemble/mix28_ps6_s1_c3_r1000_v8ep2_studio (epoch 2 depth -2 is the ensemble's) and output/v8in_compare/."""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils import predict_models  # noqa: E402
from utils.fast_figure import colorize, orient_display  # noqa: E402

SHEET = "PRIZESHEET_11_17_2"
RUN = REPO / "nader-fork/runs/v8in_mix28_4ep_ps6_s1/runs"
CK = {n: sorted(glob.glob(str(RUN / f"*/checkpoints/trained-epoch{n - 1}.ckpt")))[-1] for n in (2, 3)}
OUT = REPO / "output" / "v8in_compare"
OUT.mkdir(parents=True, exist_ok=True)
zarr_path = REPO / "ves_zarrs2" / f"{SHEET}.zarr"
mask = cv2.imread(str(REPO / "masks/extras" / f"{SHEET}.png"), cv2.IMREAD_GRAYSCALE) > 0
ens = REPO / "output/ensemble/mix28_ps6_s1_c3_r1000_v8ep2_studio" / SHEET / "cache" / "v8_in__mix28_fine_tune__epoch_2__depth-2"
jobs = [("epoch 2, depth -2", 2, -2, ens), ("epoch 3, depth -2", 3, -2, OUT / "e3_d-2"), ("epoch 2, depth 0", 2, 0, OUT / "e2_d0")]
panels = []
for label, epoch, shift, cache in jobs:
    pred = predict_models.predict_v8in(zarr_path, CK[epoch], cache, batch_size=8, shift=shift)
    out = np.full(mask.shape, np.nan, np.float32)
    r, c = min(mask.shape[0], pred.shape[0]), min(mask.shape[1], pred.shape[1])
    out[:r, :c] = pred[:r, :c]
    valid = mask & np.isfinite(out)
    lo, hi = np.percentile(out[valid], (50, 99.5))
    plain = np.where(valid, np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1), np.nan)
    rgb = colorize(orient_display(plain, SHEET), "gray_r", (0.78, 0.86, 0.95))
    name = label.replace(", ", "_").replace(" ", "")
    cv2.imwrite(str(OUT / f"v8in_{name}.png"), rgb[..., ::-1])
    cv2.putText(rgb, label, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (200, 0, 0), 6)
    panels.append(rgb)
    print("done", label, flush=True)
gap = np.full((panels[0].shape[0], 40, 3), 255, np.uint8)
cv2.imwrite(str(OUT / "v8in_compare.png"), np.vstack([cv2.resize(p, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for p in panels])[..., ::-1])
print("saved", OUT / "v8in_compare.png")
