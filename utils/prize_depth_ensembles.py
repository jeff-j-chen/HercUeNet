"""two ensembles of the prize sheet built from the 5-depth maps of utils/prize_depth_grid.py and the v8-in epoch 1 map of ensemble_eval.ipynb:

  row 1  v8-in + HercUeNet (max of the depths -2..+2)
  row 2  v8-in + HercUeNet (max of the depths) + reader-v3 (max of the depths)

columns: v8-in | HercUeNet depth max | reader-v3 depth max | z-score composite | product composite. The composites follow ensemble_eval.ipynb: z-score of the
maps as shown (reader-v3 and v8-in p50 -> 0, HercUeNet p1 -> 0, all p99.5 -> 1), product of the plain p1-p99.5 maps.

python3 -m utils.prize_depth_ensembles
"""
from __future__ import annotations

import cv2
import numpy as np

from utils import depth_ensemble, mix6, prize_depth_grid as grid
from utils.fast_figure import colorize, orient_display, save_panel_grid

V8IN_MAP = grid.ENSEMBLE_CACHE / "v8_in__mix28_fine_tune__epoch_1" / "v8in_prediction.npy"
STRETCH = (1.0, 99.5)
BAD = (0.78, 0.86, 0.95)
RED = (220, 0, 0)


def stretch(array: np.ndarray, mask: np.ndarray, floor: float) -> np.ndarray:
    valid = mask & np.isfinite(array)
    low, high = np.percentile(array[valid], (floor, STRETCH[1]))
    return np.where(valid, np.clip((array - low) / max(float(high - low), 1e-6), 0, 1), np.nan).astype(np.float32)


def z_score(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    return ((values - values[valid].mean()) / max(float(values[valid].std()), 1e-6)).astype(np.float32)


def main() -> None:
    mask = grid._mask()
    sheet = grid.SHEET
    maps = {
        "v8-in mix28 epoch 1": (grid.to_frame(np.load(V8IN_MAP).astype(np.float32), mask), 50.0),
        "HercUeNet mix28 epoch 4, max of 5 depths": (depth_ensemble.combine(np.stack([grid.hercuenet_map(d, mask) for d in grid.DEPTHS]), "max"), 1.0),
        "reader-v3 mix28 step 2000, max of 5 depths": (depth_ensemble.combine(np.stack([grid.reader_map(d, mask) for d in grid.DEPTHS]), "max"), 50.0),
    }
    shown = {name: stretch(array, mask, floor) for name, (array, floor) in maps.items()}
    plain = {name: stretch(array, mask, STRETCH[0]) for name, (array, _) in maps.items()}
    names = list(maps)

    def render(array, low=0.0, high=1.0):
        return colorize(orient_display(array, sheet), "gray_r", BAD, vmin=float(low), vmax=float(high))

    rows = []
    for title, members in (("v8-in + HercUeNet 5-depth max", names[:2]), ("v8-in + HercUeNet 5-depth max + reader-v3 5-depth max", names)):
        composites = {"z-score": np.mean([z_score(shown[name]) for name in members], axis=0),
                      "product": np.prod([plain[name] for name in members], axis=0).astype(np.float32)}
        row = []
        for name in names:
            floor = maps[name][1]
            label = f"{name} | raw p{floor:g}->0, p{STRETCH[1]:g}->1"
            row.append(((f"{label} | NOT IN THIS ENSEMBLE", RED) if name not in members else label, render(shown[name]), cv2.INTER_AREA, None))
        for label, array in composites.items():
            valid = np.isfinite(array)
            low, high = np.percentile(array[valid], STRETCH)
            np.save(grid.OUT / f"ensemble_{label}_{len(members)}models.npy", array.astype(np.float16))
            row.append((f"{title} | {label} composite" + (" of the maps as shown" if label == "z-score" else " of the plain p1-p99.5 maps"),
                        render(array, low, high), cv2.INTER_AREA, None))
        rows.append(row)
    path = grid.OUT / "ensembles_v8in_hercuenet_reader_depth_max.png"
    save_panel_grid(rows, mask.shape, str(path), gap_px=24)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
