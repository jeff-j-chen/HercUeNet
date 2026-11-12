"""figures and numbers of utils/prize_depth_tests.py from its saved maps (output/prize_depth_tests/<test>_{reader,cue}.npy).

HercUeNet's map has NaN cells (outside the eroded papyrus the sequence model does not predict), so the limits, the statistics
and the product are taken over the finite pixels only; the first version of the tests took them over the whole mask.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from utils import prize_depth_tests as t

NAMES = {"0_baseline": "baseline: flat prize sheet, depths as they are", "1_surface": "detected surface supervision",
         "2_reversed": "depths reversed", "3_shuffled": "depths shuffled"}


def main() -> None:
    mask = cv2.imread(str(t.REPO / "masks" / "extras" / f"{t.PRIZE}.png"), cv2.IMREAD_GRAYSCALE) > 0
    volume = t.predict_models.read_volume(t.predict_models.local_zarr(t.REPO / "ves_zarrs2" / f"{t.PRIZE}.zarr"))
    depth, confidence = np.load(t.SURFACE / "depth.npy"), np.load(t.SURFACE / "confidence.npy")
    valid_surface = (depth != 255) & (confidence > 0) & mask
    d_eff = np.where(valid_surface, depth, int(np.median(depth[valid_surface]))).astype(np.int16)

    def composite(reader_volume):
        return np.clip(reader_volume[t.READER_WINDOW[0]:t.READER_WINDOW[1]].max(axis=0), 0, 255).astype(np.uint8) * mask

    composites = {key: composite(volume) for key in NAMES}  # the max over a window does not depend on its order
    straightened = volume.copy()
    for layer in range(*t.READER_WINDOW):
        straightened[layer] = t.gather(volume, np.clip(layer + d_eff - t.READER_SURFACE_LAYER, 0, 27))
    composites["1_surface"] = composite(straightened)
    del straightened, volume

    maps = {key: tuple(np.load(t.OUT / f"{key}_{kind}.npy").astype(np.float32) for kind in ("reader", "cue")) for key in NAMES}
    base = maps["0_baseline"]
    limits = [np.percentile(m[mask & np.isfinite(m)], (1, 99.5)) for m in base]

    def stretched(array, index):
        low, high = limits[index]
        return np.clip((array - low) / max(float(high - low), 1e-6), 0, 1).astype(np.float32)

    base_product = stretched(base[0], 0) * stretched(base[1], 1)
    product_limit = float(np.percentile(base_product[mask & np.isfinite(base_product)], 99.5))
    stats = {}
    for key, (reader_map, cue_map) in maps.items():
        a, b = stretched(reader_map, 0), stretched(cue_map, 1)
        product = a * b
        tiles = [
            (f"reader-v3 mix18 step 1000 | {NAMES[key]}", t.colorize(t.orient_display(a, t.PRIZE), "gray_r", t.BAD, vmin=0.0, vmax=1.0), t.cv2.INTER_AREA, None),
            (f"cue_dropout mix26 epoch 2 | {NAMES[key]}", t.colorize(t.orient_display(b, t.PRIZE), "gray_r", t.BAD, vmin=0.0, vmax=1.0), t.cv2.INTER_AREA, None),
            (f"product of the two | shown 0-{product_limit:.2f} (baseline p99.5)",
             t.colorize(t.orient_display(product, t.PRIZE), "gray_r", t.BAD, vmin=0.0, vmax=product_limit), t.cv2.INTER_AREA, None),
            ("ct composite: max of the 17 layers fed to reader-v3", t.gray_rgb(t.orient_display(composites[key], t.PRIZE)), t.cv2.INTER_AREA, 9.362),
        ]
        t.save_panel_grid([tiles[:2], tiles[2:]], mask.shape, str(t.OUT / f"{key}_2x2.png"), gap_px=24)
        row = {}
        for label, current, reference, view in (("reader", reader_map, base[0], a), ("cue", cue_map, base[1], b)):
            valid = mask & np.isfinite(current) & np.isfinite(reference)
            row[f"{label}_mean_stretched"] = float(view[valid].mean())
            row[f"{label}_share_above_baseline_p90"] = float((current[valid] > np.percentile(reference[valid], 90)).mean())
            row[f"{label}_r_with_baseline"] = float(np.corrcoef(reference[valid][::50], current[valid][::50])[0, 1])
        stats[key] = row
        print(f"[{key}] " + ", ".join(f"{k} {v:.3f}" for k, v in row.items()), flush=True)
    (t.OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[done] {t.OUT}", flush=True)


if __name__ == "__main__":
    main()
