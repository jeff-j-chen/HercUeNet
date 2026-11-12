"""light ensemble: the fine-tuned reader-v3 and HercUeNet only (no v8-in, no reader-v3 base), for sheets without a flat alias.

Each map is stretched to its own p1-p99.5 over the sheet mask and the two are combined as the mean of their z-scores and as a
mid-boost composite (utils/depth_ensemble.py), as in ensemble_eval.ipynb. Predictions are cached (reader-v3 under
output/ensemble_light/<sheet>/cache/, HercUeNet under output/cue_dropout/<sheet>/ via infer_cue_dropout.py).

  python3 ensemble_light.py fiber_only_abf2_20261029c_flatboi [more sheets...] [--mask-dir masks/extras]

Checkpoints default to utils/mix6.MODEL_PATHS, falling back per model to MIX6_MODEL_PATHS when that file does not exist yet.
writes output/ensemble_light/<sheet>/{ensemble_2x2.png, ensemble_z_score.png/.npy, ensemble_mid_boost.png/.npy}
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: utils/, assemble_*.py

import cv2
import numpy as np
import torch

import infer_cue_dropout
from utils import depth_ensemble, predict_models
from utils.fast_figure import colorize, orient_display, save_panel_grid
from utils.mix6 import MIX6_MODEL_PATHS, MODEL_PATHS

REPO = Path(__file__).resolve().parent.parent
STRETCH = (1.0, 99.5)
BAD = (0.78, 0.86, 0.95)


def default_checkpoint(kind):
    path = Path(MODEL_PATHS[kind])
    if path.exists():
        return path
    print(f"[light] {path} does not exist yet; using {MIX6_MODEL_PATHS[kind]}", flush=True)
    return Path(MIX6_MODEL_PATHS[kind])


def stretch(prediction, mask):
    valid = mask & np.isfinite(prediction)
    low, high = np.percentile(prediction[valid], STRETCH)
    out = np.clip((prediction - low) / max(float(high - low), 1e-6), 0, 1)
    return np.where(valid, out, np.nan).astype(np.float32), float(low), float(high)


def z_score(values):
    valid = np.isfinite(values)
    return (values - values[valid].mean()) / max(float(values[valid].std()), 1e-6)


def frame(array, mask):
    out = np.full(mask.shape, np.nan, np.float32)
    rows, cols = min(mask.shape[0], array.shape[0]), min(mask.shape[1], array.shape[1])
    out[:rows, :cols] = array[:rows, :cols]
    out[~mask] = np.nan
    return out


def run_sheet(sheet, mask_dir, reader_checkpoint, cue_checkpoint, cue_model):
    zarr_path = REPO / "ves_zarrs2" / f"{sheet}.zarr"
    mask = cv2.imread(str(mask_dir / f"{sheet}.png"), cv2.IMREAD_GRAYSCALE) > 0
    out = REPO / "output" / "ensemble_light" / sheet
    (out / "cache").mkdir(parents=True, exist_ok=True)
    print(f"[light] {sheet}: frame {mask.shape}, {int(mask.sum()):,} papyrus pixels", flush=True)

    reader = predict_models.predict_reader(zarr_path, reader_checkpoint, out / "cache" / "reader_v3.tif")
    cue = infer_cue_dropout.cached_prediction(sheet, cue_checkpoint)
    if cue is None:
        C, model = cue_model()
        cue = infer_cue_dropout.run(C, model, sheet, mask_dir, cue_checkpoint)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    maps, panels = [], []
    for label, checkpoint, prediction in (("reader-v3 fine-tune", reader_checkpoint, reader),
                                          ("cue_dropout", cue_checkpoint, cue)):
        stretched, low, high = stretch(frame(prediction, mask), mask)
        maps.append(stretched)
        print(f"    {label}: p{STRETCH[0]:g} {low:.4f}, p{STRETCH[1]:g} {high:.4f}  ({checkpoint})", flush=True)
        panels.append((f"{label} ({Path(checkpoint).parent.name}) | stretched p{STRETCH[0]:g}-p{STRETCH[1]:g}",
                       colorize(orient_display(stretched, sheet), "gray_r", BAD), cv2.INTER_AREA, None))
    composites = {"z_score": np.mean([z_score(m) for m in maps], axis=0),
                  "mid_boost": depth_ensemble.combine(np.stack(maps), "mid-boost")}
    for label, array in composites.items():
        valid = np.isfinite(array)
        low, high = np.percentile(array[valid], STRETCH)
        rgb = colorize(orient_display(array, sheet), "gray_r", BAD, vmin=float(low), vmax=float(high))
        np.save(out / f"ensemble_{label}.npy", array.astype(np.float16))
        cv2.imwrite(str(out / f"ensemble_{label}.png"), rgb[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
        panels.append((f"{label.replace('_', '-')} composite of 2 | shown p{STRETCH[0]:g}-p{STRETCH[1]:g}", rgb, cv2.INTER_AREA, None))
    grid = out / "ensemble_2x2.png"
    save_panel_grid([panels[:2], panels[2:]], panels[2][1].shape[:2], str(grid), gap_px=24)
    print(f"[light] saved {grid}", flush=True)
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sheets", nargs="+")
    parser.add_argument("--mask-dir", default=str(REPO / "masks" / "extras"))
    parser.add_argument("--reader-checkpoint", default=None)
    parser.add_argument("--cue-checkpoint", default=None)
    args = parser.parse_args()
    reader_checkpoint = Path(args.reader_checkpoint or default_checkpoint("reader_v3")).resolve()
    cue_checkpoint = Path(args.cue_checkpoint or default_checkpoint("cue_dropout")).resolve()
    loaded = {}

    def cue_model():
        if not loaded:
            loaded["model"] = infer_cue_dropout.load_model(cue_checkpoint)
        return loaded["model"]

    for sheet in args.sheets:
        run_sheet(sheet, Path(args.mask_dir), reader_checkpoint, cue_checkpoint, cue_model)


if __name__ == "__main__":
    main()
