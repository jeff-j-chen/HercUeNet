"""reader-v3 (mix28 step 2000) and HercUeNet (mix28 epoch 4) of the mix28_h4_r2000_v1 set on the prize sheet at depths -2..+2, and one figure:

  row 1  reader-v3 at depth -2 .. +2              row 2  HercUeNet at depth -2 .. +2
  row 3  reader-v3: depth 0 | max | z-score | mid-boost | top-2 mean of the 5 depths      row 4  the same for HercUeNet

depth d shifts the window the model reads by d slices: reader-v3 slices 6+d..22+d (the 17 central + d, the volume is shifted before inference), HercUeNet
slices 9+d..16+d (surface offset -1+d) with the asf cue computed from that shifted window (cue_asf_morph_d*/).

python3 -m utils.prize_depth_grid cues|reader|hercuenet|figure
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _path in (REPO, REPO / "old"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
os.chdir(REPO)

from utils import mix6  # noqa: E402

DEPTHS = (-2, -1, 0, 1, 2)
READER_CKPT = REPO / "reader-v3" / "runs" / mix6.READER_RUN / "ckpt_002000.pth"
CUE_CKPT = mix6.CUE_EPOCH_PATHS[4]
# DEPTH_GRID_SHEET: another sheet of ensemble_eval.ipynb; one without a flat alias gets its cue from the shifted window and its norm from the ensemble cache
SHEET = os.environ.get("DEPTH_GRID_SHEET", mix6.HOLDOUT_RENDER)
ALIAS = mix6.HOLDOUT if SHEET == mix6.HOLDOUT_RENDER else None
OUT = REPO / "output" / "prize_depth_grid" if ALIAS else REPO / "output" / "prize_depth_grid_sheets" / SHEET
PREDS = OUT / "preds"
WORK = REPO / "_ves_tmp" / "prize_depth_grid" / ("" if ALIAS else SHEET)
RENDER = REPO / "ves_zarrs2" / f"{SHEET}.zarr"
ENSEMBLE_CACHE = REPO / "output" / "ensemble" / "mix28_h4_r2000_v1_ps2" / SHEET / "cache"
CUE_DIRS = {-2: "cue_asf_morph_dm2", -1: "cue_asf_morph_dm1", 0: "cue_asf_morph", 1: "cue_asf_morph_dp1", 2: "cue_asf_morph_dp2"}
FLAT_DEPTH, DEPTH_OFFSET, EVAL_STRIDE, CELL = 13, -1, 32, 16
READER_START = 6                  # first of the 17 central slices of the 28
BAD = (0.78, 0.86, 0.95)
STRETCH_HIGH = 99.5
BATCH = int(os.environ.get("DEPTH_GRID_BATCH", 16))  # a smaller batch leaves VRAM to a job sharing the GPU; the maps are the same


def _stale(path: Path) -> bool:
    """missing, or older than the render (a re-made prize sheet)."""
    from utils import predict_models

    return not path.exists() or path.stat().st_mtime < predict_models.zarr_mtime(RENDER)


def _mask() -> np.ndarray:
    import cv2

    return cv2.imread(str(REPO / "masks" / "extras" / f"{SHEET}.png"), cv2.IMREAD_GRAYSCALE) > 0


def to_frame(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape, np.nan, np.float32)
    rows, cols = min(mask.shape[0], array.shape[0]), min(mask.shape[1], array.shape[1])
    out[:rows, :cols] = array[:rows, :cols]
    out[~mask] = np.nan
    return out


# ---- asf cues of the shifted windows -----------------------------------------------------------------------------------------
def _build_cue(delta: int) -> str:
    from utils import precomputed_asf
    from utils.config import Config

    config = Config()  # mix6.cue_config only knows -1, 0, +1
    config.data.surface_window_offset_by_scroll = {mix6.HOLDOUT: mix6.window_offset(mix6.HOLDOUT) + delta}
    config.data.cue_dir = f"./{CUE_DIRS[delta]}"
    return f"shift {delta:+d}: " + precomputed_asf.build(mix6.HOLDOUT, force=True, config=config)


def cue_path(depth: int) -> Path:
    return REPO / CUE_DIRS[depth] / f"{mix6.HOLDOUT}.npy" if ALIAS else PREDS / f"cue_d{depth:+d}.npy"


def build_cues() -> None:
    todo = [d for d in DEPTHS if _stale(cue_path(d))]
    print(f"asf cues to build: {todo}", flush=True)
    if todo and not ALIAS:
        from utils import predict_models
        from utils.precomputed_asf import cue_from_window

        PREDS.mkdir(parents=True, exist_ok=True)
        volume = predict_models.read_volume(predict_models.local_zarr(RENDER))
        mask = _mask()
        for depth in todo:
            start = FLAT_DEPTH - 3 + DEPTH_OFFSET + depth  # the 8 slices the model reads at this depth
            np.save(cue_path(depth), cue_from_window(volume[start:start + 8].astype(np.float32), mask))
            print(f"[asf-cue] shift {depth:+d}: slices {start}-{start + 7}", flush=True)
    elif todo:
        with ProcessPoolExecutor(max_workers=len(todo), mp_context=multiprocessing.get_context("spawn")) as pool:
            for line in pool.map(_build_cue, todo):
                print("[asf-cue]", line, flush=True)


# ---- reader-v3 ---------------------------------------------------------------------------------------------------------------
def run_reader() -> None:
    from utils import predict_models

    PREDS.mkdir(parents=True, exist_ok=True)
    zarr_path = RENDER
    volume = None
    for depth in DEPTHS:
        if depth == 0:  # the ensemble's own prediction
            predict_models.predict_reader(zarr_path, READER_CKPT, ENSEMBLE_CACHE / "reader_v3__mix28_fine_tune__step_2000.tif", min(BATCH, 4))
            continue
        tif = PREDS / f"reader_d{depth:+d}.tif"
        scratch = WORK / f"shift{depth:+d}.zarr"
        if _stale(tif):
            import shutil

            shutil.rmtree(scratch, ignore_errors=True)
            tif.unlink(missing_ok=True)
            tif.with_suffix(".npy").unlink(missing_ok=True)
        if not tif.is_file():
            import zarr

            volume = predict_models.read_volume(predict_models.local_zarr(zarr_path)) if volume is None else volume
            index = np.clip(np.arange(volume.shape[0]) + depth, 0, volume.shape[0] - 1)
            scratch.parent.mkdir(parents=True, exist_ok=True)
            zarr.save_array(str(scratch), volume[index], chunks=(volume.shape[0], 512, 512), compressor=None)
        predict_models.predict_reader(scratch, READER_CKPT, tif, min(BATCH, 4))
        print(f"[reader] depth {depth:+d} done", flush=True)


def reader_map(depth: int, mask: np.ndarray) -> np.ndarray:
    tif = ENSEMBLE_CACHE / "reader_v3__mix28_fine_tune__step_2000.tif" if depth == 0 else PREDS / f"reader_d{depth:+d}.tif"
    return to_frame(np.load(tif.with_suffix(".npy")).astype(np.float32) / 255.0, mask)


# ---- HercUeNet ---------------------------------------------------------------------------------------------------------------
def run_hercuenet(checkpoint: Path = CUE_CKPT, depths=DEPTHS, prefix: str = "hercuenet") -> None:
    import types

    import torch
    import zarr

    from utils.chunk_cache import WindowVolume
    from utils.config import Config, ScrollConfig, resolve_run_config
    from utils.model import create_model
    from utils.norm import UNIFIED_CACHE_PATH, load_cached_norm
    from utils.platform import get_zarr_dir
    from utils.visualizer import TensorboardVisualizer, predict_tiles

    PREDS.mkdir(parents=True, exist_ok=True)
    todo = [d for d in depths if _stale(PREDS / f"{prefix}_d{d:+d}.npy") or (PREDS / f"{prefix}_d{d:+d}.npy").stat().st_mtime < checkpoint.stat().st_mtime]
    if not todo:
        return
    run_config, _ = resolve_run_config(checkpoint)
    C = Config()
    for section in ("data", "model"):
        for key, value in run_config[section].items():
            if section == "data" and key == "scrolls":
                C.data.scrolls = [ScrollConfig(**item) for item in value]
            elif key != "probe_rois" and hasattr(getattr(C, section), key):
                setattr(getattr(C, section), key, value)
    C.model.compile_model = False
    C.tra.supcon = C.tra.dann = False
    C.device = "cuda" if torch.cuda.is_available() else "cpu"
    C.data.eval_stride = EVAL_STRIDE
    C.data.eval_infer_bs = BATCH
    C.data.surface_window_offset_by_scroll = {}
    C.data.zarr_path = str(Path(get_zarr_dir()))
    depth_count = int(C.data.depth)
    model, _ = create_model(C)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module.").removeprefix("_orig_mod."): value for key, value in state.get("model_state_dict", state).items()}
    model.load_state_dict({key: value for key, value in state.items() if not key.startswith(("supcon_head.", "domain_head."))}, strict=True)
    model.eval()

    mask = _mask()
    height, width = mask.shape
    source = zarr.open(str(RENDER), mode="r")
    start = FLAT_DEPTH - (depth_count - 1) // 2 + DEPTH_OFFSET
    volume = WindowVolume(source, start + min(depths), start + depth_count + max(depths))
    if ALIAS:
        norm = load_cached_norm(ALIAS, UNIFIED_CACHE_PATH, mode=str(C.data.norm_mode))
    else:  # the flat-surface anchor norm that ensemble_eval.ipynb cached for this sheet
        norm = tuple(json.loads((ENSEMBLE_CACHE / "flat_anchor_norm.json").read_text())["norm"])
    coords = TensorboardVisualizer._gen_tile_coords(types.SimpleNamespace(c=C), (C.data.d_start, C.data.d_end), (0, height), (0, width), mask,
                                                    z_step=depth_count)
    for depth in todo:
        C.data.surface_window_offset = DEPTH_OFFSET + depth
        cue = np.load(cue_path(depth), mmap_mode="r")
        grid = predict_tiles(C, model, volume, mask, coords, (0, height), (0, width), C.data.d_start, "prize", *norm,
                             surface_depth_map=np.where(mask, FLAT_DEPTH, 255).astype(np.uint8),
                             surface_confidence_map=np.where(mask, 255, 0).astype(np.uint8), surface_depth_offset=DEPTH_OFFSET + depth, cue_map=cue)
        np.save(PREDS / f"{prefix}_d{depth:+d}.npy", np.array(grid, dtype=np.float32))
        print(f"[hercuenet] depth {depth:+d} done", flush=True)


def hercuenet_map(depth: int, mask: np.ndarray, prefix: str = "hercuenet") -> np.ndarray:
    grid = np.load(PREDS / f"{prefix}_d{depth:+d}.npy")
    return to_frame(np.repeat(np.repeat(grid, CELL, axis=0), CELL, axis=1), mask)


# ---- figure ------------------------------------------------------------------------------------------------------------------
METHODS = ("max", "z-score", "mid-boost", "mean")


def figure() -> None:
    import cv2

    from utils import depth_ensemble
    from utils.fast_figure import colorize, orient_display, save_panel_grid

    mask = _mask()
    sheet = SHEET
    models = {
        "reader-v3 mix28 step 2000": ([reader_map(d, mask) for d in DEPTHS], 50.0, lambda d: f"slices {READER_START + d}-{READER_START + 16 + d}"),
        "HercUeNet mix28 epoch 4": ([hercuenet_map(d, mask) for d in DEPTHS], 1.0, lambda d: f"slices {9 + d}-{16 + d}, own asf cue"),
    }

    def limits(array, floor):
        values = array[np.isfinite(array)]
        return tuple(float(v) for v in np.percentile(values, (floor, STRETCH_HIGH)))

    def panel(title, array, low, high, floor):
        shown = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
        rgb = colorize(orient_display(shown, sheet), "gray_r", BAD, vmin=0.0, vmax=1.0)
        return (f"{title} | p{floor:g}->0 ({low:.3f}), p{STRETCH_HIGH:g}->1 ({high:.3f})", rgb, cv2.INTER_AREA, None)

    depth_rows, ensemble_rows = [], []
    for name, (maps, floor, window) in models.items():
        low, high = limits(maps[DEPTHS.index(0)], floor)  # every depth is shown on depth 0's scale
        depth_rows.append([panel(f"{name} | depth {d:+d} ({window(d)})", array, low, high, floor) for d, array in zip(DEPTHS, maps)])
        stack = np.stack(maps)
        row = [panel(f"{name} | depth 0 alone", maps[DEPTHS.index(0)], low, high, floor)]
        for method in METHODS:
            combined = depth_ensemble.combine(stack, method)
            np.save(PREDS / f"{name.split()[0].lower()}_{method}.npy", combined.astype(np.float16))
            row.append(panel(f"{name} | {depth_ensemble.describe(list(DEPTHS), method)}", combined, *limits(combined, floor), floor))
        ensemble_rows.append(row)
    path = OUT / "prize_depth_grid_4x5.png"
    save_panel_grid([depth_rows[0], depth_rows[1], ensemble_rows[0], ensemble_rows[1]], mask.shape, str(path), gap_px=24)
    print(f"Saved {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["cues", "reader", "hercuenet", "figure"])
    step = parser.parse_args().step
    {"cues": build_cues, "reader": run_reader, "hercuenet": run_hercuenet, "figure": figure}[step]()


if __name__ == "__main__":
    main()
