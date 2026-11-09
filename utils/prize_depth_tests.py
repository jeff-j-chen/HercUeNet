"""one-off tests of reader-v3 and cue_dropout on the prize sheet (saved models only, nothing is trained):

  0 baseline      flat prize sheet, the depths as they are (cue_dropout flat depth 13, offset -1 = slices 9-16; reader-v3 central 17)
  1 surface       the detected surface supervision (utils/generate_surface_supervision.py, output/prize_depth_tests/surface_labels/)
                  instead of the flat one. cue_dropout reads it the way it reads a donor patch (window = tile surface depth - 3, offset 0);
                  reader-v3 takes no surface input, so its volume is straightened first: every pixel's column is shifted so the detected
                  surface sits at layer 15 (the layer its donor training crops are centred on)
  2 reversed      the depths of the window each model reads put in backwards (cue_dropout slices 9-16, reader-v3 slices 6-22)
  3 shuffled      the same windows in a random order (a different permutation per model)

cue_dropout's asf cue map is computed from the window it reads in every test (the stored one for the baseline, which is checked).
Every map is shown with the baseline's own p1-p99.5 limits, so a model that goes quiet looks quiet. Figures (2x2: reader-v3 |
cue_dropout / product | ct composite of the layers fed to the reader) and maps: output/prize_depth_tests/.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import zarr
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
REPO = Path(__file__).resolve().parent.parent
for _path in (REPO, REPO / "old"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
os.chdir(REPO)

from utils import predict_models  # noqa: E402
from utils.fast_figure import colorize, gray_rgb, orient_display, save_panel_grid  # noqa: E402

PRIZE = "fiber_only_20261102b_flatboi_PRIZESHEET"
NORM_ID = 20261102120000
CUE_CKPT = REPO / "models" / "runs" / "cue_dropout_labelsv6_mix26" / "epoch_2.pth"
READER_CKPT = REPO / "reader-v3" / "runs" / "mix18" / "ckpt_001000.pth"
OUT = REPO / "output" / "prize_depth_tests"
WORK = REPO / "_ves_tmp" / "prize_depth_tests"
SURFACE = OUT / "surface_labels" / PRIZE
ENSEMBLE_CACHE = REPO / "output" / "ensemble" / "mix26" / PRIZE / "cache"
FLAT_DEPTH, FLAT_OFFSET, CUE_DEPTH = 13, -1, 8
CUE_WINDOW = (FLAT_DEPTH + FLAT_OFFSET - (CUE_DEPTH - 1) // 2,) * 2  # start; end filled below
CUE_START = FLAT_DEPTH + FLAT_OFFSET - (CUE_DEPTH - 1) // 2          # 9: the flat window is slices 9-16
READER_WINDOW = (6, 23)                                             # the central 17 of the 28 layers
READER_SURFACE_LAYER = 15
BAD = (0.78, 0.86, 0.95)


def to_frame(array, mask):
    out = np.full(mask.shape, np.nan, np.float32)
    rows, cols = min(mask.shape[0], array.shape[0]), min(mask.shape[1], array.shape[1])
    out[:rows, :cols] = array[:rows, :cols]
    out[~mask] = np.nan
    return out


def gather(volume, index):
    """volume[index[y, x], y, x] for an (H, W) layer index."""
    return np.take_along_axis(volume, index[None], axis=0)[0]


def load_cue_model():
    from utils.config import Config, ScrollConfig, resolve_run_config
    from utils.model import create_model
    from utils.platform import get_zarr_dir

    run_config, _ = resolve_run_config(CUE_CKPT)
    C = Config()
    for section in ("data", "model"):
        for key, value in run_config[section].items():
            if section == "data" and key == "scrolls":
                C.data.scrolls = [ScrollConfig(**item) for item in value]
            elif key != "probe_rois" and hasattr(getattr(C, section), key):
                setattr(getattr(C, section), key, value)
    C.model.compile_model = False
    C.tra.supcon = False
    C.tra.dann = False
    C.device = "cuda"
    C.data.eval_stride = 32
    C.data.eval_infer_bs = 16
    C.data.surface_window_offset_by_scroll = {}
    C.data.zarr_path = str(get_zarr_dir())
    model, _ = create_model(C)
    state = torch.load(CUE_CKPT, map_location="cpu", weights_only=True)
    state = state.get("model_state_dict", state.get("state_dict", state))
    state = {key.removeprefix("module.").removeprefix("_orig_mod."): value for key, value in state.items()}
    state = {key: value for key, value in state.items() if not key.startswith(("supcon_head.", "domain_head."))}
    model.load_state_dict(state, strict=True)
    model.eval()
    return C, model


def predict_cue(C, model, volume, mask, depth_map, confidence_map, offset, cue_map, norm):
    from utils.visualizer import TensorboardVisualizer, predict_tiles

    C.data.surface_window_offset = int(offset)
    height, width = mask.shape
    coords = TensorboardVisualizer._gen_tile_coords(types.SimpleNamespace(c=C), (C.data.d_start, C.data.d_end), (0, height), (0, width),
                                                    mask, z_step=int(C.data.depth))
    grid = predict_tiles(C, model, volume, mask, coords, (0, height), (0, width), C.data.d_start, PRIZE, *norm,
                         surface_depth_map=depth_map, surface_confidence_map=confidence_map, surface_depth_offset=int(offset),
                         cue_map=cue_map)
    cell = int(C.model.multitile_subtile)
    return to_frame(np.repeat(np.repeat(np.asarray(grid, np.float32), cell, 0), cell, 1), mask)


def predict_reader(volume, name, mask):
    """reader-v3 map of an in-memory (28, H, W) volume (written as a zarr next to the other scratch files)."""
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"{name}.zarr"
    zarr.save_array(str(path), volume, chunks=(volume.shape[0], 512, 512), compressor=None)
    return to_frame(predict_models.predict_reader(path, READER_CKPT, WORK / f"{name}_reader.tif", 4), mask)


def main() -> None:
    from utils.norm import UNIFIED_CACHE_PATH, load_cached_norm
    from utils.precomputed_asf import cue_from_window

    OUT.mkdir(parents=True, exist_ok=True)
    mask = cv2.imread(str(REPO / "masks" / "extras" / f"{PRIZE}.png"), cv2.IMREAD_GRAYSCALE) > 0
    height, width = mask.shape
    volume = predict_models.read_volume(predict_models.local_zarr(REPO / "ves_zarrs2" / f"{PRIZE}.zarr"))
    C, model = load_cue_model()
    norm = load_cached_norm(NORM_ID, UNIFIED_CACHE_PATH, mode=str(C.data.norm_mode))
    flat_depth = np.where(mask, FLAT_DEPTH, 255).astype(np.uint8)
    flat_confidence = np.where(mask, 255, 0).astype(np.uint8)
    rng = np.random.default_rng(0)
    maps, composites = {}, {}

    def reader_composite(reader_volume):
        return np.clip(reader_volume[READER_WINDOW[0]:READER_WINDOW[1]].max(axis=0), 0, 255).astype(np.uint8) * mask

    # ---- 0 baseline: the maps of ensemble_eval.ipynb (cached) -------------------------------------------------------------
    reader_map = to_frame(predict_models.predict_reader(REPO / "ves_zarrs2" / f"{PRIZE}.zarr", READER_CKPT,
                                                        ENSEMBLE_CACHE / "reader_v3__mix18_fine_tune__step_1000.tif", 4), mask)
    saved = np.load(ENSEMBLE_CACHE / "cue_dropout__mix26__epoch_2.npz")
    cell = int(saved["cell"])
    cue_map = to_frame(np.repeat(np.repeat(saved["grid"], cell, 0), cell, 1), mask)
    maps["0_baseline"] = (reader_map, cue_map)
    composites["0_baseline"] = reader_composite(volume)
    stored_cue = np.load(REPO / "cue_asf_morph" / f"{NORM_ID}.npy", mmap_mode="r")
    flat_cue = cue_from_window(volume[CUE_START:CUE_START + CUE_DEPTH].astype(np.float32), mask)
    both = mask & (np.asarray(stored_cue) > 0)
    print(f"[check] cue recomputed from the flat window vs the stored cue: r = "
          f"{np.corrcoef(np.asarray(stored_cue)[both][::50], flat_cue[both][::50])[0, 1]:.4f}", flush=True)

    # ---- 1 surface supervision ----------------------------------------------------------------------------------------------
    depth, confidence = np.load(SURFACE / "depth.npy"), np.load(SURFACE / "confidence.npy")
    valid = (depth != 255) & (confidence > 0) & mask
    fill = int(np.median(depth[valid]))
    d_eff = np.where(valid, depth, fill).astype(np.int16)
    print(f"[surface] detected depth median {fill}, 5-95% {np.percentile(depth[valid], 5):.0f}-{np.percentile(depth[valid], 95):.0f}; "
          f"cue window = tile depth - 3 (about slices {fill - 3}-{fill + 4}); reader shift {fill - READER_SURFACE_LAYER:+d} layers", flush=True)
    window = np.stack([gather(volume, np.clip(d_eff + k - (CUE_DEPTH - 1) // 2, 0, 27)) for k in range(CUE_DEPTH)]).astype(np.float32)
    cue_surface = cue_from_window(window, mask)
    del window
    cue_1 = predict_cue(C, model, volume, mask, depth, confidence, 0, cue_surface, norm)
    straightened = volume.copy()
    for layer in range(*READER_WINDOW):
        straightened[layer] = gather(volume, np.clip(layer + d_eff - READER_SURFACE_LAYER, 0, 27))
    maps["1_surface"] = (predict_reader(straightened, "surface", mask), cue_1)
    composites["1_surface"] = reader_composite(straightened)
    del straightened, cue_surface

    # ---- 2 reversed and 3 shuffled windows ------------------------------------------------------------------------------------
    for key, label in (("2_reversed", "reversed"), ("3_shuffled", "shuffled")):
        cue_volume, reader_volume = volume.copy(), volume.copy()
        for target, (start, end) in ((cue_volume, (CUE_START, CUE_START + CUE_DEPTH)), (reader_volume, READER_WINDOW)):
            order = np.arange(end - start)[::-1] if label == "reversed" else rng.permutation(end - start)
            print(f"[{label}] slices {start}-{end - 1} fed in the order {(order + start).tolist()}", flush=True)
            target[start:end] = volume[start:end][order]
        cue_permuted = cue_from_window(cue_volume[CUE_START:CUE_START + CUE_DEPTH].astype(np.float32), mask)
        print(f"[{label}] cue from the permuted window vs the flat one: r = "
              f"{np.corrcoef(flat_cue[mask][::50], cue_permuted[mask][::50])[0, 1]:.4f}", flush=True)
        cue_2 = predict_cue(C, model, cue_volume, mask, flat_depth, flat_confidence, FLAT_OFFSET, cue_permuted, norm)
        maps[key] = (predict_reader(reader_volume, label, mask), cue_2)
        composites[key] = reader_composite(reader_volume)
        del cue_volume, reader_volume

    # ---- figures and numbers ------------------------------------------------------------------------------------------------------
    limits = [np.percentile(m[mask], (1, 99.5)) for m in maps["0_baseline"]]

    def stretched(array, index):
        low, high = limits[index]
        return np.clip((array - low) / max(float(high - low), 1e-6), 0, 1).astype(np.float32)

    product_limit = float(np.percentile((stretched(maps["0_baseline"][0], 0) * stretched(maps["0_baseline"][1], 1))[mask], 99.5))
    stats = {}
    for key, (reader_map, cue_map) in maps.items():
        np.save(OUT / f"{key}_reader.npy", reader_map.astype(np.float16))
        np.save(OUT / f"{key}_cue.npy", cue_map.astype(np.float16))
        a, b = stretched(reader_map, 0), stretched(cue_map, 1)
        product = a * b
        name = {"0_baseline": "baseline: flat prize sheet, depths as they are", "1_surface": "detected surface supervision",
                "2_reversed": "depths reversed", "3_shuffled": "depths shuffled"}[key]
        tiles = [
            (f"reader-v3 mix18 step 1000 | {name}", colorize(orient_display(a, PRIZE), "gray_r", BAD, vmin=0.0, vmax=1.0), cv2.INTER_AREA, None),
            (f"cue_dropout mix26 epoch 2 | {name}", colorize(orient_display(b, PRIZE), "gray_r", BAD, vmin=0.0, vmax=1.0), cv2.INTER_AREA, None),
            (f"product of the two | shown 0-{product_limit:.2f} (baseline p99.5)",
             colorize(orient_display(product, PRIZE), "gray_r", BAD, vmin=0.0, vmax=product_limit), cv2.INTER_AREA, None),
            ("ct composite: max of the 17 layers fed to reader-v3", gray_rgb(orient_display(composites[key], PRIZE)), cv2.INTER_AREA, 9.362),
        ]
        save_panel_grid([tiles[:2], tiles[2:]], mask.shape, str(OUT / f"{key}_2x2.png"), gap_px=24)
        base_reader, base_cue = maps["0_baseline"]
        stats[key] = {
            "reader_mean_stretched": float(np.nanmean(a[mask])), "cue_mean_stretched": float(np.nanmean(b[mask])),
            "reader_share_above_baseline_p90": float((reader_map[mask] > np.percentile(base_reader[mask], 90)).mean()),
            "cue_share_above_baseline_p90": float((cue_map[mask] > np.percentile(base_cue[mask], 90)).mean()),
            "reader_r_with_baseline": float(np.corrcoef(base_reader[mask][::50], reader_map[mask][::50])[0, 1]),
            "cue_r_with_baseline": float(np.corrcoef(base_cue[mask][::50], cue_map[mask][::50])[0, 1]),
        }
        print(f"[{key}] " + ", ".join(f"{k} {v:.3f}" for k, v in stats[key].items()), flush=True)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
