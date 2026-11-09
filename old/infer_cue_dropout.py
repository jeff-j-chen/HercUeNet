"""cue_dropout inference on rendered sheets without a flat alias (masks under masks/extras), as ensemble_eval.ipynb runs it:
flat depth-13 surface, window offset -1, surface-anchor norm and pre_clahe cue computed from the render.

  python3 infer_cue_dropout.py fiber_only_20261029d_flatboi_in_1 [more sheets...]

writes output/cue_dropout/<sheet>/{prediction.npy, prediction.png, composite.png} (prediction shown at the render's pixel grid).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "old"))  # nonml_ink.py, imported by utils.precomputed_asf

from utils import predict_models
from utils.fast_figure import colorize, gray_rgb, orient_display, save_panel_grid
from utils.mix6 import MODEL_PATHS, zarr_mtime

FLAT_DEPTH, OFFSET, EVAL_STRIDE, INFER_BATCH = 13, -1, 32, 16


def load_model(checkpoint):
    from utils.config import Config, ScrollConfig, resolve_run_config
    from utils.model import create_model
    from utils.platform import get_zarr_dir

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
    C.data.eval_infer_bs = INFER_BATCH
    C.data.surface_window_offset = OFFSET
    C.data.surface_window_offset_by_scroll = {}
    C.data.zarr_path = str(Path(get_zarr_dir()))
    if int(C.data.depth) != 8 or not C.model.surface_teacher_input or not C.data.surface_relative_depth_window:
        raise ValueError("Select a literal-surface, eight-slice checkpoint")
    if str(C.data.norm_mode) not in ("surface_anchor", "surface_anchor_knee"):
        raise ValueError(f"norm mode {C.data.norm_mode} needs a stored norm; only surface-anchor norms are computed here")
    model, _ = create_model(C)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model_state_dict", state.get("state_dict", state))
    state = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith(("supcon_head.", "domain_head."))}
    model.load_state_dict(state, strict=True)
    return C, model.eval()


def anchor_norm(zid, zarr_path, mask_dir, surface_depth, surface_confidence, cache):
    from utils.norm import ANCHOR_GAP_LEVEL, ANCHOR_PAPYRUS_LEVEL, compute_surface_anchors

    path = cache / "flat_anchor_norm.json"
    if path.is_file() and path.stat().st_mtime >= zarr_mtime(zarr_path):
        return tuple(json.loads(path.read_text())["norm"])
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        (Path(temporary) / zid).mkdir()
        np.save(Path(temporary) / zid / "depth.npy", surface_depth)
        np.save(Path(temporary) / zid / "confidence.npy", surface_confidence)
        anchors = compute_surface_anchors(zid, str(zarr_path.parent), surface_dir=temporary, mask_dir=str(mask_dir))
    span = (anchors["papyrus"] - anchors["gap"]) / (ANCHOR_PAPYRUS_LEVEL - ANCHOR_GAP_LEVEL)
    lower = anchors["gap"] - ANCHOR_GAP_LEVEL * span
    norm = (0.0, 1.0, lower, lower + span)
    path.write_text(json.dumps({"norm": norm, "anchors": anchors}, indent=2))
    return norm


def cached_prediction(zid, checkpoint):
    """the saved prediction of zid when it is newer than its render and was made by this checkpoint, else None."""
    out = REPO / "output" / "cue_dropout" / zid
    path, stamp = out / "prediction.npy", out / "checkpoint.txt"
    if (path.is_file() and stamp.is_file() and stamp.read_text() == str(Path(checkpoint).resolve())
            and path.stat().st_mtime >= max(zarr_mtime(REPO / "ves_zarrs2" / f"{zid}.zarr"), Path(checkpoint).stat().st_mtime)):
        return np.load(path).astype(np.float32)
    return None


def run(C, model, zid, mask_dir, checkpoint):
    from utils.precomputed_asf import cue_from_window
    from utils.visualizer import TensorboardVisualizer, predict_tiles

    zarr_path = REPO / "ves_zarrs2" / f"{zid}.zarr"
    mask = cv2.imread(str(mask_dir / f"{zid}.png"), cv2.IMREAD_GRAYSCALE) > 0
    out = REPO / "output" / "cue_dropout" / zid
    out.mkdir(parents=True, exist_ok=True)
    volume = predict_models.read_volume(predict_models.local_zarr(zarr_path))
    volume.setflags(write=False)
    if volume.shape[1:] != mask.shape:
        raise ValueError(f"{zid}: volume {volume.shape} vs mask {mask.shape}")
    height, width = mask.shape
    depth = int(C.data.depth)
    surface_depth = np.where(mask, FLAT_DEPTH, 255).astype(np.uint8)
    surface_confidence = np.where(mask, 255, 0).astype(np.uint8)
    norm = anchor_norm(zid, zarr_path, mask_dir, surface_depth, surface_confidence, out)
    start = FLAT_DEPTH - (depth - 1) // 2 + OFFSET
    cue = None
    if str(C.model.input_cue) == "pre_clahe":
        cue = cue_from_window(np.asarray(volume[start:start + depth], dtype=np.float32), mask)
    coords = TensorboardVisualizer._gen_tile_coords(
        types.SimpleNamespace(c=C), (C.data.d_start, C.data.d_end), (0, height), (0, width), mask, z_step=depth)
    grid = np.asarray(predict_tiles(
        C, model, volume, mask, coords, (0, height), (0, width), C.data.d_start, zid, *norm,
        surface_depth_map=surface_depth, surface_confidence_map=surface_confidence,
        surface_depth_offset=OFFSET, cue_map=cue), dtype=np.float32)
    cell = int(C.model.multitile_subtile)
    prediction = np.repeat(np.repeat(grid, cell, 0), cell, 1)[:height, :width]
    framed = np.full(mask.shape, np.nan, np.float32)
    framed[:prediction.shape[0], :prediction.shape[1]] = prediction
    framed[~mask] = np.nan
    np.save(out / "prediction.npy", framed.astype(np.float16))
    (out / "checkpoint.txt").write_text(str(Path(checkpoint).resolve()))
    composite = (np.clip(volume[start:start + depth].max(axis=0), 0, 255).astype(np.uint8) * mask)
    pred_rgb = colorize(orient_display(framed, zid), "gray_r", (0.78, 0.86, 0.95))
    cv2.imwrite(str(out / "prediction.png"), pred_rgb[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
    cv2.imwrite(str(out / "composite.png"), orient_display(composite, zid))
    save_panel_grid([[(f"{zid} | cue_dropout flat {FLAT_DEPTH} offset {OFFSET:+d}", pred_rgb, cv2.INTER_AREA, None),
                      (f"composite {start}-{start + depth - 1}", gray_rgb(orient_display(composite, zid)), cv2.INTER_AREA, 9.362)]],
                    composite.shape, str(out / "figure.png"), gap_px=24)
    print(f"[done] {zid} -> {out}", flush=True)
    return framed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sheets", nargs="+")
    parser.add_argument("--checkpoint", default=str(MODEL_PATHS["cue_dropout"]))
    parser.add_argument("--mask-dir", default=str(REPO / "masks" / "extras"))
    args = parser.parse_args()
    C, model = load_model(Path(args.checkpoint).resolve())
    for zid in args.sheets:
        run(C, model, zid, Path(args.mask_dir), args.checkpoint)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
