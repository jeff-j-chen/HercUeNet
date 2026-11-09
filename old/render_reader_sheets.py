"""prediction | composite of sheets of ves_zarrs2/ with one reader-v3 checkpoint, the figure 0846A_eval.ipynb saves for a sheet that is not trained on.

  python3 render_reader_sheets.py <checkpoint> <out_dir> <sheet> [<sheet> ...] [--mask-dir masks/extras] [--composite-only]

The sheet is the zarr name (ves_zarrs2/<sheet>.zarr) with its mask in --mask-dir (default masks/extras). Predictions are cached in
_ves_tmp/0846A_inference/<model>/<sheet>/ exactly where the notebook keeps them; --composite-only skips the model and saves the
composite of the layers reader-v3 reads.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import cv2
import numpy as np
import zarr
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "old"))
Image.MAX_IMAGE_PIXELS = None

from utils import predict_models  # noqa: E402
from utils.fast_figure import colorize, gray_rgb, orient_display, save_panel_grid  # noqa: E402

ZARR_DIR = REPO / "ves_zarrs2"
CACHE_ROOT = REPO / "_ves_tmp" / "0846A_inference"


def render(sheet: str, checkpoint: Path, out_dir: Path, mask_dir: Path, composite_only: bool) -> Path:
    volume_path = ZARR_DIR / f"{sheet}.zarr"
    mask = np.asarray(Image.open(mask_dir / f"{sheet}.png").convert("L")) > 0
    volume = zarr.open(str(volume_path), mode="r")
    if tuple(volume.shape[1:]) != mask.shape:
        raise ValueError(f"{sheet}: mask {mask.shape} and volume {volume.shape} differ")
    start, end = predict_models.native_window("reader_v3", volume.shape[0], checkpoint)
    composite = np.clip(np.asarray(volume[start:end]).max(axis=0), 0, 255).astype(np.uint8) * mask
    display_composite = orient_display(composite, sheet)
    composite_panel = (f"composite {start}-{end - 1}", gray_rgb(display_composite), cv2.INTER_AREA, 9.362)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sheet}{'_composite' if composite_only else ''}.png"
    if composite_only:
        save_panel_grid([[composite_panel]], display_composite.shape, str(path), gap_px=24)
    else:
        model = f"reader_v3_{checkpoint.parent.name}_{checkpoint.stem}"
        prediction = predict_models.predict_reader(volume_path, checkpoint, CACHE_ROOT / model / sheet / "reader_v3.tif", 4)
        framed = np.full(mask.shape, np.nan, np.float32)
        rows, cols = min(mask.shape[0], prediction.shape[0]), min(mask.shape[1], prediction.shape[1])
        framed[:rows, :cols] = prediction[:rows, :cols]
        framed[~mask] = np.nan
        pred_rgb = colorize(orient_display(framed, sheet), "gray_r", (0.78, 0.86, 0.95))
        panels = [[(f"{sheet} | forward {start}-{end - 1}", pred_rgb, cv2.INTER_AREA, None), composite_panel]]
        save_panel_grid(panels, display_composite.shape, str(path), gap_px=24)
        del prediction, framed, pred_rgb
    print(f"Saved {path}", flush=True)
    gc.collect()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    parser.add_argument("sheets", nargs="+")
    parser.add_argument("--mask-dir", default="masks/extras")
    parser.add_argument("--composite-only", action="store_true")
    args = parser.parse_args()
    for sheet in args.sheets:
        render(sheet, Path(args.checkpoint).resolve(), REPO / args.out_dir, REPO / args.mask_dir, args.composite_only)


if __name__ == "__main__":
    main()
