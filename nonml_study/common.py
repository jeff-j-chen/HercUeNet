"""site crops of the prize sheet for the non-ML letter study (native resolution, padded context)."""
import os
import numpy as np
import zarr
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = "/vesuvius"
ZID = "20261102120000"
MASK = f"{ROOT}/masks/extras/fiber_only_20261102b_flatboi_PRIZESHEET.png"
BOX = 520
PAD = 160
# name -> (x, y, letter)
SITES = {"s1_easy": (2898, 910, "N"), "s2_medium": (801, 991, "O"), "s3_hard": (1696, 1466, "M")}
_cache = {}


def load_site(name, pad=PAD):
    return load_xy(*SITES[name][:2], pad)


def load_xy(x, y, pad=PAD):
    """volume (28,H,W) float32 and mask over the box plus pad; returns (vol, mask, (y0, x0) of the box inside it)."""
    key = (x, y, pad)
    if key in _cache:
        return _cache[key]
    z = zarr.open(f"{ROOT}/ves_zarrs2/{ZID}.zarr", mode="r")
    mask = np.array(Image.open(MASK).convert("L")) > 0
    H, W = mask.shape
    y0, y1, x0, x1 = max(0, y - pad), min(H, y + BOX + pad), max(0, x - pad), min(W, x + BOX + pad)
    vol = np.asarray(z[:, y0:y1, x0:x1], dtype=np.float32)
    out = (vol, mask[y0:y1, x0:x1], (y - y0, x - x0))
    _cache[key] = out
    return out


def box(a, off):
    return a[..., off[0]:off[0] + BOX, off[1]:off[1] + BOX]
