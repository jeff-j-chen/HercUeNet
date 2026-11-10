"""precomputed whole-patch asf_morph cue (nonml_ink.py's asf_morph prior), a drop-in for the CLAHE map of
utils/precomputed_cue.py (same model.input_cue = "pre_clahe" mechanism, data.cue_dir = "./cue_asf_morph").

the prior runs on the 8-slice window the model reads at every pixel (surface-relative start plus the scroll's window
offset), on the 2x area-downsampled grid as in nonml_ink.py, is robust-z scored over the papyrus, blurred 15 px, then
mapped linearly from [-Z_CLIP, Z_CLIP] to 0..1, upsampled to the volume's pixel grid and stored as uint8 (0 outside the mask).

python3 -m utils.precomputed_asf [--only 20260927134931,...] [--workers 5] [--force]
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .precomputed_cue import WINDOW, volume_id_of

CUE_DIR = "cue_asf_morph"
BAND = 256
BLUR_NATIVE = 15


LOCAL_SIGMA = 250  # native px, nonml_ink.local_z default
COARSE = 2         # the 250 px local mean/std are smooth: estimate them on a COARSE-times smaller grid


def _coarse_nblur(x, m, sigma_native, nm):
    """nm.nblur on a COARSE-times area-downsampled grid, bilinearly upsampled back (same smoothing, ~COARSE^2 cheaper)."""
    height, width = x.shape
    size = (max(1, width // COARSE), max(1, height // COARSE))
    mf = m.astype(np.float32)
    num = cv2.resize(x * mf, size, interpolation=cv2.INTER_AREA)
    den = cv2.resize(mf, size, interpolation=cv2.INTER_AREA)
    sigma = nm.w(sigma_native) / COARSE
    num = cv2.GaussianBlur(num, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(den, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(num / np.maximum(den, 1e-3), (width, height), interpolation=cv2.INTER_LINEAR)


def _slice_prior(plane, m, nm):
    """nm.priors' per-slice term: gblur(local_z(fill(plane)), 4 px), with the wide local statistics on a coarse grid."""
    x = nm.fill(plane, m)
    mu = _coarse_nblur(x, m, LOCAL_SIGMA, nm)
    sd = np.sqrt(np.maximum(_coarse_nblur((x - mu) ** 2, m, LOCAL_SIGMA, nm), 1e-6))
    return nm.gblur(nm.fill((x - mu) / sd, m), 4)


def asf_morph_prior(v, m, nm):
    """nm.priors(v, m)["asf_morph"] without the nine other priors nobody reads (MIP over every slice of `v`)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(len(v), os.cpu_count() or 1)) as pool:
        planes = list(pool.map(lambda plane: _slice_prior(plane, m, nm), v))
    base = nm.fill(np.max(planes, axis=0), m)
    return nm.gblur(nm.asf(base), 10)


def cue_from_window(stack, frame):
    """uint8 asf_morph cue map of an (WINDOW, H, W) raw slice stack over the papyrus `frame`."""
    import nonml_ink as nm

    height, width = frame.shape
    f = nm.F
    h2, w2 = height // f, width // f
    v = stack[:, :h2 * f, :w2 * f].reshape(WINDOW, h2, f, w2, f).mean((2, 4)).astype(np.float32)
    m = frame[:h2 * f, :w2 * f].reshape(h2, f, w2, f).mean((1, 3)) > 0.999
    m = cv2.erode(m.astype(np.uint8), nm.disk(24)).astype(bool)
    asf = asf_morph_prior(v, m, nm).astype(np.float32)
    z = nm.robust_z(nm.gblur(nm.robust_z(asf, m, nm.Z_CLIP), BLUR_NATIVE), m, nm.Z_CLIP)
    small = np.clip((z + nm.Z_CLIP) / (2.0 * nm.Z_CLIP), 0.0, 1.0)
    canvas = np.zeros((height, width), dtype=np.float32)
    canvas[:h2 * f, :w2 * f] = cv2.resize(small, (w2 * f, h2 * f), interpolation=cv2.INTER_LINEAR)
    cue = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    cue[~frame] = 0
    return cue


def _first_existing(*paths):
    return next((p for p in paths if os.path.exists(p)), paths[0])


def read_zone(path) -> np.ndarray:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    return np.array(Image.open(path).convert("L")) > 0


def zone_digest(zone: np.ndarray) -> str:
    """identifies a prize-sheet zone mask; stored next to every cue built from it (<volume id>.zone)."""
    import hashlib

    return hashlib.sha1(str(zone.shape).encode() + np.packbits(zone).tobytes()).hexdigest()


def build(scroll_id, force: bool = False, config=None) -> str:
    """write <cue_dir>/<volume id>.npy for one scroll (cue_dir from `config`, default the best run's)."""
    from utils.config import Config, apply_best_run

    if config is None:
        config = Config()
        apply_best_run(config)
    return build_shifts(scroll_id, {0: config}, force)


def build_shifts(scroll_id, configs: dict, force: bool = False) -> str:
    """write the cue of every depth shift in `configs` (shift -> config; same scroll, zone and frame) from ONE read of the volume.
    The read of the 8-layer windows dominates (the cue itself takes seconds), so a sheet's three shifts cost one pass over the
    layers they need."""
    import zarr
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    cv2.setNumThreads(6)
    first = next(iter(configs.values()))
    volume_id = volume_id_of(first, scroll_id)
    todo = {}
    for shift, config in configs.items():
        out_path = os.path.join(str(config.data.cue_dir), f"{volume_id_of(config, scroll_id)}.npy")
        if force or not os.path.isfile(out_path):
            todo[shift] = (config, out_path)
    if not todo:
        return f"{volume_id}: exists"
    import nonml_ink as nm

    volume = zarr.open(os.path.join(first.data.zarr_path, f"{volume_id}.zarr"), mode="r")
    depth_count, height, width = (int(v) for v in volume.shape)
    mask_path = _first_existing(f"masks/{scroll_id}.png", f"masks/extras/{scroll_id}.png")
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    frame = np.zeros((height, width), dtype=bool)
    h, w = min(height, mask.shape[0]), min(width, mask.shape[1])
    frame[:h, :w] = mask[:h, :w]
    zone_dir = str(getattr(first.data, "cue_exclusion_dir", "") or "")
    digest = ""
    if zone_dir:
        # the prize-sheet zone leaves the frame: every term of the cue is computed over the frame only (nm.fill starts from x * m)
        zone_path = os.path.join(zone_dir, f"{scroll_id}.png")
        if not os.path.isfile(zone_path):
            raise FileNotFoundError(f"{zone_path} (python3 -m utils.holdout_exclusion)")
        zone = read_zone(zone_path)
        if zone.shape != frame.shape:
            raise ValueError(f"{zone_path}: shape {zone.shape} differs from the volume frame {frame.shape}")
        digest = zone_digest(zone)
        frame &= ~zone
    surface_dir = _first_existing(os.path.join(first.data.surface_label_dir, str(scroll_id)),
                                  os.path.join(first.data.surface_label_dir, "extras", str(scroll_id)))
    depth = np.load(os.path.join(surface_dir, "depth.npy"), mmap_mode="r")
    confidence = np.load(os.path.join(surface_dir, "confidence.npy"), mmap_mode="r")
    key = int(scroll_id) if str(scroll_id).isdigit() else scroll_id
    offsets = {shift: int(config.data.surface_window_offset_by_scroll.get(key, config.data.surface_window_offset))
               for shift, (config, _) in todo.items()}
    valid_depth = np.asarray(depth[::16, ::16]).astype(np.int64)
    valid_depth = valid_depth[valid_depth != 255]
    fallback = int(np.median(valid_depth)) if valid_depth.size else first.data.train_d_start
    stacks = {shift: np.zeros((WINDOW, height, width), dtype=np.float32) for shift in todo}
    for y0 in range(0, height, BAND):
        y1 = min(height, y0 + BAND)
        d = np.asarray(depth[y0:y1, :width]).astype(np.int64)
        c = np.asarray(confidence[y0:y1, :width])
        centre = np.where((d != 255) & (c > 0), d, fallback)
        starts = {shift: np.clip(centre - (WINDOW - 1) // 2 + offset, 0, max(depth_count - WINDOW, 0))
                  for shift, offset in offsets.items()}
        low = min(int(start.min()) for start in starts.values())
        high = min(depth_count, max(int(start.max()) for start in starts.values()) + WINDOW)
        block = np.asarray(volume[low:high, y0:y1, :], dtype=np.float32)
        for shift, start in starts.items():
            for k in range(WINDOW):
                index = np.clip(start + k, 0, depth_count - 1) - low
                stacks[shift][k, y0:y1] = np.take_along_axis(block, index[None], axis=0)[0]
    for shift, (config, out_path) in todo.items():
        cue = cue_from_window(stacks.pop(shift), frame)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        temporary = out_path + ".tmp.npy"
        np.save(temporary, cue)
        os.replace(temporary, out_path)
        stamp = out_path[:-len(".npy")] + ".zone"
        if digest:
            with open(stamp + ".tmp", "w") as handle:
                handle.write(digest + "\n")
            os.replace(stamp + ".tmp", stamp)
        elif os.path.isfile(stamp):
            os.remove(stamp)
    return f"{volume_id}: {height}x{width}, shifts {sorted(todo)}"


def _build(args):
    return build(*args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="comma-separated scroll ids (default: the iter2 corpus)")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)
    ids = [int(s) for s in args.only.split(",") if s.strip()] or list(ITER2_IDS)
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(ids)))) as pool:
        for line in pool.map(_build, [(sid, args.force) for sid in ids]):
            print(f"[asf-cue] {line}", flush=True)


# new 0846A set (train + holdout/vis) and PHerc0009B
ITER2_IDS = (
    20260927134931, 20261012073246, 20261012065813,
    20261012065817, 20261012065819, 20261012065821, 20261012065823,
    20261012065815, 20261012065825, 20261012065827, 20250919125754,
)

if __name__ == "__main__":
    main()
