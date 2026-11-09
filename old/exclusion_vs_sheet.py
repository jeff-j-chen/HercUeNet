"""how much of a new PHerc0846A sheet the mix6 models were trained on, given the exclusion masks built for straightened2.

For every trained 0846A sheet, the pixels each model kept (papyrus minus its exclusion mask: 14 vox for cue_dropout and reader-v3,
25.5 vox for v8-in) are mapped to their 3D mesh points. At each point of the new sheet this measures the distance to the nearest
kept training point (and kept inked point), and reports the new sheet's area within the depth each model reads around a training
surface (cue_dropout 4, reader-v3 8, v8-in 11.5 voxels) and within the 14 / 25.5 voxel guard. straightened2 is run as the reference
the masks were built for.

  python3 exclusion_vs_sheet.py [--sheet fiber_only_20261029d_flatboi]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: utils/, assemble_*.py

import cv2
import numpy as np
from scipy.spatial import cKDTree

from assemble_test_segments import ROTATE_180, load_mesh
from utils import holdout_exclusion as hx
from utils import mix6
from utils.dataloader import split_extra_labels

REPO = Path(__file__).resolve().parent.parent
PIXEL_CM2 = (9.362e-4) ** 2
STEP = hx.GRID_PX // hx.SHEET_DENSE  # render px per dense mesh sample
# depth (vox) each model reads on either side of a training surface
READ_DEPTH = {"cue_dropout": 4.0, "reader_v3": 8.0, "v8in": 11.5}
MODEL_MASKS = {"cue_dropout": hx.EXCLUSION_DIR, "reader_v3": hx.READER_EXCLUSION_DIR, "v8in": hx.V8IN_EXCLUSION_DIR}
MODEL_RADIUS = {"cue_dropout": hx.RADIUS_VOX, "reader_v3": hx.READER_RADIUS_VOX, "v8in": hx.V8IN_RADIUS_VOX}
THRESHOLDS = (2.0, 4.0, 8.0, 11.5, 14.0, 25.5)


def dense_mesh(mesh_dir: Path, rotate: bool, factor: int = hx.SHEET_DENSE):
    """hx._mesh for any tifxyz folder: dense points (H,W,3), valid (H,W), render frame shape."""
    X, Y, Z, valid = load_mesh(str(mesh_dir))
    if rotate:
        X, Y, Z, valid = (np.ascontiguousarray(g[::-1, ::-1]) for g in (X, Y, Z, valid))
    ys, xs = np.where(valid)
    y0, x0 = max(0, ys.min() - 8), max(0, xs.min() - 8)
    y1, x1 = min(valid.shape[0], ys.max() + 9), min(valid.shape[1], xs.max() + 9)
    X, Y, Z, valid = (g[y0:y1, x0:x1] for g in (X, Y, Z, valid))
    gh, gw = X.shape
    shape = ((gh - 1) * hx.GRID_PX + 1, (gw - 1) * hx.GRID_PX + 1)
    rows, cols = np.linspace(0, gh - 1, (gh - 1) * factor + 1), np.linspace(0, gw - 1, (gw - 1) * factor + 1)
    r0, c0 = np.minimum(rows.astype(int), gh - 2), np.minimum(cols.astype(int), gw - 2)
    fr, fc = (rows - r0)[:, None], (cols - c0)[None, :]

    def up(g):
        g = g.astype(np.float64)
        return ((g[r0][:, c0] * (1 - fc) + g[r0][:, c0 + 1] * fc) * (1 - fr)
                + (g[r0 + 1][:, c0] * (1 - fc) + g[r0 + 1][:, c0 + 1] * fc) * fr)

    return np.stack([up(X), up(Y), up(Z)], -1), up(valid.astype(np.float32)) > 0.999, shape


def at_dense(image: np.ndarray, dense_shape) -> np.ndarray:
    """render-frame image sampled at the dense mesh positions (dense i -> render px i * STEP)."""
    rows = np.minimum(np.arange(dense_shape[0]) * STEP, image.shape[0] - 1)
    cols = np.minimum(np.arange(dense_shape[1]) * STEP, image.shape[1] - 1)
    return image[rows][:, cols]


def to_frame(dense: np.ndarray, shape) -> np.ndarray:
    rows = np.minimum(np.rint(np.arange(shape[0]) / STEP).astype(int), dense.shape[0] - 1)
    cols = np.minimum(np.rint(np.arange(shape[1]) / STEP).astype(int), dense.shape[1] - 1)
    return dense[rows][:, cols]


def training_points():
    """per model: (kept points, kept ink points, sheet index per kept point); plus all trained papyrus points."""
    kept = {m: ([], [], []) for m in MODEL_MASKS}
    everything = []
    names = []
    for index, (alias, (render, name)) in enumerate(mix6.SHEETS_0846A.items()):
        _r, _a, mesh = __import__("assemble_training_segments").MESHES_0846A[name]
        points, valid, shape = dense_mesh(REPO / "tifxyz" / mesh, str(render) in ROTATE_180)
        papyrus = cv2.imread(str(REPO / "masks" / f"{render}.png"), cv2.IMREAD_GRAYSCALE) > 0
        label = cv2.imread(str(REPO / "dilated_inklabels" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)
        ink = split_extra_labels(label)[0] if label is not None else np.zeros_like(papyrus)
        if papyrus.shape != shape:
            raise ValueError(f"{name}: mask {papyrus.shape} vs mesh frame {shape}")
        base = valid & at_dense(papyrus, valid.shape)
        everything.append(points[base])
        names.append(name.replace("auto_grown_20261012065803387_", "").replace("auto_grown_", ""))
        for model, directory in MODEL_MASKS.items():
            excluded = hx.load(alias, shape, directory=directory)
            keep = base & ~at_dense(excluded, valid.shape) if excluded is not None else base
            kept[model][0].append(points[keep])
            kept[model][1].append(points[keep & at_dense(ink, valid.shape)])
            kept[model][2].append(np.full(int(keep.sum()), index))
    out = {m: (np.concatenate(p), np.concatenate(i), np.concatenate(s)) for m, (p, i, s) in kept.items()}
    return out, np.concatenate(everything), names


def analyse(sheet: str, mesh_dir: Path, mask_path: Path, train, everything, names, out_dir: Path) -> dict:
    points, valid, shape = dense_mesh(mesh_dir, sheet in ROTATE_180)
    papyrus = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
    if papyrus.shape != shape:
        raise ValueError(f"{sheet}: mask {papyrus.shape} vs mesh frame {shape}")
    query = points[valid]

    def distance_map(tree):
        dense = np.full(valid.shape, np.inf, np.float32)
        dense[valid], idx = tree.query(query, distance_upper_bound=60.0)
        return to_frame(dense, shape), idx

    area = float(papyrus.sum() * PIXEL_CM2)
    result = {"sheet": sheet, "papyrus_cm2": area, "models": {}}
    all_dist, _ = distance_map(cKDTree(everything))
    result["no_exclusion"] = {f"<= {t:g}": float((papyrus & (all_dist <= t)).sum() * PIXEL_CM2) for t in THRESHOLDS}
    maps = {}
    for model, (kept, kept_ink, sheet_of) in train.items():
        dist, idx = distance_map(cKDTree(kept))
        ink_dist, _ = distance_map(cKDTree(kept_ink))
        depth = READ_DEPTH[model]
        inside = papyrus & (dist <= depth)
        # which training sheet the nearest kept point within the read depth belongs to
        dense_sheet = np.full(valid.shape, -1)
        hit = idx < len(kept)
        dense_sheet[valid] = np.where(hit, sheet_of[np.minimum(idx, len(kept) - 1)], -1)
        frame_sheet = to_frame(dense_sheet, shape)
        result["models"][model] = {
            "guard_vox": MODEL_RADIUS[model], "read_depth_vox": depth,
            "min_distance_vox": float(dist[papyrus].min()),
            "area_within_cm2": {f"<= {t:g}": float((papyrus & (dist <= t)).sum() * PIXEL_CM2) for t in THRESHOLDS},
            "ink_within_cm2": {f"<= {t:g}": float((papyrus & (ink_dist <= t)).sum() * PIXEL_CM2) for t in THRESHOLDS},
            "inside_read_depth_cm2": float(inside.sum() * PIXEL_CM2),
            "inside_read_depth_fraction": float(inside.sum() / max(papyrus.sum(), 1)),
            "inside_guard_cm2": float((papyrus & (dist <= MODEL_RADIUS[model])).sum() * PIXEL_CM2),
            "inside_read_depth_by_sheet_cm2": {names[k]: float((inside & (frame_sheet == k)).sum() * PIXEL_CM2)
                                               for k in range(len(names)) if (inside & (frame_sheet == k)).any()},
        }
        maps[model] = (dist, ink_dist)
    figure(sheet, papyrus, maps, out_dir)
    return result


def figure(sheet, papyrus, maps, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils.fast_figure import orient_display

    fig, axes = plt.subplots(len(maps), 2, figsize=(22, 7 * len(maps)))
    for row, (model, (dist, ink_dist)) in zip(axes, maps.items()):
        depth, guard = READ_DEPTH[model], MODEL_RADIUS[model]
        near = orient_display(np.where(papyrus, np.minimum(dist, 40.0), np.nan), sheet)
        image = row[0].imshow(near, cmap="magma", vmin=0, vmax=40, interpolation="nearest")
        row[0].contour(np.nan_to_num(near, nan=999.0), levels=[depth, guard], colors=["lime", "cyan"], linewidths=0.8)
        row[0].set_title(f"{model}: 3D distance to its kept training papyrus (vox, cap 40); lime = read depth {depth:g}, cyan = guard {guard:g}")
        fig.colorbar(image, ax=row[0], shrink=0.7)
        rgb = np.full(papyrus.shape + (3,), 255, np.uint8)
        rgb[papyrus] = (205, 205, 205)
        rgb[papyrus & (dist <= guard)] = (255, 200, 120)
        rgb[papyrus & (dist <= depth)] = (240, 120, 120)
        rgb[papyrus & (ink_dist <= depth)] = (150, 0, 0)
        row[1].imshow(orient_display(rgb, sheet), interpolation="nearest")
        row[1].set_title(f"{model}: within guard (orange) / within read depth (red) / trained INK within read depth (dark red)")
        for ax in row:
            ax.axis("off")
    fig.suptitle(f"{sheet}: overlap with the mix6 training supervision left by the straightened2 exclusion masks", fontsize=15)
    path = out_dir / f"{sheet}.png"
    fig.savefig(path, dpi=60, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", default="fiber_only_20261029d_flatboi")
    args = parser.parse_args()
    out_dir = REPO / "output" / "exclusion_vs_sheet"
    out_dir.mkdir(parents=True, exist_ok=True)
    train, everything, names = training_points()
    print(f"[train] {len(everything):,} trained papyrus points; kept: "
          + ", ".join(f"{m} {len(p):,}" for m, (p, _i, _s) in train.items()), flush=True)
    from assemble_training_segments import MESHES_0846A
    holdout_render = MESHES_0846A[mix6.OLD_HOLDOUT_NAME][0]
    sheets = [
        (args.sheet, REPO / "tifxyz" / args.sheet, REPO / "masks" / "extras" / f"{args.sheet}.png"),
        ("straightened2 (reference)", REPO / "tifxyz" / MESHES_0846A[mix6.OLD_HOLDOUT_NAME][2], REPO / "masks" / f"{holdout_render}.png"),
    ]
    results = []
    for sheet, mesh_dir, mask_path in sheets:
        key = holdout_render if sheet.startswith("straightened2") else sheet
        result = analyse(key, mesh_dir, mask_path, train, everything, names, out_dir)
        result["sheet"] = sheet
        results.append(result)
        print(json.dumps(result, indent=1), flush=True)
    (out_dir / f"{args.sheet}.json").write_text(json.dumps(results, indent=1))
    print(f"[done] {out_dir / f'{args.sheet}.json'}")


if __name__ == "__main__":
    main()
