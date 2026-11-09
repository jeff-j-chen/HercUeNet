"""overlap of the fiber_only neighbour sheets with the 7 trained PHerc0846A sheets.

For each new sheet, every trained sheet's papyrus is mapped to 3D mesh points; the trained sheet with the largest area
within OVERLAP_VOX of the new sheet is picked. Its inklabel is carried onto the new sheet through the nearest 3D point and
drawn over the quick cue_dropout inference in output/cue_dropout/<sheet>/prediction.npy.

  python3 fiber_overlap.py [--vox 4]
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

import assemble_training_segments as ats
from assemble_test_segments import ROTATE_180
from exclusion_vs_sheet import PIXEL_CM2, at_dense, dense_mesh, to_frame
from utils import mix6
from utils.dataloader import split_extra_labels

REPO = Path(__file__).resolve().parent.parent
SHEETS = [f"fiber_only_20261029d_flatboi_{s}" for s in ("in_1", "in_2", "out_1", "out_2")]
OUT = REPO / "output" / "fiber_overlap"


def short(name):
    return name.replace("auto_grown_20261012065803387_", "").replace("auto_grown_", "")


def load_trained():
    sheets = []
    for alias, (render, name) in mix6.SHEETS_0846A.items():
        mesh = ats.MESHES_0846A[name][2]
        points, valid, shape = dense_mesh(REPO / "tifxyz" / mesh, str(render) in ROTATE_180)
        papyrus = cv2.imread(str(REPO / "masks" / f"{render}.png"), cv2.IMREAD_GRAYSCALE) > 0
        label = cv2.imread(str(REPO / "dilated_inklabels" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)
        ink = split_extra_labels(label)[0] if label is not None else np.zeros_like(papyrus)
        base = valid & at_dense(papyrus, valid.shape)
        sheets.append(dict(name=short(name), alias=alias, points=points, base=base, shape=shape, papyrus=papyrus,
                           ink=ink.astype(bool), ink_dense=at_dense(ink, valid.shape).astype(bool),
                           tree=cKDTree(points[base])))
    return sheets


def normalise(pred, papyrus):
    lo, hi = np.nanpercentile(pred[papyrus], [1, 99.5])
    return np.nan_to_num(np.clip((pred - lo) / max(hi - lo, 1e-6), 0, 1))


def analyse(sheet, trained, vox):
    points, valid, shape = dense_mesh(REPO / "tifxyz" / sheet, sheet in ROTATE_180)
    papyrus = cv2.imread(str(REPO / "masks" / "extras" / f"{sheet}.png"), cv2.IMREAD_GRAYSCALE) > 0
    pred = np.load(REPO / "output" / "cue_dropout" / sheet / "prediction.npy").astype(np.float32)
    assert papyrus.shape == shape == pred.shape, (papyrus.shape, shape, pred.shape)
    base = valid & at_dense(papyrus, valid.shape)
    query = points[base]
    areas, hits = {}, {}
    for t in trained:
        dist, idx = t["tree"].query(query, distance_upper_bound=4 * vox)
        hits[t["name"]] = (dist, idx)
        areas[t["name"]] = float((dist <= vox).sum() * (PIXEL_CM2 * (shape[0] / valid.shape[0]) ** 2))
    best = max(trained, key=lambda t: areas[t["name"]])
    dist, idx = hits[best["name"]]

    # new sheet: distance to best + carried ink label
    d_dense = np.full(valid.shape, np.inf, np.float32)
    d_dense[base] = dist
    ink_dense = np.zeros(valid.shape, bool)
    ok = dist <= vox
    ink_dense[np.where(base)[0][ok], np.where(base)[1][ok]] = best["ink_dense"][best["base"]][idx[ok]]
    dmap, carried = to_frame(d_dense, shape), to_frame(ink_dense, shape) & papyrus
    overlap = papyrus & (dmap <= vox)

    # trained sheet: where the new sheet overlaps it
    new_tree = cKDTree(query)
    bd, _ = new_tree.query(best["points"][best["base"]], distance_upper_bound=4 * vox)
    b_dense = np.full(best["base"].shape, np.inf, np.float32)
    b_dense[best["base"]] = bd
    b_overlap = to_frame(b_dense, best["shape"]) <= vox
    b_overlap &= best["papyrus"]

    p = normalise(pred, papyrus)
    stats = {"sheet": sheet, "best": best["name"], "best_alias": best["alias"], "overlap_vox": vox,
             "overlap_cm2_by_trained_sheet": {k: round(v, 3) for k, v in sorted(areas.items(), key=lambda kv: -kv[1])},
             "carried_ink_cm2": float(carried.sum() * PIXEL_CM2)}
    if carried.any() and (overlap & ~carried).any():
        stats["mean_pred_on_carried_ink"] = float(p[carried].mean())
        stats["mean_pred_on_overlap_non_ink"] = float(p[overlap & ~carried].mean())
        stats["pearson_pred_vs_carried_label"] = float(np.corrcoef(p[overlap], carried[overlap])[0, 1])
    figure(sheet, best, p, papyrus, dmap, overlap, carried, b_overlap, vox, stats)
    return stats


def outline(mask, k=9):
    m = mask.astype(np.uint8)
    return cv2.dilate(m, np.ones((k, k), np.uint8)) > cv2.erode(m, np.ones((k, k), np.uint8))


def figure(sheet, best, p, papyrus, dmap, overlap, carried, b_overlap, vox, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils.fast_figure import orient_display

    gray = np.repeat(p[..., None], 3, -1)
    over = gray.copy()
    over[carried] = 0.55 * over[carried] + 0.45 * np.array([1, 0, 0])
    over[outline(overlap)] = (0, 1, 1)

    d = np.where(papyrus, np.minimum(dmap, 4 * vox), np.nan)

    b_img = np.where(best["papyrus"], 0.35, 0.0)[..., None].repeat(3, -1)
    b_img[best["ink"]] = (1, 1, 1)
    b_img[b_overlap & ~best["ink"]] = (0.1, 0.45, 0.55)
    b_img[b_overlap & best["ink"]] = (1, 0.2, 0.2)
    b_img = orient_display(b_img, best["alias"])

    fig, ax = plt.subplots(2, 2, figsize=(22, 17))
    ax[0, 0].imshow(gray)
    ax[0, 0].set_title(f"{short(sheet)}: cue_dropout quick inference")
    ax[0, 1].imshow(over)
    ax[0, 1].set_title(f"inference + {best['name']} inklabel carried over (red), overlap <= {vox:g} vox (cyan)")
    im = ax[1, 0].imshow(d, cmap="viridis_r")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.03, label="3D distance to trained sheet (vox)")
    ax[1, 0].set_title(f"distance to {best['name']} ({stats['overlap_cm2_by_trained_sheet'][best['name']]:.2f} cm² within {vox:g} vox)")
    ax[1, 1].imshow(b_img)
    ax[1, 1].set_title(f"trained {best['name']}: inklabel (white), overlapped papyrus (teal), overlapped ink (red)")
    for a in ax.flat:
        a.axis("off")
    txt = "  ".join(f"{k}: {v:.2f}" for k, v in stats["overlap_cm2_by_trained_sheet"].items())
    extra = ""
    if "pearson_pred_vs_carried_label" in stats:
        extra = (f"\nin overlap: mean pred on carried ink {stats['mean_pred_on_carried_ink']:.3f} vs non-ink "
                 f"{stats['mean_pred_on_overlap_non_ink']:.3f}, r = {stats['pearson_pred_vs_carried_label']:.3f}")
    fig.suptitle(f"{sheet}  vs  {best['name']}\noverlap cm² by trained sheet: {txt}{extra}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / f"{sheet}.png", dpi=80)
    plt.close(fig)


HOLDOUT = "fiber_only_20261029d_flatboi"
CAP = 60.0  # vox; KD queries stop here
BINS = [(2, (0.45, 0, 0), "<= 2 (same sheet)"), (7, (0.9, 0.1, 0.1), "2-7"), (14, (1, 0.55, 0), "7-14"),
        (21, (1, 0.9, 0.2), "14-21"), (28, (0.55, 0.85, 0.4), "21-28"), (40, (0.45, 0.7, 1), "28-40"),
        (CAP, (0.75, 0.75, 0.9), f"40-{CAP:g}")]
FAR = (0.85, 0.85, 0.85)


def sheet_points(mesh, mask_path, rotate):
    points, valid, shape = dense_mesh(REPO / "tifxyz" / mesh, rotate)
    papyrus = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
    assert papyrus.shape == shape, (mesh, papyrus.shape, shape)
    return dict(points=points, base=valid & at_dense(papyrus, valid.shape), shape=shape, papyrus=papyrus)


def distance_to(sheet, tree):
    """per-pixel 3D distance (vox) from sheet's papyrus to the tree's points, inf beyond CAP."""
    d = np.full(sheet["base"].shape, np.inf, np.float32)
    d[sheet["base"]] = tree.query(sheet["points"][sheet["base"]], distance_upper_bound=CAP)[0]
    return to_frame(d, sheet["shape"])


def red(dist, papyrus, ink=None):
    """papyrus coloured by 3D distance bin (BINS); ink drawn darkened."""
    img = np.ones(dist.shape + (3,))
    img[papyrus] = FAR
    lo = -1.0
    for hi, colour, _ in BINS:
        img[papyrus & (dist > lo) & (dist <= hi)] = colour
        lo = hi
    if ink is not None:
        img[ink & papyrus] *= 0.3
    return img


def legend(fig):
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=f"{l} vox") for _, c, l in BINS] + [Patch(color=FAR, label=f"> {CAP:g} vox")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=14, title="3D distance (ink darkened)")


def area_within(dist, papyrus, t):
    return float((papyrus & (dist <= t)).sum() * PIXEL_CM2)


def holdout_figures(trained):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils.fast_figure import orient_display

    hold = sheet_points(HOLDOUT, REPO / "masks" / "extras" / f"{HOLDOUT}.png", HOLDOUT in ROTATE_180)
    hold_tree = cKDTree(hold["points"][hold["base"]])
    stats = {}

    # holdout vs each trained sheet: trained papyrus coloured by distance to the holdout
    fig, axes = plt.subplots(2, 4, figsize=(32, 17))
    nearest = np.full(hold["shape"], np.inf, np.float32)
    for t, ax in zip(trained, axes.flat):
        t_dist = distance_to(t, hold_tree)
        nearest = np.minimum(nearest, distance_to(hold, t["tree"]))
        s = {f"<= {v:g}": area_within(t_dist, t["papyrus"], v) for v in (4, 8, 14, 25.5)}
        s["ink <= 14"] = area_within(t_dist, t["papyrus"] & t["ink"], 14)
        stats[t["name"]] = s
        ax.imshow(orient_display(red(t_dist, t["papyrus"], t["ink"]), t["alias"]))
        ax.set_title(f"{t['name']}: {s['<= 4']:.2f} cm² <= 4 vox, {s['<= 14']:.2f} <= 14, ink {s['ink <= 14']:.2f} <= 14")
    ax = axes.flat[-1]
    ax.imshow(red(nearest, hold["papyrus"]))
    ax.set_title(f"{HOLDOUT}: distance to nearest trained sheet "
                 f"({area_within(nearest, hold['papyrus'], 4):.2f} cm² <= 4, {area_within(nearest, hold['papyrus'], 14):.2f} <= 14)")
    stats["holdout_near_any_cm2"] = {f"<= {v:g}": area_within(nearest, hold["papyrus"], v) for v in (4, 8, 14, 25.5)}
    for a in axes.flat:
        a.axis("off")
    fig.suptitle(f"3D proximity of the 7 trained 0846A sheets to the new holdout {HOLDOUT}", fontsize=15)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    legend(fig)
    fig.savefig(OUT / f"{HOLDOUT}_vs_trained.png", dpi=70)
    plt.close(fig)

    # holdout vs old holdout straightened2, both directions
    render, _alias, mesh = ats.MESHES_0846A["auto_grown_20261012065803387_straightened2_abf"]
    old = sheet_points(mesh, REPO / "masks" / f"{render}.png", render in ROTATE_180)
    d_old = distance_to(old, hold_tree)
    d_new = distance_to(hold, cKDTree(old["points"][old["base"]]))
    fig, ax = plt.subplots(1, 2, figsize=(26, 10))
    ax[0].imshow(red(d_new, hold["papyrus"]))
    ax[0].set_title(f"{HOLDOUT}: distance to straightened2 ({area_within(d_new, hold['papyrus'], 4):.2f} cm² <= 4 vox, "
                    f"{area_within(d_new, hold['papyrus'], 14):.2f} <= 14)")
    ax[1].imshow(orient_display(red(d_old, old["papyrus"]), render))
    ax[1].set_title(f"straightened2: distance to {HOLDOUT} ({area_within(d_old, old['papyrus'], 4):.2f} cm² <= 4, "
                    f"{area_within(d_old, old['papyrus'], 14):.2f} <= 14)")
    for a in ax:
        a.axis("off")
    fig.suptitle("new holdout vs old holdout straightened2", fontsize=15)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    legend(fig)
    fig.savefig(OUT / f"{HOLDOUT}_vs_straightened2.png", dpi=70)
    plt.close(fig)
    stats["straightened2"] = {"holdout_cm2": {f"<= {v:g}": area_within(d_new, hold["papyrus"], v) for v in (4, 8, 14, 25.5)},
                              "straightened2_cm2": {f"<= {v:g}": area_within(d_old, old["papyrus"], v) for v in (4, 8, 14, 25.5)},
                              "holdout_papyrus_cm2": float(hold["papyrus"].sum() * PIXEL_CM2),
                              "straightened2_papyrus_cm2": float(old["papyrus"].sum() * PIXEL_CM2)}
    # the in/out neighbours seeded from the holdout must not touch it
    fig, axes = plt.subplots(2, 2, figsize=(26, 19))
    stats["neighbours"] = {}
    for sheet, ax in zip(SHEETS, axes.flat):
        nb = sheet_points(sheet, REPO / "masks" / "extras" / f"{sheet}.png", sheet in ROTATE_180)
        d = distance_to(nb, hold_tree)
        s = {"min_vox": float(d[nb["papyrus"]].min()), "p1_vox": float(np.percentile(d[nb["papyrus"]], 1)),
             **{f"<= {v:g}_cm2": area_within(d, nb["papyrus"], v) for v in (2, 7, 14, 21, 28)}}
        stats["neighbours"][sheet] = s
        ax.imshow(red(d, nb["papyrus"]))
        ax.set_title(f"{short(sheet)}: min {s['min_vox']:.1f} vox, 1st pct {s['p1_vox']:.1f}, "
                     f"{s['<= 14_cm2']:.3f} cm² <= 14, {s['<= 28_cm2']:.2f} <= 28")
        ax.axis("off")
    fig.suptitle(f"in/out neighbour sheets: 3D distance to {HOLDOUT}", fontsize=15)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    legend(fig)
    fig.savefig(OUT / f"{HOLDOUT}_neighbours.png", dpi=70)
    plt.close(fig)
    print(json.dumps(stats, indent=1))
    (OUT / f"{HOLDOUT}_overlap.json").write_text(json.dumps(stats, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vox", type=float, default=4.0, help="3D distance counted as overlap")
    parser.add_argument("--holdout-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    trained = load_trained()
    holdout_figures(trained)
    if args.holdout_only:
        return
    results = []
    for sheet in SHEETS:
        stats = analyse(sheet, trained, args.vox)
        print(json.dumps(stats, indent=1))
        results.append(stats)
    (OUT / "summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
