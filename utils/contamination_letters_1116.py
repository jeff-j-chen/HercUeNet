"""Per-pixel count of contaminated LETTERS on PRIZESHEET_11_16, per model, for the mix28 _ps5 trainings.

A trained letter = an 8-connected ink component (>= 8 px, as utils.dataloader.reference_components) of dilated_inklabels/<alias>.png of a trained 0846A
sheet. It is contaminated for a model when a column of it that the model's ACTUAL ps5 exclusion mask kept (the _ps5 runs excluded the neighbourhood of
the ps5 test sheet, not of 11_16: _ves_tmp/backup_excl/exclusion_ps5.tgz) has voxels, at a layer the model reads in training, inside the model's
inference window of the 11_16 sheet. A pixel's value = the number of distinct such letters over its column (3 x 3 dense cells), i.e. letters
the model has seen with supervision at that place. Direct voxel overlap only (HercUeNet's 148 px image context is not counted).
Studio base was trained on PHerc0814 / 0841 / 0009B and the other-scroll groups of its mixture, never on PHerc0846A: its map is zero by construction.

  python3 utils/contamination_letters_1116.py   ->  output/contamination_letters_1116/
"""
import json
import sys
import tarfile
from pathlib import Path

# utils/platform.py would shadow the stdlib module when this file is run as a script
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path[:0] = ["/vesuvius", "/vesuvius/old"]

import cv2
import numpy as np
from scipy.spatial import cKDTree

import assemble_training_segments as ats
from assemble_test_segments import DEPTH_REVERSED
from utils import holdout_exclusion as hx
from utils import mix6
from utils.fast_figure import orient_display

REPO = Path("/vesuvius")
NEW = "PRIZESHEET_11_16"
ats.MESHES_0846A[NEW] = (NEW, "0", NEW)  # lets holdout_exclusion._mesh load the new sheet like a render
CENTER, LATERAL_MAX, LAYERS = 13.5, 2.0, 28
TRAIN_LAYERS = {"cue_dropout": range(8, 18), "reader_v3": range(3, 24), "v8in": range(2, 26)}
WINDOW = {"cue_dropout": range(9, 17), "reader_v3": range(6, 23), "v8in": range(2, 26)}
TITLE = {"studio": "Studio base", "cue_dropout": "HercUeNet", "reader_v3": "reader-v3", "v8in": "v8-in"}
BACKUP = REPO / "_ves_tmp" / "backup_excl" / "exclusion_ps5.tgz"
PS5_DIR = REPO / "_ves_tmp" / "excl_ps5"
MASK_DIR = {"cue_dropout": PS5_DIR / "holdout_exclusion", "reader_v3": PS5_DIR / "holdout_exclusion_reader", "v8in": PS5_DIR / "holdout_exclusion_v8in"}
OUT = REPO / "output" / "contamination_letters_1116"
OUT.mkdir(parents=True, exist_ok=True)
if not PS5_DIR.exists():
    with tarfile.open(BACKUP) as tar:
        tar.extractall(PS5_DIR)
import json as _json

summary_ps5 = _json.loads((PS5_DIR / "holdout_exclusion" / "summary.json").read_text())["holdout"]
print("ps5 exclusion computed against", summary_ps5, flush=True)


def split_extra_labels(raw):
    mid = ((raw >= 48) & (raw <= 207)).astype(np.uint8)
    extra = np.zeros(raw.shape, dtype=bool)
    if mid.any():
        grown = cv2.morphologyEx(mid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        if grown.any():
            step = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            for _ in range(3):
                grown = cv2.dilate(grown, step) & mid
            extra = grown > 0
    return (raw >= 128) & ~extra


def letter_ids(alias, offset):
    """int32 image: global letter id (offset + 1..) of every ink pixel of a trained sheet, 0 elsewhere; and the number of letters."""
    raw = cv2.imread(str(REPO / "dilated_inklabels" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)
    count, comp, stats, _ = cv2.connectedComponentsWithStats(split_extra_labels(raw).astype(np.uint8), 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= 8
    keep[0] = False
    remap = np.zeros(count, np.int64)
    remap[np.flatnonzero(keep)] = offset + np.arange(1, int(keep.sum()) + 1)
    return remap[comp].astype(np.int32), int(keep.sum())


def normals(points):
    dv = np.stack([np.gradient(points[..., c], axis=0) for c in range(3)], -1)
    du = np.stack([np.gradient(points[..., c], axis=1) for c in range(3)], -1)
    n = np.cross(dv, du)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.where(norm == 0, 1, norm)


def sign(name):
    render, alias, _ = ats.MESHES_0846A[name]
    return -1.0 if (str(render) in DEPTH_REVERSED or str(alias) in DEPTH_REVERSED) else 1.0


points, valid, shape = hx._mesh(NEW, hx.HOLDOUT_DENSE)
new_n = normals(np.where(valid[..., None], points, 0.0))
new_sign = sign(NEW)
tree = cKDTree(points[valid])
foot_n = new_n[valid]
flat_index = np.flatnonzero(valid.ravel())
H, W = valid.shape
print(f"{NEW}: {shape} render px, dense grid {valid.shape}, {int(valid.sum()):,} points, sign {new_sign:+.0f}", flush=True)

aliases = {model: hx.trained_aliases(model) for model in TRAIN_LAYERS}
letters, offset, totals = {}, 0, {}
for alias in sorted({a for v in aliases.values() for a in v}):
    letters[alias], n = letter_ids(alias, offset)
    offset += n
    totals[alias] = n
print(f"{offset} trained letters in {len(letters)} sheets", flush=True)
K = offset + 1

pairs_by_model = {}
for model in TRAIN_LAYERS:
    pairs = np.empty(0, np.int64)
    for alias in aliases[model]:
        name = mix6.SHEETS_0846A[int(alias)][1]
        pts, ok, _ = hx._mesh(name, hx.SHEET_DENSE)
        excluded = hx.load(alias, directory=MASK_DIR[model])
        step = hx.GRID_PX // hx.SHEET_DENSE
        rows = np.minimum(np.arange(ok.shape[0]) * step, excluded.shape[0] - 1)
        cols = np.minimum(np.arange(ok.shape[1]) * step, excluded.shape[1] - 1)
        ids = letters[alias]
        lrows, lcols = np.minimum(rows, ids.shape[0] - 1), np.minimum(cols, ids.shape[1] - 1)
        letter_grid = ids[lrows][:, lcols]
        keep = ok & ~excluded[rows][:, cols] & (letter_grid > 0)
        keep &= cv2.erode(ok.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        n = normals(np.where(ok[..., None], pts, 0.0))
        q, qn, s, lid = pts[keep], n[keep], sign(name), letter_grid[keep]
        near, _ = tree.query(q, distance_upper_bound=60.0)
        sel = np.isfinite(near)
        q, qn, lid = q[sel], qn[sel], lid[sel]
        found = []
        for layer in TRAIN_LAYERS[model]:
            x = q + (layer - CENTER) * s * qn
            d, index = tree.query(x, distance_upper_bound=40.0)
            hit = np.isfinite(d)
            v = x[hit] - tree.data[index[hit]]
            along = np.einsum("ij,ij->i", v, foot_n[index[hit]])
            lateral = np.linalg.norm(v - along[:, None] * foot_n[index[hit]], axis=1)
            good = lateral <= LATERAL_MAX
            bins = np.rint(CENTER + new_sign * along[good]).astype(int)
            inside = (bins >= WINDOW[model].start) & (bins < WINDOW[model].stop)
            cell = flat_index[index[hit][good][inside]]
            found.append(cell.astype(np.int64) * K + lid[hit][good][inside])
        got = np.unique(np.concatenate(found)) if found else np.empty(0, np.int64)
        pairs = np.unique(np.concatenate([pairs, got]))
        print(f"{model:12s} {name[-38:]:38s} kept ink columns near {len(q):8,d}, (cell, letter) pairs now {len(pairs):,}", flush=True)
    pairs_by_model[model] = pairs

mask = cv2.imread(f"/vesuvius/masks/extras/{NEW}.png", cv2.IMREAD_GRAYSCALE) > 0
scale = hx.GRID_PX / hx.HOLDOUT_DENSE
rr = np.minimum(np.rint(np.arange(shape[0]) / scale).astype(int), H - 1)
cc = np.minimum(np.rint(np.arange(shape[1]) / scale).astype(int), W - 1)
counts, stats = {"studio": np.zeros(shape, np.uint16)}, {}
stats["studio"] = {"letters_touching_sheet": 0, "pixels_contaminated_pct": 0.0, "mean_letters_per_pixel": 0.0, "max_letters_per_pixel": 0}
for model, pairs in pairs_by_model.items():
    cell, lid = pairs // K, pairs % K
    r, c = cell // W, cell % W
    expanded = []
    for dr in (-1, 0, 1):  # a trained sample covers a 5 x 5 voxel column, two dense (2.5 vx) cells
        for dc in (-1, 0, 1):
            rn, cn = np.clip(r + dr, 0, H - 1), np.clip(c + dc, 0, W - 1)
            expanded.append((rn * W + cn) * K + lid)
    unique = np.unique(np.concatenate(expanded))
    dense = np.bincount((unique // K), minlength=H * W).reshape(H, W).astype(np.uint16)
    counts[model] = dense[rr][:, cc]
    inside = counts[model][mask]
    stats[model] = {"letters_touching_sheet": int(len(np.unique(lid))), "pixels_contaminated_pct": round(100 * float((inside > 0).mean()), 2),
                    "mean_letters_per_pixel": round(float(inside.mean()), 3), "max_letters_per_pixel": int(inside.max())}
    print(f"{model:12s} {stats[model]}", flush=True)
for model, array in counts.items():
    np.save(OUT / f"{model}_contaminated_letters.npy", array)
(OUT / "summary.json").write_text(json.dumps({"sheet": NEW, "ps5_exclusion_computed_against": summary_ps5, "trained_letters_total": offset, "models": stats,
                                              "definition": __doc__}, indent=2))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

VMAX = max(1, max(int(counts[m][mask].max()) for m in counts))
colors = ["#e8e8e8"] + list(plt.get_cmap("turbo")(np.linspace(0.12, 0.98, VMAX)))
cmap = ListedColormap(colors)
cmap.set_bad("white")
DOWN = 4
fig, axes = plt.subplots(4, 1, figsize=(18, 22))
for axis, model in zip(axes, ("studio", "cue_dropout", "reader_v3", "v8in")):
    oriented = orient_display(np.where(mask, counts[model].astype(np.float32), np.nan), NEW)
    small = cv2.resize(np.nan_to_num(oriented, nan=-1), (oriented.shape[1] // DOWN, oriented.shape[0] // DOWN), interpolation=cv2.INTER_NEAREST)
    shown = np.ma.masked_where(small < 0, small)
    image = axis.imshow(shown, cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, VMAX + 1.5), cmap.N), interpolation="nearest")
    s = stats[model]
    axis.set_title(f"{TITLE[model]}: {s['letters_touching_sheet']} trained letters touch the sheet, {s['pixels_contaminated_pct']:.1f}% of pixels contaminated, "
                   f"mean {s['mean_letters_per_pixel']:.2f}, max {s['max_letters_per_pixel']}", fontsize=12)
    axis.set_xticks([]), axis.set_yticks([])
    bar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.01, ticks=np.arange(0, VMAX + 1, max(1, VMAX // 8)))
    bar.set_label("contaminated letters")
    rgb = cv2.cvtColor(cv2.applyColorMap(np.clip(oriented * 255.0 / VMAX, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[np.nan_to_num(oriented, nan=-1) == 0] = (232, 232, 232)
    rgb[np.isnan(oriented)] = (255, 255, 255)
    cv2.imwrite(str(OUT / f"{model}_contaminated_letters.png"), rgb[..., ::-1])
fig.tight_layout()
fig.savefig(OUT / "contamination_letters_map.png", dpi=80)
print("saved", OUT / "contamination_letters_map.png", flush=True)
