"""discard the parts of the trained PHerc0846A sheets near the prize sheet (utils/mix6.py HOLDOUT_NAME, now
fiber_only_20261102b_flatboi_PRIZESHEET; every earlier prize sheet is superseded).

the neighbour sheets were traced next to the prize sheet and in places run onto it, so their renders and labels would leak
it into training. For every trained 0846A sheet this measures, at each mesh point, the 3D distance to the prize-sheet mesh
(the zone = distance <= the model's radius) and writes one discard mask per model in the render frame (255 = discard):

  cue_dropout  holdout_exclusion/<alias>.png         12 vx zone, dilated by the model's full context reach (CUE_CONTEXT_PX:
                                                     a kept 16 px cell never has a 192 px context window that touches the zone);
               holdout_exclusion/zone/<alias>.png    the 12 vx zone itself: the asf cue of the sheet is built without it
  reader-v3    holdout_exclusion_reader/<alias>.png  18 vx zone
  v8-in        holdout_exclusion_v8in/<alias>.png    25 vx zone

train.py, finetune_readerv3.py and finetune_v8in.py drop supervision there (the two fine-tunes also blank the zone voxels of
their staged inputs). Each folder has a summary.json naming the prize sheet and radius; problems() fails a training whose
masks were computed against another prize sheet or radius. Figures go to output/holdout_exclusion*/.

  python3 -m utils.holdout_exclusion [--force]
  python3 -m utils.holdout_exclusion --export exclusion_visualization   # every model's figures + a per-sheet comparison
"""
from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXCLUSION_DIR = REPO / "holdout_exclusion"
CUE_ZONE_DIR = EXCLUSION_DIR / "zone"
FIGURE_DIR = REPO / "output" / "holdout_exclusion"
RADIUS_VOX = 12.0
READER_RADIUS_VOX = 18.0
READER_EXCLUSION_DIR = REPO / "holdout_exclusion_reader"
READER_FIGURE_DIR = REPO / "output" / "holdout_exclusion_reader"
V8IN_RADIUS_VOX = 25.0
V8IN_EXCLUSION_DIR = REPO / "holdout_exclusion_v8in"
V8IN_FIGURE_DIR = REPO / "output" / "holdout_exclusion_v8in"
HOLDOUT_DENSE = 8   # prize-sheet mesh sampled every 20/8 = 2.5 voxels
SHEET_DENSE = 4     # trained-sheet mesh sampled every 5 voxels (= render pixels)
GRID_PX = 20        # render pixels per tifxyz grid cell (scale 0.05, 9.362 um volume)
VERSION = 2
MIN_KEPT_INK_CM2 = 0.001  # a trained sheet must keep at least this much ink after the exclusion


@functools.lru_cache(maxsize=1)
def cue_context_reach() -> int:
    """render px from any pixel of a supervised cue_dropout cell to the farthest pixel its input can read (Chebyshev):
    half of (context window + prediction core) plus the context jitter, from the training config itself (148 for mix26)."""
    from utils.config import Config, apply_mix6_run

    config = Config()
    apply_mix6_run(config)
    data, model = config.data, config.model
    context = int(data.context_size) * int(data.context_downsample)
    core = int(model.multitile_grid) * int(model.multitile_subtile)
    jitter = int(data.ctx_jitter) if float(getattr(data, "arbitrary_rotation_prob", 0) or 0) == 0 else None
    if jitter is None:
        raise ValueError("arbitrary rotations enlarge the context reach; recompute CUE_CONTEXT_PX for them")
    return (context + core + 1) // 2 + jitter


def specs() -> dict:
    """model -> (radius in voxels, context dilation in px, mask folder, figure folder)."""
    return {
        "cue_dropout": (RADIUS_VOX, cue_context_reach() + 1, EXCLUSION_DIR, FIGURE_DIR),
        "reader_v3": (READER_RADIUS_VOX, 0, READER_EXCLUSION_DIR, READER_FIGURE_DIR),
        "v8in": (V8IN_RADIUS_VOX, 0, V8IN_EXCLUSION_DIR, V8IN_FIGURE_DIR),
    }


def _mesh(name: str, factor: int):
    """dense mesh points of a 0846A sheet on its render's cropped grid: points (H,W,3), valid (H,W), render shape."""
    from assemble_test_segments import ROTATE_180, load_mesh
    from assemble_training_segments import MESHES_0846A

    render, _alias, mesh = MESHES_0846A[name]
    X, Y, Z, valid = load_mesh(str(REPO / "tifxyz" / mesh))
    if render in ROTATE_180:
        X, Y, Z, valid = (np.ascontiguousarray(grid[::-1, ::-1]) for grid in (X, Y, Z, valid))
    # the renderer's crop: valid bounding box + 8 cells
    ys, xs = np.where(valid)
    y0, x0 = max(0, ys.min() - 8), max(0, xs.min() - 8)
    y1, x1 = min(valid.shape[0], ys.max() + 9), min(valid.shape[1], xs.max() + 9)
    X, Y, Z, valid = (grid[y0:y1, x0:x1] for grid in (X, Y, Z, valid))
    gh, gw = X.shape
    shape = ((gh - 1) * GRID_PX + 1, (gw - 1) * GRID_PX + 1)
    height, width = (gh - 1) * factor + 1, (gw - 1) * factor + 1
    # corner-aligned bilinear upsampling, so dense index i sits exactly at grid row i / factor
    rows, cols = np.linspace(0, gh - 1, height), np.linspace(0, gw - 1, width)
    r0 = np.minimum(rows.astype(int), gh - 2)
    c0 = np.minimum(cols.astype(int), gw - 2)
    fr, fc = (rows - r0)[:, None], (cols - c0)[None, :]

    def up(grid):
        grid = grid.astype(np.float64)
        top = grid[r0][:, c0] * (1 - fc) + grid[r0][:, c0 + 1] * fc
        bottom = grid[r0 + 1][:, c0] * (1 - fc) + grid[r0 + 1][:, c0 + 1] * fc
        return top * (1 - fr) + bottom * fr

    points = np.stack([up(X), up(Y), up(Z)], axis=-1)
    dense_valid = up(valid.astype(np.float32)) > 0.999
    return points, dense_valid, shape


def dense_distance(name: str, holdout_tree) -> tuple[np.ndarray, tuple]:
    """3D distance (voxels) of every dense mesh point of a sheet to the prize sheet (inf off the mesh), and the render shape."""
    points, valid, shape = _mesh(name, SHEET_DENSE)
    distance = np.full(valid.shape, np.inf)
    nearest, _ = holdout_tree.query(points[valid], distance_upper_bound=200.0)
    distance[valid] = nearest
    return distance, shape


def _to_render(grid: np.ndarray, shape: tuple) -> np.ndarray:
    step = GRID_PX // SHEET_DENSE
    rows = np.minimum(np.rint(np.arange(shape[0]) / step).astype(int), grid.shape[0] - 1)
    cols = np.minimum(np.rint(np.arange(shape[1]) / step).astype(int), grid.shape[1] - 1)
    return grid[rows][:, cols]


def zone(distance: np.ndarray, shape: tuple, radius: float) -> np.ndarray:
    """render-frame mask of the mesh points within `radius` voxels of the prize sheet."""
    # nearest prize-sheet sample overestimates the surface distance by at most half its 2.5 voxel spacing
    near = distance <= radius + 0.5 * GRID_PX / HOLDOUT_DENSE
    # one dense step of margin (5 px) covers the nearest-neighbour mapping to render pixels
    near = cv2.dilate(near.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return _to_render(near, shape)


def dilate_square(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """every pixel within Chebyshev distance radius_px of the mask (a square kernel of side 2 radius + 1, exact)."""
    if radius_px <= 0:
        return mask.copy()
    side = 2 * int(radius_px) + 1
    return cv2.dilate(mask.astype(np.uint8), np.ones((side, side), np.uint8)) > 0


def exclusion_path(alias, directory: Path = EXCLUSION_DIR) -> Path:
    return Path(directory) / f"{int(alias)}.png"


def load(alias, shape=None, directory: Path = EXCLUSION_DIR) -> np.ndarray | None:
    """bool discard mask of a trained alias, or None when it has none; checked against the frame shape."""
    path = exclusion_path(alias, directory)
    if not path.is_file():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
    if shape is not None and tuple(mask.shape) != tuple(shape):
        raise ValueError(f"{path}: shape {mask.shape} differs from the frame {tuple(shape)}")
    return mask


def expected_aliases() -> list[int]:
    from utils import mix6

    return list(mix6.TRAIN_0846A)


def trained_aliases(model: str) -> list[int]:
    from utils import mix6

    return list(mix6.CUE_TRAIN_0846A if model == "cue_dropout" else mix6.TRAIN_0846A)


def _wanted(model: str) -> dict:
    from utils import mix6

    radius, context, _directory, _figures = specs()[model]
    return {"version": VERSION, "holdout": mix6.HOLDOUT_NAME, "radius_vox": radius, "context_px": context}


def problems(model: str = "cue_dropout") -> list[str]:
    """why the model's masks are not usable: missing for a trained 0846A sheet, or computed against another prize sheet,
    radius or context dilation (empty = ready to train)."""
    _radius, context, directory, _figures = specs()[model]
    summary_path = Path(directory) / "summary.json"
    if not summary_path.is_file():
        return [f"{summary_path} is missing (python3 -m utils.holdout_exclusion)"]
    summary = json.loads(summary_path.read_text())
    found = [f"{summary_path}: {key} {summary.get(key)!r}, expected {value!r} (python3 -m utils.holdout_exclusion)"
             for key, value in _wanted(model).items() if summary.get(key) != value]
    for alias in trained_aliases(model):
        paths = [exclusion_path(alias, directory)] + ([exclusion_path(alias, CUE_ZONE_DIR)] if context else [])
        found += [f"{path} is missing (python3 -m utils.holdout_exclusion)" for path in paths if not path.is_file()]
        row = summary.get("sheets", {}).get(str(alias))
        if row is None:
            found.append(f"{summary_path}: no entry for trained sheet {alias} (python3 -m utils.holdout_exclusion)")
        elif row["ink_cm2"] - row["discarded_ink_cm2"] < MIN_KEPT_INK_CM2:
            found.append(f"{model}: sheet {alias} keeps {row['ink_cm2'] - row['discarded_ink_cm2']:.4f} cm2 of ink after the exclusion "
                         f"(< {MIN_KEPT_INK_CM2}); training would crash on it, drop it from the corpus (utils/mix6.py CUE_DROPPED)")
    return found


def all_problems() -> list[str]:
    return [line for model in specs() for line in problems(model)]


def _figure(alias, name, render, distance, excluded, papyrus, ink, summary_row, radius, figure_dir, context=0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils.fast_figure import orient_display

    def orient(image):
        return orient_display(image, render)

    rgb = np.full(papyrus.shape + (3,), 255, np.uint8)
    rgb[papyrus] = (205, 205, 205)
    rgb[papyrus & ink] = (40, 40, 40)
    rgb[papyrus & excluded] = (240, 120, 120)
    rgb[papyrus & excluded & ink] = (150, 0, 0)
    near = np.where(papyrus, np.minimum(distance, 60.0), np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    axes[0].imshow(orient(rgb), interpolation="nearest")
    reach = f" + {context} px context reach" if context else ""
    axes[0].set_title("kept papyrus (grey, ink black) / DISCARDED within "
                      f"{radius:g} vox of the prize sheet{reach} (red, ink dark red)")
    image = axes[1].imshow(orient(near), cmap="magma", vmin=0, vmax=60, interpolation="nearest")
    axes[1].contour(orient(np.nan_to_num(near, nan=999.0)), levels=[radius], colors="cyan", linewidths=1.0)
    axes[1].set_title(f"3D distance to the prize-sheet mesh (voxels, capped at 60); cyan = {radius:g}")
    fig.colorbar(image, ax=axes[1], shrink=0.7)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{name.replace('auto_grown_', '')} (alias {alias}): discarded {summary_row['discarded_cm2']:.2f} of "
        f"{summary_row['papyrus_cm2']:.2f} cm2 ({100 * summary_row['discarded_fraction']:.0f}%), ink "
        f"{summary_row['discarded_ink_cm2']:.3f} of {summary_row['ink_cm2']:.3f} cm2", fontsize=14)
    fig.savefig(Path(figure_dir) / f"{alias}.png", dpi=80, bbox_inches="tight")
    plt.close(fig)


def _overview(summary: dict, radius, figure_dir, holdout: str, context: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary)
    cols = 4
    rows_count = (len(names) + 1 + cols - 1) // cols  # one panel per sheet + the table
    fig, axes = plt.subplots(rows_count, cols, figsize=(7 * cols, 6.5 * rows_count))
    for ax in axes.flat:
        ax.axis("off")
    for ax, alias in zip(axes.flat, names):
        ax.imshow(cv2.cvtColor(cv2.imread(str(Path(figure_dir) / f"{alias}.png")), cv2.COLOR_BGR2RGB))
    table = axes.flat[len(names)]
    rows = [["sheet", "discarded cm2", "%", "ink discarded cm2", "ink %"]]
    for alias, row in summary.items():
        rows.append([row["name"].replace("auto_grown_20261012065803387_", "").replace("auto_grown_", ""),
                     f"{row['discarded_cm2']:.2f}", f"{100 * row['discarded_fraction']:.0f}",
                     f"{row['discarded_ink_cm2']:.3f}", f"{100 * row['discarded_ink_fraction']:.0f}"])
    cells = table.table(cellText=rows, loc="center", cellLoc="left")
    cells.auto_set_font_size(False)
    cells.set_fontsize(10)
    cells.scale(1, 2.0)
    reach = f" (+ {context} px context reach)" if context else ""
    fig.suptitle(f"trained PHerc0846A sheets: supervision discarded within {radius:g} voxels{reach} of the prize sheet {holdout}",
                 fontsize=16)
    fig.savefig(Path(figure_dir) / "overview.png", dpi=70, bbox_inches="tight")
    plt.close(fig)


def _papyrus(render) -> np.ndarray:
    for folder in ("masks", "masks/extras"):
        papyrus = cv2.imread(str(REPO / folder / f"{render}.png"), cv2.IMREAD_GRAYSCALE)
        if papyrus is not None:
            return papyrus > 0
    raise FileNotFoundError(f"masks/{render}.png (python3 assemble_training_segments.py --mix26)")


def build_all(force: bool = False, models=None, aliases=None) -> None:
    """every model's masks that are missing or computed against another prize sheet / radius (all of them with force):
    one distance map per trained sheet, then cue_dropout 12 vx + context reach (+ its zone), reader-v3 18 vx, v8-in 25 vx.
    aliases: compute only these sheets and merge them into the existing summaries (the masks of the other sheets are left
    untouched, so their asf cues do not become older than their zone); ignored for a model whose summary is for another prize
    sheet, radius or context."""
    from scipy.spatial import cKDTree

    from utils import mix6
    from utils.dataloader import split_extra_labels

    todo = [model for model in (models or specs()) if force or problems(model)]
    for model in specs():
        if model not in todo:
            print(f"[exclusion] {model}: {specs()[model][2]} is up to date (prize sheet {mix6.HOLDOUT_NAME})")
    if not todo:
        return
    wanted = {int(alias) for alias in aliases} if aliases else None
    previous = {}
    for model in todo:
        _radius, context, directory, figure_dir = specs()[model]
        Path(directory).mkdir(parents=True, exist_ok=True)
        Path(figure_dir).mkdir(parents=True, exist_ok=True)
        summary_path = Path(directory) / "summary.json"
        if wanted and summary_path.is_file():
            old = json.loads(summary_path.read_text())
            if all(old.get(key) == value for key, value in _wanted(model).items()):
                previous[model] = old.get("sheets", {})  # kept; the summary is rewritten with the new sheets at the end
        if model not in previous:
            summary_path.unlink(missing_ok=True)  # an interrupted build stays not ready
        if context:
            CUE_ZONE_DIR.mkdir(parents=True, exist_ok=True)
    sheets = {alias: value for alias, value in mix6.SHEETS_0846A.items() if wanted is None or alias in wanted}
    print(f"[exclusion] prize sheet {mix6.HOLDOUT_NAME}; building {', '.join(todo)} for {len(sheets)} of {len(mix6.SHEETS_0846A)} trained sheets",
          flush=True)
    holdout_points, holdout_valid, _ = _mesh(mix6.HOLDOUT_NAME, HOLDOUT_DENSE)
    tree = cKDTree(holdout_points[holdout_valid])
    pixel_cm2 = (9.362e-4) ** 2
    summaries = {model: dict(previous.get(model, {})) for model in todo}
    for alias, (render, name) in sheets.items():
        distance, shape = dense_distance(name, tree)
        papyrus = _papyrus(render)
        if papyrus.shape != tuple(shape):
            raise ValueError(f"{name}: render mask {papyrus.shape} differs from the mesh frame {tuple(shape)}")
        distance_px = _to_render(distance, shape).astype(np.float32)
        label = cv2.imread(str(REPO / "dilated_inklabels" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)
        ink = split_extra_labels(label)[0] if label is not None else np.zeros_like(papyrus)
        for model in todo:
            radius, context, directory, figure_dir = specs()[model]
            near = zone(distance, shape, radius)
            excluded = dilate_square(near, context)
            cv2.imwrite(str(exclusion_path(alias, directory)), (excluded & papyrus).astype(np.uint8) * 255)
            if context:
                cv2.imwrite(str(exclusion_path(alias, CUE_ZONE_DIR)), (near & papyrus).astype(np.uint8) * 255)
            row = {
                "name": name, "render": str(render),
                "papyrus_cm2": float(papyrus.sum() * pixel_cm2),
                "zone_cm2": float((papyrus & near).sum() * pixel_cm2),
                "discarded_cm2": float((papyrus & excluded).sum() * pixel_cm2),
                "ink_cm2": float((papyrus & ink).sum() * pixel_cm2),
                "discarded_ink_cm2": float((papyrus & ink & excluded).sum() * pixel_cm2),
                "min_distance_vox": float(np.min(distance_px[papyrus])) if papyrus.any() else None,
            }
            row["discarded_fraction"] = row["discarded_cm2"] / max(row["papyrus_cm2"], 1e-9)
            row["discarded_ink_fraction"] = row["discarded_ink_cm2"] / max(row["ink_cm2"], 1e-9)
            summaries[model][str(alias)] = row
            _figure(alias, name, render, distance_px, excluded, papyrus, ink, row, radius, figure_dir, context)
            print(f"[exclusion] {model} {alias} {name}: zone {row['zone_cm2']:.2f}, discard {row['discarded_cm2']:.2f}/"
                  f"{row['papyrus_cm2']:.2f} cm2 ({100 * row['discarded_fraction']:.0f}%), ink {row['discarded_ink_cm2']:.3f}/"
                  f"{row['ink_cm2']:.3f} cm2, min distance {row['min_distance_vox']:.1f} vox", flush=True)
    for model in todo:
        radius, context, directory, figure_dir = specs()[model]
        _overview(summaries[model], radius, figure_dir, mix6.HOLDOUT_NAME, context)
        (Path(directory) / "summary.json").write_text(json.dumps({**_wanted(model), "sheets": summaries[model]}, indent=1))
        print(f"[exclusion] {model}: masks in {directory}, figures in {figure_dir}")


# who dropped a pixel. reader-v3's zone lies inside v8-in's, but cue_dropout's (12 vx + context reach) follows its own geometry,
# so the zones are not nested: every combination is its own category
DROP_CATEGORIES = (
    ("kept by all three", (220, 0, 0)),
    ("dropped by cue_dropout only", (255, 140, 0)),
    ("dropped by v8-in only", (255, 220, 0)),
    ("dropped by cue_dropout and v8-in", (0, 170, 0)),
    ("dropped by v8-in and reader-v3", (0, 160, 220)),
    ("dropped by all three", (110, 40, 160)),
)


def drop_category(cue, reader, v8in) -> np.ndarray:
    """per pixel index into DROP_CATEGORIES from the three discard masks."""
    v8in = v8in | reader  # reader-v3's zone is inside v8-in's (checked on every trained sheet)
    index = np.zeros(cue.shape, np.uint8)
    index[cue & ~v8in] = 1
    index[~cue & v8in & ~reader] = 2
    index[cue & v8in & ~reader] = 3
    index[~cue & reader] = 4
    index[cue & reader] = 5
    return index


def export_visualization(out_dir) -> None:
    """every model's discard figures (copied from output/holdout_exclusion*/) into out_dir/<model>/, and per sheet a comparison of
    the three discarded regions plus who dropped what into out_dir/comparison/. Masks and summaries are only read."""
    import shutil

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from utils import mix6
    from utils.fast_figure import orient_display

    out = Path(out_dir)
    (out / "comparison").mkdir(parents=True, exist_ok=True)
    summaries = {}
    for model, (_radius, _context, directory, figures) in specs().items():
        (out / model).mkdir(exist_ok=True)
        for path in sorted(Path(figures).glob("*.png")):
            shutil.copy2(path, out / model / path.name)
        summaries[model] = json.loads((Path(directory) / "summary.json").read_text())["sheets"]
    for alias, (render, name) in mix6.SHEETS_0846A.items():
        masks = {model: cv2.imread(str(Path(directory) / f"{alias}.png"), cv2.IMREAD_GRAYSCALE) > 0
                 for model, (_r, _c, directory, _f) in specs().items()}
        ink = cv2.imread(str(REPO / "dilated_inklabels" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE) > 0
        papyrus = _papyrus(render)
        rows, cols = (min(array.shape[axis] for array in (*masks.values(), ink, papyrus)) for axis in (0, 1))
        masks = {model: mask[:rows, :cols] for model, mask in masks.items()}
        ink, papyrus = ink[:rows, :cols], papyrus[:rows, :cols]
        step = max(1, max(rows, cols) // 1800)
        small = (cols // step, rows // step)

        def reduce(mask):
            return cv2.resize(mask.astype(np.uint8), small, interpolation=cv2.INTER_NEAREST) > 0

        ink_s, papyrus_s = reduce(ink), reduce(papyrus)
        reduced = {model: reduce(mask) for model, mask in masks.items()}
        fig, axes = plt.subplots(1, 4, figsize=(32, 7))
        for ax, (model, discarded) in zip(axes, reduced.items()):
            radius, context, _d, _f = specs()[model]
            rgb = np.full(papyrus_s.shape + (3,), 255, np.uint8)
            rgb[papyrus_s] = (205, 205, 205)
            rgb[papyrus_s & ink_s] = (40, 40, 40)
            rgb[papyrus_s & discarded] = (240, 120, 120)
            rgb[papyrus_s & discarded & ink_s] = (150, 0, 0)
            row = summaries[model][str(alias)]
            reach = f" + {context} px context" if context else ""
            ax.imshow(orient_display(rgb, render), interpolation="nearest")
            ax.set_title(f"{model}: within {radius:g} vox{reach}\ndiscarded {100 * row['discarded_fraction']:.0f}% of papyrus, "
                         f"{100 * row['discarded_ink_fraction']:.0f}% of ink")
        category = drop_category(reduced["cue_dropout"], reduced["reader_v3"], reduced["v8in"])
        rgb = np.full(papyrus_s.shape + (3,), 255, np.uint8)
        rgb[papyrus_s] = (205, 205, 205)
        for index, (_label, color) in enumerate(DROP_CATEGORIES):
            rgb[papyrus_s & ink_s & (category == index)] = color
        axes[3].imshow(orient_display(rgb, render), interpolation="nearest")
        axes[3].set_title("ink by which models dropped it")
        axes[3].legend(handles=[Patch(color=np.array(color) / 255.0, label=label) for label, color in DROP_CATEGORIES],
                       loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=10)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{name.replace('auto_grown_', '')} (alias {alias}): discarded regions near the prize sheet {mix6.HOLDOUT_NAME}",
                     fontsize=14)
        fig.savefig(out / "comparison" / f"{alias}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[export] {alias}", flush=True)
    print(f"[export] figures in {out}/<model>/ and {out}/comparison/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--export", default="", help="copy every model's figures into DIR/<model>/ and write DIR/comparison/")
    parser.add_argument("--only", default="", help="comma-separated aliases: compute just these sheets and merge them into the summaries")
    args = parser.parse_args()
    os.chdir(REPO)
    if args.export:
        export_visualization(args.export)
        return
    build_all(force=args.force, aliases=[int(alias) for alias in args.only.split(",") if alias.strip()] or None)
    for line in all_problems():
        print("[exclusion] PROBLEM:", line)


if __name__ == "__main__":
    main()
