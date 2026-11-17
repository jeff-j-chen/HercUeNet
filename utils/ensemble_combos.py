"""Every product and z-score ensemble of HercUeNet, reader-v3, v8-in and the base Studio model on one prize sheet.

The three trained models run through ensemble_eval.ipynb's own code cells (settings, caches, depth-shift maxing, stretching; the cells are
executed here, the notebook is not modified) and the Studio map comes from studio-multiscroll/infer_ckpts.py base_map. Per model the maps are
stretched as in the notebook: `plain` p1 -> 0 / p99.5 -> 1 (the product), `shown` with p50 -> 0 for reader-v3 and v8-in (the z-score).
Combination of 2..4 models: z-score = mean of the per-map z-scores of the shown maps, product = product of the plain maps; both are saved
from their p1 to p99.5 as PNGs into one folder, with the single maps and an overview.

  ENSEMBLE_CUE=... ENSEMBLE_READER=... ENSEMBLE_V8IN=... COMBO_STUDIO=<studio .npy> python3 utils/ensemble_combos.py [sheet]
"""
import itertools
import json
import os
import sys
from pathlib import Path

# utils/platform.py would shadow the stdlib module when this file is run as a script
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import cv2
import numpy as np

import nbformat  # noqa: E402

SHEET = sys.argv[1] if len(sys.argv) > 1 else "PRIZESHEET_11_16"
OUT = REPO / "output" / f"ensemble_combos_{SHEET}"
OUT.mkdir(parents=True, exist_ok=True)

notebook = nbformat.read(REPO / "ensemble_eval.ipynb", as_version=4)
cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"][:-2]  # without the two one-line figure cells
ns = {"__name__": "ensemble_eval"}
for source in cells:
    exec(compile(source, "<ensemble_eval>", "exec"), ns)
ns["SHEETS"][SHEET] = {"zid": SHEET, "mask_dir": REPO / "masks" / "extras"}

sheet = ns["Sheet"](SHEET)
MODELS, RUNNERS = ns["MODELS"], ns["RUNNERS"]
stretch, stretch_from, z_score, STRETCH = ns["stretch"], ns["stretch_from"], ns["z_score"], ns["STRETCH"]
colorize, orient_display = ns["colorize"], ns["orient_display"]
ID = {"cue_dropout": "hercuenet", "reader_v3": "readerv3", "v8in": "v8in"}
print(f"{SHEET}: frame {sheet.frame}, {int(sheet.mask.sum()):,} papyrus pixels", flush=True)
ns["link_cached"](sheet, MODELS)

plain, shown, info = {}, {}, {}
for name, spec in MODELS.items():
    if not spec.get("in_ensemble", True) or spec["kind"] == "studio":
        continue
    key = ID[spec["kind"]]
    print(f"[{key}] {name}: {spec['checkpoint']}", flush=True)
    raw = sheet.to_frame(RUNNERS[spec["kind"]](sheet, name, spec))
    plain[key], _, _ = stretch(raw, sheet.mask)
    shown[key] = plain[key] if spec.get("floor") is None else stretch_from(raw, sheet.mask, spec["floor"])[0]
    info[key] = {"name": name, "checkpoint": str(spec["checkpoint"]), "depth_shifts": list(spec.get("depths", (0,))), "floor": spec.get("floor")}
    del raw

studio = np.load(os.environ["COMBO_STUDIO"]).astype(np.float32)
raw = sheet.to_frame(studio)
plain["studio"], _, _ = stretch(raw, sheet.mask)
shown["studio"] = plain["studio"]
info["studio"] = {"name": "Studio base (released soft-loss EMA)", "checkpoint": os.environ["COMBO_STUDIO"], "depth_shifts": [0], "floor": None}
del raw, studio

bad = (0.78, 0.86, 0.95)


def save_map(array, path, vmin=None, vmax=None):
    valid = np.isfinite(array)
    low, high = np.percentile(array[valid], STRETCH) if vmin is None else (vmin, vmax)
    rgb = colorize(orient_display(array, sheet.name), "gray_r", bad, vmin=float(low), vmax=float(high))
    cv2.imwrite(str(path), rgb[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])


order = ["hercuenet", "readerv3", "v8in", "studio"]
for key in order:
    save_map(shown[key], OUT / f"single__{key}.png", 0.0, 1.0)

made = []
for size in (4, 3, 2):
    for subset in itertools.combinations(order, size):
        tag = "+".join(subset)
        z = np.mean([z_score(shown[k]) for k in subset], axis=0)
        p = np.prod([plain[k] for k in subset], axis=0).astype(np.float32)
        save_map(z, OUT / f"zscore__{tag}.png")
        save_map(p, OUT / f"product__{tag}.png")
        made.append(tag)
        print("[combo]", tag, flush=True)
        del z, p

# overview: the four single maps, then one row per combination (z-score | product), 900 px wide panels
WIDTH = 900


def panel(path, label):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (WIDTH, round(img.shape[0] * WIDTH / img.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (min(WIDTH, 30 + 12 * len(label)), 26), (0, 0, 0), -1)
    cv2.putText(img, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


rows = [np.hstack([panel(OUT / f"single__{k}.png", k) for k in order[:2]]), np.hstack([panel(OUT / f"single__{k}.png", k) for k in order[2:]])]
for tag in made:
    rows.append(np.hstack([panel(OUT / f"zscore__{tag}.png", f"z-score {tag}"), panel(OUT / f"product__{tag}.png", f"product {tag}")]))
cv2.imwrite(str(OUT / "overview.png"), np.vstack(rows))
(OUT / "manifest.json").write_text(json.dumps({"sheet": SHEET, "models": info, "combinations": made, "stretch_percentiles": list(STRETCH)}, indent=2))
print(f"[done] {len(made)} combinations x 2 methods + 4 singles in {OUT}", flush=True)
