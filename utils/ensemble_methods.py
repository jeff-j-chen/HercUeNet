"""Other ways to combine the four maps of ensemble_combos.py (cached predictions, no inference): all 4 and hercuenet+readerv3+studio.

Maps: `plain` = p1 -> 0 / p99.5 -> 1, `shown` = as in the figures (p50 -> 0 for reader-v3 and v8-in). Methods (n = models):
  mean         mean of shown                              (the notebook's "mean")
  rank         mean of each shown map's rank score (0..1) (the notebook's "rank")
  harmonic     harmonic mean of plain (+0.02): a low map drags it down
  max          max of shown: any model suffices (union)
  min          min of shown: all models must agree (intersection)
  median       median of shown: robust to one outlier model
  logit_mean   mean of logit(plain clipped to 0.02..0.98): agreement in odds space, a soft AND / OR
  mean_minus_std  mean of shown - 0.5 * std of shown across models: disagreement is penalised
  vote         share of models whose shown map is above its own 90th percentile, blurred (sigma 2 px): a consensus count
Each composite is shown from its p1 to p99.5 (vote: 0 to 1) like the product / z-score figures.

  ENSEMBLE_* (as ensemble_combos.py) COMBO_STUDIO=<studio .npy> python3 utils/ensemble_methods.py [sheet]
"""
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
import nbformat
import numpy as np

SHEET = sys.argv[1] if len(sys.argv) > 1 else "PRIZESHEET_11_19_flatboi"
OUT = REPO / "output" / f"ensemble_methods_{SHEET}"
OUT.mkdir(parents=True, exist_ok=True)

notebook = nbformat.read(REPO / "ensemble_eval.ipynb", as_version=4)
ns = {"__name__": "ensemble_eval"}
for source in [cell.source for cell in notebook.cells if cell.cell_type == "code"][:-2]:
    exec(compile(source, "<ensemble_eval>", "exec"), ns)
ns["SHEETS"][SHEET] = {"zid": SHEET, "mask_dir": REPO / "masks" / "extras"}
sheet = ns["Sheet"](SHEET)
MODELS, RUNNERS = ns["MODELS"], ns["RUNNERS"]
stretch, stretch_from, z_score, rank_score, STRETCH = ns["stretch"], ns["stretch_from"], ns["z_score"], ns["rank_score"], ns["STRETCH"]
colorize, orient_display = ns["colorize"], ns["orient_display"]
ID = {"cue_dropout": "hercuenet", "reader_v3": "readerv3", "v8in": "v8in"}
ns["link_cached"](sheet, MODELS)

plain, shown = {}, {}
for name, spec in MODELS.items():
    if not spec.get("in_ensemble", True) or spec["kind"] == "studio":
        continue
    key = ID[spec["kind"]]
    raw = sheet.to_frame(RUNNERS[spec["kind"]](sheet, name, spec))
    plain[key], _, _ = stretch(raw, sheet.mask)
    shown[key] = plain[key] if spec.get("floor") is None else stretch_from(raw, sheet.mask, spec["floor"])[0]
    del raw
raw = sheet.to_frame(np.load(os.environ["COMBO_STUDIO"]).astype(np.float32))
plain["studio"], _, _ = stretch(raw, sheet.mask)
shown["studio"] = plain["studio"]
del raw

EPS = 0.02


def blur_nan(array, sigma=2.0):
    valid = np.isfinite(array)
    filled = np.where(valid, array, 0).astype(np.float32)
    weight = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sigma)
    blurred = cv2.GaussianBlur(filled, (0, 0), sigma) / np.maximum(weight, 1e-6)
    return np.where(valid, blurred, np.nan).astype(np.float32)


def vote(maps):
    flags = []
    for m in maps:
        valid = np.isfinite(m)
        flags.append(np.where(valid, (m > np.percentile(m[valid], 90)).astype(np.float32), np.nan))
    return blur_nan(np.mean(flags, axis=0))


METHODS = {
    "mean": lambda s, p: np.mean(s, axis=0),
    "rank": lambda s, p: np.mean([rank_score(m) for m in s], axis=0),
    "harmonic": lambda s, p: len(p) / np.sum([1.0 / (m + EPS) for m in p], axis=0),
    "max": lambda s, p: np.max(s, axis=0),
    "min": lambda s, p: np.min(s, axis=0),
    "median": lambda s, p: np.median(s, axis=0),
    "logit_mean": lambda s, p: np.mean([np.log(np.clip(m, EPS, 1 - EPS) / (1 - np.clip(m, EPS, 1 - EPS))) for m in p], axis=0),
    "mean_minus_std": lambda s, p: np.mean(s, axis=0) - 0.5 * np.std(s, axis=0),
    "vote": lambda s, p: vote(s),
}
COMBOS = {"all4": ("hercuenet", "readerv3", "v8in", "studio"), "no_v8in": ("hercuenet", "readerv3", "studio")}
bad = (0.78, 0.86, 0.95)
for combo, keys in COMBOS.items():
    s = [shown[k] for k in keys]
    p = [plain[k] for k in keys]
    for method, fn in METHODS.items():
        array = fn(s, p).astype(np.float32)
        array[~np.isfinite(shown[keys[0]])] = np.nan
        valid = np.isfinite(array)
        low, high = (0.0, 1.0) if method == "vote" else np.percentile(array[valid], STRETCH)
        rgb = colorize(orient_display(array, sheet.name), "gray_r", bad, vmin=float(low), vmax=float(high))
        cv2.imwrite(str(OUT / f"{method}__{'+'.join(keys)}.png"), rgb[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
        print("[method]", combo, method, flush=True)
        del array

WIDTH = 900
REF = REPO / "output" / f"ensemble_combos_{SHEET}"


def panel(path, label):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (WIDTH, round(img.shape[0] * WIDTH / img.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (min(WIDTH, 30 + 12 * len(label)), 26), (0, 0, 0), -1)
    cv2.putText(img, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


rows = []
names = [(f"{kind}", REF / f"{kind}__{'+'.join(COMBOS['all4'])}.png", REF / f"{kind}__{'+'.join(COMBOS['no_v8in'])}.png") for kind in ("zscore", "product")]
names += [(m, OUT / f"{m}__{'+'.join(COMBOS['all4'])}.png", OUT / f"{m}__{'+'.join(COMBOS['no_v8in'])}.png") for m in METHODS]
for label, left, right in names:
    rows.append(np.hstack([panel(left, f"{label} all 4"), panel(right, f"{label} hercuenet+readerv3+studio")]))
cv2.imwrite(str(OUT / "overview.png"), np.vstack(rows))
(OUT / "methods.json").write_text(json.dumps({"combos": {k: list(v) for k, v in COMBOS.items()}, "methods": list(METHODS), "doc": __doc__}, indent=2))
print(f"[done] {len(METHODS)} methods x {len(COMBOS)} combos in {OUT}", flush=True)
