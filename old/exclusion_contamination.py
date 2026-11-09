"""how much of each model's kept training supervision a new prize-sheet candidate would contaminate.

For every trained 0846A sheet (utils/mix6.SHEETS_0846A) this applies each model's own exclusion rule (utils/holdout_exclusion.specs:
cue_dropout 12 vox + its context reach, reader-v3 18 vox, v8-in 25 vox) against the candidate sheet's mesh and reports the
papyrus and ink that the current masks KEPT (trained on) but that rule would discard. The current prize sheet is run as a
check (expected 0).

  python3 exclusion_contamination.py fiber_only_20261102b_flatboi [more candidates...]
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

from assemble_test_segments import ROTATE_180
from exclusion_vs_sheet import dense_mesh
from utils import holdout_exclusion as hx
from utils import mix6
from utils.dataloader import split_extra_labels

REPO = Path(__file__).resolve().parent.parent
PIXEL_CM2 = (9.362e-4) ** 2


def candidate_tree(sheet: str) -> cKDTree:
    points, valid, _shape = dense_mesh(REPO / "tifxyz" / sheet, sheet in ROTATE_180, factor=hx.HOLDOUT_DENSE)
    return cKDTree(points[valid])


def contamination(sheet: str) -> dict:
    tree = candidate_tree(sheet)
    specs = hx.specs()
    totals = {m: {"kept_cm2": 0.0, "kept_ink_cm2": 0.0, "contaminated_cm2": 0.0, "contaminated_ink_cm2": 0.0} for m in specs}
    per_sheet = {m: {} for m in specs}
    for alias, (render, name) in mix6.SHEETS_0846A.items():
        distance, shape = hx.dense_distance(name, tree)
        papyrus = hx._papyrus(render)
        label = cv2.imread(str(REPO / "dilated_inklabels" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)
        ink = split_extra_labels(label)[0] if label is not None else np.zeros_like(papyrus)
        for model, (radius, context, directory, _figures) in specs.items():
            kept = papyrus & ~hx.load(alias, shape, directory=directory)
            hit = kept & hx.dilate_square(hx.zone(distance, shape, radius), context)
            row = {"kept_cm2": kept.sum() * PIXEL_CM2, "kept_ink_cm2": (kept & ink).sum() * PIXEL_CM2,
                   "contaminated_cm2": hit.sum() * PIXEL_CM2, "contaminated_ink_cm2": (hit & ink).sum() * PIXEL_CM2}
            for key, value in row.items():
                totals[model][key] += float(value)
            if row["contaminated_cm2"] > 0:
                per_sheet[model][name.replace("auto_grown_20261012065803387_", "").replace("auto_grown_", "")] = {
                    k: round(float(v), 4) for k, v in row.items()}
        print(f"  [{sheet}] {name}", flush=True)
    for model, row in totals.items():
        row["contaminated_fraction"] = row["contaminated_cm2"] / max(row["kept_cm2"], 1e-9)
        row["contaminated_ink_fraction"] = row["contaminated_ink_cm2"] / max(row["kept_ink_cm2"], 1e-9)
        row["rule"] = f"{specs[model][0]:g} vox" + (f" + {specs[model][1]} px context" if specs[model][1] else "")
        row["by_sheet"] = per_sheet[model]
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sheets", nargs="+")
    args = parser.parse_args()
    results = {}
    for sheet in [*args.sheets, mix6.HOLDOUT_NAME]:
        results[sheet] = contamination(sheet)
        for model, row in results[sheet].items():
            print(f"{sheet} {model} ({row['rule']}): {row['contaminated_cm2']:.3f} of {row['kept_cm2']:.2f} cm2 kept "
                  f"({100 * row['contaminated_fraction']:.1f}%), ink {row['contaminated_ink_cm2']:.3f} of "
                  f"{row['kept_ink_cm2']:.3f} cm2 ({100 * row['contaminated_ink_fraction']:.1f}%)", flush=True)
    out = REPO / "output" / "exclusion_contamination" / f"{args.sheets[0]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
