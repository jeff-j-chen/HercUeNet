"""Seed-averaged ensemble on the prize sheet (utils/queue_ps4.sh step 4).

Per training seed (utils/mix6.py SEEDS, tags s1 / s2): HercUeNet epoch 3 and reader-v3 step 1000 predicted at the depth shifts -2, -1, 0 and
reduced by the pixelwise max, v8-in epoch 3 at the shifts -2, -1 and reduced by the max (the selection of utils/mix6.py, the same as
ensemble_eval.ipynb makes for one seed). The seeds are then averaged model by model and the result is run through
ensemble_eval.ipynb as the three models of one ensemble (z-score / product composites, 3x2 and 2x2 figures):

  python3 utils/seed_ensemble.py [--sheet <name>] [--seeds s1,s2] [--subdir <output/ensemble folder>]

Every per-seed map is cached in output/ensemble/mix28_ps4_seedmaps/<sheet>/cache, so a rerun only computes what is missing or older than
the render or its checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("ENSEMBLE_SUBDIR", "mix28_ps4_seedmaps")  # per-seed maps; the notebook reads this at import
from utils import mix6  # noqa: E402

CUE_EPOCH, READER_STEP, V8IN_EPOCH = 3, 1000, 3
SHIFTS = mix6.CUE_DEPTHS  # HercUeNet and reader-v3 (-2, -1, 0)
V8IN_SHIFTS = mix6.V8IN_DEPTHS  # v8-in (-2, -1)
# the reader reads layers [6 + shift, 23 + shift) of the 28-layer render
READER_START, READER_END = 6, 23


def checkpoints(seed: str) -> dict:
    return {
        "cue": REPO / "models" / "runs" / f"cue_dropout_labelsv6_{mix6.FINETUNE_CORPUS}{mix6.RUN_TAG}_{seed}" / f"epoch_{CUE_EPOCH}.pth",
        "reader": REPO / "reader-v3" / "runs" / f"{mix6.FINETUNE_CORPUS}{mix6.RUN_TAG}_{seed}" / f"ckpt_{READER_STEP:06d}.pth",
        "v8in": REPO / "nader-fork" / "runs" / f"v8in_{mix6.FINETUNE_CORPUS}_{mix6.V8IN_EPOCHS}ep{mix6.RUN_TAG}_{seed}" / "runs" / "*" / "checkpoints"
                / f"trained-epoch{V8IN_EPOCH - 1}.ckpt",
    }


def load_notebook() -> dict:
    notebook = json.loads((REPO / "ensemble_eval.ipynb").read_text())
    namespace = {"__name__": "nb"}
    for index in (1, 2, 3):
        exec("".join(notebook["cells"][index]["source"]), namespace)
    return namespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", default=mix6.HOLDOUT_RENDER)
    parser.add_argument("--seeds", default=",".join(mix6.SEEDS))
    parser.add_argument("--subdir", default=f"mix28{mix6.RUN_TAG}_c{CUE_EPOCH}_r{READER_STEP}_v{V8IN_EPOCH}_seedavg")
    args = parser.parse_args()
    seeds = args.seeds.split(",")
    os.chdir(REPO)
    g = load_notebook()
    import cv2
    import numpy as np
    from utils import predict_models

    sheet, slug = g["Sheet"](args.sheet), g["slug"]
    newest_render = predict_models.zarr_mtime(sheet.zarr_path)
    reader_python = predict_models.venv_python(predict_models.READER_DIR / ".venv")
    cue_maps, reader_maps, v8in_maps = [], [], []
    for seed in seeds:
        paths = checkpoints(seed)
        for kind in ("cue", "reader"):
            if not paths[kind].is_file():
                raise SystemExit(f"[seed ensemble] missing {paths[kind]}")
        if not list(Path("/").glob(str(paths["v8in"]).lstrip("/"))):
            raise SystemExit(f"[seed ensemble] missing {paths['v8in']}")
        cells = []
        for shift in SHIFTS:
            name = f"HercUeNet (mix28, epoch {CUE_EPOCH}, {seed}, depth {shift:+d})"
            spec = {"checkpoint": paths["cue"], "depth_offset": -1 + shift, "infer_batch": 16}
            g["run_cue_dropout"](sheet, name, spec)
            saved = np.load(sheet.cache / f"{slug(name)}.npz")
            cells.append((saved["grid"], int(saved["cell"])))
            print(f"[seed ensemble] HercUeNet {seed} depth {shift:+d} done", flush=True)
        cue_maps.append((np.fmax.reduce([grid for grid, _ in cells]), cells[0][1]))
        layers = []
        for shift in SHIFTS:
            tif = sheet.cache / f"reader_v3__mix28_fine_tune__step_{READER_STEP}__{seed}__depth{shift:+d}.tif"
            npy = tif.with_suffix(".npy")
            newest = max(newest_render, paths["reader"].stat().st_mtime)
            if not npy.is_file() or npy.stat().st_mtime < newest:
                tif.unlink(missing_ok=True)
                npy.unlink(missing_ok=True)
                subprocess.run([reader_python, "-m", "koine_machines.inference.infer", str(predict_models.local_zarr(sheet.zarr_path)),
                                str(paths["reader"]), str(tif), "--overlap", "0.5", "--blend-mode", "hann", "--direction", "forward",
                                "--no-compile", "--batch-size", "4", "--layer-start", str(READER_START + shift),
                                "--layer-end", str(READER_END + shift)], cwd=predict_models.READER_DIR, check=True)
                subprocess.run([reader_python, "-c", "import sys, numpy as np, tifffile; np.save(sys.argv[2], tifffile.imread(sys.argv[1]))",
                                str(tif), str(npy)], check=True)
            layers.append(np.load(npy))
            print(f"[seed ensemble] reader-v3 {seed} depth {shift:+d} done", flush=True)
        reader_maps.append(np.maximum.reduce(layers))
        v8in_layers = [predict_models.predict_v8in(sheet.zarr_path, paths["v8in"],
                                                   sheet.cache / f"v8_in__mix28_fine_tune__epoch_{V8IN_EPOCH}__{seed}__depth{shift:+d}", shift=shift)
                       for shift in V8IN_SHIFTS]
        v8in_maps.append(np.maximum.reduce(v8in_layers))
        print(f"[seed ensemble] v8-in {seed} done", flush=True)

    count = len(seeds)
    labels = {
        "cue": f"epoch {CUE_EPOCH}, max depth -2..0, mean of {count} seeds",
        "reader": f"step {READER_STEP}, max depth -2..0, mean of {count} seeds",
        "v8in": f"epoch {V8IN_EPOCH}, max depth {min(V8IN_SHIFTS)}..{max(V8IN_SHIFTS)}, mean of {count} seeds",
    }
    out = REPO / "output" / "ensemble" / args.subdir / sheet.name / "cache"
    out.mkdir(parents=True, exist_ok=True)
    cell = cue_maps[0][1]
    cue_name, reader_name, v8in_name = (f"HercUeNet (mix28, {labels['cue']})", f"reader-v3 (mix28 fine-tune, {labels['reader']})",
                                        f"v8-in (mix28 fine-tune, {labels['v8in']})")
    np.savez(out / f"{slug(cue_name)}.npz", grid=np.mean([grid for grid, _ in cue_maps], axis=0).astype(np.float32), cell=cell)
    tif = out / f"{slug(reader_name)}.tif"
    reader_mean = np.mean([m.astype(np.float32) for m in reader_maps], axis=0)
    cv2.imwrite(str(tif), np.clip(np.rint(reader_mean), 0, 255).astype(np.uint8))
    np.save(tif.with_suffix(".npy"), np.clip(np.rint(reader_mean), 0, 255).astype(np.uint8))
    v8in_dir = out / slug(v8in_name)
    v8in_dir.mkdir(exist_ok=True)
    np.save(v8in_dir / "v8in_prediction.npy", np.mean([m.astype(np.float32) for m in v8in_maps], axis=0).astype(np.float16))
    print(f"[seed ensemble] averaged caches written to {out}", flush=True)

    first = checkpoints(seeds[0])
    v8in_ckpt = sorted(Path("/").glob(str(first["v8in"]).lstrip("/")))[-1]
    env = {**os.environ, "ENSEMBLE_SHEET": sheet.name, "ENSEMBLE_SUBDIR": args.subdir,
           "ENSEMBLE_CUE": str(first["cue"]), "ENSEMBLE_CUE_LABEL": labels["cue"], "ENSEMBLE_CUE_CORPUS": "mix28",
           "ENSEMBLE_READER": str(first["reader"]), "ENSEMBLE_READER_LABEL": labels["reader"], "ENSEMBLE_READER_CORPUS": "mix28",
           "ENSEMBLE_V8IN": str(v8in_ckpt), "ENSEMBLE_V8IN_LABEL": labels["v8in"], "ENSEMBLE_V8IN_CORPUS": "mix28",
           # the averaged maps are final: the notebook must not max them over depths again
           "ENSEMBLE_CUE_DEPTHS": "0", "ENSEMBLE_READER_DEPTHS": "0", "ENSEMBLE_V8IN_DEPTHS": "0"}
    subprocess.run([sys.executable, "utils/run_notebook.py", "ensemble_eval.ipynb"], cwd=REPO, env=env, check=True)
    print(f"[seed ensemble] done: output/ensemble/{args.subdir}/{sheet.name}", flush=True)


if __name__ == "__main__":
    main()
