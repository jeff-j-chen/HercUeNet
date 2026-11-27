"""Evaluation of the _ps5 trainings (utils/mix6.py PS5, utils/queue_ps5.sh), with ensemble_eval.ipynb's setup and runners.

  python3 utils/ps5_eval.py prize       every saved checkpoint of both seeds on the test sheet (utils/mix6.py TEST_SHEET), depth 0:
                                        output/ps5/prize/{hercuenet,reader_v3,v8in}.png (rows: seed s1, s2; columns: epoch / step)
  python3 utils/ps5_eval.py reader      the same for reader-v3 only (PS5_BATCH=2 for a small batch)
  python3 utils/ps5_eval.py fragments   v8-in epoch 3 and reader-v3 step 2000 of both seeds on every training sheet of HercUeNet:
                                        output/ps5/fragments/<sheet>.png (rows: reader-v3, v8-in; columns: s1, s2, mean of the two)
"""
import glob
import json
import os
import sys
from pathlib import Path

os.environ["VESUVIUS_PS5"] = "1"
REPO = Path(__file__).resolve().parent.parent
os.environ["ENSEMBLE_SUBDIR"] = "ps5_eval"
os.environ.setdefault("ENSEMBLE_SHEET", "PRIZESHEET_11_11_flatboi_flatboi_abf_copy_copy_abf_copy_abf_flatboi_flatboi")
sys.path.insert(0, str(REPO))
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
notebook = json.loads((REPO / "ensemble_eval.ipynb").read_text())
g = {"__name__": "ps5_eval"}
for index in (1, 2, 3, 4):
    exec(compile("".join(notebook["cells"][index]["source"]), f"ensemble_eval cell {index}", "exec"), g)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from utils import mix6  # noqa: E402

SCALE = 4
BAD = (0.78, 0.86, 0.95)
OUT = REPO / "output" / ("ps5" if mix6.CORPUS == "mix28" else f"ps5_{mix6.CORPUS}")
SEEDS = tuple(mix6.SEEDS)
CUE_RUN = lambda seed: REPO / "models" / "runs" / f"cue_dropout_labelsv6_{mix6.CORPUS}{mix6.RUN_TAG}_{seed}"  # noqa: E731
READER_RUN = lambda seed: REPO / "reader-v3" / "runs" / f"{mix6.FINETUNE_CORPUS}{mix6.RUN_TAG}_{seed}"  # noqa: E731
V8IN_CKPTS = lambda seed: REPO / "nader-fork" / "runs" / f"v8in_{mix6.FINETUNE_CORPUS}_{mix6.V8IN_EPOCHS}ep{mix6.RUN_TAG}_{seed}" / "runs" / "*" / "checkpoints"  # noqa: E731


def v8in_ckpt(seed, epoch):
    found = sorted(glob.glob(str(V8IN_CKPTS(seed) / f"trained-epoch{epoch - 1}.ckpt")))
    return Path(found[-1]) if found else None


def checkpoints(kind, seed):
    """[(label, path)] of every saved checkpoint of the kind and seed."""
    if kind == "hercuenet":
        return [(f"epoch {n}", CUE_RUN(seed) / f"epoch_{n}.pth") for n in mix6.CUE_SAVE_EPOCHS]
    if kind == "reader_v3":
        return [(f"step {n}", READER_RUN(seed) / f"ckpt_{n:06d}.pth") for n in range(1000, mix6.READER_STEPS + 1, 1000)]
    return [(f"epoch {n}", v8in_ckpt(seed, n)) for n in range(1, mix6.V8IN_EPOCHS + 1)]


def predict(sheet, kind, seed, label, path):
    name = {"hercuenet": "HercUeNet", "reader_v3": "reader-v3", "v8in": "v8-in"}[kind] + f" (ps5{'' if mix6.CORPUS == 'mix28' else ' ' + mix6.CORPUS} {seed} {label})"
    if kind == "hercuenet":
        return sheet.to_frame(g["run_cue_dropout"](sheet, name, {"checkpoint": path, "depth_offset": -1}))
    runner = g["run_reader_v3"] if kind == "reader_v3" else g["run_v8in"]
    spec = {"checkpoint": path, **({"batch_size": int(os.environ["PS5_BATCH"])} if "PS5_BATCH" in os.environ else {})}
    return sheet.to_frame(runner(sheet, name, spec))


def panel(sheet, array, title):
    shown, low, high = g["stretch"](array, sheet.mask)
    rgb = g["colorize"](g["orient_display"](shown, sheet.name), "gray_r", BAD, vmin=0.0, vmax=1.0)
    rgb = cv2.resize(np.ascontiguousarray(rgb), (rgb.shape[1] // SCALE, rgb.shape[0] // SCALE), interpolation=cv2.INTER_AREA)
    cv2.putText(rgb, f"{title} | p1->0 ({low:.3f}), p99.5->1 ({high:.3f})", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return rgb


def grid(rows, path, gap=24):
    width = max(sum(p.shape[1] for p in row) + gap * (len(row) - 1) for row in rows)
    lines = []
    for row in rows:
        line = np.hstack([np.hstack([p, np.full((p.shape[0], gap, 3), 255, np.uint8)]) if i < len(row) - 1 else p for i, p in enumerate(row)])
        lines.append(np.pad(line, ((0, 0), (0, width - line.shape[1]), (0, 0)), constant_values=255))
    full = np.vstack([np.vstack([line, np.full((gap, width, 3), 255, np.uint8)]) if i < len(lines) - 1 else line for i, line in enumerate(lines)])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), full[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"could not write {path}")
    print(f"Saved {path} {full.shape[1]}x{full.shape[0]}", flush=True)


def prize(kinds=("hercuenet", "reader_v3", "v8in")):
    sheet = g["Sheet"](os.environ["ENSEMBLE_SHEET"])
    for kind in kinds:
        rows = []
        for seed in SEEDS:
            row = []
            for label, path in checkpoints(kind, seed):
                if path is None or not Path(path).is_file():
                    print(f"[prize] MISSING {kind} {seed} {label}: {path}", flush=True)
                    continue
                print(f"[prize] {kind} {seed} {label}", flush=True)
                array = predict(sheet, kind, seed, label, path)
                row.append(panel(sheet, array, f"{kind} {seed} {label}"))
                del array
            if row:
                rows.append(row)
        if rows:
            grid(rows, OUT / "prize" / f"{kind}.png")


def fragments():
    reader = {seed: READER_RUN(seed) / "ckpt_002000.pth" for seed in SEEDS}
    v8in = {seed: v8in_ckpt(seed, 3) for seed in SEEDS}
    missing = [str(p) for p in (*reader.values(), *v8in.values()) if p is None or not Path(p).is_file()]
    if missing:
        raise SystemExit("missing checkpoints: " + ", ".join(missing))
    failed = []
    for index, name in enumerate(g["TRAIN_SHEETS"], start=1):
        print(f"[fragments {index}/{len(g['TRAIN_SHEETS'])}] {name}", flush=True)
        try:
            sheet = g["Sheet"](name)
            rows = []
            for kind, paths, title, label in (("reader_v3", reader, "reader-v3 step 2000", "step 2000"), ("v8in", v8in, "v8-in epoch 3", "epoch 3")):
                maps = [predict(sheet, kind, seed, label, paths[seed]) for seed in SEEDS]
                shown = [g["stretch"](m, sheet.mask)[0] for m in maps]
                row = [panel(sheet, m, f"{title} {seed}") for m, seed in zip(maps, SEEDS)]
                row.append(panel(sheet, np.mean(shown, axis=0), f"{title} mean of {len(SEEDS)} seeds"))
                rows.append(row)
            grid(rows, OUT / "fragments" / f"{name}.png")
        except Exception as error:  # noqa: BLE001 - the next sheet still runs
            print(f"[fragments] FAILED {name}: {error!r}", flush=True)
            failed.append(name)
    print(f"[fragments] {len(failed)} failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "prize":
        prize()
    elif mode == "reader":
        prize(("reader_v3",))
    elif mode == "cue_reader":
        prize(("hercuenet", "reader_v3"))
    elif mode == "fragments":
        sys.exit(fragments())
    else:
        raise SystemExit(__doc__)
