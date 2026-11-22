"""usage: viz_single.py <ckpt> <label> <sheet> [<sheet>...]  -- one model's map on a sheet, using ensemble_eval.ipynb's own runners."""
import json
import os
import sys

# utils/platform.py would shadow the stdlib module when this file is run as a script
sys.path = [entry for entry in sys.path if os.path.abspath(entry or ".") != os.path.dirname(os.path.abspath(__file__))]

os.environ.setdefault("VESUVIUS_REPO", "/vesuvius")
os.chdir("/vesuvius")
nb = json.load(open("/vesuvius/ensemble_eval.ipynb"))
ns = {"__name__": "viz"}
for i in (1, 2, 3, 4):
    exec(compile("".join(nb["cells"][i]["source"]), f"cell{i}", "exec"), ns)

ckpt, label, sheets = ns["Path"](sys.argv[1]).resolve(), sys.argv[2], sys.argv[3:]
kind = ns["predict_models"].detect_kind(ckpt)
cv2, np = ns["cv2"], ns["np"]
out = ns["REPO"] / "output" / "viz_retrain"
out.mkdir(parents=True, exist_ok=True)
for sheet_name in sheets:
    ns["SHEETS"][sheet_name] = {"zid": sheet_name, "mask_dir": ns["REPO"] / "masks" / "extras"}
    if sheet_name == ns["HOLDOUT_RENDER"]:
        pass
    sheet = ns["Sheet"](sheet_name)
    spec = {"kind": kind, "checkpoint": ckpt, "depth_offset": -1, "infer_batch": 16, "batch_size": 4 if kind == "reader_v3" else 16}
    name = f"viz {label} {sheet_name}"
    raw = sheet.to_frame(ns["RUNNERS"][kind](sheet, name, spec))
    plain, lo, hi = ns["stretch"](raw, sheet.mask)
    rgb = ns["colorize"](ns["orient_display"](plain, sheet.name), "gray_r", (0.78, 0.86, 0.95))
    path = out / f"{kind}_{label}_{sheet_name}.png"
    cv2.imwrite(str(path), rgb[..., ::-1])
    print("saved", path, flush=True)
