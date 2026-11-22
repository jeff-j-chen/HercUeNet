"""The released Studio model (finetune_studio.py setup) on every PHerc0846A sheet: 16 ascending planes around surface layer 13, stride 64 + Hann.
-> output/studio_base_0846A/<alias>.png (probability x 255)   python3 utils/studio_all_0846A.py"""
import subprocess, sys, tempfile
from pathlib import Path
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
import cv2, numpy as np, zarr
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils.mix6 import HOLDOUT, HOLDOUT_RENDER, SHEETS_0846A
OUT = REPO / "output" / "studio_base_0846A"; OUT.mkdir(parents=True, exist_ok=True)
SHEETS = {**{str(a): str(REPO / "ves_zarrs2" / f"{r}.zarr") for a, (r, _) in SHEETS_0846A.items()}, str(HOLDOUT): str(REPO / "ves_zarrs2" / f"{HOLDOUT_RENDER}.zarr")}
for alias, path in SHEETS.items():
    png = OUT / f"{alias}.png"
    if png.is_file():
        continue
    planes = np.clip(np.asarray(zarr.open(path, mode="r")[5:21]), 0, 255).astype(np.uint8)
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        np.save(f"{tmp}/p.npy", planes)
        subprocess.run([sys.executable, str(REPO / "utils/predict_studio.py"), "--input", f"{tmp}/p.npy", "--output", f"{tmp}/o.npy"], check=True)
        prob = np.load(f"{tmp}/o.npy")
    cv2.imwrite(str(png), (prob * 255).astype(np.uint8))
    print("[done]", alias, prob.shape, round(float(prob.mean()), 3), flush=True)
