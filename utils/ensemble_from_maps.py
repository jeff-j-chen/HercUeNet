"""A 3x2 / 2x2 ensemble (ensemble_eval.ipynb) from maps that are already cached, with no inference: the maps are written into the notebook's cache under the names it
looks for, and the notebook is run headlessly with the GPU hidden (so nothing can start an inference).

  python3 -m utils.ensemble_from_maps --sheet S --subdir D --studio studio_step500.npy --reader reader_step1000.npy --v8in v8in_epoch.npy --base studio_base.npy \\
      --studio-ckpt ... --reader-ckpt ... --v8in-ckpt ... --v8in-label "epoch 1"

Maps are raw model outputs of the prize sheet (utils/viz_checkpoints.py caches). Output: output/ensemble/<D>/<S>/ (ensemble_3x2.png, zscore_product_ct_2x2.png, ...)."""
import argparse, os, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from utils import mix6  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--sheet", required=True)
parser.add_argument("--subdir", required=True)
for name in ("studio", "reader", "v8in", "base"):
    parser.add_argument(f"--{name}", required=True, help="cached raw map (.npy)")
for name in ("studio", "reader", "v8in"):
    parser.add_argument(f"--{name}-ckpt", required=True)
parser.add_argument("--studio-label", default="step 500")
parser.add_argument("--reader-label", default="step 1000")
parser.add_argument("--v8in-label", default="epoch 1")
args = parser.parse_args()


def slug(text):
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


cache = REPO / "output" / "ensemble" / args.subdir / args.sheet / "cache"
cache.mkdir(parents=True, exist_ok=True)
corpus = mix6.CORPUS
np.save(cache / (slug("Studio base (released soft-loss EMA)") + ".npy"), np.load(args.base).astype(np.float32))
np.save(cache / (slug(f"Studio fine-tune ({corpus}, {args.studio_label})") + ".npy"), np.load(args.studio).astype(np.float32))
reader = np.load(args.reader)
u8 = np.clip(np.rint(reader * 255), 0, 255).astype(np.uint8)
if np.abs(u8 / 255.0 - reader).max() > 1e-6:
    raise SystemExit("the reader map is not a uint8 / 255 map")
name = slug(f"reader-v3 ({corpus} fine-tune, {args.reader_label})")
cv2.imwrite(str(cache / f"{name}.tif"), u8)
np.save(cache / f"{name}.npy", u8)
folder = cache / slug(f"v8-in ({corpus} fine-tune, {args.v8in_label})")
folder.mkdir(exist_ok=True)
np.save(folder / "v8in_prediction.npy", np.load(args.v8in).astype(np.float16))
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", ENSEMBLE_SHEETS=args.sheet, ENSEMBLE_SUBDIR=args.subdir, ENSEMBLE_STUDIO_CKPT=args.studio_ckpt, ENSEMBLE_READER_CKPT=args.reader_ckpt,
           ENSEMBLE_V8IN_CKPT=args.v8in_ckpt, ENSEMBLE_STUDIO_LABEL=args.studio_label, ENSEMBLE_READER_LABEL=args.reader_label, ENSEMBLE_V8IN_LABEL=args.v8in_label)
subprocess.run([sys.executable, str(REPO / "utils" / "run_notebook.py"), str(REPO / "ensemble_eval.ipynb")], check=True, env=env, cwd=REPO)
print("[ensemble] ", REPO / "output" / "ensemble" / args.subdir / args.sheet / "ensemble_3x2.png")
