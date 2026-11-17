"""Set up the released Studio model for ensemble_eval.ipynb. Studio is used straight out of the box: no fine-tuning, no training data.

  python3 stage_studio.py              # download + verify the weights, build .venv-studio, run a self-test
  python3 stage_studio.py --no-venv    # weights only (the venv is only needed to predict)

Weights: the soft-loss EMA release of priyankaranke/vesuvius-studio-multiscroll-cpt (tag studio-softloss-20261023, 893 MB, CC BY-NC 4.0),
SHA-256 checked, -> models/studio/studio-softloss-20261023-ema.safetensors. The environment (.venv-studio, torch 2.13.0 + numpy +
safetensors, the pins of the release's requirements-inference.txt) is what utils/predict_studio.py runs in; it is built with uv.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
URL = ("https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/releases/download/studio-softloss-20261023/"
       "studio-softloss-20261023-ema.safetensors")
SHA256 = "96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90"
WEIGHTS = REPO / "models" / "studio" / "studio-softloss-20261023-ema.safetensors"
VENV = REPO / ".venv-studio"
PYTHON = VENV / "bin" / "python"
REQUIREMENTS = ["torch==2.13.0", "numpy==2.4.6", "safetensors==0.8.0"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
UV_ENV = dict(os.environ, UV_CACHE_DIR=os.environ.get("UV_CACHE_DIR", str(REPO / "_ves_tmp" / "uv_cache")))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weights():
    if WEIGHTS.is_file() and sha256(WEIGHTS) == SHA256:
        print(f"[studio] weights ok: {WEIGHTS}")
        return
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    partial = WEIGHTS.with_suffix(".partial")
    print(f"[studio] downloading {URL}", flush=True)
    with urllib.request.urlopen(URL) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, 8 << 20)
    if sha256(partial) != SHA256:
        partial.unlink()
        raise SystemExit("[studio] downloaded weights do not match the released SHA-256")
    partial.replace(WEIGHTS)
    print(f"[studio] weights verified and saved to {WEIGHTS}")


def venv_ready():
    if not PYTHON.exists():
        return False
    probe = "import torch, numpy, safetensors; assert torch.__version__.startswith('2.13')"
    return subprocess.run([str(PYTHON), "-c", probe], capture_output=True).returncode == 0


def venv():
    if venv_ready():
        print(f"[studio] environment ok: {VENV}")
        return
    if shutil.which("uv") is None:
        raise SystemExit("[studio] uv is needed to build .venv-studio: pip install uv")
    subprocess.run(["uv", "venv", "--python", "3.11", str(VENV)], check=True, env=UV_ENV)
    subprocess.run(["uv", "pip", "install", "--python", str(PYTHON), "--extra-index-url", TORCH_INDEX, "--index-strategy", "unsafe-best-match",
                    *REQUIREMENTS], check=True, env=UV_ENV)


def selftest():
    subprocess.run([str(PYTHON), str(REPO / "utils" / "predict_studio.py"), "--selftest"], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-venv", action="store_true", help="only download the weights")
    args = parser.parse_args()
    weights()
    if not args.no_venv:
        venv()
        selftest()
    print("[studio] ready: ensemble_eval.ipynb reads", WEIGHTS)
