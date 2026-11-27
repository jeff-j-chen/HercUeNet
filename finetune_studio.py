"""Studio: set up the released model, stage the corpus, fine-tune it on the PHerc0846 sheets (models/studio/studio-softloss-20261023-ema.safetensors, 223M parameters).

  python3 finetune_studio.py              # setup -> stage -> train (all of it can be re-run; setup and stage skip what is done)
  python3 finetune_studio.py setup        # system python: download + verify the released weights, build .venv-studio (uv), self-test (--no-venv: weights only)
  python3 finetune_studio.py stage        # system python: 16 planes + label + supervision mask of every sheet -> _ves_tmp/studio_data
  .venv-studio/bin/python finetune_studio.py train [--steps 2000] [--stop-step 500] [--micro-batch 4] [--accum 4] [--lr 2e-5]

Released weights: the soft-loss EMA release of priyankaranke/vesuvius-studio-multiscroll-cpt (tag studio-softloss-20261023, 893 MB, CC BY-NC 4.0), SHA-256 checked. The
environment (.venv-studio, torch 2.13.0 + numpy + safetensors, the pins of the release's requirements-inference.txt) is what utils/predict_studio.py and train run in.

Defaults = the model the ensemble uses: corpus mix28x_new, the learning-rate schedule of a 2000-step run stopped (and saved) at step 500 (--stop-step), jitter 40,
background weight 1, soft nnPU (weight 0.5, prior 0.2); the checkpoint is models/studio/mix28x_new_ps7/step_000500.safetensors.

Corpus (utils/mix6.py, corpus mix28x_new = the regenerated x export; mix28x = the first one): the 0846A / 0846B sheets only, no donors. Two groups sampled half and half: HAND (the 14 mix28 sheets + the 2
hand-labelled 0846B prize sheets, reliable) and X (the 101 HerculeX sheets). Inside a group every sheet is equally likely.
Data: the 16 planes of a render in ascending layer order around the flat surface (layers 5..20 of the 28, surface at index 8, as ensemble_eval.ipynb
run_studio feeds Studio); windows of 128 px, the central 96 px supervised (the tiling of utils/predict_studio.py); input clip(raw, 0, 200) / 255.
Labels: 255 strokes of the dilated inklabels are ink; half-brightness strokes (48..207) are NOT ink and stay supervised background. Supervision is the
papyrus mask. The mix28 sheets lose their papyrus near the prize sheet (the 18 vox mask of holdout_exclusion_reader/, shared with reader-v3; those voxels are
zeroed in the staged planes): nothing trains on the prize sheet.
Loss: BCE (background weight --neg-weight, default 1; 0.8 in the first fine-tune, which over-predicted ink) + 0.25 * soft dice over the supervised pixels, plus
optional --ohem-frac (hard-negative mining) and --pu-weight (non-negative PU) terms. Half of the windows contain a sampled ink pixel, half a random
supervised pixel, each placed anywhere in the supervised core (--jitter); the only augmentation is a small intensity jitter.
Optimiser: AdamW, lr 2e-5 with 100 warm-up steps and a cosine decay to 10% over --steps (2000); one step = micro-batch x accumulation windows (4 x 4 = 16).
bf16 autocast. The model is huge: micro-batch 4 is the largest that fits next to nothing else on the 20 GB card; lower --micro-batch (and raise --accum)
if it does not.
Output: models/studio/mix28x_new_ps7/step_<n>.safetensors (the keys of the released file) every --save-every steps (500) and at --stop-step; the training stops there
with the learning rate of the full --steps schedule (a 2000-step run has the same learning rate at step 500).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "_ves_tmp" / "studio_data"
OUT = REPO / "models" / "studio" / "mix28x_new_ps7"  # --out; utils/mix6.py STUDIO_RUN
RELEASED = REPO / "models" / "studio" / "studio-softloss-20261023-ema.safetensors"
URL = ("https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/releases/download/studio-softloss-20261023/"
       "studio-softloss-20261023-ema.safetensors")
SHA256 = "96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90"
VENV = REPO / ".venv-studio"
PYTHON = VENV / "bin" / "python"
REQUIREMENTS = ["torch==2.13.0", "numpy==2.4.6", "safetensors==0.8.0"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
UV_ENV = dict(os.environ, UV_CACHE_DIR=os.environ.get("UV_CACHE_DIR", str(REPO / "_ves_tmp" / "uv_cache")))
FIRST_PLANE, PLANES = 5, 16  # flat depth 13 - 8


# ---- setup (system python) ---------------------------------------------------------------------------------------------------

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weights():
    if RELEASED.is_file() and sha256(RELEASED) == SHA256:
        print(f"[studio] weights ok: {RELEASED}")
        return
    RELEASED.parent.mkdir(parents=True, exist_ok=True)
    partial = RELEASED.with_suffix(".partial")
    print(f"[studio] downloading {URL}", flush=True)
    with urllib.request.urlopen(URL) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, 8 << 20)
    if sha256(partial) != SHA256:
        partial.unlink()
        raise SystemExit("[studio] downloaded weights do not match the released SHA-256")
    partial.replace(RELEASED)
    print(f"[studio] weights verified and saved to {RELEASED}")


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


def setup(with_venv: bool = True) -> None:
    weights()
    if with_venv:
        venv()
        subprocess.run([str(PYTHON), str(REPO / "utils" / "predict_studio.py"), "--selftest"], check=True)
    print("[studio] ready: ensemble_eval.ipynb reads", RELEASED)


# ---- stage (system python) -------------------------------------------------------------------------------------------------

def stage_one(item) -> str:
    import cv2
    import numpy as np

    from utils import mix6, predict_models
    from utils.dataloader import split_extra_labels

    alias, segment = item
    folder = DATA / alias
    stamp = folder / "done.json"
    zarr_path = mix6.volume_path(alias)
    mask = cv2.imread(str(mix6.render_mask_path(segment.render)), cv2.IMREAD_GRAYSCALE) > 0
    label = cv2.imread(str(REPO / "dilated_inklabels" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)
    exclusion = REPO / "holdout_exclusion_reader" / f"{alias}.png"  # Studio and reader-v3 share one zone (18 vox of the prize sheet)
    first = segment.layer - 8  # the surface layer at plane index 8 (5 for the flat depth-13 PHerc0846 renders)
    state = {"label": int(label.sum()), "mask": int(mask.sum()),
             "render": mix6.zarr_mtime(zarr_path), "excl": exclusion.stat().st_mtime if exclusion.is_file() else 0}
    if first != FIRST_PLANE:
        state["first"] = first
    signature = json.dumps(state)
    if stamp.is_file() and stamp.read_text() == signature:
        return f"{alias}: up to date"
    rows, cols = min(mask.shape[0], label.shape[0]), min(mask.shape[1], label.shape[1])
    mask, label = mask[:rows, :cols], label[:rows, :cols]
    ink, extra = split_extra_labels(label)
    if segment.split_masked:  # donors: the train region only (255 trains, 119 = hard negative, black unused), half-brightness strokes unused
        split = cv2.imread(str(REPO / "train_masks" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)[:rows, :cols]
        mask = mask & ((split >= 240) | ((split >= 100) & (split <= 143))) & ~extra
    if exclusion.is_file():
        discard = cv2.imread(str(exclusion), cv2.IMREAD_GRAYSCALE)[:rows, :cols] > 0
        mask, ink = mask & ~discard, ink & ~discard
    else:
        discard = None
    volume = predict_models.read_volume(zarr_path, first, first + PLANES)[:, :rows, :cols]
    if discard is not None:
        volume[:, discard] = 0  # the zone reads as outside the sheet, like the staged volumes of reader-v3 and v8-in
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "planes.npy", np.ascontiguousarray(np.clip(volume, 0, 255).astype(np.uint8)))
    np.save(folder / "ink.npy", ink.astype(np.uint8))
    np.save(folder / "mask.npy", mask.astype(np.uint8))
    stamp.write_text(signature)
    return f"{alias}: {rows}x{cols} ink={int(ink.sum())} supervised={int(mask.sum())}"


def stage(jobs: int = 4, corpus: str = "mix28x_new") -> None:
    sys.path.insert(0, str(REPO))
    from concurrent.futures import ThreadPoolExecutor

    from utils import mix6

    # every sheet any corpus of --corpus uses: hand-grown 0846A/B sheets ("hand"), the HerculeX ones ("x") and the labelled donor patches
    # ("donor:<scroll>"); the manifest is written at the end (atomically), so a run that is already training keeps the one it read
    segments = dict(mix6.corpus_segments(corpus))
    manifest = {alias: {"group": "x" if segment.scroll == "0846Ax" else "hand" if segment.scroll == "0846A" else f"donor:{segment.scroll}",
                        "name": segment.assembler} for alias, segment in segments.items()}
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"[studio-stage] {len(manifest)} sheets: {sum(v['group'] == 'hand' for v in manifest.values())} hand, "
          f"{sum(v['group'] == 'x' for v in manifest.values())} x, {sum(v['group'].startswith('donor') for v in manifest.values())} donor", flush=True)
    with ThreadPoolExecutor(jobs) as pool:
        for line in pool.map(stage_one, segments.items()):
            print("[studio-stage]", line, flush=True)
    (DATA / "manifest.tmp").write_text(json.dumps(manifest, indent=1))
    os.replace(DATA / "manifest.tmp", DATA / "manifest.json")


# ---- train (.venv-studio) ----------------------------------------------------------------------------------------------------

def region(name: str) -> str:
    """which half of the scroll a 0846 sheet belongs to: B (0846B partials / blanks and the two B prize sheets) or A."""
    return "B" if name.startswith(("PHerc0846B", "PRIZESHEET_B")) else "A"


CORPORA = {
    # name: (what it is) -- the sheets and the group weights are chosen in select()
    "legacy": "the 117 sheets of the first experiments: hand 16 + x 101 (no blank export), no donors; hand : x = 1 : 1",
    "a_hand": "the 14 hand-grown 0846A sheets of mix28 only",
    "ab_hand": "the 14 hand-grown 0846A sheets + the 2 hand-labelled 0846B prize sheets",
    "a_all": "0846A only: the 14 hand-grown sheets + the 54 HerculeX marks sheets, hand : x = 1 : 1",
    "ab_all": "all 0846A / 0846B sheets: 16 hand + every HerculeX sheet (54 + 47 + 20), hand : x = 1 : 1",
    "mix28": "the mix28 corpus: the 14 hand-grown 0846A sheets (weight 2) + the labelled donor patches (weight 1 per scroll)",
    "mix28x": "mix28x: 16 hand + every HerculeX sheet + the donors, hand : x : each donor scroll = 4 : 2 : 1 (0846A/B are 43% of the draws)",
    "mix28x_new": "mix28x_new: 16 hand + the regenerated x export (253 sheets) + the donors, hand : x : each donor scroll = 4 : 2 : 1 (0846A/B are 43% of the draws)",
}


def select(manifest: dict, corpus: str):
    """-> (groups {name: [alias]}, weights {name: float}) of a corpus."""
    def in_group(group):
        return [a for a, row in manifest.items() if row["group"] == group]

    hand, x = in_group("hand"), in_group("x")
    scrolls = sorted({row["group"] for row in manifest.values() if row["group"].startswith("donor:")})
    donors = {g: in_group(g) for g in scrolls}
    hand_a = [a for a in hand if region(manifest[a]["name"]) == "A"]
    if corpus == "legacy":
        groups = {"hand": hand, "x": [a for a in x if not manifest[a]["name"].startswith("PHerc0846B_blank")]}
        weights = {"hand": 1, "x": 1}
    elif corpus == "a_hand":
        groups, weights = {"hand": hand_a}, {"hand": 1}
    elif corpus == "ab_hand":
        groups, weights = {"hand": hand}, {"hand": 1}
    elif corpus == "a_all":
        groups = {"hand": hand_a, "x": [a for a in x if region(manifest[a]["name"]) == "A"]}
        weights = {"hand": 1, "x": 1}
    elif corpus == "ab_all":
        groups, weights = {"hand": hand, "x": x}, {"hand": 1, "x": 1}
    elif corpus == "mix28":
        groups, weights = {"hand": hand_a, **donors}, {"hand": 2, **{g: 1 for g in donors}}
    elif corpus in ("mix28x", "mix28x_new"):
        groups, weights = {"hand": hand, "x": x, **donors}, {"hand": 4, "x": 2, **{g: 1 for g in donors}}
    else:
        raise SystemExit(f"unknown corpus {corpus!r}: {sorted(CORPORA)}")
    return groups, weights


def train(args) -> None:
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from torch.nn import functional as F

    sys.path.insert(0, str(REPO / "utils"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("predict_studio", REPO / "utils" / "predict_studio.py")
    studio = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(studio)

    manifest = json.loads((DATA / "manifest.json").read_text())
    groups, weights = select(manifest, args.corpus)
    groups = {g: aliases for g, aliases in groups.items() if aliases}
    used = [a for aliases in groups.values() for a in aliases]
    sheets = {alias: {k: np.load(DATA / alias / f"{k}.npy", mmap_mode="r") for k in ("planes", "ink", "mask")} for alias in used}
    names = list(groups)
    probability = np.array([weights[g] for g in names], dtype=float)
    probability /= probability.sum()
    print(f"[studio] corpus {args.corpus} ({CORPORA[args.corpus]}): " + ", ".join(
        f"{g} {len(groups[g])} sheets p={p:.2f}" for g, p in zip(names, probability)), flush=True)
    torch.manual_seed(args.seed)

    class Windows(torch.utils.data.IterableDataset):
        """endless random windows; every worker has its own generator."""

        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            rng = np.random.default_rng([args.seed, info.id if info else 0])
            ink_cache = {}
            while True:
                group = groups[names[int(rng.choice(len(names), p=probability))]]
                alias = group[int(rng.integers(len(group)))]
                sheet = sheets[alias]
                height, width = sheet["mask"].shape
                if alias not in ink_cache:
                    ys, xs = np.nonzero(sheet["ink"][::2, ::2])
                    ink_cache[alias] = (ys * 2, xs * 2)
                ys, xs = ink_cache[alias]
                centre = None
                if len(ys) and rng.random() < 0.5:
                    k = int(rng.integers(len(ys)))
                    centre = (int(ys[k]), int(xs[k]))
                else:
                    for _ in range(40):
                        y, x = int(rng.integers(height)), int(rng.integers(width))
                        if sheet["mask"][y, x]:
                            centre = (y, x)
                            break
                if centre is None:
                    continue
                # the chosen pixel lands anywhere in the supervised 96 px core, not on the window centre: a centred ink pixel taught the model
                # "the window centre is ink", which paints a blob on every tile centre of the 64 px inference lattice (the blocky maps of the
                # first fine-tune)
                dy, dx = (int(v) for v in rng.integers(-args.jitter, args.jitter + 1, 2)) if args.jitter else (0, 0)
                y0, x0 = centre[0] - 64 + dy, centre[1] - 64 + dx
                tile = _crop(sheet["planes"], y0, x0, "reflect", np)
                label = _crop(sheet["ink"][None], y0, x0, "constant", np)[0, 16:112, 16:112]
                valid = _crop(sheet["mask"][None], y0, x0, "constant", np)[0, 16:112, 16:112]
                if not valid.any():
                    continue
                tile = np.clip(tile.astype(np.float32) * rng.uniform(0.92, 1.08) + rng.uniform(-6, 6), 0, 255)
                yield (torch.from_numpy(tile / 255.0).unsqueeze(0), torch.from_numpy(label.astype(np.float32)),
                       torch.from_numpy(valid.astype(np.float32)))

    loader = torch.utils.data.DataLoader(Windows(), batch_size=args.micro_batch, num_workers=args.workers, prefetch_factor=4,
                                         persistent_workers=True)
    device = "cuda"
    model = studio.DeepResidualUNet()
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(args.init), device="cpu"), strict=True)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)

    def lr_at(step):  # step counted from 0
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()}, "corpus": args.corpus, "groups": {g: len(a) for g, a in groups.items()}}, indent=1))
    def extra_loss(logits, label, valid):
        """terms that punish ink on background (the first fine-tune over-predicted ink and made the maps unreadable)."""
        total = logits.new_zeros(())
        background = valid * (label < 0.5)
        if args.ohem_frac > 0:
            # hard-negative mining: the worst ohem_frac of the background pixels of each window, weighted ohem_weight
            loss_map = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none") * background
            flat = loss_map.flatten(1)
            k = max(1, int(args.ohem_frac * flat.shape[1]))
            total = total + args.ohem_weight * flat.topk(k, dim=1).values.mean()
        if args.pu_weight > 0:
            # non-negative PU risk (as HercUeNet's strong nnPU): the labelled ink is the positive class, everything else is unlabelled and
            # holds an unknown share pi of hidden ink; the background term is max(0, E_u[l(-)] - pi * E_p[l(-)]) so unlabelled pixels that
            # look like ink are only forgiven up to the prior
            positive = valid * (label > 0.5)
            neg_loss = F.softplus(logits)  # -log(1 - sigmoid(x)) = loss of predicting ink
            if positive.sum() > 0:
                e_p = (neg_loss * positive).sum() / positive.sum().clamp_min(1.0)
                e_u = (neg_loss * background).sum() / background.sum().clamp_min(1.0)
                total = total + args.pu_weight * torch.clamp(e_u - args.pu_prior * e_p, min=0)
        return total

    iterator = iter(loader)
    start = time.time()
    running = []
    for step in range(min(args.steps, args.stop_step)):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.accum):
            tile, label, valid = (t.to(device, non_blocking=True) for t in next(iterator))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(tile)
            logits = logits[:, 0, 16:112, 16:112].float()
            weight = valid * torch.where(label > 0.5, torch.ones_like(label), torch.full_like(label, args.neg_weight))
            bce = (F.binary_cross_entropy_with_logits(logits, label, reduction="none") * weight).sum() / weight.sum().clamp_min(1.0)
            probability = logits.sigmoid() * valid
            dice = 1 - (2 * (probability * label).sum() + 1) / (probability.sum() + (label * valid).sum() + 1)
            loss = (bce + 0.25 * dice + extra_loss(logits, label, valid)) / args.accum
            loss.backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running.append(total)
        if (step + 1) % 20 == 0:
            elapsed = time.time() - start
            print(f"[studio] step {step + 1}/{min(args.steps, args.stop_step)} (schedule {args.steps}) loss {np.mean(running):.4f} lr {lr_at(step):.2e} "
                  f"{elapsed / (step + 1):.1f}s/step, eta {elapsed / (step + 1) * (min(args.steps, args.stop_step) - step - 1) / 60:.0f} min, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            running = []
        if (step + 1) % args.save_every == 0 or step + 1 == min(args.steps, args.stop_step):
            path = OUT / f"step_{step + 1:06d}.safetensors"
            save_file({k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}, str(path))
            print(f"[studio] saved {path}", flush=True)


def _crop(array, y, x, mode, np):
    """array[..., 128, 128] window at (y, x), padded with `mode` where it leaves the sheet (as utils/predict_studio.py crop)."""
    h, w = array.shape[-2:]
    tile = np.asarray(array[..., max(0, y):min(h, y + 128), max(0, x):min(w, x + 128)])
    padding = [(0, 0)] * (tile.ndim - 2) + [(max(0, -y), max(0, y + 128 - h)), (max(0, -x), max(0, x + 128 - w))]
    if any(a or b for a, b in padding):
        tile = np.pad(tile, padding, mode=mode)
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="all", choices=("all", "setup", "stage", "train"))
    parser.add_argument("--no-venv", action="store_true", help="setup: only download the weights")
    parser.add_argument("--steps", type=int, default=2000, help="length of the learning-rate schedule")
    parser.add_argument("--stop-step", type=int, default=500, help="stop (and save) after this step: the checkpoint the ensemble uses")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--neg-weight", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init", default=str(RELEASED))
    parser.add_argument("--jitter", type=int, default=40, help="the sampled pixel is moved by up to +-jitter px inside the window (0 = centred, the first fine-tune)")
    parser.add_argument("--ohem-frac", type=float, default=0.0, help="hard-negative loss on the worst fraction of background pixels per window")
    parser.add_argument("--ohem-weight", type=float, default=1.0)
    parser.add_argument("--pu-weight", type=float, default=0.5, help="non-negative PU term weight (0 = off)")
    parser.add_argument("--pu-prior", type=float, default=0.2, help="assumed share of hidden ink among unlabelled pixels for the PU term")
    parser.add_argument("--corpus", default="mix28x_new", choices=sorted(CORPORA), help="; ".join(f"{k}: {v}" for k, v in CORPORA.items()).replace("%", "%%"))
    parser.add_argument("--out", default="mix28x_new_ps7", help="folder under models/studio for the checkpoints (utils/mix6.py STUDIO_RUN)")
    parser.add_argument("--jobs", type=int, default=4, help="stage: sheets read in parallel")
    args = parser.parse_args()
    os.chdir(REPO)
    if args.command in ("all", "setup"):
        setup(not args.no_venv)
    if args.command in ("all", "stage"):
        sys.path.insert(0, str(REPO))
        os.environ.setdefault("VESUVIUS_CORPUS", args.corpus)
        stage(args.jobs, args.corpus)
    global OUT
    OUT = REPO / "models" / "studio" / args.out
    if args.command == "train":
        train(args)
    elif args.command == "all":
        options = [arg for arg in sys.argv[1:] if arg != "all"]
        subprocess.run([str(PYTHON), str(Path(__file__)), "train", *options], check=True)


if __name__ == "__main__":
    main()
