"""Fine-tune reader-v3 on the mix28 corpus (utils/mix6.py) for a 4000-step schedule (utils/queue_ps3.sh stops it after step 2000), from the released checkpoint.
(mix28 = the corpus of all three trainings: the 14 PHerc0846A sheets + the labelled donor patches of PHerc0009B, 0343, 1447, 0841,
0172, 0139, 1667, 0500P2 and PHerc0306B; a donor still without ink or train mask is skipped.
--corpus mix26 | mix25 | mix22 | mix18 | 0846a trains another corpus on the same staged data; --steps and
--run-name change the run.)

one command does everything and can be re-run (finished steps are skipped):

  python3 finetune_readerv3.py              # setup -> prep -> train
  python3 finetune_readerv3.py --smoke      # the same with 20 steps into runs/mix25_smoke (data and environment are shared)
  python3 finetune_readerv3.py setup|prep|train

setup  clones DomRusso2/reader-v3 (train_config.json) and the koine_machines training code (ScrollPrize/villa, sparse) at the
       pinned commits into /vesuvius/reader-v3 and /vesuvius/villa-ink, builds reader-v3/.venv (uv, python 3.12, the locked
       requirements_readerv3.txt) and downloads the released checkpoint reader-v3/reader-v3.pth from Hugging Face.
prep   stages every corpus segment the way the reader reads a corpus: a 21-layer crop of the 28-layer render centred on the surface
       layer (label plane = the middle slice; the trainer jitters a 17-layer window by +-2), the ink label and a supervision mask.
       Fine scans (w013) are read from their spectrally matched <id>.matched volume, like train.py.
       Half-brightness strokes of dilated_inklabels are not ink (split_extra_labels, as train.py reads them) and stay supervised
       background in 0846A; every other scroll trains on its train_masks region only (255 trains, 119 = hard negative inside the
       papyrus mask, black = unused; half-brightness strokes unused). A segment is re-staged when its labels or masks change.
       The trained 0846A sheets lose everything within 18 voxels of the fiber_only_20261029d_flatboi prize sheet
       (holdout_exclusion_reader/, utils/holdout_exclusion.py): no supervision there, and those voxels of the staged volume are
       zeroed (like the area outside the sheet) so no tile reads the prize sheet. prep and train refuse to run until the masks
       were computed against the current prize sheet.
train  batch 13 (0846A 4, each of the other scrolls 1, the slots of a missing scroll go to 0009B; sgd, lr 0.003, 200 warmup steps), 384 px
       tiles, fp16; checkpoints every 1000 steps.

output: reader-v3/runs/mix28_ps3/ckpt_<step>.pth (printed at the end; a checkpoint every 1000 steps; the models used are steps 1000 and 2000). ensemble_eval.ipynb and 0846A_eval.ipynb read it through
utils/mix6.MODEL_PATHS.
Needs the corpus data (python3 assemble_training_segments.py --mix28), git, uv and a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from utils import holdout_exclusion, mix6  # noqa: E402

READER = Path(os.environ.get("READER_V3_DIR", REPO / "reader-v3"))
VILLA = Path(os.environ.get("VILLA_DIR", REPO / "villa-ink"))
READER_URL, READER_COMMIT = "https://github.com/DomRusso2/reader-v3", "4a9f1c633d2f30d9b522922ec1cb52c082013183"
VILLA_URL, VILLA_COMMIT = "https://github.com/ScrollPrize/villa.git", "47a9c37f6f8e99afdab47bb46ad425aa726f5dc0"  # branch merge-ink-pipelines
HF_REPO, CHECKPOINT_FILE, CHECKPOINT_BYTES = "domenicor046/reader-v3", "reader-v3.pth", 138379659
PYTHON = READER / ".venv" / "bin" / "python"
CHECKPOINT = READER / CHECKPOINT_FILE
DATA = READER / "data" / "mix25"
DEPTH = 21  # crop depth; label plane = DEPTH // 2
# samples per batch of each scroll (fixed_scroll_prior sampling); 0846A has fourteen sheets, every other scroll one to five patches
BATCH_COUNTS = {"0846A": 4, "0343": 1, "0009B": 1, "1447": 1, "0841": 1, "0814": 1, "0172": 1, "0139": 1, "1667": 1, "0500P2": 1,
                "0306B": 1}
BATCH = 13
STEPS = mix6.READER_STEPS
LEARNING_RATE, WARMUP_STEPS, SAVE_EVERY = 0.003, 200, 1000
EXCLUSION = holdout_exclusion.READER_EXCLUSION_DIR
# set from the command line: which corpus (utils/mix6.py corpus_segments) is trained and where it goes
CORPUS, RUN_NAME = mix6.FINETUNE_CORPUS, mix6.READER_RUN


# the torch wheels are several GB: keep uv's cache next to the repo (the container's home disk is small)
UV_ENV = dict(os.environ, UV_CACHE_DIR=os.environ.get("UV_CACHE_DIR", str(REPO / "_ves_tmp" / "uv_cache")))


def run(command, **kwargs) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def run_dir(smoke: bool) -> Path:
    return READER / "runs" / (f"{RUN_NAME}_smoke" if smoke else RUN_NAME)


def final_checkpoint(smoke: bool) -> Path:
    return run_dir(smoke) / f"ckpt_{(20 if smoke else STEPS):06d}.pth"  # the full mix25 run's is mix6.MODEL_PATHS["reader_v3"]


def batch_counts() -> dict:
    """samples per batch of each scroll of the (labelled part of the) corpus; the batch stays 13."""
    scrolls = list(dict.fromkeys(segment.scroll for segment in mix6.corpus_segments(CORPUS).values()))
    if CORPUS == "0846a":
        return {"0846A": BATCH}
    counts = {scroll: BATCH_COUNTS[scroll] for scroll in scrolls}
    if CORPUS == "mix18":  # every donor scroll is one patch: 2 slots keep the study's 0846A 2 : donor 1 domain weight
        return {scroll: 4 if scroll == "0846A" else 2 for scroll in scrolls}
    # a scroll missing from the corpus (PHerc0814, or a donor without labels) leaves its slot to 0009B, the strongest donor
    counts["0009B"] += BATCH - sum(counts.values())
    return counts


# ---- setup ---------------------------------------------------------------------------------------------------------------

def setup() -> None:
    if not (READER / ".git").exists():
        run(["git", "clone", READER_URL, READER])
        run(["git", "-C", READER, "checkout", "-q", READER_COMMIT])
    exclude = READER / ".git" / "info" / "exclude"
    known = exclude.read_text().split() if exclude.is_file() else []
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with open(exclude, "a", encoding="utf-8") as handle:
        handle.writelines(f"{name}\n" for name in (".venv/", "data/", "runs/", ".cache/") if name not in known)

    if not (VILLA / "ink-detection" / "koine_machines").is_dir():
        VILLA.mkdir(parents=True, exist_ok=True)
        if not (VILLA / ".git").exists():
            run(["git", "init", "-q", VILLA])
            run(["git", "-C", VILLA, "remote", "add", "origin", VILLA_URL])
        run(["git", "-C", VILLA, "fetch", "--depth", "1", "origin", VILLA_COMMIT])
        run(["git", "-C", VILLA, "sparse-checkout", "set", "--cone", "ink-detection", "vesuvius"])  # koine_machines + its vesuvius dependency
        run(["git", "-C", VILLA, "checkout", "-q", VILLA_COMMIT])

    if not PYTHON.exists():
        if shutil.which("uv") is None:
            raise SystemExit("uv is needed to build the reader environment: pip install uv")
        run(["uv", "venv", "--python", "3.12", READER / ".venv"], env=UV_ENV)
        run(["uv", "pip", "install", "--python", PYTHON, "-r", REPO / "requirements_readerv3.txt"], env=UV_ENV)
        run(["uv", "pip", "install", "--python", PYTHON, "--no-deps", "-e", VILLA / "vesuvius", "-e", VILLA / "ink-detection"], env=UV_ENV)
    run([PYTHON, "-c", "import koine_machines.training.train"])

    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size != CHECKPOINT_BYTES:
        run([PYTHON, "-c", f"from huggingface_hub import hf_hub_download; hf_hub_download({HF_REPO!r}, {CHECKPOINT_FILE!r}, local_dir={str(READER)!r})"])
    print(f"[setup] reader-v3 {READER}, koine_machines {VILLA}, checkpoint {CHECKPOINT}", flush=True)


# ---- prep ----------------------------------------------------------------------------------------------------------------

def _plane_zarr(path: Path, shape, chunks, dtype):
    import zarr

    group = zarr.open_group(str(path), mode="w")
    return group.create_dataset("0", shape=shape, chunks=chunks, dtype=dtype, fill_value=0, dimension_separator=".")


def _signature(alias: str) -> str:
    """hash of everything a staged segment is made from."""
    segment = mix6.ALL_SEGMENTS[alias]
    digest = hashlib.sha1(f"{segment.layer}:{DEPTH}:{segment.split_masked}:blank-zone".encode())
    paths = [REPO / "masks" / f"{segment.render}.png", REPO / "dilated_inklabels" / f"{alias}.png"]
    if segment.split_masked:
        paths.append(REPO / "train_masks" / f"{alias}.png")
    exclusion = holdout_exclusion.exclusion_path(alias, EXCLUSION)
    if exclusion.is_file():
        paths.append(exclusion)
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prep() -> None:
    import cv2
    import numpy as np
    import zarr

    from utils.dataloader import split_extra_labels

    for alias, segment in mix6.corpus_segments(CORPUS).items():
        folder = DATA / "labels" / alias
        stamp = folder / "prep.json"
        signature = _signature(alias)
        staged = (DATA / "volumes" / f"{alias}.zarr").exists() and (folder / f"{alias}_inklabels.zarr").exists()
        # labels and masks are hashed; a regenerated render (new zarr metadata) is newer than the stamp
        rendered = mix6.zarr_mtime(mix6.volume_path(alias))
        if staged and stamp.is_file() and json.loads(stamp.read_text()).get("signature") == signature and rendered <= stamp.stat().st_mtime:
            print(f"[prep] {alias}: up to date")
            continue
        for stale in (DATA / "volumes" / f"{alias}.zarr", folder):
            shutil.rmtree(stale, ignore_errors=True)
        source = zarr.open(str(mix6.volume_path(alias)), mode="r")
        mask = cv2.imread(str(REPO / "masks" / f"{segment.render}.png"), cv2.IMREAD_GRAYSCALE) > 0
        label = cv2.imread(str(REPO / "dilated_inklabels" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)
        rows = min(source.shape[1], mask.shape[0], label.shape[0])
        cols = min(source.shape[2], mask.shape[1], label.shape[1])
        if (rows, cols) != tuple(source.shape[1:]):
            print(f"[prep] {alias}: cropping to {rows}x{cols} (volume {source.shape[1:]}, mask {mask.shape}, label {label.shape})")
        mask, label = mask[:rows, :cols], label[:rows, :cols]
        ink, extra = split_extra_labels(label)
        if segment.split_masked:
            split = cv2.imread(str(REPO / "train_masks" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)[:rows, :cols]
            mask = mask & ((split >= 240) | ((split >= 100) & (split <= 143))) & ~extra
        discard = holdout_exclusion.load(alias, directory=EXCLUSION)
        if discard is None and int(alias) in mix6.SHEETS_0846A:
            raise SystemExit(f"[prep] {alias}: no prize-sheet exclusion in {EXCLUSION} (python3 -m utils.holdout_exclusion)")
        if discard is not None:
            discard = discard[:rows, :cols]
            mask &= ~discard
            ink &= ~discard
            print(f"[prep] {alias}: {int(discard.sum())} px within {holdout_exclusion.READER_RADIUS_VOX:g} vox of the prize sheet "
                  "discarded and blanked")
        z0 = segment.layer - DEPTH // 2
        if z0 < 0 or z0 + DEPTH > source.shape[0]:
            raise ValueError(f"{alias}: surface layer {segment.layer} leaves the volume")
        folder.mkdir(parents=True, exist_ok=True)
        (DATA / "volumes").mkdir(parents=True, exist_ok=True)
        shape = (DEPTH, rows, cols)
        volume = _plane_zarr(DATA / "volumes" / f"{alias}.zarr", shape, (DEPTH, 128, 128), source.dtype)
        for y in range(0, rows, 512):
            block = np.asarray(source[z0:z0 + DEPTH, y:min(y + 512, rows), :cols])
            if discard is not None:
                block[:, discard[y:y + 512]] = 0  # the prize-sheet zone reads as outside the sheet
            volume[:, y:y + 512, :] = block
        for kind, plane in (("inklabels", ink), ("supervision_mask", mask)):
            array = _plane_zarr(folder / f"{alias}_{kind}.zarr", shape, (DEPTH, 128, 128), np.uint8)
            array[DEPTH // 2, :, :] = plane.astype(np.uint8) * 255
        stamp.write_text(json.dumps({"signature": signature}))
        print(f"[prep] {alias} ({segment.scroll}): {shape} ink={int(ink.sum())} supervised={int(mask.sum())}", flush=True)


# ---- train ---------------------------------------------------------------------------------------------------------------

def patch_cache(base: dict, out_dir: Path) -> Path:
    """one patch index per staged corpus, shared by the smoke and the full run and replaced whenever a segment is re-staged
    (the trainer would otherwise reuse the base config's single shared cache for any data)."""
    stamps = [DATA / "labels" / alias / "prep.json" for alias in mix6.corpus_segments(CORPUS)]
    size = "x".join(str(int(v)) for v in base["patch_size"][1:])
    if not all(stamp.is_file() for stamp in stamps):
        return out_dir / f"patches_{size}.json"
    digest = hashlib.sha1("".join(json.loads(stamp.read_text())["signature"] for stamp in stamps).encode()).hexdigest()[:10]
    return DATA / f"patches_{size}_{digest}.json"


def config(smoke: bool) -> Path:
    out_dir = run_dir(smoke)
    base = json.loads((READER / "train_config.json").read_text())
    datasets = []
    corpus = mix6.corpus_segments(CORPUS)
    for scroll in dict.fromkeys(segment.scroll for segment in corpus.values()):
        aliases = [alias for alias, segment in corpus.items() if segment.scroll == scroll]
        family = f"native_9p36_{scroll}"
        datasets.append({
            "segments_path": str(DATA / "labels"),
            "segments": aliases,
            "surface_volume_paths": {alias: str(DATA / "volumes" / f"{alias}.zarr") for alias in aliases},
            "volume_scale": 0,
            "source_family": family,
            "sampling_scroll": scroll,
            "sampling_physical_segment_keys": {alias: f"{scroll}:{alias}" for alias in aliases},
            "sampling_representation_keys": {alias: f"{family}:{alias}" for alias in aliases},
        })
    base["datasets"] = datasets
    base["fixed_scroll_prior"]["target_batch_counts"] = batch_counts()
    base.update({
        "checkpoint": str(CHECKPOINT),
        "weights_only": True,
        "batch_size": sum(batch_counts().values()),
        "grad_acc_steps": 1,
        "num_iterations": 20 if smoke else STEPS,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": 5 if smoke else WARMUP_STEPS,
        "save_every": 10 if smoke else SAVE_EVERY,
        "val_every": 10**9,
        "dataloader_workers": 6,
        "out_dir": str(out_dir),
        "patch_cache_filename": str(patch_cache(base, out_dir)),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.json"
    path.write_text(json.dumps(base, indent=1))
    return path


def train(smoke: bool) -> Path:
    final = final_checkpoint(smoke)
    if final.is_file():
        print(f"[train] {final} exists already; delete {run_dir(smoke)} to train again", flush=True)
        return final
    if run_dir(smoke).exists():
        if smoke:
            shutil.rmtree(run_dir(smoke))
        elif list(run_dir(smoke).glob("ckpt_*.pth")):
            raise SystemExit(f"[train] {run_dir(smoke)} holds the checkpoints of an interrupted run; delete it to train again")
    path = config(smoke)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    run([PYTHON, "-m", "koine_machines.training.train", path], cwd=READER, env=env)
    if not final.is_file():
        raise SystemExit(f"[train] finished without {final}")
    return final


def main() -> None:
    global CORPUS, STEPS, RUN_NAME
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="all", choices=("all", "setup", "prep", "train"))
    parser.add_argument("--smoke", action="store_true", help="20 steps into runs/mix25_smoke")
    parser.add_argument("--corpus", default=CORPUS, choices=mix6.CORPORA, help="segments to train on (staged data is shared)")
    parser.add_argument("--steps", type=int, default=STEPS, help="optimizer steps (a checkpoint every 1000)")
    parser.add_argument("--run-name", default=None, help="folder under reader-v3/runs (default: mix28 for the default corpus and steps, else <corpus>_<steps>)")
    parser.add_argument("--dry-run", action="store_true", help="train: write the config and print what would train, then stop")
    args = parser.parse_args()
    CORPUS, STEPS = args.corpus, args.steps
    RUN_NAME = args.run_name or (mix6.READER_RUN if (CORPUS, STEPS) == (mix6.FINETUNE_CORPUS, mix6.READER_STEPS) else f"{CORPUS}_{STEPS}")
    os.chdir(REPO)
    final = None
    if args.command in ("all", "setup"):
        setup()
    if args.command in ("all", "prep", "train"):
        # NOTHING trains on the prize sheet: the 18-voxel masks must exist for every trained 0846A sheet, computed against it
        problems = holdout_exclusion.problems("reader_v3")
        if problems:
            raise SystemExit("prize-sheet exclusion not ready (python3 -m utils.holdout_exclusion):\n  " + "\n  ".join(problems[:10]))
    if args.command in ("all", "prep"):
        problems = mix6.status([int(alias) for alias in mix6.corpus_segments(CORPUS)], derived=False)
        if problems:
            raise SystemExit("corpus data is incomplete (python3 assemble_training_segments.py --mix28):\n  " + "\n  ".join(problems[:10]))
        prep()
    if args.command in ("all", "train"):
        if args.dry_run:
            path = config(args.smoke)
            settings = json.loads(path.read_text())
            print(f"[dry-run] corpus {CORPUS}: {sum(len(d['segments']) for d in settings['datasets'])} segments, batch {settings['batch_size']}, "
                  f"counts {settings['fixed_scroll_prior']['target_batch_counts']}, {settings['num_iterations']} steps, "
                  f"save every {settings['save_every']}, out {settings['out_dir']}")
            return
        final = train(args.smoke)
    if final is not None:
        print(f"\n[done] reader-v3 {CORPUS} fine-tune ({'smoke, 20' if args.smoke else str(STEPS)} steps): {final}", flush=True)


if __name__ == "__main__":
    main()
