"""Fine-tune v8-in on the mix28 corpus (utils/mix6.py) for a 4-epoch schedule ($V8IN_EPOCHS overrides; utils/queue_ps4.sh runs it to the end; the ensemble uses epoch 3), from the released checkpoint.
(mix28 = the corpus of all three trainings: the 14 PHerc0846A sheets + the labelled donor patches of PHerc0009B, 0343, 1447, 0841,
0172, 0139, 1667, 0500P2 and PHerc0306B; a donor still without ink or train mask is skipped.
--corpus mix26 | mix25 | mix22 | mix18 | 0846a trains another corpus on the same staged data; --epochs and
--run-name change the run.)

one command does everything and can be re-run (finished steps are skipped):

  python3 finetune_v8in.py              # setup -> stage -> train -> export
  python3 finetune_v8in.py --smoke      # one epoch of 6 batches into runs/v8in_mix25_smoke (data and environment are shared)
  python3 finetune_v8in.py setup|stage|train

setup  clones jeff-j-chen/nader-fork at the pinned commit into /vesuvius/nader-fork (git lfs pulls the v8-in weights
       model.safetensors and the r3d50 backbone), builds nader-fork/.venv (uv, python 3.12, the locked requirements_v8in.txt).
stage  writes what v8-in reads, per segment, under nader-fork/runs/v8in_mix25_data/data/<alias>/: the central 24 layers of the
       render as 00.tif..23.tif (uint8, forward order; fine scans such as w013 from their spectrally matched <id>.matched volume),
       <alias>_inklabels.png and <alias>_mask.png. Half-brightness strokes of
       dilated_inklabels are not ink (split_extra_labels, as train.py reads them) and stay supervised background in 0846A; every
       other scroll trains on its train_masks region only (255 trains, 119 = hard negative inside the papyrus mask, black =
       unused; half-brightness strokes unused). Ink specks under 8 px are dropped in the three validation segments (the
       stitched validation metrics crash on them). The trained 0846A sheets lose everything within 25 voxels of the
       fiber_only_20261029d_flatboi prize sheet (holdout_exclusion_v8in/, utils/holdout_exclusion.py): no supervision there, and
       those pixels of the staged layers are zeroed (like the area outside the sheet). stage and train refuse to run until the
       masks were computed against the current prize sheet.
train  the PH1447 recipe of nader-fork (finetune_v8in_with_local_segment.py) on the staged corpus: 4 epochs (cosine schedule), micro-batch 2 x accumulation 16 (= batch 4 x 8), NEG_WEIGHT 25,
       group-balanced sampling (0846A split into two groups, so it weighs 2 against 1 for every other scroll). The last epoch
       also scores the stitched validation sheets. The final checkpoint is exported to a model folder.

output: nader-fork/runs/v8in_mix28_4ep_ps4_<seed>/model/ (<seed> = s1 / s2, VESUVIUS_SEED_TAG or --seed; NEG_WEIGHT 20; model.safetensors + config.json, exported by a run that is not stopped early, printed at the end; ensemble_eval.ipynb and
0846A_eval.ipynb read it through utils/mix6.MODEL_PATHS); the Lightning checkpoints
(a trained-epoch<N>.ckpt after every epoch) are nader-fork/runs/v8in_mix28_4ep_ps4_<seed>/runs/*/checkpoints/ (the model used is epoch 3 = trained-epoch2.ckpt).
Needs the corpus data (python3 assemble_training_segments.py --mix28), git, git-lfs, uv and a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
TRAINER_FLAG = "--trainer"  # internal: train() re-runs this file under nader-fork/.venv with this flag first


def trainer_main() -> None:
    """train_resnet3d.py (nader-fork/.venv) with cuDNN autotuning, channels-last conv weights (same maths, faster mixed-precision
    convs) and a trained-epoch{N}.ckpt saved after each epoch's last optimizer step, so a crash in validation keeps the weights."""
    import pytorch_lightning as pl
    import torch

    class ChannelsLast(pl.Callback):
        def on_fit_start(self, trainer, pl_module):
            converted = 0
            for parameter in pl_module.parameters():
                fmt = {5: torch.channels_last_3d, 4: torch.channels_last}.get(parameter.dim())
                if fmt is not None:
                    parameter.data = parameter.data.contiguous(memory_format=fmt)
                    converted += 1
            print(f"[trainer] {converted} conv weights in channels-last layout; cudnn.benchmark={torch.backends.cudnn.benchmark}",
                  flush=True)

    class PreValidationCheckpoint(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if batch_idx + 1 != trainer.num_training_batches or trainer.checkpoint_callback is None:
                return
            path = os.path.join(trainer.checkpoint_callback.dirpath, f"trained-epoch{trainer.current_epoch}.ckpt")
            trainer.save_checkpoint(path)
            print(f"[trainer] saved {path}", flush=True)

    init = pl.Trainer.__init__

    def trainer_init(self, *args, **kwargs):
        kwargs["benchmark"] = True
        kwargs["callbacks"] = [*(kwargs.get("callbacks") or []), ChannelsLast(), PreValidationCheckpoint()]
        init(self, *args, **kwargs)

    pl.Trainer.__init__ = trainer_init
    torch.backends.cudnn.benchmark = True
    fork = Path(os.environ.get("NADER_FORK_DIR", REPO / "nader-fork"))
    sys.path.insert(0, os.environ.get("V8IN_TRAIN_LIB", str(fork / "training" / "lib")))
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    import train_resnet3d

    train_resnet3d.main()


# the trainer runs in nader-fork/.venv, which lacks this repo's dependencies: dispatch before importing utils
if __name__ == "__main__" and sys.argv[1:2] == [TRAINER_FLAG]:
    trainer_main()
    sys.exit(0)

sys.path.insert(0, str(REPO))
from utils import holdout_exclusion, mix6, prize_finetune  # noqa: E402
from utils.predict_models import venv_python  # noqa: E402

FORK = Path(os.environ.get("NADER_FORK_DIR", REPO / "nader-fork"))
FORK_URL, FORK_COMMIT = "https://github.com/jeff-j-chen/nader-fork.git", "8a811a86b751734223b3febff9adba24ecc549ad"
LFS_FILES = ("model.safetensors", "training/r3d50_KM_200ep.safetensors")
PYTHON = FORK / ".venv" / "bin" / "python"
DATA = FORK / "runs" / "v8in_mix25_data" / "data"
EXCLUSION = holdout_exclusion.V8IN_EXCLUSION_DIR
EPOCHS = int(os.environ.get("V8IN_EPOCHS", mix6.V8IN_EPOCHS))
DEPTH = 24
NEG_WEIGHT = f"{mix6.V8IN_NEG_WEIGHT:g}"  # background weight of the BCE term, 25 before (utils/mix6.py)
SEED = mix6.SEED["v8in"]  # utils/mix6.py SEEDS, VESUVIUS_SEED_TAG
# the recipe's batch 4 x accumulation 8 needs more than 20 GB; same effective batch of 32 in micro-batches of this size
MICRO_BATCH = int(os.environ.get("V8IN_MICRO_BATCH", 2))
# the stitched validation sheets (trained segments, as in the author's recipe): two 0846A sheets and the 343 first letters
VAL = ("20261012065813", "20261012065823", "20261007000000")
# the container's home disk is small: keep uv's cache next to the repo
UV_ENV = dict(os.environ, UV_CACHE_DIR=os.environ.get("UV_CACHE_DIR", str(REPO / "_ves_tmp" / "uv_cache")))
# the sampler gives every group the same share: two 0846A groups make 0846A weigh 2 against 1 for each other scroll, as in train.py
# (a corpus of 0846A sheets alone is just those two groups)
CORPUS, RUN_NAME = mix6.FINETUNE_CORPUS, mix6.V8IN_RUN  # set from the command line
# --init / --lr: start from this model folder, safetensors or Lightning .ckpt instead of the released v8-in; learning-rate override
INIT, LEARNING_RATE = None, None


def groups(corpus: dict) -> dict:
    sheets = [alias for alias, segment in corpus.items() if segment.scroll == "0846A"]
    mapping = {alias: f"PHerc{segment.scroll}" for alias, segment in corpus.items()}
    mapping.update({alias: "PHerc0846A_b" for alias in sheets[(len(sheets) + 1) // 2:]})
    return mapping


def run(command, **kwargs) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def run_dir(smoke: bool) -> Path:
    return FORK / "runs" / ("v8in_mix25_smoke" if smoke else RUN_NAME)


def model_dir(smoke: bool) -> Path:
    return run_dir(smoke) / "model"  # the full run's is mix6.MODEL_PATHS["v8in"]


# ---- setup ---------------------------------------------------------------------------------------------------------------

def setup() -> None:
    if not (FORK / ".git").exists():
        run(["git", "clone", FORK_URL, FORK])
        run(["git", "-C", FORK, "checkout", "-q", FORK_COMMIT])
    if shutil.which("git-lfs") is None:
        raise SystemExit("git-lfs is needed for the v8-in weights: apt install git-lfs && git lfs install")
    pointers = [name for name in LFS_FILES if not (FORK / name).is_file() or (FORK / name).stat().st_size < 1_000_000]
    if pointers:
        run(["git", "-C", FORK, "lfs", "pull", f"--include={','.join(pointers)}"])
    if not PYTHON.exists():
        if shutil.which("uv") is None:
            raise SystemExit("uv is needed to build the v8-in environment: pip install uv")
        run(["uv", "venv", "--python", "3.12", FORK / ".venv"], env=UV_ENV)
        run(["uv", "pip", "install", "--python", PYTHON, "-r", REPO / "requirements_v8in.txt"], env=UV_ENV)
    run([PYTHON, "-c", "import pytorch_lightning, segmentation_models_pytorch, safetensors"])
    print(f"[setup] nader-fork {FORK}, weights {FORK / 'model.safetensors'}", flush=True)


# ---- stage ---------------------------------------------------------------------------------------------------------------

def _layers(alias: str, destination: Path, discard) -> None:
    """the central DEPTH slices of the volume the corpus reads for this alias as 00.tif..23.tif (uint8, forward order), zero
    where `discard` is set (the prize-sheet zone)."""
    import hashlib

    import numpy as np
    import tifffile
    import zarr

    source = mix6.volume_path(alias)
    volume = zarr.open(str(source), mode="r")
    start = (volume.shape[0] - DEPTH) // 2
    tiffs = [destination / f"{index:02d}.tif" for index in range(DEPTH)]
    blank = "none" if discard is None else hashlib.sha1(np.packbits(discard).tobytes()).hexdigest()
    stamp = destination / "blank.json"
    # layers staged before the render was regenerated, or with another blanked zone, are dropped (a symlink to an older run's
    # folder is unlinked, never written through)
    staged = all(path.is_file() for path in tiffs)
    stale_render = staged and min(path.stat().st_mtime for path in tiffs) < mix6.zarr_mtime(source)
    stale_blank = staged and (not stamp.is_file() or json.loads(stamp.read_text()).get("blank") != blank)
    if (destination.exists() or destination.is_symlink()) and (stale_render or stale_blank or not staged):
        reason = "the render changed" if stale_render else "the blanked prize-sheet zone changed" if stale_blank else "incomplete"
        print(f"[stage] {alias}: {reason} since its layers were staged; re-staging", flush=True)
        if destination.is_symlink():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    missing = [index for index, path in enumerate(tiffs) if not path.is_file()]
    if missing:
        # the render is ~100k small chunk files on the network drive: read it once with the threaded reader of predict_models
        from utils.predict_models import read_volume

        slab = np.clip(read_volume(source)[start:start + DEPTH], 0, 255).astype(np.uint8)
        for index in missing:
            layer = slab[index]
            if discard is not None:
                rows, cols = min(layer.shape[0], discard.shape[0]), min(layer.shape[1], discard.shape[1])
                layer[:rows, :cols][discard[:rows, :cols]] = 0
            tifffile.imwrite(tiffs[index], layer)
        del slab
    stamp.write_text(json.dumps({"blank": blank}))


def stage() -> None:
    import cv2
    import numpy as np

    from utils.dataloader import split_extra_labels

    for alias, segment in mix6.corpus_segments(CORPUS).items():
        folder = DATA / alias
        mask = cv2.imread(str(REPO / "masks" / f"{segment.render}.png"), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(REPO / "dilated_inklabels" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None or label is None:
            raise FileNotFoundError(f"{alias}: mask or label missing (python3 assemble_training_segments.py --mix28)")
        mask = mask > 0
        if label.shape != mask.shape:
            print(f"[stage] {alias}: cropping label {label.shape} / mask {mask.shape} to the common shape", flush=True)
            rows, cols = min(label.shape[0], mask.shape[0]), min(label.shape[1], mask.shape[1])
            label, mask = label[:rows, :cols], mask[:rows, :cols]
        ink, extra = split_extra_labels(label)
        discard = holdout_exclusion.load(alias, directory=EXCLUSION)
        if discard is None and (int(alias) in mix6.SHEETS_0846A or CORPUS == "prize"):
            raise SystemExit(f"[stage] {alias}: no exclusion in {EXCLUSION} (python3 -m utils.{'prize_finetune' if CORPUS == 'prize' else 'holdout_exclusion'})")
        if discard is not None:
            discard = discard[:mask.shape[0], :mask.shape[1]]
            mask &= ~discard
            ink &= ~discard
            what = "the half-brightness region of the label" if CORPUS == "prize" else f"{holdout_exclusion.V8IN_RADIUS_VOX:g} vox of the prize sheet"
            print(f"[stage] {alias}: {int(discard.sum())} px of {what} discarded and blanked")
        # after the exclusion cut, which leaves new slivers of ink at its edge
        if alias in VAL:
            _, components, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
            tiny = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < 8) + 1
            if len(tiny):
                ink[np.isin(components, tiny)] = False
                print(f"[stage] {alias}: dropped {len(tiny)} ink specks under 8 px", flush=True)
        if segment.split_masked:
            split = cv2.imread(str(REPO / "train_masks" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)[:mask.shape[0], :mask.shape[1]]
            mask = mask & ((split >= 240) | ((split >= 100) & (split <= 143))) & ~extra
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{alias}_inklabels.png"), ink.astype(np.uint8) * 255)
        cv2.imwrite(str(folder / f"{alias}_mask.png"), mask.astype(np.uint8) * 255)
        _layers(alias, folder / "layers", discard)
        print(f"[stage] {alias}: {mask.shape} ink={int(ink.sum())} mask={int(mask.sum())}", flush=True)


# ---- train ---------------------------------------------------------------------------------------------------------------

def init_weights() -> str:
    """the safetensors the fine-tune starts from: the released model, or --init (a Lightning .ckpt is exported once)."""
    if INIT is None:
        return str((FORK / "model.safetensors").resolve())
    path = Path(INIT).resolve()
    if path.is_file() and path.suffix == ".safetensors":
        return str(path)
    from utils import predict_models

    key = "".join(ch if ch.isalnum() else "_" for ch in str(path))[-120:]
    return str(predict_models.v8in_model_dir(path, REPO / "_ves_tmp" / "v8in_exports" / key).resolve() / "model.safetensors")


def recipe(smoke: bool) -> dict:
    sys.path.insert(0, str(FORK))
    import finetune_v8in_with_local_segment as fork

    config = json.loads(json.dumps(fork.PH1447_RECIPE))
    corpus = mix6.corpus_segments(CORPUS)
    group = groups(corpus)
    # the prize corpus has one sheet, which the trainer also needs as its (blanked, masked) validation segment
    val = list(corpus) if CORPUS == "prize" else [alias for alias in VAL if alias in corpus]
    config["description"] = f"v8-in finetune on the {CORPUS} corpus."
    config["segments"] = {
        alias: {"base_path": group[alias], "layer_range": [0, DEPTH], "reverse_layers": False} for alias in corpus
    }
    config["wandb"]["enabled"] = False
    config["wandb"]["tags"] = [CORPUS, "v8in-finetune"]
    epochs = 1 if smoke else EPOCHS
    config["training"].update({
        "dataset_root": str(DATA),
        "train_segments": list(corpus),
        "val_segments": val,
        "init_ckpt_path": init_weights(),
        # stitched metrics on the validation sheets once, at the last epoch
        "stitching_schedule": {"train_every_n_epochs": 1, "eval_every_n_epochs": epochs, "eval_every_n_epochs_plus_one": False},
    })
    config["training_hyperparameters"]["training"]["epochs"] = epochs
    training = config["training_hyperparameters"]["training"]
    effective = int(training["train_batch_size"]) * int(training.get("accumulate_grad_batches", 1))
    if effective % MICRO_BATCH:
        raise ValueError(f"micro-batch {MICRO_BATCH} does not divide the effective batch {effective}")
    training["train_batch_size"] = MICRO_BATCH
    training["accumulate_grad_batches"] = effective // MICRO_BATCH
    training["valid_batch_size"] = min(int(training["valid_batch_size"]), 4)
    if LEARNING_RATE:
        training["lr"] = LEARNING_RATE
    training["seed"] = SEED
    # the stitched validation metrics crashed the last epoch on slivers of ink; they are diagnostics only
    training["eval_stitch_metrics"] = False
    return config


def last_checkpoint(smoke: bool) -> Path | None:
    """the final-epoch checkpoint (epochepoch=N.ckpt), not best-*.ckpt."""
    candidates = sorted((run_dir(smoke) / "runs").glob(f"{run_dir(smoke).name}*/**/epochepoch=*.ckpt"),
                        key=lambda path: int(path.stem.split("=")[-1]))
    final = (1 if smoke else EPOCHS) - 1
    return candidates[-1] if candidates and int(candidates[-1].stem.split("=")[-1]) == final else None


def trained_checkpoint(smoke: bool) -> Path | None:
    """the final epoch's weights saved before its validation (trainer_main), if that epoch finished training."""
    final = (1 if smoke else EPOCHS) - 1
    candidates = sorted((run_dir(smoke) / "runs").glob(f"{run_dir(smoke).name}*/**/trained-epoch{final}.ckpt"))
    return candidates[-1] if candidates else None


def train(smoke: bool, resume: str | None = None) -> Path:
    existing = last_checkpoint(smoke)
    if existing is not None:
        print(f"[train] {existing} exists already; delete {run_dir(smoke)} to train again", flush=True)
        return existing
    if run_dir(smoke).exists() and not resume:
        if smoke:
            shutil.rmtree(run_dir(smoke))
        elif list((run_dir(smoke) / "runs").glob("*/**/*.ckpt")):
            raise SystemExit(f"[train] {run_dir(smoke)} holds the checkpoints of an interrupted run; delete it to train again")
    run_dir(smoke).mkdir(parents=True, exist_ok=True)
    config_path = run_dir(smoke) / "config.json"
    config_path.write_text(json.dumps(recipe(smoke), indent=2) + "\n", encoding="utf-8")
    entry = FORK / "training" / "lib" / "train_resnet3d.py"
    # the author's finetune_v8in_with_local_segment.py settings
    env = dict(os.environ, INPUT_NORM_TYPE="instance", DECODER_TYPE="2d", NEG_WEIGHT=NEG_WEIGHT, VAL_STRIDE="64",
               SCHED_WORLD_SIZE="1", NO_ALBUMENTATIONS_UPDATE="1", WANDB_MODE="disabled",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    command = [venv_python(FORK / ".venv"), Path(__file__).resolve(), TRAINER_FLAG, "--metadata_json", config_path, "--outputs_path", run_dir(smoke),
               "--devices", "1", "--accelerator", "gpu", "--precision", "16-mixed", "--run_name", run_dir(smoke).name]
    if resume:
        command += ["--resume_from_ckpt", resume]
    if smoke:
        command += ["--limit_train_batches", "6"]
    try:
        run(command, cwd=entry.parent, env=env)
    except subprocess.CalledProcessError:
        if trained_checkpoint(smoke) is None:
            raise
        print(f"[train] training finished but its last validation failed; using {trained_checkpoint(smoke)}", flush=True)
    final = last_checkpoint(smoke) or trained_checkpoint(smoke)
    if final is None:
        raise SystemExit(f"[train] finished without the final-epoch checkpoint under {run_dir(smoke) / 'runs'}")
    return final


def export(smoke: bool, checkpoint: Path) -> Path:
    """model.safetensors + config.json of the final checkpoint, the folder InkDetector.from_pretrained reads."""
    target = model_dir(smoke)
    if not (target / "model.safetensors").is_file():
        run([venv_python(FORK / ".venv"), FORK / "training" / "export_checkpoint.py", checkpoint, target])
    return target


def main() -> None:
    global CORPUS, EPOCHS, RUN_NAME, INIT, LEARNING_RATE, EXCLUSION, SEED, NEG_WEIGHT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="all", choices=("all", "setup", "stage", "train"))
    parser.add_argument("--smoke", action="store_true", help="one epoch of 6 batches into runs/v8in_mix25_smoke")
    parser.add_argument("--resume", default=None, help="train: continue from this Lightning .ckpt (epoch, optimizer, scheduler)")
    parser.add_argument("--corpus", default=CORPUS, choices=mix6.CORPORA, help="segments to train on (staged data is shared)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="epochs (a trained-epoch<N>.ckpt is saved after each)")
    parser.add_argument("--run-name", default=None, help="folder under nader-fork/runs (default: v8in_mix28_4ep_ps4_<seed> for the default corpus and epochs, else v8in_<corpus>_<epochs>ep)")
    parser.add_argument("--dry-run", action="store_true", help="train: write the config and print what would train, then stop")
    parser.add_argument("--init", default=None, help="train from this model folder / .safetensors / Lightning .ckpt instead of the released v8-in")
    parser.add_argument("--lr", type=float, default=None, help="learning rate (the recipe's 1e-5 by default)")
    parser.add_argument("--seed", type=int, default=None, help=f"training seed (default {SEED}: utils/mix6.py SEEDS[VESUVIUS_SEED_TAG])")
    parser.add_argument("--neg-weight", type=float, default=None, help=f"background weight of the BCE term (default {NEG_WEIGHT})")
    args = parser.parse_args()
    SEED = args.seed if args.seed is not None else SEED
    NEG_WEIGHT = f"{args.neg_weight:g}" if args.neg_weight is not None else NEG_WEIGHT
    CORPUS, EPOCHS = args.corpus, args.epochs
    INIT, LEARNING_RATE = args.init, args.lr
    if INIT and not Path(INIT).exists():
        raise SystemExit(f"--init {INIT} does not exist")
    if CORPUS == "prize":  # the prize sheet alone: its discard mask is the half-brightness region of its label
        EXCLUSION = prize_finetune.DIRS["v8in"]
    RUN_NAME = args.run_name or (mix6.V8IN_RUN if (CORPUS, EPOCHS) == (mix6.FINETUNE_CORPUS, mix6.V8IN_EPOCHS) else f"v8in_{CORPUS}_{EPOCHS}ep")
    os.chdir(REPO)
    exported = None
    if args.command in ("all", "setup"):
        setup()
    if args.command in ("all", "stage", "train"):
        # NOTHING trains on the prize sheet: the 25-voxel masks must exist for every trained 0846A sheet, computed against it
        # (the prize corpus is the prize sheet itself: its region mask must match its label instead)
        problems = prize_finetune.problems("v8in") if CORPUS == "prize" else holdout_exclusion.problems("v8in")
        if problems:
            raise SystemExit("prize-sheet exclusion not ready (python3 -m utils."
                             + ("prize_finetune" if CORPUS == "prize" else "holdout_exclusion") + "):\n  " + "\n  ".join(problems[:10]))
    if args.command in ("all", "stage"):
        problems = mix6.status([int(alias) for alias in mix6.corpus_segments(CORPUS)], derived=False)
        if problems:
            raise SystemExit("corpus data is incomplete (python3 assemble_training_segments.py --mix28):\n  " + "\n  ".join(problems[:10]))
        stage()
    if args.command in ("all", "train"):
        if args.dry_run:
            run_dir(args.smoke).mkdir(parents=True, exist_ok=True)
            settings = recipe(args.smoke)
            training = settings["training"]
            batch = settings["training_hyperparameters"]["training"]
            sizes = {}
            for name in groups(mix6.corpus_segments(CORPUS)).values():
                sizes[name] = sizes.get(name, 0) + 1
            print(f"[dry-run] corpus {CORPUS}: {len(training['train_segments'])} train segments, val {training['val_segments']}, "
                  f"groups {sizes}, "
                  f"{batch['epochs']} epochs, micro-batch {batch['train_batch_size']} x {batch['accumulate_grad_batches']}, out {run_dir(args.smoke)}")
            return
        checkpoint = train(args.smoke, args.resume)
        exported = export(args.smoke, checkpoint)
        print(f"\n[done] v8-in {CORPUS} fine-tune ({'smoke, 1 epoch of 6 batches' if args.smoke else f'{EPOCHS} epochs'}): "
              f"model {exported}\n       Lightning checkpoint {checkpoint}", flush=True)
        if not args.smoke and CORPUS == mix6.FINETUNE_CORPUS:
            print(mix6.describe_selection(), flush=True)


if __name__ == "__main__":
    main()
