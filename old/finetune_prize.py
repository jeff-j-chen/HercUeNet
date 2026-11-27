"""the special fine-tune of HercUeNet, reader-v3 and v8-in on the prize sheet alone (utils/prize_finetune.py has the label contract).

dilated_inklabels/<prize alias>.png (the prize sheet's frame): 255 = ink, black = background, a solid half-brightness (48..207) REGION that no
model may see: HercUeNet discards it plus its 149 px context reach and builds its cue maps without it; reader-v3 and v8-in blank its voxels
in their staged inputs and drop its supervision (their patches may overlap it, the region reads as outside the sheet). Everything else of
the label trains normally. Nothing else is in the corpus.

  python3 finetune_prize.py build                      # masks, cue maps (HercUeNet), train mask and figure; rerun after changing the label
  python3 finetune_prize.py hercuenet [--dry-run]      # from the mix28 _ps3 epoch 4         -> models/runs/prize_ft_hercuenet/ (epoch_<n>.pth)
  python3 finetune_prize.py reader [--dry-run]         # from the mix28 _ps3 step 2000       -> reader-v3/runs/prize_ft/ (ckpt_<step>.pth)
  python3 finetune_prize.py v8in [--dry-run]           # from the mix28 _ps3 epoch 1         -> nader-fork/runs/v8in_prize_ft/ (trained-epoch<n>.ckpt)
  python3 finetune_prize.py all                        # build, then the three one after another

--epochs / --draws / --steps / --lr / --init change a model's run (defaults: the recommended lengths below).
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from utils import mix6, prize_finetune as pf  # noqa: E402

HERCUENET_INIT = REPO / "models" / "runs" / mix6.CUE_DROPOUT_RUN / "epoch_4.pth"
READER_INIT = REPO / "reader-v3" / "runs" / mix6.READER_RUN / "ckpt_002000.pth"
V8IN_INIT_GLOB = str(REPO / "nader-fork" / "runs" / mix6.V8IN_RUN / "runs" / "*" / "checkpoints" / "trained-epoch0.ckpt")
# HercUeNet: an epoch = 6,500 draws (a mix28 epoch gives each 0846A sheet 5,500 of its 66,238); reader-v3: 13 tiles of 384 px per step;
# v8-in: an epoch = every 64 px tile of the sheet once (stride 48)
DEFAULTS = {
    "hercuenet": {"epochs": 2, "draws": 4000, "lr": 3e-5, "name": "prize_ft_hercuenet"},
    "reader": {"steps": 60, "lr": 1e-3, "warmup": 6, "save_every": 20, "name": "prize_ft"},
    "v8in": {"epochs": 2, "lr": 1e-5, "name": "v8in_prize_ft"},
}


def v8in_init() -> str:
    matches = sorted(glob.glob(V8IN_INIT_GLOB))
    if not matches:
        raise SystemExit(f"no v8-in epoch 1 checkpoint under {V8IN_INIT_GLOB}")
    return matches[-1]


def build(args) -> None:
    pf.build(force=args.force, cues=not args.no_cues)


def hercuenet(args) -> None:
    problems = pf.problems("cue_dropout")
    if problems:
        raise SystemExit("prize-finetune inputs not ready:\n  " + "\n  ".join(problems))
    init = Path(args.init or HERCUENET_INIT)
    if not init.is_file():
        raise SystemExit(f"{init} is missing")
    import train as trainer

    sys.argv = ["train.py", "--run-name", args.name, *(["--dry-run"] if args.dry_run else []), *(["--smoke"] if args.smoke else [])]
    trainer.main(configure=lambda config: pf.configure_hercuenet(config, init, args.epochs, args.draws, args.lr, args.name))


def external(script: str, extra: list[str], args) -> None:
    command = [sys.executable, script, "all", "--corpus", "prize", "--run-name", args.name, *extra,
               *(["--dry-run"] if args.dry_run else []), *(["--smoke"] if args.smoke else [])]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=REPO)


def reader(args) -> None:
    init = args.init or str(READER_INIT)
    external("finetune_readerv3.py", ["--init", init, "--steps", str(args.steps), "--lr", str(args.lr), "--warmup", str(args.warmup),
                                      "--save-every", str(args.save_every)], args)


def v8in(args) -> None:
    external("finetune_v8in.py", ["--init", args.init or v8in_init(), "--epochs", str(args.epochs), "--lr", str(args.lr)], args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=("build", "hercuenet", "reader", "v8in", "all"))
    parser.add_argument("--dry-run", action="store_true", help="check the inputs and print the plan, train nothing")
    parser.add_argument("--smoke", action="store_true", help="a tiny run of the trainer's own smoke mode")
    parser.add_argument("--force", action="store_true", help="build: rewrite the masks and rebuild the cue maps")
    parser.add_argument("--no-cues", action="store_true", help="build: skip the HercUeNet cue maps")
    for flag, kind in (("--epochs", int), ("--draws", int), ("--steps", int), ("--lr", float), ("--warmup", int), ("--save-every", int)):
        parser.add_argument(flag, type=kind, default=None)
    parser.add_argument("--init", default=None, help="checkpoint to start from (default: the mix28 _ps3 epoch 4 / step 2000 / epoch 1)")
    parser.add_argument("--name", default=None, help="run name (default: prize_ft_hercuenet / prize_ft / v8in_prize_ft)")
    args = parser.parse_args()
    if args.model == "build":
        return build(args)
    models = ("hercuenet", "reader", "v8in") if args.model == "all" else (args.model,)
    if args.model == "all":
        build(args)
    for model in models:
        if args.model == "all":  # one process per model, so HercUeNet's cached VRAM is freed before the next one starts
            subprocess.run([sys.executable, __file__, model, *(["--dry-run"] if args.dry_run else []),
                            *(["--smoke"] if args.smoke else [])], check=True, cwd=REPO)
            continue
        values = argparse.Namespace(**vars(args))
        if args.model == "all":  # one name / init per model
            values.name = values.init = None
        for key, value in DEFAULTS[model].items():
            if getattr(values, key, None) is None:
                setattr(values, key, value)
        {"hercuenet": hercuenet, "reader": reader, "v8in": v8in}[model](values)


if __name__ == "__main__":
    main()
