"""leave-all-but-one PHerc0846A runs of HercUeNet: the mix26 corpus with ONE PHerc0846A sheet kept in, 2 epochs.

One run per sheet of the HercUeNet corpus (utils/mix6.py CUE_TRAIN_0846A, the 12 sheets; straightened2 abf in_1 / out_1 are
already dropped from it), each trained on that sheet plus the 11 donor patches (the other 11 sheets are removed from the
training config like the dropped sheets of the k_folds.py leave-patches-out arms). Everything else is `apply_mix6_run`
(utils/config.py): 192 px field, asf cue head, nnPU, ring labels, seed 17; warm-up epoch, then lr 1e-4. An epoch is what
`data.epoch_draws` sets here: 169,173 draws, half of a mix26 epoch (338,345), about 5.3k steps at batch 32.

After every run its final.pth (= epoch 2) renders the prize sheet through 0846A_eval.ipynb.

checkpoints   models/runs_single_sheet/only_<alias>/   (final.pth, epoch_1.pth, epoch_2.pth)
TensorBoard   runs_single_sheet/only_<alias>/
logs          logs_single_sheet/only_<alias>.log (training), only_<alias>_figure.log (figure)
figure        output/single_sheet/only_<alias>/<prize sheet>.png

usage: python3 -m utils.single_sheet_runs [--only 20261012073246,...] [--dry-run]
  tmux new -d -s single_sheet 'python3 -m utils.single_sheet_runs'
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG_ROOT = REPO / "logs_single_sheet"
MODEL_ROOT = REPO / "models" / "runs_single_sheet"
FIGURE_ROOT = REPO / "output" / "single_sheet"
N_EPOCHS = 2
EPOCH_DRAWS = 169_173  # half of a mix26 epoch (338,345 draws, see logs_mix26/04_cue_dropout.log)
# every cell of 0846A_eval.ipynb except the setup cells and the prize-sheet cell
SKIP_CELLS = ('^render\\("', "^train_0846a_outputs", "^train_other_outputs", "^quick_patch_outputs")


def sheets() -> list[int]:
    from utils import mix6

    return [int(alias) for alias in mix6.CUE_TRAIN_0846A]


def run_name(alias: int) -> str:
    return f"only_{alias}"


def configure(config, keep: int) -> None:
    """the mix26 run with every 0846A sheet but `keep` removed from the training corpus, N_EPOCHS epochs."""
    from utils.config import apply_mix6_run

    apply_mix6_run(config)
    dropped = {alias for alias in sheets() if alias != keep}
    data, tra = config.data, config.tra
    data.scrolls = [scroll for scroll in data.scrolls if int(scroll.scroll_id) not in dropped]
    data.train_scroll_dict = {
        name: [sid for sid in ids if int(sid) not in dropped] for name, ids in data.train_scroll_dict.items()
    }
    data.train_only_scroll_ids = [sid for sid in data.train_only_scroll_ids if int(sid) not in dropped]
    data.extra_hard_negative_scroll_ids = [sid for sid in data.extra_hard_negative_scroll_ids if int(sid) not in dropped]
    leftover = {
        name: value for name, value in vars(data).items()
        if isinstance(value, (list, tuple)) and dropped & {sid for sid in value if isinstance(sid, int)}
    }
    if leftover:
        raise RuntimeError(f"dropped sheets are still referenced by data.{sorted(leftover)}")
    config.run_name = config.exp_name = run_name(keep)
    config.model_dir = os.path.join("models", "runs_single_sheet", run_name(keep))
    tra.log_dir = "runs_single_sheet"
    tra.n_epochs = tra.eval_int = tra.save_int = N_EPOCHS
    data.epoch_draws = EPOCH_DRAWS


def run_worker(alias: int, dry_run: bool) -> None:
    import train

    sys.argv = ["train.py", "--run-name", run_name(alias), *(["--dry-run"] if dry_run else [])]
    train.main(configure=lambda config: configure(config, alias))


def result_dir(alias: int) -> Path | None:
    """the newest run folder of this sheet that has a final.pth (a rerun lands in _2, _3 ...)."""
    folders = sorted(MODEL_ROOT.glob(f"{run_name(alias)}*/final.pth"), key=lambda path: path.stat().st_mtime)
    return folders[-1].parent if folders else None


def stream(command: list, log_path: Path, env: dict | None = None) -> int:
    with open(log_path, "a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return process.wait()


def render_prize_sheet(folder: Path) -> int:
    """0846A_eval.ipynb's prize-sheet cell with this run's final.pth -> output/single_sheet/<run>/<prize sheet>.png"""
    out_dir = FIGURE_ROOT / folder.name
    env = dict(os.environ, VESUVIUS_CHECKPOINT=str(folder / "final.pth"), VESUVIUS_OUT_DIR=str(out_dir))
    command = [sys.executable, "utils/run_notebook.py", "0846A_eval.ipynb", *(arg for skip in SKIP_CELLS for arg in ("--skip", skip))]
    return stream(command, LOG_ROOT / f"{folder.name}_figure.log", env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="comma-separated sheet aliases (default: all 12)")
    parser.add_argument("--dry-run", action="store_true", help="preflight every run, no training and no figures")
    parser.add_argument("--worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    if args.worker:
        run_worker(int(args.worker), args.dry_run)
        return

    aliases = [int(item) for item in args.only.split(",") if item.strip()] or sheets()
    unknown = sorted(set(aliases) - set(sheets()))
    if unknown:
        raise SystemExit(f"unknown sheets {unknown}; choose from {sheets()}")
    LOG_ROOT.mkdir(exist_ok=True)
    failed = []
    for index, alias in enumerate(aliases, start=1):
        folder = None if args.dry_run else result_dir(alias)
        if folder is not None:
            print(f"[single] {index}/{len(aliases)} {alias}: trained already ({folder}), skipping training", flush=True)
        else:
            log_path = LOG_ROOT / f"{run_name(alias)}{'_dry' if args.dry_run else ''}.log"
            command = [sys.executable, "-u", "-m", "utils.single_sheet_runs", "--worker", str(alias),
                       *(["--dry-run"] if args.dry_run else [])]
            print(f"[single] {index}/{len(aliases)} {alias} -> {log_path}", flush=True)
            started = time.time()
            # each run in its own process, so RAM caches and CUDA state never carry over
            code = stream(command, log_path)
            print(f"[single] {alias}: {'ok' if code == 0 else f'FAILED (exit {code})'} after {(time.time() - started) / 60:.1f} min", flush=True)
            if args.dry_run:
                if code != 0:
                    failed.append(alias)
                continue
            folder = result_dir(alias)
            if code != 0 or folder is None:
                failed.append(alias)
                continue
        started = time.time()
        code = render_prize_sheet(folder)
        print(f"[single] {alias}: prize-sheet figure {'ok' if code == 0 else f'FAILED (exit {code})'} "
              f"after {(time.time() - started) / 60:.1f} min -> {FIGURE_ROOT / folder.name}", flush=True)
        if code != 0:
            failed.append(alias)
    if failed:
        raise SystemExit(f"[single] failed: {failed}")
    print("[single] all runs finished", flush=True)


if __name__ == "__main__":
    main()
