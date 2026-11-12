"""generalization test: HercUeNet trained on the mix26 corpus WITHOUT any PHerc0846A sheet (the 11 donor patches only).

Same recipe as the leave-all-but-one runs (utils/single_sheet_runs.py): `apply_mix6_run` with the 0846A domain removed (so no
equalize_domain, uniform donor weights), 2 epochs of 169,173 draws. Afterwards final.pth renders the prize sheet.

checkpoints   models/runs_single_sheet/no_0846a/       TensorBoard runs_single_sheet/no_0846a/       log logs_single_sheet/no_0846a.log
figure        output/single_sheet_results/prize_sheet/00_no_0846a.png   (next to the 01..12 figures of single_sheet_figures.py)

usage: python3 -m utils.no_0846a_run [--dry-run] [--wait-for single_sheet --wait-for single_sheet_figures]
  tmux new -d -s no_0846a 'python3 -m utils.no_0846a_run --wait-for single_sheet --wait-for single_sheet_figures'
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils import mix6  # noqa: E402
from utils.single_sheet_runs import EPOCH_DRAWS, FIGURE_ROOT, LOG_ROOT, MODEL_ROOT, N_EPOCHS, render_prize_sheet, stream  # noqa: E402

RUN = "no_0846a"
RESULTS = REPO / "output" / "single_sheet_results"


def configure(config) -> None:
    """the mix26 run without its 0846A domain: only the donor patches train."""
    from utils.config import apply_mix6_run

    apply_mix6_run(config)
    dropped = {int(alias) for alias in mix6.CUE_TRAIN_0846A}
    data, tra = config.data, config.tra
    data.scrolls = [scroll for scroll in data.scrolls if int(scroll.scroll_id) not in dropped]
    data.train_scroll_dict = {name: ids for name, ids in data.train_scroll_dict.items() if not dropped & {int(sid) for sid in ids}}
    data.train_scroll_weights = [1] * len(data.train_scroll_dict)
    data.equalize_domain = ""
    data.train_only_scroll_ids = [sid for sid in data.train_only_scroll_ids if int(sid) not in dropped]
    data.extra_hard_negative_scroll_ids = []
    tra.dann_n_domains = len(data.train_scroll_dict)
    leftover = {
        name: value for name, value in vars(data).items()
        if isinstance(value, (list, tuple)) and dropped & {sid for sid in value if isinstance(sid, int)}
    }
    if leftover:
        raise RuntimeError(f"0846A sheets are still referenced by data.{sorted(leftover)}")
    config.run_name = config.exp_name = RUN
    config.model_dir = os.path.join("models", "runs_single_sheet", RUN)
    tra.log_dir = "runs_single_sheet"
    tra.n_epochs = tra.eval_int = tra.save_int = N_EPOCHS
    data.epoch_draws = EPOCH_DRAWS


def run_worker(dry_run: bool) -> None:
    import train

    sys.argv = ["train.py", "--run-name", RUN, *(["--dry-run"] if dry_run else [])]
    train.main(configure=configure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="preflight only")
    parser.add_argument("--wait-for", action="append", default=[], help="tmux session to wait for (repeatable)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(REPO)
    if args.worker:
        run_worker(args.dry_run)
        return
    while any(subprocess.run(["tmux", "has-session", "-t", f"={name}"], capture_output=True).returncode == 0 for name in args.wait_for):
        time.sleep(60)
    LOG_ROOT.mkdir(exist_ok=True)
    final = MODEL_ROOT / RUN / "final.pth"
    if final.is_file() and not args.dry_run:
        print(f"[no_0846a] trained already ({final})", flush=True)
    else:
        started = time.time()
        code = stream([sys.executable, "-u", "-m", "utils.no_0846a_run", "--worker", *(["--dry-run"] if args.dry_run else [])],
                      LOG_ROOT / f"{RUN}{'_dry' if args.dry_run else ''}.log")
        print(f"[no_0846a] training {'ok' if code == 0 else f'FAILED (exit {code})'} after {(time.time() - started) / 60:.1f} min", flush=True)
        if args.dry_run or code != 0 or not final.is_file():
            raise SystemExit(code if code else int(not args.dry_run))
    code = render_prize_sheet(MODEL_ROOT / RUN)
    figure = FIGURE_ROOT / RUN / f"{mix6.PRIZE_SHEET}.png"
    if code == 0 and figure.is_file():
        (RESULTS / "prize_sheet").mkdir(parents=True, exist_ok=True)
        shutil.copy2(figure, RESULTS / "prize_sheet" / "00_no_0846a.png")
    print(f"[no_0846a] prize-sheet figure {'ok' if code == 0 else f'FAILED (exit {code})'} -> {RESULTS / 'prize_sheet' / '00_no_0846a.png'}", flush=True)
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
