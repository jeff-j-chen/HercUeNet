"""Fine-tune HercUeNet from an existing checkpoint on the corpus of VESUVIUS_CORPUS (mix28x = mix28 + the HerculeX sheets).

  VESUVIUS_CORPUS=mix28x VESUVIUS_STOP_AFTER_EPOCH=3 python3 -m utils.finetune_hercuenet [--init CKPT] [--dry-run]

The recipe is apply_mix6_run unchanged (4-epoch schedule, 2 warm-up epochs); only the start weights differ (default: the mix28 epoch 3), and
VESUVIUS_STOP_AFTER_EPOCH keeps the schedule's learning rates but stops once that epoch is saved (epoch_1..3.pth in models/runs/<run>/).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

DEFAULT_INIT = REPO / "models" / "runs" / "cue_dropout_labelsv6_mix28_ps6_s1" / "epoch_3.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", default=str(DEFAULT_INIT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    init = Path(args.init).resolve()
    if not init.is_file():
        raise SystemExit(f"--init {init} does not exist")
    import train
    from utils.config import apply_mix6_run

    def configure(config) -> None:
        apply_mix6_run(config)
        config.init_weights = str(init)
        # the mix28x corpus keeps ~31 GiB private + ~11 GiB shared RAM at 8 train workers + 4 validation workers; the 62 GiB container limit
        # killed a worker twice (utils/queue_mix28x.sh clears /dev/shm/ves_zarr_cache first, 14 GiB of unreclaimable shmem)
        config.dl.num_workers = int(os.environ.get("VES_TRAIN_WORKERS", 6))

    sys.argv = ["train.py", *(["--dry-run"] if args.dry_run else [])]
    train.main(configure=configure)


if __name__ == "__main__":
    main()
