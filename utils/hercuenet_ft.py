"""HercUeNet trained WITHOUT any PHerc0846A sheet, then fine-tunes of the saved model, then a prize-sheet grid.

  base      4 epochs on mix28 minus the 0846A domain (the 9 donor scrolls, 14 patches); each epoch keeps the donor draws of a mix28 epoch
            (298,071 = 364,309 - 66,238), so a donor sees what it saw in the mix28 run        -> models/runs/hercuenet_no0846a/
  ft_mix28  1 epoch of the regular mix28 (0846A included) starting from base epoch 4           -> models/runs/hercuenet_no0846a_ft_mix28_1ep/
  ft_0846a  2 epochs of the 12 trained 0846A sheets only (one epoch = their 66,238 draws)      -> models/runs/hercuenet_no0846a_ft_0846a_2ep/
            (epoch_1.pth and epoch_2.pth both kept)
  ft_0846a_ext  2 more epochs of ft_0846a from its epoch_2.pth (= epochs 3-4; fresh optimizer, sampling seed 19)  -> models/runs/hercuenet_no0846a_ft_0846a_ext2ep/
  ft_0846a_9b   3 epochs of the 12 0846A sheets + all 4 PHerc0009B patches from base epoch 4, 0846A : 0009B = 2 : 1 draws as in mix28
                (an epoch = the 0846A 66,238 draws + 33,119 of 0009B)                           -> models/runs/hercuenet_no0846a_ft_0846a_9b_3ep/
  ft_0846a_9b_neg  ft_0846a_9b again with tra.neg_weight 5 (train.py: v8-in's NEG_WEIGHT; labelled background cells weigh 5x an ink cell)
                                                                                                  -> models/runs/hercuenet_no0846a_ft_0846a_9b_neg5_3ep/
  ft_sharp         ft_0846a_9b_neg + the cue flat in 60% of the 0846A / 0009B samples (0.2 in mix28) + cosine lr decay to 0 (tra.lr_cosine_final)
                                                                                                  -> models/runs/hercuenet_no0846a_ft_0846a_9b_sharp_3ep/
  ft_sharp_replay  ft_sharp plus the 7 other donor scrolls at weight 1 (0846A 6, 0009B 2); 2 epochs of 150,000 draws
                                                                                                  -> models/runs/hercuenet_no0846a_ft_sharp_replay_2ep/

Everything else is `apply_mix6_run` (utils/config.py): 192 px field, asf cue head, nnPU, ring labels, seed 17, lr 1e-4. The fine-tunes have no
warm-up (a learning-rate ramp over 1-2 epochs would never reach lr). The prize sheet is never trained or validated on.

grid  output/hercuenet_ft/prize_sheet_1x3.png: ft_0846a_9b_neg epoch 3 | ft_sharp epoch 3 | ft_sharp_replay epoch 2 (real depth-0 asf cue, flat depth-13 surface,
      window offset -1, as ensemble_eval.ipynb runs HercUeNet). Earlier figures: prize_sheet_2x2.png, _3x2.png, _1x2.png.

  python3 -m utils.hercuenet_ft queue [--dry-run]     # every stage that is not done, then the grid; logs in logs_hercuenet_ft/
  python3 -m utils.hercuenet_ft train base|ft_mix28|ft_0846a|ft_0846a_ext|ft_0846a_9b|ft_0846a_9b_neg|ft_sharp|ft_sharp_replay [--dry-run] [--smoke] [--init CKPT]
  python3 -m utils.hercuenet_ft figure
  tmux new -d -s hercuenet_ft 'python3 -m utils.hercuenet_ft queue'
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

MODEL_ROOT = REPO / "models" / "runs"
LOG_ROOT = REPO / "logs_hercuenet_ft"
OUT = REPO / "output" / "hercuenet_ft"
DONOR_EPOCH_DRAWS = 298_071  # a mix28 epoch (364,309 draws) minus its 0846A share (66,238), see logs_mix28/01_cue_dropout.log
STAGES = {
    "base": {"run": "hercuenet_no0846a", "epochs": 4},
    "ft_mix28": {"run": "hercuenet_no0846a_ft_mix28_1ep", "epochs": 1},
    "ft_0846a": {"run": "hercuenet_no0846a_ft_0846a_2ep", "epochs": 2},
    # init_from: (stage, checkpoint) the stage starts from instead of the base model's epoch_4.pth
    "ft_0846a_ext": {"run": "hercuenet_no0846a_ft_0846a_ext2ep", "epochs": 2, "init_from": ("ft_0846a", "epoch_2.pth"), "seed": 19},
    "ft_0846a_9b": {"run": "hercuenet_no0846a_ft_0846a_9b_3ep", "epochs": 3, "corpus": "0846a_9b"},
    # the same fine-tune with v8-in's NEG_WEIGHT idea: labelled background cells weigh neg_weight times an ink cell in the cell loss
    "ft_0846a_9b_neg": {"run": "hercuenet_no0846a_ft_0846a_9b_neg5_3ep", "epochs": 3, "corpus": "0846a_9b", "neg_weight": 5.0},
    # against the fuzzy maps of the 0846A + 0009B fine-tunes (cue reliance, lifted background): the cue is flat in `HercUeNet` of the 0846A and
    # 0009B samples (0.2 in mix28) so the raw slices must carry the prediction, and the lr decays by cosine to 0 over the run
    "ft_sharp": {"run": "hercuenet_no0846a_ft_0846a_9b_sharp_3ep", "epochs": 3, "corpus": "0846a_9b", "neg_weight": 5.0,
                 "cue_dropout": 0.6, "lr_cosine_final": 0.0},
    # the same plus replay of the 7 other donor scrolls (always flat cue, they pin the background) at weight 1 against 6 (0846A) and 2 (0009B)
    "ft_sharp_replay": {"run": "hercuenet_no0846a_ft_sharp_replay_2ep", "epochs": 2, "corpus": "replay", "neg_weight": 5.0,
                        "cue_dropout": 0.6, "lr_cosine_final": 0.0},
}
REPLAY_WEIGHTS = {"pherc0846a": 6, "pherc0009b": 2}  # every other domain 1
REPLAY_EPOCH_DRAWS = 150_000  # 2 epochs = 300,000 draws, the size of the 3-epoch 0846A + 0009B fine-tunes (298,071)


def run_dir(stage: str) -> Path:
    return MODEL_ROOT / STAGES[stage]["run"]


def init_path(stage: str) -> Path:
    parent, name = STAGES[stage].get("init_from", ("base", "epoch_4.pth"))
    return run_dir(parent) / name


def configure(config, stage: str, init: str | None = None) -> None:
    from utils import mix6
    from utils.config import apply_mix6_run

    apply_mix6_run(config)
    data, tra = config.data, config.tra
    sheets = {int(alias) for alias in mix6.CUE_TRAIN_0846A}
    if stage == "base":
        data.scrolls = [scroll for scroll in data.scrolls if int(scroll.scroll_id) not in sheets]
        data.train_scroll_dict = {name: ids for name, ids in data.train_scroll_dict.items() if not sheets & {int(sid) for sid in ids}}
        data.train_scroll_weights = [1] * len(data.train_scroll_dict)
        data.equalize_domain = ""
        data.epoch_draws = DONOR_EPOCH_DRAWS
        data.extra_hard_negative_scroll_ids = []
    else:
        config.init_weights = init or str(init_path(stage))
        tra.warmup_epochs = 0
        if "seed" in STAGES[stage]:
            tra.seed = STAGES[stage]["seed"]
        if "neg_weight" in STAGES[stage]:
            tra.neg_weight = STAGES[stage]["neg_weight"]
        if stage in ("ft_0846a", "ft_0846a_ext"):
            data.scrolls = [scroll for scroll in data.scrolls if int(scroll.scroll_id) in sheets]
            data.train_scroll_dict = {"pherc0846a": [int(sid) for sid in mix6.CUE_TRAIN_0846A]}
            data.train_scroll_weights = [1]
            data.equalize_domain = ""
        elif STAGES[stage].get("corpus") == "0846a_9b":
            # 0846A weight 2 and 0009B 1 with the 0846A draws kept, as in the mix28 epoch
            data.train_scroll_dict = {name: ids for name, ids in data.train_scroll_dict.items() if name in ("pherc0846a", "pherc0009b")}
            keep = {int(sid) for ids in data.train_scroll_dict.values() for sid in ids}
            data.scrolls = [scroll for scroll in data.scrolls if int(scroll.scroll_id) in keep]
            data.train_scroll_weights = [2, 1]
        elif STAGES[stage].get("corpus") == "replay":
            data.train_scroll_weights = [REPLAY_WEIGHTS.get(name, 1) for name in data.train_scroll_dict]
            data.equalize_domain = ""
            data.epoch_draws = REPLAY_EPOCH_DRAWS
        if "lr_cosine_final" in STAGES[stage]:
            tra.lr_cosine_final = STAGES[stage]["lr_cosine_final"]
    data.train_only_scroll_ids = [sid for sid in data.train_only_scroll_ids if int(sid) in {int(s.scroll_id) for s in data.scrolls}]
    data.cue_dropout_prob_by_scroll = mix6.cue_dropout_by_scroll(data.train_scroll_dict)
    if "cue_dropout" in STAGES[stage]:
        for name in ("pherc0846a", "pherc0009b"):
            for sid in data.train_scroll_dict.get(name, []):
                data.cue_dropout_prob_by_scroll[int(sid)] = STAGES[stage]["cue_dropout"]
    tra.dann_n_domains = len(data.train_scroll_dict)
    if stage == "base":
        leftover = {name: value for name, value in vars(data).items()
                    if isinstance(value, (list, tuple)) and sheets & {sid for sid in value if isinstance(sid, int)}}
        if leftover:
            raise RuntimeError(f"0846A sheets are still referenced by data.{sorted(leftover)}")
    epochs = STAGES[stage]["epochs"]
    tra.n_epochs = tra.eval_int = tra.save_int = epochs
    tra.save_epochs = tuple(range(1, epochs + 1))
    config.run_name = config.exp_name = STAGES[stage]["run"]
    config.model_dir = f"models/runs/{STAGES[stage]['run']}"
    config.save_final = f"{config.model_dir}/final.pth"


def train(stage: str, extra: list[str], init: str | None) -> None:
    import train as trainer

    sys.argv = ["train.py", "--run-name", STAGES[stage]["run"], *extra]
    trainer.main(configure=lambda config: configure(config, stage, init))


def stream(command: list[str], log: Path) -> int:
    LOG_ROOT.mkdir(exist_ok=True)
    with open(log, "w") as handle:
        return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=REPO).returncode


def queue(dry_run: bool) -> None:
    def status(text):
        print(f"[hercuenet_ft {time.strftime('%F %T')}] {text}", flush=True)

    for stage in STAGES:
        folder, final = run_dir(stage), run_dir(stage) / "final.pth"
        if final.is_file() and not dry_run:
            status(f"{stage}: done already ({final})")
            continue
        if folder.is_dir() and any(folder.iterdir()) and not dry_run:
            status(f"{stage}: {folder} holds an unfinished run; move it away, then rerun the queue")
            continue
        if stage != "base" and not dry_run and not init_path(stage).is_file():
            status(f"{stage}: SKIP, its init weights {init_path(stage)} are missing")
            continue
        started = time.time()
        init = None if stage == "base" or not dry_run else str(MODEL_ROOT / "cue_dropout_labelsv6_mix28" / "epoch_4.pth")
        command = [sys.executable, "-u", "-m", "utils.hercuenet_ft", "train", stage, *(["--dry-run"] if dry_run else []),
                   *(["--init", init] if init else [])]
        status(f"{stage}: start")
        code = stream(command, LOG_ROOT / f"{stage}{'_dry' if dry_run else ''}.log")
        status(f"{stage}: {'ok' if code == 0 else f'FAILED (exit {code})'} after {(time.time() - started) / 60:.1f} min")
    if not dry_run:
        code = stream([sys.executable, "-u", "-m", "utils.hercuenet_ft", "figure"], LOG_ROOT / "figure.log")
        status(f"figure: {'ok' if code == 0 else f'FAILED (exit {code})'} -> {OUT / 'prize_sheet_1x3.png'}")


# ---- inference ---------------------------------------------------------------------------------------------------------------
FLAT_DEPTH, DEPTH_OFFSET, EVAL_STRIDE = 13, -1, 32


def predict_prize_sheet(checkpoint: Path) -> "np.ndarray":
    """HercUeNet prediction of the prize sheet, one value per 16 px cell (cached next to the figure, redone for a newer checkpoint)."""
    import numpy as np

    cache = OUT / "preds" / f"{checkpoint.parent.name}__{checkpoint.stem}.npy"
    if cache.is_file() and cache.stat().st_mtime > checkpoint.stat().st_mtime:
        return np.load(cache)
    import types

    import torch
    import zarr
    from PIL import Image

    from utils import mix6
    from utils.chunk_cache import WindowVolume
    from utils.config import Config, ScrollConfig, resolve_run_config
    from utils.model import create_model
    from utils.norm import UNIFIED_CACHE_PATH, load_cached_norm
    from utils.platform import get_zarr_dir
    from utils.visualizer import TensorboardVisualizer, predict_tiles

    run_config, _ = resolve_run_config(checkpoint)
    C = Config()
    for section in ("data", "model"):
        for key, value in run_config[section].items():
            if section == "data" and key == "scrolls":
                C.data.scrolls = [ScrollConfig(**item) for item in value]
            elif key != "probe_rois" and hasattr(getattr(C, section), key):
                setattr(getattr(C, section), key, value)
    C.model.compile_model = False
    C.tra.supcon = C.tra.dann = False
    C.device = "cuda" if torch.cuda.is_available() else "cpu"
    C.data.eval_stride = EVAL_STRIDE
    C.data.eval_infer_bs = 16
    C.data.surface_window_offset = DEPTH_OFFSET
    C.data.surface_window_offset_by_scroll = {}
    C.data.zarr_path = str(Path(get_zarr_dir()))
    depth = int(C.data.depth)
    model, _ = create_model(C)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module.").removeprefix("_orig_mod."): value for key, value in state.get("model_state_dict", state).items()}
    model.load_state_dict({key: value for key, value in state.items() if not key.startswith(("supcon_head.", "domain_head."))}, strict=True)
    model.eval()

    mask = np.asarray(Image.open(REPO / "masks" / "extras" / f"{mix6.HOLDOUT_RENDER}.png").convert("L")) > 0
    height, width = mask.shape
    source = zarr.open(str(REPO / "ves_zarrs2" / f"{mix6.HOLDOUT_RENDER}.zarr"), mode="r")
    start = FLAT_DEPTH - (depth - 1) // 2 + DEPTH_OFFSET
    volume = WindowVolume(source, start, start + depth)
    norm = load_cached_norm(mix6.HOLDOUT, UNIFIED_CACHE_PATH, mode=str(C.data.norm_mode))
    coords = TensorboardVisualizer._gen_tile_coords(types.SimpleNamespace(c=C), (C.data.d_start, C.data.d_end), (0, height), (0, width), mask, z_step=depth)
    cue = np.load(REPO / mix6.CUE_DIRS[0] / f"{mix6.HOLDOUT}.npy", mmap_mode="r") if str(C.model.input_cue) == "pre_clahe" else None
    grid = predict_tiles(C, model, volume, mask, coords, (0, height), (0, width), C.data.d_start, "prize", *norm,
                         surface_depth_map=np.where(mask, FLAT_DEPTH, 255).astype(np.uint8),
                         surface_confidence_map=np.where(mask, 255, 0).astype(np.uint8), surface_depth_offset=DEPTH_OFFSET, cue_map=cue)
    grid = np.array(grid, dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, grid)
    return grid


def render_grid(grid: "np.ndarray", mask: "np.ndarray") -> tuple:
    """the p1 -> p99.5 gray rendering of a per-cell prediction over the sheet mask: (rgb, low, high)."""
    import numpy as np

    from utils.fast_figure import colorize

    height, width = mask.shape
    pixels = np.repeat(np.repeat(grid, 16, axis=0), 16, axis=1)
    frame = np.full((height, width), np.nan, np.float32)
    rows, cols = min(height, pixels.shape[0]), min(width, pixels.shape[1])
    frame[:rows, :cols] = pixels[:rows, :cols]
    frame[~mask] = np.nan
    low, high = np.percentile(frame[np.isfinite(frame)], (1.0, 99.5))
    return colorize(frame, "gray_r", (0.78, 0.86, 0.95), vmin=float(low), vmax=float(high)), float(low), float(high)


def predict_figure(checkpoint: Path) -> None:
    import cv2
    import numpy as np
    from PIL import Image

    from utils import mix6
    from utils.fast_figure import save_panel_grid

    mask = np.asarray(Image.open(REPO / "masks" / "extras" / f"{mix6.HOLDOUT_RENDER}.png").convert("L")) > 0
    rgb, low, high = render_grid(predict_prize_sheet(checkpoint), mask)
    title = f"{checkpoint.parent.name} {checkpoint.stem} on {mix6.HOLDOUT_NAME} | raw p1->0 ({low:.3f}), p99.5->1 ({high:.3f})"
    path = OUT / f"prize_sheet_{checkpoint.parent.name}__{checkpoint.stem}.png"
    save_panel_grid([[(title, rgb, cv2.INTER_AREA, None)]], mask.shape, str(path), gap_px=24)
    print(f"Saved {path}")


def figure() -> None:
    import cv2
    import numpy as np
    from PIL import Image

    from utils import mix6
    from utils.fast_figure import save_panel_grid

    panels = [
        ("fine-tune: 0846A + all PHerc0009B, x5 background weight (the previous best)", run_dir("ft_0846a_9b_neg") / "epoch_3.pth"),
        ("+ cue flat in 60%, cosine lr decay", run_dir("ft_sharp") / "epoch_3.pth"),
        ("+ replay of the 7 other donor scrolls (2 epochs)", run_dir("ft_sharp_replay") / "epoch_2.pth"),
    ]
    missing = [str(path) for _, path in panels if not path.is_file()]
    if missing:
        raise FileNotFoundError("checkpoints missing: " + ", ".join(missing))
    mask = np.asarray(Image.open(REPO / "masks" / "extras" / f"{mix6.HOLDOUT_RENDER}.png").convert("L")) > 0
    height, width = mask.shape
    images = []
    for title, path in panels:
        rgb, low, high = render_grid(predict_prize_sheet(path), mask)
        images.append((f"{title} | raw p1->0 ({low:.3f}), p99.5->1 ({high:.3f})", rgb, cv2.INTER_AREA, None))
    path = OUT / "prize_sheet_1x3.png"
    save_panel_grid([images], (height, width), str(path), gap_px=24)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    queue_parser = sub.add_parser("queue")
    queue_parser.add_argument("--dry-run", action="store_true")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("stage", choices=list(STAGES))
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.add_argument("--smoke", action="store_true")
    train_parser.add_argument("--init", default=None, help="init weights of a fine-tune (default: the base model's epoch_4.pth)")
    sub.add_parser("figure")
    predict_parser = sub.add_parser("predict", help="prize sheet prediction + figure of one cue_dropout checkpoint")
    predict_parser.add_argument("checkpoint")
    args = parser.parse_args()
    if args.command == "queue":
        queue(args.dry_run)
    elif args.command == "predict":
        predict_figure(Path(args.checkpoint).resolve())
    elif args.command == "train":
        train(args.stage, [flag for flag, on in (("--dry-run", args.dry_run), ("--smoke", args.smoke)) if on], args.init)
    else:
        figure()


if __name__ == "__main__":
    main()
