"""the special fine-tune of the three models on the prize sheet alone (utils/mix6.py HOLDOUT, corpus "prize").

The label is dilated_inklabels/<HOLDOUT>.png, drawn by hand in the prize sheet's frame (same size as masks/<HOLDOUT>.png): 255 = ink,
black = background, and a half-brightness REGION (48..207, a solid area: utils/dataloader.split_extra_labels) that NO model may see.
Everything else trains normally. From that region this module writes, per model, the pixels its trainer discards (255 = discard):

  HercUeNet   prize_finetune/hercuenet/<alias>.png   the region + MARGIN_PX, dilated by the model's full context reach (as holdout_exclusion
                                                      does for the neighbour sheets: no supervised cell has an input window that touches it);
              prize_finetune/zone/<alias>.png        the region + MARGIN_PX itself: the asf cue is built without it
              cue_asf_morph_prize{,_dm1,_dp1}/       those cue maps (the prize sheet's own cue_asf_morph/ stays as the evaluation cue)
  reader-v3   prize_finetune/reader/<alias>.png      the region + MARGIN_PX: its voxels are zeroed in the staged volume and unsupervised
  v8-in       prize_finetune/v8in/<alias>.png        the same

Everything is rebuilt when the label changes (summary.json stores its hash); the trainers refuse to start on stale masks.

  python3 -m utils.prize_finetune [--force]     # masks, train mask, cue maps and a report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from utils import mix6

REPO = Path(__file__).resolve().parent.parent
ALIAS = mix6.HOLDOUT
LABEL = REPO / "dilated_inklabels" / f"{ALIAS}.png"
TRAIN_MASK = REPO / "train_masks" / f"{ALIAS}.png"
ROOT = REPO / "prize_finetune"
DIRS = {"cue_dropout": ROOT / "hercuenet", "reader_v3": ROOT / "reader", "v8in": ROOT / "v8in"}
ZONE_DIR = ROOT / "zone"
FIGURE_DIR = REPO / "output" / "prize_finetune"
CUE_DIRS = {0: "cue_asf_morph_prize", -1: "cue_asf_morph_prize_dm1", 1: "cue_asf_morph_prize_dp1"}
MARGIN_PX = 4  # the drawn rim of the region is antialiased and the opening of split_extra_labels trims it
VERSION = 1


def label_sha1() -> str:
    return hashlib.sha1(LABEL.read_bytes()).hexdigest()


def mask() -> np.ndarray:
    return cv2.imread(str(mix6.render_mask_path(mix6.HOLDOUT_RENDER)), cv2.IMREAD_GRAYSCALE) > 0


def split() -> tuple[np.ndarray, np.ndarray]:
    """(ink, region): the regular ink and the half-brightness region of the label, both cut to the sheet frame."""
    from utils.dataloader import split_extra_labels

    raw = cv2.imread(str(LABEL), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"{LABEL} is missing: draw the prize-sheet label (255 ink, 127 = the region no model may see)")
    frame = mask()
    if raw.shape != frame.shape:
        raise ValueError(f"{LABEL} is {raw.shape}, the prize sheet's frame is {frame.shape}")
    return split_extra_labels(raw)


def discard_masks() -> dict[str, np.ndarray]:
    """model -> bool discard mask, from the label's region."""
    from utils import holdout_exclusion

    _ink, region = split()
    zone = holdout_exclusion.dilate_square(region, MARGIN_PX)
    reach = holdout_exclusion.cue_context_reach() + 1
    return {"zone": zone, "cue_dropout": holdout_exclusion.dilate_square(zone, reach), "reader_v3": zone, "v8in": zone}


def _wanted() -> dict:
    from utils import holdout_exclusion

    return {"version": VERSION, "label_sha1": label_sha1(), "margin_px": MARGIN_PX, "context_px": holdout_exclusion.cue_context_reach() + 1}


def cue_config(delta: int):
    config = mix6.cue_config(ALIAS, delta)
    config.data.cue_dir = f"./{CUE_DIRS[delta]}"
    config.data.cue_exclusion_dir = str(ZONE_DIR)
    return config


def cue_file(delta: int) -> Path:
    return REPO / CUE_DIRS[delta] / f"{mix6.volume_id(ALIAS)}.npy"


def problems(model: str) -> list[str]:
    """why the model's prize-finetune inputs are not usable (empty = ready)."""
    if not LABEL.is_file():
        return [f"{LABEL} is missing: draw the prize-sheet label (255 ink, 127 = the region no model may see)"]
    summary_path = ROOT / "summary.json"
    if not summary_path.is_file():
        return [f"{summary_path} is missing (python3 -m utils.prize_finetune)"]
    summary = json.loads(summary_path.read_text())
    found = [f"{summary_path}: {key} {summary.get(key)!r}, expected {value!r} (python3 -m utils.prize_finetune)"
             for key, value in _wanted().items() if summary.get(key) != value]
    paths = [DIRS[model] / f"{ALIAS}.png"]
    if model == "cue_dropout":
        from utils import precomputed_asf

        paths += [ZONE_DIR / f"{ALIAS}.png", TRAIN_MASK]
        for delta in CUE_DIRS:
            path = cue_file(delta)
            stamp = path.with_suffix(".zone")
            if not path.is_file():
                found.append(f"{path} is missing (python3 -m utils.prize_finetune)")
            elif not stamp.is_file() or stamp.read_text().strip() != precomputed_asf.zone_digest(precomputed_asf.read_zone(ZONE_DIR / f"{ALIAS}.png")):
                found.append(f"{path} was not built without the current region (python3 -m utils.prize_finetune)")
    found += [f"{path} is missing (python3 -m utils.prize_finetune)" for path in paths if not path.is_file()]
    return found


def build(force: bool = False, cues: bool = True) -> None:
    os.chdir(REPO)
    ROOT.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    wanted = _wanted()
    summary_path = ROOT / "summary.json"
    ink, region = split()
    frame = mask()
    if not region.any():
        print(f"[prize] WARNING: {LABEL.name} has no half-brightness region (48..207); nothing is excluded", flush=True)
    masks = discard_masks()
    for name, folder in (("cue_dropout", DIRS["cue_dropout"]), ("reader_v3", DIRS["reader_v3"]), ("v8in", DIRS["v8in"]), ("zone", ZONE_DIR)):
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{ALIAS}.png"), masks[name].astype(np.uint8) * 255)
    if not TRAIN_MASK.is_file():
        cv2.imwrite(str(TRAIN_MASK), np.full(frame.shape, 255, np.uint8))  # a 0846A sheet trains on its whole papyrus mask
    area = max(int(frame.sum()), 1)
    px_cm2 = (9.362e-4) ** 2
    report = {"alias": ALIAS, "papyrus_cm2": round(area * px_cm2, 3), "ink_px": int(ink.sum()), "region_px": int(region.sum()), **wanted, "models": {}}
    for model in ("cue_dropout", "reader_v3", "v8in"):
        discard = masks[model]
        kept_ink = int((ink & ~discard).sum())
        report["models"][model] = {"discarded_cm2": round(int((discard & frame).sum()) * px_cm2, 3),
                                   "discarded_fraction": round(int((discard & frame).sum()) / area, 4),
                                   "ink_kept_px": kept_ink}
        print(f"[prize] {model}: discards {report['models'][model]['discarded_cm2']} of {report['papyrus_cm2']} cm2 "
              f"({100 * report['models'][model]['discarded_fraction']:.1f}%), ink kept {kept_ink:,}/{int(ink.sum()):,} px", flush=True)
        if kept_ink == 0:
            raise SystemExit(f"[prize] {model} would keep no ink: nothing to train on")
    if cues:
        # the cue maps are built without the zone, three depth shifts from one read of the volume
        import sys

        sys.path[:0] = [str(REPO / "old")]  # nonml_ink.py lives in old/
        from utils import precomputed_asf

        outdated = [delta for delta in CUE_DIRS if not cue_file(delta).is_file() or not cue_file(delta).with_suffix(".zone").is_file()
                    or cue_file(delta).with_suffix(".zone").read_text().strip() != precomputed_asf.zone_digest(masks["zone"])]
        if outdated or force:
            print(f"[prize] building the asf cue maps of shifts {sorted(CUE_DIRS)} without the region", flush=True)
            print("[prize]", precomputed_asf.build_shifts(ALIAS, {delta: cue_config(delta) for delta in CUE_DIRS}, force=True), flush=True)
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    _figure(frame, ink, region, masks)
    print(f"[prize] masks in {ROOT}, figure {FIGURE_DIR / 'regions.png'}", flush=True)


def configure_hercuenet(config, init, epochs: int, epoch_draws: int, lr: float, run_name: str) -> None:
    """HercUeNet (apply_mix6_run) on the prize sheet alone, from `init`: the region is a discard mask (dilated by the context reach)
    and a cue-free zone, not a hard negative; no warm-up, lr cosine to 0."""
    from utils.config import ScrollConfig, apply_mix6_run

    apply_mix6_run(config)
    data, tra = config.data, config.tra
    config.init_weights = str(init)
    data.scrolls = [ScrollConfig(ALIAS, split_axis="x", train_split_frac=1.0)]
    data.train_scroll_dict = {"pherc0846a": [ALIAS]}
    data.train_scroll_weights = [1]
    data.train_only_scroll_ids = [ALIAS]
    data.equalize_domain = ""
    data.epoch_draws = int(epoch_draws)
    data.extra_hard_negative_scroll_ids = []
    data.whole_patch_valid_scroll_ids = []
    data.label_free_vis_scroll_ids = []
    data.holdout_exclusion_dir = f"./{DIRS['cue_dropout'].relative_to(REPO)}"
    data.cue_dir = f"./{CUE_DIRS[0]}"
    data.depth_cue_dirs = {"-1": f"./{CUE_DIRS[-1]}", "1": f"./{CUE_DIRS[1]}"}
    data.cue_dropout_prob_by_scroll = mix6.cue_dropout_by_scroll(data.train_scroll_dict)
    tra.dann_n_domains = 1
    tra.warmup_epochs = 0
    tra.lr = lr
    tra.lr_cosine_final = 0.0
    tra.n_epochs = tra.eval_int = tra.save_int = epochs
    tra.save_epochs = tuple(range(1, epochs + 1))
    config.run_name = config.exp_name = run_name
    config.model_dir = f"models/runs/{run_name}"
    config.save_final = f"{config.model_dir}/final.pth"


def _figure(frame, ink, region, masks) -> None:
    """papyrus grey, ink white, the region red and each model's extra discard (HercUeNet's context) orange, downscaled."""
    from utils.fast_figure import orient_display

    scale = 8
    rgb = np.zeros((*frame.shape, 3), np.uint8)
    rgb[frame] = (70, 70, 70)
    rgb[masks["cue_dropout"] & frame] = (200, 120, 20)
    rgb[masks["zone"] & frame] = (210, 30, 30)
    rgb[ink & ~masks["cue_dropout"]] = (255, 255, 255)
    rgb[ink & masks["cue_dropout"] & ~masks["zone"]] = (240, 220, 120)
    small = cv2.resize(rgb, (rgb.shape[1] // scale, rgb.shape[0] // scale), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(FIGURE_DIR / "regions.png"), orient_display(small, mix6.HOLDOUT_RENDER)[..., ::-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rewrite the masks and rebuild the cue maps")
    parser.add_argument("--no-cues", action="store_true", help="masks only (reader-v3 and v8-in need no cue)")
    args = parser.parse_args()
    build(force=args.force, cues=not args.no_cues)


if __name__ == "__main__":
    main()
