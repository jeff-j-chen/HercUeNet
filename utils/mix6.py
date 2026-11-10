"""the training corpus (mix28; this module keeps its historical mix6 name): one definition shared by the downloaders,
train.py, and the reader-v3 / v8-in fine-tunes. cue_dropout (HercUeNet), reader-v3 and v8-in all train on the same corpus.

mix28 (the default, CORPUS and FINETUNE_CORPUS) = mix26 + three fragments registered 2026-11-04: the PHerc0009B patches 642 and 639
(20250919130642, 20250510172639) and the PHerc0306B first-letters sheet (20261104000001, Prodigle/pherc0306b-first-letters), all hand-labelled
(dilated_inklabels / train_masks). A donor whose label has no ink or whose train mask has no
supervised pixel is skipped by every trainer (`is_labelled`, `corpus_segments`, `trained_domains`) instead of crashing cue_dropout's
character sampling, so a freshly registered fragment with placeholder (all-black) labels only costs the assembly until it is drawn.
cue_dropout additionally drops two straightened2 sheets (CUE_DROPPED), which lose all their ink to its zone, and sees the asf cue per
domain (CUE_DROPOUT_BY_DOMAIN).

mix26 = mix22 + 3 train-only PHerc0846A sheets = 25 trained patches: 14 PHerc0846A sheets + 11 donor patches
from 8 other scrolls (PHerc0009B 487/592, PHerc1447 w060, PHerc0841 405/w00, PHerc0172 w087, PHerc0343 first letters (not 343P),
PHerc0139 w041, PHerc1667 w013, PHerc0500P2 -1/-2). mix22 is mix25 without the 0139 title and the two PHerc0814 patches (the
patch-correlation study, runs_corr/report/summary.md, found them neutral or harmful for 0846A). The three new sheets (2026-11-02) are
fiber_only_abf2_20261029c_flatboi, its +28 voxel normal offset fiber_only_abf2_20261029c_flatboi_p28 and PHerc0846A test surface 6
(20260921094413, now trained on).

Older corpora stay selectable and frozen (corpus_segments): mix25 = the 25 patches of the earlier runs (11 sheets + 14 donors),
mix22, and 0846a = the 11 sheets of mix25. They share the staged data of the fine-tunes.

mix25 = 25 trained patches: 11 PHerc0846A sheets + 14 donor patches from 9 other scrolls (PHerc0009B 487/592, PHerc1447 w060,
PHerc0841 405/w00, PHerc0814 46527/459, PHerc0172 w087, PHerc0343 first letters (not 343P), PHerc0139 w041/title, PHerc1667 w013,
PHerc0500P2 -1/-2). A fibre-only prize sheet (since superseded) was held out with its 4 fibre
neighbour sheets (_in_1, _out_1, _in_2, _out_2) as train-only 0846A sheets. straightened2, the earliest prize sheet, was obsolete
by then. mix25 = mix21 + those 4 sheets; mix21 replaced mix6 (15 patches) after the
patch-correlation study (runs_corr/report/summary.md).

Every trained 0846A sheet loses its papyrus near the prize sheet (utils/holdout_exclusion.py: cue_dropout 12 vx + its context
reach, reader-v3 18 vx, v8-in 25 vx); the trainings refuse to start until those masks were computed against HOLDOUT_NAME.

The prize sheet is PRIZESHEET_NEW_flatboi_flatboi_flatboi_flatboi_copy_flatboi
(alias 20261109120000, the 2026-11-09 remake of PRIZESHEET_NEW, rendered upright, never trained or validated on) and the only prize
sheet that is assembled: every earlier one (straightened2, fiber_only_20261029d_flatboi, the 20261030 / 20261102b renderings,
fiber_only_20261102b_flatboi_PRIZESHEET and PRIZESHEET_NEW of 2026-11-06) is superseded and stays on disk as an evaluation render.
`assemble_training_segments.py --mix28` sets up exactly mix28's 28 registered patches + this sheet = 29.

`python3 -m utils.mix6` prints the state of every artifact the three trainings read (read-only);
`--build` makes the missing derived ones (flat 1447 surfaces, surface anchors, prize-sheet exclusions, 3 depths of asf cues).
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent

# 0846A sheets: flat alias (the label / cue / norm / train-mask name) -> render zarr id and display name (= the name in
# assemble_training_segments.SEGMENTS); the aliases are symlinks to the render (assemble_training_segments.MESHES_0846A) and read a
# flat depth-13 surface. The fibre sheets render like extras: zarr id = mesh name, uint8, mask in masks/extras (+ a masks/ link)
FIBER_NEIGHBOUR_BASE = "fiber_only_20261029d_flatboi"  # the four train-only neighbours are <base>_<suffix>
FIBER_NEIGHBOURS = {20261029120001: "in_1", 20261029120002: "out_1", 20261029120003: "in_2", 20261029120004: "out_2"}
SHEETS_0846A = {
    20260927134931: (20260927134928, "auto_grown_20260927134928530_abf"),
    20261012073246: (20261012073244, "auto_grown_20261012073244544_abf"),
    20261012065813: (20261012065812, "auto_grown_20261012065803387_straight_abf_m28"),
    20261012065817: (20261012065816, "auto_grown_20261012065803387_straightened2_abf_in_1"),
    20261012065819: (20261012065818, "auto_grown_20261012065803387_straightened2_abf_out_1"),
    20261012065821: (20261012065820, "auto_grown_20261012065803387_straightened2_abf_out_2"),
    20261012065823: (20261012065822, "auto_grown_20261012065803387_straightened2_abf_out_3"),
    # the prize sheet's fibre neighbours: train only (whole sheet, never validated on)
    **{alias: (f"{FIBER_NEIGHBOUR_BASE}_{suffix}", f"{FIBER_NEIGHBOUR_BASE}_{suffix}") for alias, suffix in FIBER_NEIGHBOURS.items()},
    # mix26 additions (2026-11-02), train only: the abf2 fibre-only sheet, that sheet moved +28 voxels along its normal, and test
    # surface 6 (a uint16 render, not an extras one)
    20261029130001: ("fiber_only_abf2_20261029c_flatboi", "fiber_only_abf2_20261029c_flatboi"),
    20261029130002: ("fiber_only_abf2_20261029c_flatboi_p28", "fiber_only_abf2_20261029c_flatboi_p28"),
    20260921094415: (20260921094413, "auto_grown_20260921094413486_flatboi"),
}
MIX26_NEW_SHEETS = (20261029130001, 20261029130002, 20260921094415)
TRAIN_0846A = tuple(SHEETS_0846A)
# straightened2 abf in / out_1 lie on the prize sheet: the 12 vx + context exclusion of cue_dropout leaves them no ink (the fine-tunes
# keep 1-24%), so the cue_dropout corpus omits them
CUE_DROPPED = (20261012065817, 20261012065819)
CUE_TRAIN_0846A = tuple(alias for alias in TRAIN_0846A if alias not in CUE_DROPPED)
# cue_dropout only: the asf cue is replaced by the flat map (128 over the papyrus) in this share of the samples of each domain. It is
# useful on 0846A, 0009B and 0172 (20%); on every other scroll it is always flat (100%), also in validation, so the cue head only learns
# from the domains it helps on
CUE_DROPOUT_BY_DOMAIN = {"pherc0846a": 0.2, "pherc0009b": 0.2, "pherc0172": 0.2}
CUE_DROPOUT_OTHER = 1.0
# the prize sheet (the only one assembled): rendered and predicted, never trained or validated on. It is the 2026-11-09 remake of
# PRIZESHEET_NEW (alias 20261106120000, PS2_*) which extends the 2026-11-02 sheet fiber_only_20261102b_flatboi_PRIZESHEET
# (OLD_PRIZE_SHEET, alias 20261102120000); both stay on disk for comparison. Its mesh grid is upright like PRIZESHEET_NEW's, so it is not
# in ROTATE_180
OLD_PRIZE_SHEET = "fiber_only_20261102b_flatboi_PRIZESHEET"
OLD_HOLDOUT = 20261102120000
PS2_PRIZE_SHEET = "PRIZESHEET_NEW"
PS2_HOLDOUT = 20261106120000
PRIZE_SHEET = "PRIZESHEET_NEW_flatboi_flatboi_flatboi_flatboi_copy_flatboi"
HOLDOUT = 20261109120000
HOLDOUT_RENDER = PRIZE_SHEET
HOLDOUT_NAME = PRIZE_SHEET
FIBER_RENDERS = (PRIZE_SHEET, *(f"{FIBER_NEIGHBOUR_BASE}_{suffix}" for suffix in FIBER_NEIGHBOURS.values()))
# straightened2 out_4 / out_5: whole-patch validation entries of the training config only (evaluation renders, not assembled by --mix26)
OUT_VIS = (20261012065825, 20261012065827)
FLAT_DEPTH_0846A = 13
WINDOW_OFFSET_0846A = -1

# train_scroll_dict domains after "pherc0846a" (training order); weights: 0846A 2, every other domain 1 (the mix26 donors)
# the PHerc0009B 642 / 639 patches and the PHerc0306B sheet (mix28) were registered 2026-11-04
P9B_NEW = (20250919130642, 20250510172639)
P306B_LETTERS = 20261104000001  # synthetic id of the PHerc0306B first-letters sheet (assembled from the Hugging Face render)
MIX28_NEW = (*(str(sid) for sid in P9B_NEW), str(P306B_LETTERS))
DOMAINS = {
    "pherc0009b": [20250919125754, 20250919132115, *P9B_NEW],
    "pherc1447": [20260930144760],
    "pherc0841": [20260221022814, 20260220213127],
    "pherc0172": [20251112000002],
    "pherc0343": [20261007000000],  # first letters, NOT 343P (20250511003658)
    "pherc0139": [20260108000000],
    "pherc1667": [20240304141531],  # w013: trained on its spectrally matched <id>.matched volume
    "pherc0500p2": [20250825181859, 20250825181917],
    "pherc0306b": [P306B_LETTERS],
}


class Segment(NamedTuple):  # keyed by the alias: the label / cue / norm name
    render: str         # zarr / mask name
    scroll: str         # sampling group of the fine-tunes
    layer: int          # centre layer of the 28-layer render (median of the detected depth; flat for 0846A and 1447)
    split_masked: bool  # trains on its train_masks region only (255 trains, 119 = hard negative, black = unused)
    assembler: str      # name in assemble_training_segments.SEGMENTS


# every segment any corpus ever trained on, in the fine-tune corpus order of reader-v3 and v8-in (the staged data covers all of them)
ALL_SEGMENTS = {
    **{str(alias): Segment(str(render), "0846A", FLAT_DEPTH_0846A, False, name) for alias, (render, name) in SHEETS_0846A.items()},
    "20261007000000": Segment("20261007000000", "0343", 15, True, "p343_letters"),
    "20250919125754": Segment("20250919125754", "0009B", 15, True, "p9b_487"),
    "20250919132115": Segment("20250919132115", "0009B", 15, True, "p9b_592"),
    "20250919130642": Segment("20250919130642", "0009B", 15, True, "p9b_642"),
    "20250510172639": Segment("20250510172639", "0009B", 15, True, "p9b_639"),
    "20260930144760": Segment("20260930144760", "1447", 15, True, "ph1447_w060"),
    "20260221022814": Segment("20260221022814", "0841", 15, True, "p841"),
    "20260220213127": Segment("20260220213127", "0841", 15, True, "p841_w00"),
    "20260226000000": Segment("20260226000000", "0814", 16, True, "seg46527"),
    "20250925222237": Segment("20250925222237", "0814", 15, True, "seg0814_459"),
    "20251112000002": Segment("20251112000002", "0172", 15, True, "w087"),
    "20260108000000": Segment("20260108000000", "0139", 15, True, "w041"),
    "20260422000000": Segment("20260422000000", "0139", 15, True, "title"),
    "20240304141531": Segment("20240304141531", "1667", 15, True, "w013"),
    "20250825181859": Segment("20250825181859", "0500P2", 16, True, "500P2_m1"),
    "20250825181917": Segment("20250825181917", "0500P2", 16, True, "500P2_m2"),
    str(P306B_LETTERS): Segment(str(P306B_LETTERS), "0306B", 15, True, "p306b_letters"),
}
# the default corpus (mix28): the 0846A sheets of SHEETS_0846A and the donors of DOMAINS
_DEFAULT_ALIASES = {*(str(alias) for alias in SHEETS_0846A), *(str(sid) for ids in DOMAINS.values() for sid in ids)}
SEGMENTS = {alias: segment for alias, segment in ALL_SEGMENTS.items() if alias in _DEFAULT_ALIASES}

# corpora of the trainings and of the reader-v3 / v8-in comparison runs (--corpus of finetune_readerv3.py / finetune_v8in.py); a
# corpus only selects segments, the staged data is shared. The older ones are frozen: they do not follow SEGMENTS / DOMAINS.
#   mix28  the default of all three trainings: mix26 + the 0009B 642 / 639 patches and the PHerc0306B sheet
#   mix26  mix22 + the 3 new train-only sheets (frozen: the interim cue_dropout epoch 2 and reader-v3 / v8-in mix18 models)
#   mix25  the corpus of the earlier mix25 trainings (11 sheets + 14 donors)
#   mix22  mix25 without 0139 title, 0814 seg46527 and 0814 459 (the PHerc0814 domain)
#   0846a  the 11 sheets of mix25 only
#   mix18  the reader-v3 / v8-in default: the 14 mix26 sheets + 4 donors (0009B 487, 0139 w041, 1447 w060, 0500P2 -1)
CORPORA = ("mix28", "mix26", "mix25", "mix22", "mix18", "0846a")
CORPUS = "mix28"  # train.py (cue_dropout)
FINETUNE_CORPUS = "mix28"  # finetune_readerv3.py and finetune_v8in.py: the same corpus as train.py
MIX18_DONORS = ("20250919125754", "20260108000000", "20260930144760", "20250825181859")
MIX22_DROPPED = ("20260422000000", "20260226000000", "20250925222237")


def _frozen(segments: dict[str, Segment]) -> dict[str, Segment]:
    """the segments of the older corpora: without the mix28 additions."""
    return {alias: segment for alias, segment in segments.items() if alias not in MIX28_NEW}


@functools.lru_cache(maxsize=None)
def _labelled(alias: str, label_stamp: tuple, mask_stamp: tuple) -> bool:
    import cv2
    import numpy as np

    label = cv2.imread(str(REPO / "dilated_inklabels" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)
    split = cv2.imread(str(REPO / "train_masks" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)
    if label is None or split is None:
        return False
    # a stroke (255 or half-brightness 48..207) to learn from, and a train region (255) or a hard-negative one (119)
    return bool(np.any(label >= 48)) and bool(np.any((split >= 240) | ((split >= 100) & (split <= 143))))


def is_labelled(alias) -> bool:
    """False for a donor whose label has no ink or whose train mask supervises nothing (a placeholder): the trainers skip it.
    The PHerc0846A sheets always count (their labels and masks are the flat aliases of the renders)."""
    alias = str(alias)
    segment = ALL_SEGMENTS.get(alias)
    if segment is None or not segment.split_masked:
        return True
    stamps = tuple((path.stat().st_mtime_ns, path.stat().st_size) if path.is_file() else None
                   for path in (REPO / "dilated_inklabels" / f"{alias}.png", REPO / "train_masks" / f"{alias}.png"))
    return _labelled(alias, stamps[0], stamps[1])


def unlabelled_ids() -> list[int]:
    return [int(alias) for alias in SEGMENTS if not is_labelled(alias)]


def trained_domains() -> dict[str, list[int]]:
    """DOMAINS of the labelled donor patches (a domain left without any is dropped): what train.py trains on after the 0846A sheets."""
    domains = {name: [sid for sid in ids if is_labelled(sid)] for name, ids in DOMAINS.items()}
    return {name: ids for name, ids in domains.items() if ids}


def cue_dropout_by_scroll(domains: dict[str, list[int]]) -> dict[int, float]:
    """cue_dropout's per-scroll share of samples with a flat cue: CUE_DROPOUT_BY_DOMAIN for its domains, CUE_DROPOUT_OTHER for the rest.
    `domains` is the train_scroll_dict (domain name -> scroll ids)."""
    return {int(sid): CUE_DROPOUT_BY_DOMAIN.get(name, CUE_DROPOUT_OTHER) for name, ids in domains.items() for sid in ids}


def corpus_segments(name: str, labelled_only: bool = True) -> dict[str, Segment]:
    """the segments of a corpus; the trainers get only the labelled ones (labelled_only), the assembler and the data check all of them."""
    segments = _corpus_segments(name)
    return {alias: segment for alias, segment in segments.items() if not labelled_only or is_labelled(alias)}


def _corpus_segments(name: str) -> dict[str, Segment]:
    if name == "mix28":
        return dict(SEGMENTS)
    if name == "mix26":
        return _frozen(SEGMENTS)
    if name == "mix18":
        return {alias: segment for alias, segment in _frozen(SEGMENTS).items() if segment.scroll == "0846A" or alias in MIX18_DONORS}
    mix25 = {alias: segment for alias, segment in _frozen(ALL_SEGMENTS).items() if int(alias) not in MIX26_NEW_SHEETS}
    if name == "mix25":
        return mix25
    if name == "mix22":
        return {alias: segment for alias, segment in mix25.items() if alias not in MIX22_DROPPED}
    if name == "0846a":
        return {alias: segment for alias, segment in mix25.items() if segment.scroll == "0846A"}
    raise ValueError(f"unknown corpus {name!r}; choose from {CORPORA}")


# PHerc1447 has no usable surface detection: its surface is flat at the layer with the highest mean intensity (w058 is no longer
# trained but keeps its flat surface)
FLAT_SURFACE = {20260930144758: 14, 20260930144760: 15}
TRAINING_NAMES = tuple(seg.assembler for seg in SEGMENTS.values() if seg.assembler)
# everything `assemble_training_segments.py --mix28` assembles: the 28 registered patches (25 of mix26 + 0009B 642 / 639 + the
# PHerc0306B sheet) and the prize sheet (29)
ASSEMBLER_NAMES = (*TRAINING_NAMES, HOLDOUT_NAME)


# where the trainings of the default corpora leave their model: cue_dropout on CORPUS, the two fine-tunes on FINETUNE_CORPUS.
# RUN_TAG marks the trainings made against the current prize sheet (their exclusion masks, cues and staged data). _ps3 = the 2026-11-09
# remake; the runs tagged _ps2 were made against PRIZESHEET_NEW and the runs without a tag against the 2026-11-02 sheet; they stay on disk
# (models/runs/cue_dropout_labelsv6_mix28_ps2, reader-v3/runs/mix28_ps2, nader-fork/runs/v8in_mix28_4ep_ps2; their schedules were 4000 steps / 4 epochs, cut at step 2000 / epoch 1)
RUN_TAG = "_ps3"
CUE_DROPOUT_RUN = f"cue_dropout_labelsv6_{CORPUS}{RUN_TAG}"
READER_RUN = f"{FINETUNE_CORPUS}{RUN_TAG}"
# the fine-tunes keep the learning-rate schedule of a 4000-step / 4-epoch run; utils/queue_ps3.sh stops them once step READER_STOP_STEPS /
# epoch V8IN_STOP_EPOCHS is written (the weights equal the full run's at that point)
READER_STEPS = 4000
V8IN_EPOCHS = 4
READER_STOP_STEPS = 2000
V8IN_STOP_EPOCHS = 2
V8IN_RUN = f"v8in_{FINETUNE_CORPUS}_{V8IN_EPOCHS}ep{RUN_TAG}"
# train.py also saves the weights after these epochs as models/runs/<run>/epoch_<n>.pth (next to best_model_*.pth and final.pth)
CUE_SAVE_EPOCHS = (1, 2, 3, 4)
CUE_EPOCH_PATHS = {n: REPO / "models" / "runs" / CUE_DROPOUT_RUN / f"epoch_{n}.pth" for n in CUE_SAVE_EPOCHS}
# VESUVIUS_CUE_EPOCH=<n> points the notebooks and utils/queue_mix28_eval.sh at epoch_<n>.pth; unset = final.pth (the last epoch)
_cue_epoch = os.environ.get("VESUVIUS_CUE_EPOCH", "").strip()
if _cue_epoch and int(_cue_epoch) not in CUE_EPOCH_PATHS:
    raise ValueError(f"VESUVIUS_CUE_EPOCH must be one of {CUE_SAVE_EPOCHS}, got {_cue_epoch}")
CUE_CHECKPOINT = CUE_EPOCH_PATHS[int(_cue_epoch)] if _cue_epoch else REPO / "models" / "runs" / CUE_DROPOUT_RUN / "final.pth"
# the models the notebooks (ensemble_eval.ipynb, 0846A_eval.ipynb) read (utils/predict_models.KINDS): the LAST epoch / step of the
# default trainings (not the best-validation checkpoint); they exist once utils/queue_mix28.sh has run
MODEL_PATHS = {
    "cue_dropout": CUE_CHECKPOINT,    # train.py: final.pth = weights after the last epoch (4), or epoch_<n>.pth with VESUVIUS_CUE_EPOCH
    "reader_v3": REPO / "reader-v3" / "runs" / READER_RUN / f"ckpt_{READER_STOP_STEPS:06d}.pth",    # finetune_readerv3.py, step 2000 of the 4000-step schedule
    "v8in": REPO / "nader-fork" / "runs" / V8IN_RUN / "model",                                # finetune_v8in.py, exported only by a run that is not stopped early
}
# the interim models of the mix26 / mix18 corpora (cue_dropout stopped after epoch 2, reader-v3 and v8-in on mix18), kept for comparison
MIX26_CUE_EPOCH_PATHS = {n: REPO / "models" / "runs" / "cue_dropout_labelsv6_mix26" / f"epoch_{n}.pth" for n in (1, 2, 3, 5)}
MIX26_MODEL_PATHS = {
    "cue_dropout": MIX26_CUE_EPOCH_PATHS[2],
    "reader_v3": REPO / "reader-v3" / "runs" / "mix18" / "ckpt_003000.pth",
    "v8in": REPO / "nader-fork" / "runs" / "v8in_mix18_3ep" / "model",
}
# the previous run (mix25 corpus: 11 sheets + 14 donors, old prize sheet), kept for comparison
MIX25_MODEL_PATHS = {
    "cue_dropout": REPO / "models" / "runs" / "cue_dropout_labelsv6_mix25" / "final.pth",
    "reader_v3": REPO / "reader-v3" / "runs" / "mix25" / "ckpt_003000.pth",
    "v8in": REPO / "nader-fork" / "runs" / "v8in_mix25_5ep" / "model",
}
# the earlier corpora's models, kept for comparison (mix21 never trained)
MIX6_MODEL_PATHS = {
    "cue_dropout": REPO / "models" / "runs" / "cue_dropout_labelsv6_mix6" / "best_model_loss.pth",
    "reader_v3": REPO / "reader-v3" / "runs" / "mix6" / "ckpt_002000.pth",
    "v8in": REPO / "nader-fork" / "runs" / "v8in_mix6_5ep" / "model",
}


def train_ids() -> list[int]:
    """the 25 trained aliases in the order of the training config (0846A first, then the other domains)."""
    return [*TRAIN_0846A, *(sid for ids in DOMAINS.values() for sid in ids)]


def vis_ids() -> list[int]:
    return [*TRAIN_0846A, HOLDOUT]


def all_ids() -> list[int]:
    return [*train_ids(), HOLDOUT]


def render_id(alias) -> str:
    alias = int(alias)
    if alias == HOLDOUT:
        return str(HOLDOUT_RENDER)
    return str(SHEETS_0846A[alias][0]) if alias in SHEETS_0846A else str(alias)


def render_mask_path(render) -> Path:
    """papyrus mask of a render: masks/<render>.png (the fibre renders keep theirs in masks/extras, linked into masks/)."""
    path = REPO / "masks" / f"{render}.png"
    return path if path.exists() or str(render).isdigit() else REPO / "masks" / "extras" / f"{render}.png"


def render_ids_0846a() -> list[str]:
    """render zarr ids of the 0846A part of the corpus (11 trained sheets + the prize sheet)."""
    return [render_id(alias) for alias in (*TRAIN_0846A, HOLDOUT)]


@functools.lru_cache(maxsize=1)
def _zarr_suffixes() -> dict:
    from utils.config import Config

    return {str(key): value for key, value in (Config().data.zarr_suffix or {}).items()}


def volume_id(alias) -> str:
    """the volume a training reads (zarr / norm / anchor / cue name): the alias, or <alias>.matched for a fine scan (w013)."""
    return f"{alias}{_zarr_suffixes().get(str(alias), '')}"


def volume_path(alias) -> Path:
    """the zarr every training reads for this alias (the 0846A sheets read their render)."""
    return zarr_dir() / f"{render_id(alias) if int(alias) in SHEETS_0846A or int(alias) == HOLDOUT else volume_id(alias)}.zarr"


def zarr_dir() -> Path:
    from utils.platform import get_zarr_dir

    return Path(get_zarr_dir())


def zarr_mtime(path) -> float:
    """newest change of a zarr's metadata: a re-render recreates it (rewriting chunks alone does not count)."""
    path = Path(path)
    return max(((path / name).stat().st_mtime for name in (".zarray", ".zattrs", ".zgroup") if (path / name).exists()), default=0.0)


# ---- surface -----------------------------------------------------------------------------------------------------------

def centre_layer(render: str) -> tuple[int, "object"]:
    """smoothed argmax of the mean intensity per layer over the sheet (one strided read); the first and last layer are skipped."""
    import cv2
    import numpy as np
    import zarr

    volume = zarr.open(str(zarr_dir() / f"{render}.zarr"), mode="r")
    mask = cv2.imread(str(REPO / "masks" / f"{render}.png"), cv2.IMREAD_GRAYSCALE)[::16, ::16] > 0
    stack = np.asarray(volume[:, ::16, ::16]).astype(np.float32)
    rows, cols = min(mask.shape[0], stack.shape[1]), min(mask.shape[1], stack.shape[2])
    profile = stack[:, :rows, :cols][:, mask[:rows, :cols]].mean(axis=1)
    valid = np.flatnonzero(profile > 0)  # the 24 released PHerc1447 layers cover layers 3 to 24 of the 28
    smooth = np.convolve(profile, np.ones(3) / 3, mode="same")
    inner = valid[1:-1]
    return int(inner[smooth[inner].argmax()]), profile


def is_flat_surface(alias) -> bool:
    meta = REPO / "surface_labels" / str(alias) / "metadata.json"
    return meta.is_file() and "flat_depth" in json.loads(meta.read_text())


def write_flat_surface(alias, force: bool = False) -> str:
    """surface_labels/<alias>: a flat surface at the measured centre layer (PHerc1447). a detected surface already there
    is kept under surface_labels_detected/<alias>; cached surface anchors are dropped."""
    import shutil

    import cv2
    import numpy as np

    alias = str(alias)
    if is_flat_surface(alias) and not force:
        return f"{alias}: flat surface exists"
    import zarr

    volume = zarr.open(str(zarr_dir() / f"{alias}.zarr"), mode="r")
    if not volume.attrs.get("depth_flipped", False):
        raise RuntimeError(f"{alias}: the stored stack is not the depth-reversed one (attrs depth_flipped missing)")
    depth, _ = centre_layer(alias)
    folder = REPO / "surface_labels" / alias
    backup = REPO / "surface_labels_detected" / alias
    if folder.exists() and not backup.exists() and not is_flat_surface(alias):
        backup.parent.mkdir(exist_ok=True)
        shutil.copytree(folder, backup)
    folder.mkdir(parents=True, exist_ok=True)
    mask = cv2.imread(str(REPO / "masks" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE) > 0
    np.save(folder / "depth.npy", np.where(mask, depth, 255).astype(np.uint8))
    np.save(folder / "confidence.npy", np.where(mask, 255, 0).astype(np.uint8))
    (folder / "metadata.json").write_text(json.dumps({
        "scroll_id": int(alias), "flat_depth": depth, "flat_rule": "smoothed argmax of the mean layer intensity",
        "invalid_depth_value": 255}, indent=2))
    cache = REPO / "utils" / "surface_anchor_cache.json"
    if cache.is_file():
        data = json.loads(cache.read_text())
        if data.pop(alias, None) is not None:
            cache.write_text(json.dumps(data, indent=2))
    return f"{alias}: flat surface written at layer {depth}"


def flat_surface_problems(ids=None) -> list[str]:
    """PHerc1447 trains on a flat surface of its depth-reversed stack."""
    problems = []
    for alias in (ids or [alias for alias in FLAT_SURFACE if alias in all_ids()]):
        if int(alias) in FLAT_SURFACE and not is_flat_surface(alias):
            problems.append(f"{alias}: surface is not the flat PHerc1447 one (surface_labels/{alias}/metadata.json); "
                            "run python3 -m utils.mix6 --build")
    return problems


def ensure_anchors(ids=None) -> None:
    """surface-anchor normalisation of every corpus scroll (cached in surface_anchor_cache.json)."""
    from utils.norm import ensure_surface_anchors

    os.chdir(REPO)
    ensure_surface_anchors([volume_id(sid) for sid in (ids or all_ids())], str(zarr_dir()), surface_dir="./surface_labels")


# ---- asf cues: depth shifts 0 (cue_dir), -1 and +1 (the windows depth jitter reads) -------------------------------------------

CUE_DIRS = {0: "cue_asf_morph", -1: "cue_asf_morph_dm1", 1: "cue_asf_morph_dp1"}


def window_offset(alias) -> int:
    return WINDOW_OFFSET_0846A if int(alias) in (*TRAIN_0846A, HOLDOUT) else 0


def cue_file(alias, delta: int = 0) -> Path:
    return REPO / CUE_DIRS[delta] / f"{volume_id(alias)}.npy"


def cue_config(alias, delta: int):
    from utils.config import Config

    config = Config()
    config.data.surface_window_offset_by_scroll = {int(alias): window_offset(alias) + delta}
    config.data.cue_dir = f"./{CUE_DIRS[delta]}"
    if int(alias) in SHEETS_0846A:  # trained 0846A sheets: the cue never sees the prize-sheet zone
        config.data.cue_exclusion_dir = str(cue_zone_path(alias).parent)
    return config


def cue_zone_path(alias) -> Path:
    from utils.holdout_exclusion import CUE_ZONE_DIR

    return CUE_ZONE_DIR / f"{int(alias)}.png"


def cue_shifts(alias) -> tuple[int, ...]:
    """trained scrolls read the depth-jittered windows too; the holdout is only rendered."""
    return (0,) if int(alias) == HOLDOUT else tuple(CUE_DIRS)


def _stale(path: Path, alias) -> bool:
    sources = [REPO / "surface_labels" / str(alias) / "depth.npy", REPO / "masks" / f"{alias}.png"]
    rendered = zarr_mtime(zarr_dir() / f"{render_id(alias)}.zarr")
    return rendered > path.stat().st_mtime or any(source.exists() and source.stat().st_mtime > path.stat().st_mtime for source in sources)


def _zone_newer(path: Path, alias) -> bool:
    """a trained 0846A sheet's cue was not built from its current prize-sheet zone (a new prize sheet or radius): it must be rebuilt.
    A cue with a <volume id>.zone digest (utils/precomputed_asf.py) is current while the zone mask is unchanged, whatever the file
    times (a sheet far from a new prize sheet keeps its cue); one without a digest compares file times."""
    if int(alias) not in SHEETS_0846A:
        return False
    zone = cue_zone_path(alias)
    if not zone.is_file():
        return True
    stamp = path.with_suffix(".zone")
    if stamp.is_file() and stamp.stat().st_mtime >= path.stat().st_mtime:
        from utils import precomputed_asf

        return stamp.read_text().strip() != precomputed_asf.zone_digest(precomputed_asf.read_zone(zone))
    return zone.stat().st_mtime > path.stat().st_mtime


def cue_jobs(ids=None, force: bool = False) -> list[tuple[int, int, bool]]:
    """(alias, shift, force) for every missing cue map, every trained 0846A cue older than its zone (every one with force)."""
    jobs = []
    for alias in (ids or all_ids()):
        for delta in cue_shifts(alias):
            path = cue_file(alias, delta)
            if force or not path.is_file() or _zone_newer(path, alias):
                jobs.append((int(alias), delta, True))
    return jobs


def _build_cue(job) -> str:
    alias, delta, force = job
    os.chdir(REPO)
    for path in (str(REPO), str(REPO / "old")):  # nonml_ink.py lives in old/
        if path not in sys.path:
            sys.path.insert(0, path)
    from utils import precomputed_asf

    return f"{alias} shift {delta:+d}: " + precomputed_asf.build(alias, force=force, config=cue_config(alias, delta))


def _build_cue_sheet(job) -> str:
    """all the listed depth shifts of one sheet from a single read of its volume."""
    alias, shifts, force = job
    os.chdir(REPO)
    for path in (str(REPO), str(REPO / "old")):  # nonml_ink.py lives in old/
        if path not in sys.path:
            sys.path.insert(0, path)
    from utils import precomputed_asf

    return f"{alias}: " + precomputed_asf.build_shifts(alias, {delta: cue_config(alias, delta) for delta in shifts}, force=force)


def ensure_cues(ids=None, workers: int = 8, force: bool = False) -> None:
    jobs = cue_jobs(ids, force)
    sheets: dict[int, list[int]] = {}
    for alias, delta, _ in jobs:
        sheets.setdefault(alias, []).append(delta)
    print(f"[mix28] {len(jobs)} asf cue map(s) to build ({len(sheets)} sheet(s), one volume read each)", flush=True)
    if not jobs:
        return
    import multiprocessing

    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(sheets))), mp_context=multiprocessing.get_context("spawn")) as pool:
        for line in pool.map(_build_cue_sheet, [(alias, tuple(shifts), True) for alias, shifts in sheets.items()]):
            print("[mix28] asf cue", line, flush=True)


# ---- state -------------------------------------------------------------------------------------------------------------------

def stale_cues(ids=None) -> list[str]:
    """cue maps older than their render, surface labels or mask (rebuilt only with --rebuild-cues)."""
    return [f"{alias}: asf cue {cue_file(alias, delta)} is older than its render, surface labels or mask"
            for alias in (ids or all_ids()) for delta in cue_shifts(alias)
            if cue_file(alias, delta).is_file() and _stale(cue_file(alias, delta), alias)]


def layer_warnings(ids=None) -> list[str]:
    """the fine-tunes pin the surface layer of every scroll; flag a detected surface whose median differs."""
    import numpy as np

    warnings = []
    for alias in (ids or all_ids()):
        path = REPO / "surface_labels" / str(alias) / "depth.npy"
        if str(alias) not in SEGMENTS or not path.is_file():
            continue
        depth = np.asarray(np.load(path, mmap_mode="r")[::16, ::16])
        depth = depth[depth != 255]
        median = int(np.median(depth)) if depth.size else None
        if median != SEGMENTS[str(alias)].layer:
            warnings.append(f"{alias}: median surface layer {median}, the fine-tunes pin {SEGMENTS[str(alias)].layer}")
    return warnings


def status(ids=None, derived: bool = True) -> list[str]:
    """read-only: one line per missing artifact of the given corpus aliases (default all 26).
    derived=False checks only the downloaded and tracked inputs (render, mask, label, train mask), not what is built from them."""
    zarrs = zarr_dir()
    norms = json.loads((REPO / "utils" / "norm_cache.json").read_text()) if (REPO / "utils" / "norm_cache.json").is_file() else {}
    anchors = json.loads((REPO / "utils" / "surface_anchor_cache.json").read_text()) if (REPO / "utils" / "surface_anchor_cache.json").is_file() else {}
    problems = []
    for alias in (ids or all_ids()):
        alias, render = str(alias), render_id(alias)
        paths = [zarrs / f"{render}.zarr", render_mask_path(render)]
        if int(alias) != HOLDOUT:  # the prize sheet has no label
            paths += [REPO / "dilated_inklabels" / f"{alias}.png", REPO / "train_masks" / f"{alias}.png"]
        if derived:
            paths += [REPO / "surface_labels" / alias / "depth.npy", REPO / "surface_labels" / alias / "confidence.npy"]
            if volume_id(alias) != alias:
                paths.append(zarrs / f"{volume_id(alias)}.zarr")
            if int(alias) in SHEETS_0846A or int(alias) == HOLDOUT:
                paths += [zarrs / f"{alias}.zarr", REPO / "masks" / f"{alias}.png"]
        problems += [f"{alias}: missing {path}" for path in paths if not path.exists()]
        if not derived:
            continue
        for name, table in (("norm_cache.json", norms), ("surface_anchor_cache.json", anchors)):
            if volume_id(alias) not in table:
                problems.append(f"{alias}: no {volume_id(alias)} entry in {name}")
        for delta in cue_shifts(alias):
            if not cue_file(alias, delta).is_file():
                problems.append(f"{alias}: missing asf cue {cue_file(alias, delta)}")
            elif _zone_newer(cue_file(alias, delta), alias):
                problems.append(f"{alias}: asf cue {cue_file(alias, delta)} predates its prize-sheet zone {cue_zone_path(alias)}")
        problems += flat_surface_problems([alias])
    if derived and any(int(alias) in SHEETS_0846A for alias in (ids or all_ids())):
        from utils import holdout_exclusion

        problems += [f"exclusion: {line}" for line in holdout_exclusion.all_problems()]
    return problems


def report(ids=None) -> list[str]:
    """print the read-only state of the given aliases (default all 26) and return the problems."""
    ids = ids or all_ids()
    stale = stale_cues(ids)
    if stale:
        print(f"  WARN: {len(stale)} asf cue map(s) are older than their render, surface labels or mask; they are kept "
              f"(--rebuild-cues rebuilds them): {', '.join(line.split(': ')[0] for line in stale)}")
    for line in layer_warnings(ids):
        print("  WARN:", line)
    problems = status(ids)
    for line in problems:
        print("  PROBLEM:", line)
    print(f"[mix28] {len(ids)} scrolls checked: {'ok' if not problems else f'{len(problems)} problem(s)'}")
    return problems


def finalize(ids=None, workers: int = 8, rebuild_cues: bool = False) -> None:
    """flat PHerc1447 surfaces, surface anchors, the prize-sheet exclusions (before the cues: the trained 0846A cues are built
    without the zone) and the 3 depths of asf cues for the given aliases (default all 26)."""
    os.chdir(REPO)
    ids = [int(sid) for sid in (ids or all_ids())]
    for alias in ids:
        if alias in FLAT_SURFACE:
            print("[mix28]", write_flat_surface(alias), flush=True)
    ensure_anchors(ids)
    from utils import holdout_exclusion

    holdout_exclusion.build_all()
    ensure_cues(ids, workers, force=rebuild_cues)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true", help="make the missing derived artifacts (needs the zarrs and masks)")
    parser.add_argument("--workers", type=int, default=8, help="parallel asf cue builds (one sheet each)")
    parser.add_argument("--rebuild-cues", action="store_true", help="with --build: rebuild every asf cue, not only the missing ones")
    args = parser.parse_args()
    os.chdir(REPO)
    if args.build:
        finalize(workers=args.workers, rebuild_cues=args.rebuild_cues)
    sys.exit(1 if report() else 0)


if __name__ == "__main__":
    main()
