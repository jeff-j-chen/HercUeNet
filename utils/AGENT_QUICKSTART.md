# Agent quickstart: the three default trainings (mix28)

Run everything from `/vesuvius`. Do NOT run smoke tests (`--smoke`): the user does not want them. Run long jobs in `tmux`, and never `pkill -f` a pattern that
also appears in a tmux command line (kill the session by name instead).

Base directory: only `README.md`, `train.py`, `finetune_v8in.py`, `finetune_readerv3.py`, `assemble_test_segments.py`, `assemble_training_segments.py`,
`k_folds.py`, the four notebooks (`0846A_eval`, `ensemble_eval`, `eval_scrolls`, `eval_test`) and the three requirements files live there. Pipeline helpers and
caches are in `utils/` (`norm_cache.json`, `surface_anchor_cache.json`, `depth_reversed.json`, `precompute_norm.py`, `generate_surface_supervision.py`,
`build_domain_cues.py`, `run_notebook.py`, `queue_mix28.sh`, `queue_mix28_eval.sh`), unused scripts in `old/`. Call helper scripts as modules: `python3 -m utils.<name>`.

## Corpora and the prize sheet

- **mix28** (`utils/mix6.py` `CORPUS` and `FINETUNE_CORPUS`: HercUeNet, reader-v3 and v8-in all train on it; HercUeNet omits `CUE_DROPPED` = straightened2 abf in / out_1,
  whose ink the 12 vx + context exclusion removes entirely): the 14 PHerc0846A sheets (7 auto_grown + the 4 train-only fibre neighbours of the earlier prize sheet + the abf2 fibre-only
  sheet, its +28 voxel offset and test surface 6) + the donors 0009B 487/592, 1447 w060, 0841 405/w00, 0172 w087, 343 first letters (NOT 343P), 0139 w041, 1667 w013 (fine scan, read as
  `.matched`), 0500P2 -1/-2 and three fragments registered 2026-11-04 (hand-labelled): 0009B 642 (`20250919130642`) and 639 (`20250510172639`) and the PHerc0306B first-letters
  sheet (`20261104000001`, `p306b_letters`). A donor whose `dilated_inklabels` has no ink or whose `train_masks` supervises nothing is skipped by all three trainers (`utils/mix6.py`
  `is_labelled`, `corpus_segments`, `trained_domains`); it trains as soon as its labels are drawn (255 trains, 119 hard negative, black unused).
- **mix26** (frozen; `--corpus mix26` of the fine-tunes): the 25 patches before 2026-11-04. The interim models (HercUeNet `cue_dropout_labelsv6_mix26/epoch_2.pth`, reader-v3 / v8-in on the 18-patch
  **mix18**) are `MIX26_MODEL_PATHS`; the evaluation notebooks default to them until the mix28 trainings have run.
- The PRIZE sheet is `PRIZESHEET_11_11_flatboi_flatboi_abf` (alias 20261112120000, the 2026-11-11 sheet made canonical on 2026-11-12, rendered upright). It is the only prize sheet: every earlier one
  (the 2026-11-09 sheet `PRIZESHEET_NEW_flatboi_flatboi_flatboi_flatboi_copy_flatboi`, `fiber_only_20261102b_flatboi_PRIZESHEET`, `PRIZESHEET_NEW`) is superseded and kept only as an evaluation render. NOTHING trains or validates on it.
  `RUN_TAG` in `utils/mix6.py` (`_ps4`) names the runs made against it; `_ps3` runs were made against the 2026-11-09 sheet, `_ps2` runs against `PRIZESHEET_NEW`, untagged ones against the 2026-11-02 sheet. Every `_ps4` model is trained twice (seed tags `s1`, `s2`, `VESUVIUS_SEED_TAG`), which ends the run name; the notebooks default to `s1`: HercUeNet epoch 3, reader-v3 step 1000, v8-in epoch 3.

## Prize-sheet exclusion (must be ready before any training)

`utils/holdout_exclusion.py` discards, on every trained 0846A sheet, the papyrus near the prize sheet (3D mesh distance): HercUeNet 12 vx dilated by its context reach
of 148 px (`holdout_exclusion/`; the zone itself in `holdout_exclusion/zone/`, and the sheet's asf cue is built without it), reader-v3 18 vx (`holdout_exclusion_reader/`),
v8-in 25 vx (`holdout_exclusion_v8in/`). reader-v3 and v8-in also zero those voxels in their staged inputs. Each folder's `summary.json` names the prize sheet and radius;
the three trainings refuse to start if they don't match. `python3 -m utils.holdout_exclusion` rebuilds stale masks.

## 0. Data (one command)

```bash
python3 assemble_training_segments.py --mix28         # downloads + sets up everything; add --check to only report what is missing
python3 -m utils.mix6                                 # the same read-only report (python3 -m utils.mix6 --build makes the derived files)
```

Per scroll: volume + mask (`ves_zarrs2/<id>.zarr`, `masks/<id>.png`), norm, surface detection (`surface_labels/<id>/`; PHerc1447 is a flat surface at its
measured centre layer), surface anchors, the prize-sheet exclusions, and 3 depths of asf cues (`cue_asf_morph`, `_dm1`, `_dp1`; the prize sheet only the first).
The PHerc0846A sheets (`MESHES_0846A` in `assemble_training_segments.py`) are rendered from their tifxyz meshes; the fibre sheets as uint8 extras (mask in `masks/extras/`,
linked into `masks/`). Labels (`dilated_inklabels/`, `train_masks/`) and the meshes of the other scrolls (`tifxyz/`) are tracked in git; the PHerc0846A meshes (incl. the prize sheet) are in the passphrase-protected `utils/tifxyz_0846A.tar.enc` (the assembler prompts for the passphrase or reads `TIFXYZ_PASSPHRASE`, `python3 -m utils.unlock_tifxyz_0846A` extracts it by hand, an empty or wrong answer skips the 0846A sheets).
Aliases like `20261012065815` are symlinks to the render `20261012065814`: the label/cue name is the alias, the zarr/mask name is the render id.
`--rebuild-cues` rebuilds existing cue maps (they are kept otherwise). After a new prize sheet only the cues of 0846A sheets whose zone mask changed are rebuilt (each cue stores a `<id>.zone` digest of the zone it was built from; a cue without one falls back to file times); a sheet's three depth shifts come from one volume read (`precomputed_asf.build_shifts`, the read is ~all of the time), 8 sheets in parallel (`--workers`). The norm and anchor caches are `utils/norm_cache.json` and `utils/surface_anchor_cache.json`.

## The queue

```bash
tmux new -d -s queue_ps4 'bash /vesuvius/utils/queue_ps4.sh'   # waits for tmux prep_ps4 (utils.mix6 --build), then check -> per seed s1, s2: HercUeNet (4 epochs, epoch 3 used) -> reader-v3 (4000 steps, step 1000 used) -> v8-in (4 epochs, epoch 3 used); full schedules, EARLY_STOP=1 stops at the used checkpoint
                                                              # -> seed-averaged depth-max ensemble (HercUeNet / reader-v3 maxed over -2..0, v8-in over -2..-1) (output/ensemble/mix28_ps4_c3_r1000_v3_seedavg) -> seed s1 on every training sheet + utils/label_overlay_figures; logs in logs_ps4/
```

## 1. cue_dropout_labelsv6_mix28: `train.py`

```bash
python3 train.py --dry-run    # input preflight only (includes the prize-sheet exclusion check)
python3 train.py              # 4 epochs, seed 17 (builds models/init_mae192_cue4_seed17.pth first if it is missing)
```

Output `models/runs/cue_dropout_labelsv6_mix28/` (`final.pth` = the last epoch is what the notebooks read, also `epoch_1/2/3/4.pth` (`VESUVIUS_CUE_EPOCH=<n>` selects one for the notebooks and `queue_mix28_eval.sh`), `best_model_loss.pth`); TensorBoard in `runs/<run>/`.
An existing run is never overwritten (`_2`, `_3`). `utils/config.py` `apply_mix6_run` is the whole recipe.

## 2. reader-v3 fine-tune: `finetune_readerv3.py`

```bash
python3 finetune_readerv3.py            # setup -> prep -> train, mix28, 2000 optimizer steps, batch 13, a checkpoint every 1000
```

`setup` clones `DomRusso2/reader-v3` and the koine_machines training code (ScrollPrize/villa, sparse) into `reader-v3/` and `villa-ink/` at pinned commits, builds
`reader-v3/.venv` with uv from `requirements_readerv3.txt` and downloads `reader-v3/reader-v3.pth` from Hugging Face. `prep` stages 21-layer crops around each scroll's
surface layer (`reader-v3/data/mix25`, a historical folder name shared by every corpus; a segment is re-staged when its label, train mask, mask or exclusion changes).
Output `reader-v3/runs/mix28_ps4_<seed>/ckpt_001000.pth` (the 4000-step schedule runs to the end; step 1000 is the one the ensemble uses). The BCE background weight is 0.8 (`mix6.READER_NEG_WEIGHT`; `patch_loss()` adds `neg_weight` to the pinned vesuvius loss).
uv's cache goes to `_ves_tmp/uv_cache` (the container's home disk is small; set `UV_CACHE_DIR` to change it).

## 3. v8-in fine-tune: `finetune_v8in.py`

```bash
python3 finetune_v8in.py                # setup -> stage -> train -> export, mix28, 4-epoch cosine schedule (run to the end; epoch 3 is the one the ensemble uses; `NEG_WEIGHT` 20)
```

`setup` clones `jeff-j-chen/nader-fork` (git lfs pulls the v8-in weights), builds `nader-fork/.venv` from `requirements_v8in.txt`. `stage` writes the central 24 layers as tiffs and the label/mask PNGs
to `nader-fork/runs/v8in_mix25_data/data` (also a historical name; layers are re-staged when the blanked prize-sheet zone changes). Output `nader-fork/runs/v8in_mix28_4ep_ps4_<seed>/` (checkpoints `runs/*/checkpoints/trained-epoch<N>.ckpt`; `model/` only for a run that is not stopped)
(exported epoch 4; the Lightning checkpoints are under `runs/*/checkpoints/`, 0-based: `trained-epoch<N>.ckpt` after every epoch, `epochepoch=3.ckpt` is the last).

## Labels differ per model

- HercUeNet (train.py): half-brightness strokes of `dilated_inklabels` are hard negatives in 0846A and unused letters elsewhere; ring negatives and nnPU far background around the ink.
- reader-v3 and v8-in: half-brightness strokes are not ink and stay supervised background in 0846A; every other scroll trains on its `train_masks` region only (255 trains,
  119 = hard negative inside the papyrus mask, black = unused, half-brightness inklabel strokes unused). reader-v3 sees a 21-layer crop centred on the surface layer (17-layer window jittered +-2),
  v8-in the central 24 layers of the render.

## Evaluate

- `ensemble_eval.ipynb`: the three last-epoch models on the prize sheet, caches each prediction under `output/ensemble/<ENSEMBLE_SUBDIR>/<sheet>/cache/` (delete a file to recompute), saves the 3x2 grid
  (reader-v3 base | reader-v3 / HercUeNet | v8-in / z-score | product of the three) and `product_ct_2x2.png` (product composite | ct composite view x3). Headless: `python3 utils/run_notebook.py ensemble_eval.ipynb`.
- `0846A_eval.ipynb`: set `CHECKPOINT` to any of the three models (HercUeNet `.pth`, reader-v3 `.pth`, v8-in model folder or `.ckpt`; default the reader-v3 fine-tune); the kind is detected, the figures are the same.
  Headless without the remaining quick patches: `python3 utils/run_notebook.py 0846A_eval.ipynb --keep-going --skip '^quick_patch_outputs ='`.
- Locations of the three models: `utils/mix6.py` `MODEL_PATHS` (last epoch / step of the default trainings); the previous run's are in `MIX25_MODEL_PATHS`, the older ones in `MIX6_MODEL_PATHS`.

## hecate 9.6 um (`hecate/`, runs in the reader venv; not part of the ensemble)

```bash
reader-v3/.venv/bin/python - <<'EOF'      # hecate needs an unnormalised uint8 (Z, Y, X) array
import numpy as np, zarr
a = np.asarray(zarr.open("ves_zarrs2/PRIZESHEET_11_11_flatboi_flatboi_abf.zarr", mode="r")[:])
np.save("_ves_tmp/render_u8.npy", np.clip(a, 0, 255).astype(np.uint8))
EOF
TMPDIR=/vesuvius/_ves_tmp reader-v3/.venv/bin/python hecate/hecate.py --checkpoint hecate/hecate_9.6um.pth \
  --input _ves_tmp/render_u8.npy --spacing-um 9.6 --precision bf16 --batch-size 16 --device cuda --output output/hecate/out.png
```

Output: a uint8 PNG (value = 255 x ink probability), same H x W as the render; it reads the central 16 of the 28 planes. Forward depth looked better than `--reverse`.
