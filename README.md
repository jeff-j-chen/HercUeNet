# HercUeNet: PHerc0846A submission

Three models read the same PHerc0846A surface volumes and are ensembled on one prize sheet: a **Studio** fine-tune (`finetune_studio.py`), **reader-v3** (`finetune_readerv3.py`) and **v8-in** (`finetune_v8in.py`). The released Studio model (`python3 finetune_studio.py setup` downloads its weights into `models/studio/` and builds `.venv-studio`) is the starting point of the Studio fine-tune and is shown next to it in the figures, but is not part of the ensemble.

- **Studio** (`finetune_studio.py`): the released 223M-parameter model, fine-tuned.
- **reader-v3** (`finetune_readerv3.py`): the released reader-v3, fine-tuned (twice: on mix28, then on mix28x).
- **v8-in** (`finetune_v8in.py`): the released v8-in, fine-tuned (twice: on mix28, then on mix28x).
- **HercUeNet** (`train.py`) is **retired**: nnunet3d baseline, not trained any more and not in the ensemble. `train.py`, `utils/config.py` and the asf cues stay for reference.

Experiment history, the legacy corpora and fragments, label conventions, data layout and normalization are kept in `old/KNOWLEDGE.md`.

## Current setup (2026-11-21)

Every script defaults to this; nothing needs a flag. **All three models start from the RELEASED models and train on the same corpus, mix28x_new** (`utils/mix6.py`; `python3 -m utils.mix6` checks it): the 14 mix28 PHerc0846A
sheets + the 2 hand-labelled 0846B sheets + the regenerated HerculeX x export of 2026-11-21 (253 of its 259 sheets, see below) + the labelled donor patches, sampled hand-grown 0846A/B : x : each donor scroll = 4 : 2 : 1. Each script keeps the learning-rate
schedule of its **full** run and stops itself (`utils/stop_at.py`) at the checkpoint the ensemble uses, so a model made by these defaults has the learning rate of the full schedule at that checkpoint. The trainings are very short on purpose.

| model | script | schedule, stopped at | output (the ensemble reads it) |
|---|---|---|---|
| Studio fine-tune | `python3 finetune_studio.py` (setup -> stage -> train, `.venv-studio`) | 2000-step schedule (AdamW lr 2e-5, 100 warm-up steps, cosine to 10%, micro-batch 4 x 4, bf16), **stopped at step 500**; jitter 40, soft nnPU (weight 0.5, prior 0.2), background weight 1 | `models/studio/mix28x_new_ps7/step_000500.safetensors` |
| reader-v3 fine-tune | `python3 finetune_readerv3.py` | 4000-step schedule (sgd lr 0.003, 200 warm-up steps, cosine), batch 14, background weight 0.8, **stopped at step 1000** | `reader-v3/runs/mix28x_new_ps7_s1/ckpt_001000.pth` |
| v8-in fine-tune | `python3 finetune_v8in.py` | 4-epoch cosine schedule (lr 1e-5, 15% warm-up), `NEG_WEIGHT` 10, **stopped after epoch 2** (a checkpoint every epoch; the ensembles use epoch 1 or epoch 2) | `nader-fork/runs/v8in_mix28x_new_4ep_ps7_s1/runs/*/checkpoints/trained-epoch0.ckpt` (epoch 1) and `trained-epoch1.ckpt` (epoch 2) |

`--stop-step` / `--stop-epoch` move the stopping point, `--steps` / `--epochs` change the length of the schedule, `--init <checkpoint>` starts reader-v3 / v8-in from another model than the released one, `--corpus mix28x` selects the first x export.
`VESUVIUS_V8IN_EPOCH=2` makes the notebooks read v8-in epoch 2 (default 1). The earlier models (corpus mix28x, the first x export; a two-stage chain from mix28 fine-tunes; a one-stage run) are kept under their own run names (`..._mix28x_ps7...`).

**The ensemble** (`ensemble_eval.ipynb`; the models are not depth-maxed; `utils/ensemble_from_maps.py` makes the figures from maps that are already cached, with no inference):

| model | checkpoint | stretch |
|---|---|---|
| Studio fine-tune | step 500 | plain, p1 -> 0, p99.5 -> 1 |
| reader-v3 | step 1000 | plain, p1 -> 0, p99.5 -> 1 |
| v8-in | epoch 1 (or 2) | **p50 -> 0**, p99.5 -> 1 (v8-in always needs it) |
| Studio base (released) | - | shown, **does not contribute** |

The default 3x2 figure (`ensemble_3x2.png`): Studio base (labelled in red as not in the ensemble) | Studio step 500, reader-v3 | v8-in, z-score composite | rank composite. The default 2x2 figure (`zscore_product_ct_2x2.png`): z-score composite | ct composite view, product composite | ct composite view. Run
`python3 utils/run_notebook.py ensemble_eval.ipynb`; output goes to `output/ensemble/mix28x_new_ps7_s1_studio500_reader1000_v8in1/<sheet>/`.

**The prize sheet is `PRIZESHEET_11_19_flatboi`** (alias `20261119120000`, the 2026-11-19 sheet, canonical since 2026-11-20, rendered upright). It is the only prize sheet: every earlier one is superseded and gone from the repository (its mesh was removed from `utils/tifxyz_0846A.tar.enc`, which now holds this sheet). Nothing trains or validates on it.
Runs are tagged by the prize sheet they were made against (`RUN_TAG` in `utils/mix6.py`, which also fixes the output folders above): `_ps7` = this sheet. Runs of earlier sheets (`_ps2` to `_ps6`) may still be on disk; their exclusion zones differ, so only `_ps7` runs are clean on this prize sheet.
Every run can be made with a second fixed seed (`s2`; `utils/mix6.py` `SEEDS`, `VESUVIUS_SEED_TAG=s2` or `--seed`), which ends the run names (`..._ps7_s1`, `..._ps7_s2`); `s1` uses each trainer's own default seed. The background (papyrus) pixels weigh less in the BCE term of the fine-tunes, so that predicting ink on papyrus is punished less: reader-v3 0.8 (was 1.0), v8-in `NEG_WEIGHT` 10 (25 in the `_ps3` runs, 20 in `_ps6`) (`mix6.V8IN_NEG_WEIGHT`, `mix6.READER_NEG_WEIGHT`; `finetune_readerv3.py` patches `LabelSmoothedDCAndBCELoss` in its pinned checkout to take `bce_kwargs.neg_weight`).

mix28 = mix26 (25 trained patches) + three fragments registered 2026-11-04 (the PHerc0009B patches `p9b_642` and `p9b_639` and the PHerc0306B first-letters sheet `p306b_letters`, see [new fragments](#new-fragments)); mix28x = mix28 + the HerculeX sheets (see below).
`--corpus mix28 | mix26 | mix25 | mix22 | mix18 | 0846a` on the fine-tune scripts selects the older corpora, and `MIX26_MODEL_PATHS` in `utils/mix6.py` names the interim models.

## Quick start

The project normally runs in a Docker image with system Python. No project virtual environment is needed for HercUeNet; the reader-v3 and v8-in fine-tunes build
their own environments (`reader-v3/.venv`, `nader-fork/.venv`) on first run. Run long jobs in `tmux`.

```bash
# everything the three trainings read, in one command. --mix28 sets up exactly the 29 sheets / patches of mix28 (the 28 registered patches plus the prize sheet): it renders the 14
# PHerc0846A sheets and the prize sheet (their meshes come from the passphrase-protected archive: it prompts, see below), fetches the 14 donors (the PHerc0306B sheet from Hugging Face), and builds masks, norms, surface detections (PHerc1447: flat;
# PHerc0846A: flat depth-13 aliases), the w013 .matched volume, surface anchors, the prize-sheet exclusions and 3 depths of asf cues.
# --check only reports what is missing; python3 -m utils.mix6 prints the same state read-only.
python3 assemble_training_segments.py --mix28

# the three models, defaults only (each prints where its model is)
python3 finetune_studio.py                            # Studio, mix28x, 2000-step schedule stopped at step 500
python3 finetune_readerv3.py                          # reader-v3, mix28x, 4000-step schedule stopped at step 1000
python3 finetune_v8in.py                              # v8-in, mix28x, 4-epoch schedule stopped after epoch 1
VESUVIUS_SEED_TAG=s2 python3 finetune_readerv3.py     # the same with the second seed (reader-v3 and v8-in)

# or all of it in one sequential queue (logs in logs_herculex/): corpus check -> Studio -> reader-v3 -> v8-in, each followed by a visualization of every saved checkpoint on the prize sheet
tmux new -d -s queue_mix28x 'bash /vesuvius/utils/queue_mix28x.sh'
python3 utils/run_notebook.py ensemble_eval.ipynb     # then the ensemble figures (3x2 and 2x2)
```

- **The PHerc0846A meshes are encrypted.** The tifxyz meshes of every PHerc0846A sheet (the 14 trained sheets, the prize sheet and the evaluation renders) are not in git: they ship as `utils/tifxyz_0846A.tar.enc` (openssl aes-256-cbc, pbkdf2 with 600000 iterations, over a tar of `tifxyz/<dir>` paths). When a sheet's render is missing and its mesh is not in `tifxyz/`, `assemble_training_segments.py` asks for the passphrase (up to 3 prompts, or `TIFXYZ_PASSPHRASE` in the environment) and extracts the archive; an empty or wrong answer skips the 0846A sheets, everything else is assembled, and `--mix28` then stops before finalizing the corpus (exclusions, anchors, cues), so rerun it with the passphrase. `python3 -m utils.unlock_tifxyz_0846A [--dest DIR]` extracts the archive by hand. The prize-sheet mesh `tifxyz/PRIZESHEET_11_19_flatboi/` has to be in the archive, so whoever holds the passphrase rebuilds it when the prize sheet changes.
- `utils/queue_mix28x.sh` runs the three trainings strictly one after another with the defaults of the scripts (Studio, reader-v3, v8-in; each stops itself at its checkpoint) and visualizes every saved checkpoint on the prize sheet after each; a failed step is retried once and does not stop the next one, and nothing trains unless the corpus check (`python3 -m utils.mix6`) passes. `utils/queue_ps6.sh` (HercUeNet, mix28, seeds `s1` / `s2`) is the earlier queue and stays for reference.
- `utils/queue_mix28_eval.sh` (the earlier untagged mix28 runs) waits for `queue_mix28.sh`'s session to end, aborts if a checkpoint is missing, runs `ensemble_eval.ipynb` on the prize sheet (3x2 and 2x2 figures), then `0846A_eval.ipynb` with the last reader-v3 checkpoint on the prize sheet, the 14 trained sheets, the donors and the held-out patches (not the remaining quick patches). Both use `python3 utils/run_notebook.py` (the executed copy goes to `output/notebook_runs/`).
- Every training refuses to start without current prize-sheet exclusions (`python3 -m utils.holdout_exclusion` rebuilds them for a new prize sheet).
- The fine-tunes stage their data once (`reader-v3/data/mix25`, `nader-fork/runs/v8in_mix25_data`; the folder names are historical and shared by every corpus).
- Checkpoints go to `models/runs/<run>/` and TensorBoard logs to `runs/<run>/`. An existing HercUeNet run is never overwritten (`_2`, `_3`, ...). Experiment runners call `train.main(configure)` to edit the plain `Config()` instead.
- `python3 k_folds.py` runs letter k-fold and leave-patches-out arms of the earlier best run (`iter2_asf_cue`) on the PHerc0846A patches.

## Corpus

`utils/mix6.py` defines the corpora (the module keeps its historical name so imports don't break). Older corpora stay selectable and frozen with `--corpus mix26 | mix25 | mix22 | mix18 | 0846a` on the two fine-tune scripts.

| Alias (label / cue / norm id) | Render (zarr / mask id) | Fragment | Physical scroll | Used by |
|---|---|---|---|---|
| `20260927134931` | `20260927134928` | ABF patch (`auto_grown_20260927134928530_abf`) | PHerc0846A | mix28 |
| `20261012073246` | `20261012073244` | 073244 patch (`auto_grown_20261012073244544_abf`) | PHerc0846A | mix28 |
| `20261012065813` | `20261012065812` | straight sheet -28 voxels (`..._straight_abf_m28`) | PHerc0846A | mix28 |
| `20261012065817` | `20261012065816` | straightened2 in_1 | PHerc0846A | mix28 |
| `20261012065819` | `20261012065818` | straightened2 out_1 | PHerc0846A | mix28 |
| `20261012065821` | `20261012065820` | straightened2 out_2 | PHerc0846A | mix28 |
| `20261012065823` | `20261012065822` | straightened2 out_3 | PHerc0846A | mix28 |
| `20261029120001` | `fiber_only_20261029d_flatboi_in_1` | fibre sheet in_1 | PHerc0846A | mix28 |
| `20261029120002` | `fiber_only_20261029d_flatboi_out_1` | fibre sheet out_1 | PHerc0846A | mix28 |
| `20261029120003` | `fiber_only_20261029d_flatboi_in_2` | fibre sheet in_2 | PHerc0846A | mix28 |
| `20261029120004` | `fiber_only_20261029d_flatboi_out_2` | fibre sheet out_2 | PHerc0846A | mix28 |
| `20261029130001` | `fiber_only_abf2_20261029c_flatboi` | abf2 fibre-only sheet | PHerc0846A | mix28 |
| `20261029130002` | `fiber_only_abf2_20261029c_flatboi_p28` | abf2 sheet moved +28 voxels along its normal | PHerc0846A | mix28 |
| `20260921094415` | `20260921094413` | PHerc0846A test surface 6 (`auto_grown_20260921094413486_flatboi`, uint16 render) | PHerc0846A | mix28 |
| `20250919125754` | same | patch 487 | PHerc0009B | mix28 |
| `20260108000000` | same | w041 | PHerc0139 | mix28 |
| `20260930144760` | same | w060 (flat surface, layer 15) | PHerc1447 | mix28 |
| `20250825181859` | same | 500P2 -1 (`500P2_m1`) | PHerc0500P2 | mix28 |
| `20250919132115` | same | patch 592 (`p9b_592`) | PHerc0009B | mix28 |
| `20261007000000` | same | w047-w048 first letters (NOT 343P) | PHerc0343 | mix28 |
| `20260221022814` | same | auto-grown 405 | PHerc0841 | mix28 |
| `20260220213127` | same | w00 (`p841_w00`) | PHerc0841 | mix28 |
| `20251112000002` | same | w087 | PHerc0172 | mix28 |
| `20240304141531` | same (read as `20240304141531.matched`) | w013 (pooled 2.4µm fine scan) | PHerc1667 | mix28 |
| `20250825181917` | same | 500P2 -2 (`500P2_m2`) | PHerc0500P2 | mix28 |
| `20250919130642` | same | patch 642 (`p9b_642`, a fractured sheet: many holes in its papyrus mask) | PHerc0009B | mix28 |
| `20250510172639` | same | patch 639 (`p9b_639`) | PHerc0009B | mix28 |
| `20261104000001` | same | first-letters sheet (`p306b_letters`, the 41-layer claim-surface render of Prodigle/pherc0306b-first-letters) | PHerc0306B | mix28 |
| `20261119120000` | `PRIZESHEET_11_19_flatboi` | **the prize sheet** (2026-11-19 sheet, canonical since 2026-11-20) | PHerc0846A | **held out, never trained or validated on** |

- **PHerc0846A sheets** are rendered from their tifxyz meshes (`MESHES_0846A` in `assemble_training_segments.py`) and read through a **flat depth-13 alias**: `<alias>.zarr` and the mask / label / train-mask PNGs are symlinks to the render, and `surface_labels/<alias>/` is a constant depth 13 over the mask (no detected surface). With `surface_window_offset = -1` HercUeNet reads slices 9-16. Aliases get their own norm and surface-anchor entries. Hand-drawn labels are in `dilated_inklabels/`, named after the alias; all 14 sheets are train only (no validation split).
- The fibre sheets (the prize sheet, the four `fiber_only_20261029d` sheets, abf2 and its +28 offset) render like extras: a uint8 zarr named after the mesh, the mask in `masks/extras/` linked into `masks/`. The four `fiber_only_20261029d` neighbours are rotated 180° (`ROTATE_180`, zarr attribute `rotated_180`) so every stored array is upright; the prize sheet's mesh grid is already upright, and it has no label.
- The other scrolls use their stored surface labels with window offset 0. w013 is a fine scan, so every model reads its spectrally matched sibling `ves_zarrs2/20240304141531.matched.zarr` (with its own norm, anchors and cues). Donors train on their `train_masks/` region only.
- For every PHerc0846A sheet the hard negatives are drawn on the inklabels at half brightness. `split_extra_labels` takes any solid stroke of 48..207 for one (127 +- 80), so a brush that drifted from 127 still counts; 208 and above is ink.
- The study behind the corpus choice (single-patch transfer to 0846A) is `runs_corr/report/summary.md`.

### New fragments

`p9b_642`, `p9b_639` and `p306b_letters` (registered 2026-11-04, ids `20250919130642`, `20250510172639`, `20261104000001`) are donors like the others: hand-drawn
`dilated_inklabels/<id>.png` and `train_masks/<id>.png` (255 trains, 119 or 127 hard negative, black unused, so everything unpainted is that patch's validation split in
HercUeNet and unused in reader-v3 / v8-in; `inklabels/<id>.png` is not read by the trainings). A donor with no ink in `dilated_inklabels` or no train-mask pixel is
**skipped** by the three trainers instead of crashing HercUeNet's character sampling (`character-aware split has no positive characters`): `utils/mix6.py`
`is_labelled`, `corpus_segments`, `trained_domains`; `train.py` prints the skipped ids. A scroll with no labelled fragment gets no domain, batch slot or group.
`p9b_642` is a fractured sheet (its papyrus mask has many holes); windows that reach a hole load like any other (the cue slice is zero there and the loss is masked).
`p306b_letters` is the 41-layer claim-surface render of [Prodigle/pherc0306b-first-letters](https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters)
(`results/claim_surface_render_41layers.zarr`, level 0, 8.64 µm, layer 20 = the surface), fetched with the Hugging Face CLI and resampled like the other 8.64 µm fragments.

### HerculeX marks export and the 2026-11-20 setup (corpus mix28x, prize sheet PRIZESHEET_11_19_flatboi, tag `_ps7`)

`VESUVIUS_CORPUS=mix28x` is mix28 plus 123 more PHerc0846 sheets, all registered in `utils/mix6.py` and built by `python3 -m utils.assemble_herculex` (renders uint8 over the whole mesh grid, copies the labels to `dilated_inklabels/`, writes the flat aliases, norms, anchors; `--check` only reports):

- **`HERCULEX_SHEETS`, the "x" sheets (121):** `tifxyz/HerculeX_marks_export_0846A` (54 sheets, 0846A volume), `..._0846B` (47) and `..._0846B_blank` (20, the 0846B volume; "blank" is only the labelling style). Aliases `20261118010001..`, `20261118020001..`, `20261118040001..`. Less reliable labels than the hand-grown sheets.
- **`NEW_HAND_SHEETS` (2):** the hand-labelled 0846B prize sheets `PRIZESHEET_B_11_18_abf_flatboi_flatboi` and `PRIZESHEET_B_2` (aliases `20261118030001/2`; labels `dilated_inklabels/<name>.png`).
- **Orientation:** every mesh whose normal points away from the scroll axis (the prize sheet's points toward it) is listed in `utils/depth_reversed.json` and rendered depth-reversed (`flipped_normals.txt` of each export; the three mix28 sheets `fiber_only_abf2_20261029c_flatboi`, `..._p28` and test surface 6 were found reversed too). The PRIZESHEET_B sheets are reversed as well.
- **Weights.** HAND = the 14 mix28 0846A sheets + the 2 hand 0846B prize sheets; X = the 121 x sheets; donors = every labelled donor scroll. Every training draws HAND : X : each donor scroll = 4 : 2 : 1, so 0846A/B are 43% of the draws and the hand-grown sheets twice the x ones. Studio: those probabilities (`finetune_studio.py` `select`); reader-v3: batch 14 = 4 + 2 + 1 per donor scroll (`BATCH_COUNTS`); v8-in: groups = HAND split in two halves, X, one per donor scroll (2 : 1 : 1 each).
- **Exclusion.** Only the 14 mix28 0846A sheets carry prize-sheet masks, plus the x sheets that come within a radius of it (`holdout_exclusion.hx_sheets`: six x sheets lose a sliver to v8-in's 25 vox; none to reader-v3's 18 vox). The 0846B sheets are in another volume: no exclusion. **Studio shares reader-v3's mask** (`holdout_exclusion_reader/`, 18 vox; its voxels are zeroed in the staged planes). `python3 -m utils.holdout_exclusion --force` rebuilds all three models' masks.
- **Sealed meshes.** The prize sheet, the three export folders and the two 0846B prize sheets are inside `utils/tifxyz_0846A.tar.enc` (`python3 -m utils.seal_tifxyz --add tifxyz/<dir> ... --remove tifxyz/<dir>` re-seals it from `TIFXYZ_PASSPHRASE`, `--verify` compares it with the working copies, `--list` names its contents; the working copies are git-ignored). `assemble_training_segments.py` and `assemble_herculex.py` extract it when a mesh is missing (passphrase prompt or `TIFXYZ_PASSPHRASE`). The old prize sheet `PRIZESHEET_11_17_2` is gone from the archive and the working tree.
- **The regenerated x export (2026-11-21, corpus mix28x_new).** `tifxyz/new_x.zip` (or the sealed archive) holds 45 PHerc0846A and 214 PHerc0846B sheets (`tifxyz/HerculeX_labels_0846{A,B}/`) that replace the first x export. `python3 -m utils.assemble_herculex2` builds all 259 under new names (zarr id `hx2_<mesh name>`, aliases `202611210100NN` / `202611210200NN`, `mix6.HERCULEX2_SHEETS`): uint8 renders, labels, flat aliases, norms and anchors, the rendered shape checked against `manifest.json`, depth reversed for the meshes in `flipped_normals.txt`. The A sheets are already cut at least 26 voxels from the prize sheet. Six B sheets (`mix6.HERCULEX2_DROPPED`) lie within 30 voxels of a held-out evaluation sheet and are not trained on, so the corpus has 253 of the 259. `python3 -m utils.mix6` and `utils/stage_mix28x_new.sh` check and stage it.
- **Disk quota.** The pod's quota counts the files of the whole volume in the filesystem's own units, not the sizes `du` shows (small zarr chunks count much more), and it was hit once during staging ("Disk quota exceeded", staging killed). Check free space with a test write of a few GB before and during big jobs; `utils/stage_watchdog.sh` does so while staging next to a training. Safe to delete when space is short: staged data of the old x export, experiment checkpoints, chunk caches in `_ves_tmp`.
- **Local-only sheets.** A sheet that is not published is rendered with `python3 -m utils.render_sheet <name> B --reversed --secret` (depth reversal in the git-excluded `utils/depth_reversed_local.json`); `utils/local_eval_sheets.txt` (git-excluded) lists evaluation sheets the queue visualizes in addition to the prize sheet.
- **HercUeNet is retired** (not trained, not in the queue or the ensemble; `train.py` and the asf cues stay for reference; the data check ignores the cues unless `VESUVIUS_CUES=1`).
- **Studio fine-tune** (`python3 finetune_studio.py setup|stage|train`, trains in `.venv-studio`): the released 223M-parameter model on mix28x, a 2000-step schedule stopped at step 500 (micro-batch 4 x accumulation 4, bf16, 14.5 GiB), AdamW lr 2e-5, background-weighted BCE + dice, plus **jitter** (the sampled pixel lands anywhere in the 96 px supervised core; the first fine-tune centred it and painted a blob on every tile centre of the 64 px inference lattice) and a **soft non-negative PU** term (`--pu-weight 0.5 --pu-prior 0.2`) against over-predicting ink. Experiments: `output/studio_ft_exps/` (`report.txt`).
- **Queue** `utils/queue_mix28x_new.sh` (reader-v3, Studio, v8-in from the released models on mix28x_new with the defaults of the scripts, a visualization of every checkpoint on the evaluation sheets, then two ensembles from the cached maps, one per v8-in epoch; `utils/smoke_mix28x_new.sh` tests every step first) and the earlier `utils/queue_mix28x.sh` (one model after the other with the defaults of the scripts, retries once, no ensembles): Studio -> reader-v3 -> v8-in, each followed by `python3 -m utils.viz_checkpoints studio|reader|v8in`, which runs every saved checkpoint on the prize sheet, caches the maps in `output/prize_PRIZESHEET_11_19_flatboi/cache/` and writes one image per model. `ensemble_eval.ipynb` then makes the ensemble figures: `ENSEMBLE_STUDIO_CKPT`, `ENSEMBLE_READER_CKPT`, `ENSEMBLE_V8IN_CKPT`, `ENSEMBLE_SHEETS` choose the models and sheets.
- **Memory.** The container has 62 GiB of RAM, not the 251 GiB `free` shows. HercUeNet's workers were OOM-killed twice (its mix28x run needs ~42 GiB that cannot be reclaimed, plus 14 GiB of `/dev/shm/ves_zarr_cache` left by inference runs). The queue clears that cache between steps; never run a second heavy job next to a training.

### Prize-sheet exclusion

Every trained PHerc0846A sheet loses its papyrus near the prize sheet, by model (`utils/holdout_exclusion.py`; the distance is the 3D distance in voxels from each mesh point of the trained sheet to the prize-sheet mesh):

| model | discarded | masks |
|---|---|---|
| HercUeNet | within **12 voxels**, dilated by the model's full context reach: 148 px = (192 px context + 64 px prediction core) / 2 + 20 px context jitter, +1 px margin. A kept 16 px cell never has a 192 px input window that touches the zone. The asf cue of the sheet is also built over its papyrus minus the 12-voxel zone (`holdout_exclusion/zone/`), so it carries nothing from there | `holdout_exclusion/` |
| reader-v3 | within **18 voxels**; those voxels are also zeroed in the staged volume | `holdout_exclusion_reader/` |
| v8-in | within **25 voxels**; those pixels are also zeroed in the staged layers | `holdout_exclusion_v8in/` |

Each folder has a `summary.json` naming the prize sheet, the radius and the context dilation. All three trainings refuse to start unless the masks of all 14 trained 0846A sheets exist and were computed against the current prize sheet (`train.py` preflight, `finetune_readerv3.py`, `finetune_v8in.py`), and `train.py` also refuses a cue that predates its zone. The same check fails if a trained sheet keeps less than 0.001 cm² of ink after its exclusion, which crashes the data setup: straightened2 abf in / out_1 lie on the prize sheet and lose all their ink to HercUeNet's 12 vx + context zone, so `CUE_DROPPED` (`utils/mix6.py`) removes them from the HercUeNet corpus only. `python3 -m utils.holdout_exclusion` rebuilds whatever is stale; figures (per sheet and an overview) go to `output/holdout_exclusion*/`.

## The three trainings

**Studio** (`finetune_studio.py`): see the setup above and the Studio bullet of the mix28x section; reads 16 planes (layers 5..20 of the 28-layer render) in 128 px windows.

**Sampling.** HercUeNet weights the 0846A domain 2 and each of the nine donor scrolls 1 (`equalize_domain`; a scroll without a labelled fragment is skipped), draws every patch of a domain equally often and rotates through the letters (8-connected ink components) of a patch with a positive window and a ring-negative window each. reader-v3 draws a fixed quota per scroll (batch 13: 0846A 4, each of the nine donor scrolls 1; with fewer scrolls 0009B takes the freed slots) uniformly over 384 px windows. v8-in balances eleven groups (two halves of 7 sheets for 0846A, one per labelled donor scroll) and draws 64 px tiles in proportion to their supervised area. 0846A therefore gets 18% (HercUeNet), 31% (reader-v3) and 18% (v8-in) of the draws.

**HercUeNet** (`apply_mix6_run` in `utils/config.py`):

- 192 px field (batch 32, lr 1e-4), dual scale (mix 0.5), strong nnPU (`pu_lambda` 3, 50% of the negative draws taken at least 96 px from any label), text-region gate, ring close 2 / gap 2 / shell 4, the 192 px native-noise bank, seed 17, 4 epochs (2 warmup)
- a fourth gated input head, next to raw / lcn / dz, reads the precomputed asf_morph map (`model.input_cue = "pre_clahe"`, `data.cue_dir = ./cue_asf_morph`) and trains at 5x the learning rate. The cue is replaced by a flat map (cue dropout) in 20% of the samples of 0846A, 0009B and 0172, which it helps on, and in 100% of the samples of every other scroll (also in validation), so the cue head only learns from those three (`CUE_DROPOUT_BY_DOMAIN` in `utils/mix6.py`, `data.cue_dropout_prob_by_scroll`); run the model on any other scroll with the flat cue. Depth jitter reads the shifted maps `cue_asf_morph_dm1` / `_dp1`. A trained 0846A sheet's cue is built without its 12-voxel zone (`data.cue_exclusion_dir`); the prize sheet only gets depth 0
- half-brightness strokes in the PHerc0846A labels become hard negatives (`extra_hard_negative_scroll_ids`)
- starts from `models/init_mae192_cue4_seed17.pth`, the 192 px campaign-40 MAE grown to four input heads (`utils/cue_expand.py`); `train.py` builds it when it is missing
- the prize sheet is rendered in the end-of-training figures and scored over its whole patch; it is never trained or validated on

**reader-v3** (`finetune_readerv3.py`, `setup` -> `prep` -> `train`): `setup` clones `DomRusso2/reader-v3` and the koine_machines training code (ScrollPrize/villa, sparse) into `reader-v3/` and `villa-ink/` at pinned commits, builds `reader-v3/.venv` with uv from `requirements_readerv3.txt` and downloads `reader-v3/reader-v3.pth`. `prep` stages a 21-layer crop around each scroll's surface layer (the trainer jitters a 17-layer window by +-2), the ink label and a supervision mask. Training: sgd, lr 0.003 with 200 warmup steps, 4000-step cosine schedule stopped at step 1000, batch 14, 384 px tiles, fp16, a checkpoint every 1000 steps (`ckpt_001000.pth`). uv's cache goes to `_ves_tmp/uv_cache` (set `UV_CACHE_DIR` to change it).

**v8-in** (`finetune_v8in.py`, `setup` -> `stage` -> `train` -> `export`): `setup` clones `jeff-j-chen/nader-fork` (git lfs pulls the weights) and builds `nader-fork/.venv` from `requirements_v8in.txt`. `stage` writes the central 24 layers as tiffs plus label and mask PNGs. Training: the PH1447 recipe of nader-fork with a cosine schedule over the 4 epochs, micro-batch 2 x accumulation 16 (effective batch 32), `NEG_WEIGHT` 10, stopped after epoch 1. A `trained-epoch<N>.ckpt` is saved after every epoch; the epoch used is exported to a model folder on first use. `V8IN_EPOCHS` overrides the epoch count.

**Labels differ per model.** HercUeNet: half-brightness strokes of `dilated_inklabels` are hard negatives in 0846A and unused letters elsewhere; ring negatives and nnPU far background surround the ink. reader-v3 and v8-in: half-brightness strokes are not ink and stay supervised background in 0846A; every other scroll trains on its `train_masks` region only (255 trains, 119 = hard negative inside the papyrus mask, black = unused, half-brightness strokes unused).

## Evaluation notebooks

| Notebook | Role |
|---|---|
| `ensemble_eval.ipynb` | Ensembles the Studio fine-tune (step 500, p1 -> 0), reader-v3 (step 1000, p1 -> 0) and v8-in (epoch 1, p50 -> 0) of the mix28x_new `_ps7_s1` runs on the prize sheet `PRIZESHEET_11_19_flatboi` (checkpoints at the paths of `utils/mix6.py` `default_models()`; the released Studio is shown but does not contribute; `ENSEMBLE_*_DEPTHS` change the depth shifts, `ENSEMBLE_SUBDIR` the output folder). Saves a 3x2 grid `ensemble_3x2.png` (Studio base, in red as not contributing | Studio step 500 / reader-v3 | v8-in / z-score composite | rank composite) plus the composites as plain PNGs, then a 2x2 figure `zscore_product_ct_2x2.png`: z-score composite | ct composite view, product composite | ct composite view. Output goes to `output/ensemble/<ENSEMBLE_SUBDIR>/<sheet>/` (default `mix28x_new_ps7_s1_studio500_reader1000_v8in1`) |
| `0846A_eval.ipynb` | Runs one of the three models (the kind is detected from `CHECKPOINT`, default the reader-v3 step 1000 of the ensemble) on the prize sheet, the 14 trained sheets, the donor patches, the held-out patches and the remaining PHerc0846A quick patches |
| `eval_scrolls.ipynb` | Runs a HercUeNet checkpoint on every labeled training fragment; useful for relabeling and sanity checks |
| `eval_test.ipynb` | Runs a HercUeNet checkpoint on the 13 test surfaces (see [Test surfaces](#test-surfaces)) |

**Ensemble.** Each model predicts the sheet at its own resolution (Studio, reader-v3 and v8-in per pixel). Studio and reader-v3 are stretched p1 -> 0, p99.5 -> 1 over the sheet mask; v8-in p50 -> 0, p99.5 -> 1 (its background pedestal is the median). The z-score composite is the mean of the per-model z-scores of the stretched maps, the rank composite the mean of their rank scores, the product composite multiplies the three stretched maps, so it is high only where all three agree. Predictions are cached under `output/ensemble/<ENSEMBLE_SUBDIR>/<sheet>/cache/` (delete a file to recompute that model). The ct composite view is the masked max projection of the 17 central layers reader-v3 reads.

**0846A notebook.** Settings cell: `CHECKPOINT` (any of the three; `VESUVIUS_CHECKPOINT` overrides it), the depth-ensemble options and `DEPTH_OFFSET = -1` (HercUeNet only: slices 9-16 of the flat depth-13 surface). HercUeNet replays its run config including its input heads (restart a long-lived kernel if the 4-head check fails); reader-v3 reads the 17 central layers and v8-in the 24 central layers, both per pixel in their own virtualenvs (`utils/predict_models.py`). Cells, in order: the prize sheet, the 11 older trained sheets (one loop cell), the three sheets added in mix26 (abf2, abf2 +28, test surface 6; one cell each), the donor patches (one loop cell), the held-out patches (straightened2 out_4 / out_5, three quick patches), and the remaining quick patches into a `quick_patches/` subfolder. Trained patches get a 2x2 figure (prediction, composite, prediction with the inklabel coloured by which models trained on it, asf_morph cue); the others get prediction | composite. Figures go to `output/0846A/<model>/`. The inklabel colours: red = used by all three, orange = dropped by HercUeNet only, yellow = dropped by HercUeNet and v8-in, dark yellow = dropped by all three.

**Headless runs.** `python3 utils/run_notebook.py <notebook> [--keep-going] [--skip REGEX]` executes a notebook and saves the executed copy to `output/notebook_runs/`; `0846A_eval.ipynb` also reads `VESUVIUS_OUT_DIR` and `VESUVIUS_V8IN_BATCH`.

The HercUeNet notebooks start with two depth-ensemble options: `depth_ensemble = []` predicts at one depth, `[low, high]` at every offset `low..high` around the usual depth (`[-3, 3]` = 7 depths); `ensemble_method` combines them (`'z-score'`, `'max'`, `'mid-boost'`, `'mean'`; empty means `'mid-boost'`, see `utils/depth_ensemble.py`).

`eval_test.ipynb` renders `raw inference | inference | fiber composite | composite with 1 cm scale bar` for each test fragment into `output/test_visualizations/<run_id>/`. Missing zarrs, masks, norms and surface labels are generated on demand. The figure needs some small edits before it can be submitted for a prize.

## Repository layout

Only these files live in the base directory: `README.md`, `train.py`, `finetune_studio.py`, `finetune_readerv3.py`, `finetune_v8in.py`, `assemble_training_segments.py`, `assemble_test_segments.py`, `k_folds.py`, the four notebooks above, and `requirements.txt` / `requirements_readerv3.txt` / `requirements_v8in.txt` (`LICENSE` as well). Call helper scripts as modules from the repo root: `python3 -m utils.<name>`.

| Path | Purpose |
|---|---|
| `utils/mix6.py` | The corpora (mix28 and the frozen older ones), the prize sheet, the model locations (`MODEL_PATHS`), data checks (`python3 -m utils.mix6 [--build]`), flat PHerc1447 surfaces and the 3-depth asf cues |
| `utils/holdout_exclusion.py` | The prize-sheet exclusion masks of the trained 0846A sheets and their readiness check |
| `utils/unlock_tifxyz_0846A.py`, `utils/tifxyz_0846A.tar.enc` | The passphrase-protected archive of the PHerc0846A meshes and its extractor |
| `utils/config.py` | Shared configuration; `apply_mix6_run` is what `train.py` trains |
| `utils/model.py`, `dataloader.py`, `training_utils.py`, `visualizer.py`, `surface.py` | The 3D nnU-Net, labels / rings / splits / sampling, optimizers and losses, TensorBoard figures, surface-relative depth targets |
| `utils/norm.py`, `utils/norm_cache.json`, `utils/surface_anchor_cache.json` | Chunk-aligned masked normalization and surface-anchor normalization, with their per-fragment caches |
| `utils/precomputed_asf.py`, `utils/build_domain_cues.py` | The whole-patch asf_morph cue maps in `cue_asf_morph/` (+ `_dm1`, `_dp1`) |
| `utils/predict_models.py` | Inference of reader-v3 and v8-in in their own environments; detects which of the three models a checkpoint is |
| `utils/generate_surface_supervision.py`, `utils/precompute_norm.py`, `utils/depth_reversed.json`, `utils/depth_flip.py` | Literal surface maps, norm precompute, the list of depth-reversed renders and the in-place flip |
| `utils/run_notebook.py`, `utils/queue_ps6.sh`, `utils/seed_ensemble.py`, `utils/label_overlay_figures.py`, `utils/AGENT_QUICKSTART.md` | Headless notebook runs, the current queue, the training-sheet label figures, short operating notes |
| `old/` | Unused scripts, campaign history and finished studies (`old/KNOWLEDGE.md` is the long record) |

| Directory | Contents |
|---|---|
| `ves_zarrs2/` | Local 28-layer surface volumes (aliases are symlinks to the renders) |
| `masks/` | Valid papyrus footprints (`masks/extras/` for the fibre sheets and quick patches) |
| `dilated_inklabels/`, `train_masks/` | Training labels and explicit train / hard-negative regions (tracked in git) |
| `surface_labels/` | Literal surface depth and confidence maps |
| `cue_asf_morph*/` | Precomputed asf_morph cue maps (git-ignored; rebuilt by `--mix28`) |
| `holdout_exclusion*/` | Prize-sheet exclusion masks per model |
| `tifxyz/` | The meshes every patch is rendered from (the PHerc0846A meshes are not in git: `utils/tifxyz_0846A.tar.enc`) |
| `models/`, `runs*/`, `reader-v3/`, `nader-fork/` | Checkpoints, TensorBoard logs and the fine-tune workspaces |
| `output/`, `logs_mix28/`, `_ves_tmp/` | Figures and reviews, queue logs, regenerable scratch data |

## Test surfaces

The thirteen VC3D-grown test surfaces are the final discovery targets. They come from physical scrolls outside the supervised corpus. The same thirteen IDs are registered in `DEFAULT_TEST_SCROLL_IDS` (`utils/config.py`), `FRAGMENTS` in `assemble_test_segments.py`, and `FRAGMENTS` in `eval_test.ipynb` (notebook keys are the tifxyz names, values are the zarr IDs).

Shape `(H,W)` and mask coverage are measured from the current files in `masks/`; every zarr has 28 layers. Surface area is geometric mesh area from `meta.json`; `~` values are approximate because merged surfaces lack a reliable `area_cm2`.

| # | ID | Physical scroll | Shape `(H,W)` | Mask coverage | Surface area | Scan | tifxyz mesh |
|---|---|---|---:|---:|---:|---|---|
| 1 | `20260814140748` | PHerc0813 | `(5081, 5701)` | 43.4% | 33.31 cm² | 9.362µm / 113keV | `auto_grown_20260814140748456_flatboi` (updated patch) |
| 2 | `20260717193517` | PHerc0211 | `(7181, 6501)` | 70.5% | ~28.60 cm² | 9.362µm / 113keV | `auto_grown_20260717193517520_0_1_2_3_4_merged_flatboi` (five patches) |
| 3 | `20260720090842` | PHerc1203 | `(3421, 3821)` | 65.1% | 7.90 cm² | 9.362µm / 113keV | `auto_grown_20260720090842117_flatboi` |
| 4 | `20250703034159` | PHerc1447 | `(7181, 9401)` | 70.0% | 51.27 cm² | 8.640µm / 116keV | `20250703034159_flatboi` |
| 5 | `20260723112922` | PHerc0826 | `(5981, 3201)` | 60.0% | ~18.93 cm² | 9.362µm / 113keV | `auto_grown_20260723112922652_merged_flatboi` |
| 6 | `20260921094413` | PHerc0846A | `(5021, 7721)` | 63.3% | 22.88 cm² | 9.362µm / 113keV | `auto_grown_20260921094413486_flatboi` |
| 7 | `20260918132724` | PHerc0175A | `(4501, 5141)` | 61.0% | 11.95 cm² | 8.640µm / 116keV | `auto_grown_20260918132724424_flatboi` |
| 8 | `20260922073234` | PHerc0306B | `(5121, 6181)` | 56.9% | 14.13 cm² | 8.640µm / 116keV | `auto_grown_20260922073234974_flatboi` |
| 9 | `20260922161631` | PHerc0800 | `(6921, 8081)` | 55.9% | ~23.36 cm² | 8.640µm / 116keV | `auto_grown_20260922161631422_merged_flatboi` |
| 10 | `20260925085345` | PHerc1447 | `(4901, 4621)` | 64.3% | 10.90 cm² | 8.640µm / 116keV | `auto_grown_20260925085345806_abf` (second surface) |
| 11 | `20260928000001` | PHerc0800 | `(2321, 4141)` | 77.4% | 5.50 cm² | 8.640µm / 116keV | `PHerc0800_z9264_w040_abf` (expanded auto-winder segment) |
| 12 | `20260928000002` | PHerc0125 | `(1161, 3141)` | 77.3% | 2.50 cm² | 9.362µm / 113keV | `PHerc0125_z15344_w040_abf` (expanded auto-winder segment) |
| 13 | `20260928000003` | PHerc0211 | `(1941, 3641)` | 53.5% | 26.229 cm² | 9.362µm / 113keV | `PHerc0211_z7312_w080_abf` (expanded auto-winder segment) |

`assemble_test_segments.py` renders each surface from its tifxyz coordinates and the listed raw CT volume; the scan column is the volume the mesh was built on. Every render lands on the 9.362µm training grid (in-plane pixels and depth steps): 9.362µm volumes are nearest-sampled, 8.640µm volumes are trilinearly resampled, like the 8.64µm training fragments. Zarrs record `voxel_um`/`source_voxel_um`; shapes in the table above for 8.640µm surfaces predate this (~8% larger), and stale surface labels, norms and composites are dropped on re-render.

### Flattened meshes

`tifxyz/` holds only the SLIM-flattened `_flatboi` meshes (`x/y/z.tif` + `meta.json`, no `mask.tif`); the unflattened originals were removed. They were produced headlessly with the patched VC3D build in `~/Downloads/squashfs-root/usr/bin`, reproducing the GUI's SLIM-flatten action (keep 1.5%, 50 iterations, `symmetric_dirichlet`, tol 1e-5):

```bash
vc_tifxyz2obj <seg> <seg>_coarse.obj --keep=1.5
flatboi <seg>_coarse.obj 50 symmetric_dirichlet --tol=1e-05
vc_tifxyz2obj <seg> <seg>_fine.obj
vc_obj_uv_lift <seg>_coarse.obj <seg>_coarse_flatboi.obj <seg>_fine.obj <seg>_lifted.obj
vc_obj2tifxyz <seg>_lifted.obj <seg>_flatboi --tifxyz-source=<seg>
```

Notes:

- `vc_obj2tifxyz` copies the source `meta.json`, so `area_cm2` in `_flatboi` folders is inherited, not recomputed
- PHerc0800 (`auto_grown_20260922161631422_merged_flatboi`) was flattened in the VC3D GUI; the other eight headlessly
- the second PHerc1447 surface (`auto_grown_20260925085345806_abf`) is used as supplied, without a `_flatboi` pass, and keeps its `mask.tif`
- test surfaces 11-13 are expanded versions of the atlas extras `PHerc0800_z9264_w040`, `PHerc0125_z15344_w040` and `PHerc0211_z7312_w080`, used as supplied; their numeric IDs are synthetic, and their areas are computed from the mesh because `meta.json` has no `area_cm2` or a stale one
- `assemble_test_segments.py` reads the `_flatboi` meshes, but zarrs in `ves_zarrs2/` rendered before 2026-09-23 came from the unflattened meshes; re-render with `--force`

### Extra segment sets

Besides the numbered test surfaces, `assemble_test_segments.py --include-extras` renders two unlabeled extra sets for broader test-time review. They are keyed by their folder name and stored as uint8. Their masks go in `masks/extras/`, surface labels in `surface_labels/extras/` and composites in `output/composite_cache/extras/`. None of these are committed: each set's surface labels and composites are synced through R2 as `<set>_surface_labels.zip` and `<set>_composites.zip`. Extras are not in `DEFAULT_SCROLLS` or `DEFAULT_TEST_SCROLL_IDS`, so MAE pretraining never samples them.

- `atlas`: auto-winder meshes, downloaded from R2 as `atlas.zip` into `tifxyz/atlas/`
- `quick_patches_from_D`: meshes tracked in git under `tifxyz/quick_patches_from_D/<scroll>/<segment>/`. Any segment folder added there is assembled automatically, provided its scroll has a raw volume in `EXTRA_VOLUMES`.

| Scroll | quick_patches_from_D segments | Scan |
|---|---:|---|
| PHerc0846A | 37 | 9.362µm / 113keV |
| PHerc0175A | 25 | 8.640µm / 116keV |

A few PHerc0846A meshes reach up to ~4% past the top of the scan; those points are clamped to the last slice.

```bash
# render missing extras, then pull or generate their surface labels + composites
python3 assemble_test_segments.py --include-extras
# re-zip and upload every set's surface labels + composites (needs R2_PUBLIC_URL + ACCESS_KEY/SECRET_KEY)
python3 assemble_test_segments.py --include-extras --upload-extras
```

Do not combine `--only` with an R2-configured `--include-extras` run: the automatic re-upload would replace a set's R2 zips with only the selected segments.
