# HercUeNet — PHerc 0846A First Letters

Winning ink-detection pipeline for the **Vesuvius Challenge PHerc 0846A First Letters Prize**.

Three pretrained multi-scroll ink models—**Studio**, **reader-v3**, and **v8-in**—are fine-tuned on a shared multi-scroll corpus (`mix28x_new`) and **stopped early along their full cosine schedules** to adapt to PHerc 0846 without overfitting, then ensembled per pixel on the held-out prize sheet (`PRIZESHEET_11_19_flatboi`). *(The original 3D nnU-Net in `train.py` is retired and excluded from the ensemble.)*

---

## Quickstart

Requires Linux, an NVIDIA GPU (~20 GB VRAM), ~64 GB RAM, Python 3, `git`, `git-lfs`, and `uv`. Each trainer bootstraps its own isolated virtualenv (`.venv-studio`, `reader-v3/.venv`, `nader-fork/.venv`) and base weights on first run. Encrypted PHerc 0846A/B meshes in `utils/tifxyz_0846A.tar.enc` use `TIFXYZ_PASSPHRASE` (or prompt interactively).

```bash
# 1. Assemble the mix28x_new corpus + prize-sheet exclusion masks
export TIFXYZ_PASSPHRASE="LuaZYY9dHUJZr8jhtpOyBxZFBuP0Yqah"
python3 assemble_training_segments.py --mix28                                   # 14 0846A sheets, prize sheet, donors, exclusions
python3 -m utils.assemble_herculex --only 20261118030001 20261118030002 --no-cues  # 2 hand-labelled 0846B sheets
python3 -m utils.assemble_herculex2                                             # 259 regenerated HerculeX 0846A/B sheets
python3 -m utils.mix6                                                           # verify corpus state (read-only)

# 2. Fine-tune all three models (defaults match the winning ensemble; run sequentially in tmux)
python3 finetune_studio.py       # setup -> stage -> train (stops at step 500, ~20 min)
python3 finetune_readerv3.py     # setup -> prep  -> train (stops at step 1000, ~25 min)
python3 finetune_v8in.py         # setup -> stage -> train (stops after epoch 2, ~17 h/epoch)

# 3. Predict and build ensemble composites on PRIZESHEET_11_19_flatboi
python3 utils/run_notebook.py ensemble_eval.ipynb
```

Outputs write to `output/ensemble/mix28x_new_ps7_s1_studio500_reader1000_v8in1/PRIZESHEET_11_19_flatboi/` (`ensemble_3x2.png` and `zscore_product_ct_2x2.png`).

---

## Models & Training Duration

All three scripts keep the learning-rate trajectory of a longer cosine schedule and halt early via `utils/stop_at.py` before memorizing papyrus texture:

| Model | Script | Base Checkpoint | Schedule & Stopping Point | Wall Time (1 GPU) | Output Checkpoint |
|---|---|---|---|---|---|
| **Studio** (223M) | `finetune_studio.py` | `vesuvius-studio-multiscroll-cpt` (`softloss-20261023-ema`) | 2,000-step cosine (AdamW `2e-5`, 100 warmup, batch `4×4`, bf16, 128 px / 96 px core, jitter 40, soft nnPU), **stopped at step 500** | **~20 min** (14.5 GiB VRAM) | `models/studio/mix28x_new_ps7/step_000500.safetensors` |
| **reader-v3** | `finetune_readerv3.py` | `domenicor046/reader-v3` (`koine_machines`) | 4,000-step cosine (SGD `3e-3`, 200 warmup, batch 14, 384 px, fp16, BCE neg weight `0.8`), **stopped at step 1,000** | **~25 min** | `reader-v3/runs/mix28x_new_ps7_s1/ckpt_001000.pth` |
| **v8-in** (r3d50) | `finetune_v8in.py` | `jeff-j-chen/nader-fork` (PH1447 recipe) | 4-epoch cosine (`1e-5`, 15% warmup, batch `2×16`, 64 px, `NEG_WEIGHT=10`), **stopped after epoch 2** (ensemble defaults to **epoch 1**; `VESUVIUS_V8IN_EPOCH=2` selects epoch 2) | **~17 h / epoch** | `nader-fork/runs/v8in_mix28x_new_4ep_ps7_s1/runs/*/checkpoints/trained-epoch{0,1}.ckpt` |

- **Ensemble (`ensemble_eval.ipynb`):** Per-pixel depth-0 predictions are linearly stretched over the papyrus mask (`p1 -> 0, p99.5 -> 1` for Studio and reader-v3; `p50 -> 0, p99.5 -> 1` for v8-in) and combined by **z-score mean**, **rank mean**, and **product**. Released Studio base is plotted alongside for comparison only.
- **Common flags:** `--stop-step` / `--stop-epoch` adjust early stopping; `--corpus` selects older frozen corpora (`mix28x`, `mix28`, `mix26`, `mix25`); `VESUVIUS_SEED_TAG=s2` trains a second seed (`s1` default).

---

## Training Corpus (`mix28x_new`)

Defined in `utils/mix6.py`, standardized to 28-layer surface volumes on a **9.362 µm** grid (PHerc 0846A/B read via flat depth-13 aliases):

- **Hand-labelled PHerc 0846 (`HAND`, 16 sheets):** 14 hand-grown PHerc 0846A sheets + 2 hand-labelled PHerc 0846B prize sheets.
- **HerculeX export (`X`, 253 sheets):** Regenerated 2026-11-21 export (45 PHerc 0846A + 208 PHerc 0846B; 6 B sheets within 30 voxels of held-out evaluation surfaces are excluded).
- **Multi-scroll donors (13–14 patches across 8–9 scrolls):** Hand-labelled `train_masks/` regions from `PHerc0009B` (4), `0841` (2), `0500P2` (2), `0343` (1), `1447` (1), `0172` (1), `0139` (1), `0306B` (1), and spectrally matched `1667` `w013` (unlabelled donors are skipped automatically).
- **Sampling weights:** `HAND : X : each donor scroll = 4 : 2 : 1` (~43% of draws from PHerc 0846A/B, weighting curated hand-grown sheets 2× over HerculeX exports).
- **Strict holdout & 3D exclusion:** `PRIZESHEET_11_19_flatboi` (alias `20261119120000`) is never trained or validated on. Any papyrus on 0846A training sheets within **18 voxels** (Studio & reader-v3) or **25 voxels** (v8-in) of the prize-sheet 3D mesh is masked and zeroed in staged volumes (`utils/holdout_exclusion.py`); trainers abort preflight if `_ps7` exclusions are missing.

---

## Key Files

| Path | Role |
|---|---|
| `finetune_studio.py`, `finetune_readerv3.py`, `finetune_v8in.py` | Self-contained `setup` / `stage` (`prep`) / `train` entrypoints for the three ensemble models |
| `assemble_training_segments.py`, `utils/assemble_herculex*.py` | Surface volume downloaders, tifxyz mesh renderers, flat aliases, norms, and exclusions |
| `ensemble_eval.ipynb`, `0846A_eval.ipynb` | 3-model prize-sheet ensemble and single-model sheet/donor evaluation (`python3 utils/run_notebook.py <nb>`) |
| `utils/mix6.py`, `utils/holdout_exclusion.py` | Corpus registry, default checkpoint resolution, readiness checks, and 3D holdout exclusion masks |
| [old/KNOWLEDGE.md](old/KNOWLEDGE.md) | Full corpus tables, experiment history, retired `HercUeNet` (`train.py`) docs, PHerc 0175A notes, and test surfaces |

