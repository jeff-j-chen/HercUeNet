#!/usr/bin/env bash
# the orig7 experiment queue: all three models train on the same corpus, the 7 original PHerc0846A sheets + the 11 mix26 donors
# (utils/mix6.py orig7; HercUeNet still omits the two sheets CUE_DROPPED that lose all ink to its exclusion zone), with the labels of
# commit 458f8f3 and the current prize-sheet exclusion. Strictly sequential (one GPU job at a time); logs in logs_orig7/.
#   1. reader-v3   python3 finetune_readerv3.py --corpus orig7   3000 steps  -> reader-v3/runs/orig7_3000/ckpt_003000.pth
#   2. HercUeNet python3 train.py (VESUVIUS_CORPUS=orig7)       3 epochs    -> models/runs/cue_dropout_labelsv6_orig7/final.pth
#   3. v8-in       python3 finetune_v8in.py --corpus orig7        3 epochs    -> nader-fork/runs/v8in_orig7_3ep/model
#   4. ensemble_eval.ipynb on the prize sheet with the three results (3x2 and 2x2 figures) -> output/ensemble/orig7/<sheet>/
#   5. 0846A_eval.ipynb with the HercUeNet result on every PHerc0846A figure of the notebook except the remaining quick patches
#      -> output/0846A/runs_cue_dropout_labelsv6_orig7_final/
# A failed training does not stop the next one; the evaluations run only when all three results exist.
#   tmux new -d -s orig7 'bash /vesuvius/utils/queue_orig7.sh'
set -u
cd /vesuvius
LOG=logs_orig7
mkdir -p "$LOG"
export VESUVIUS_CORPUS=orig7 VESUVIUS_CUE_EPOCHS=3
status() { echo "[orig7 $(date '+%F %T')] $*" | tee -a "$LOG/queue.log"; }

step() {  # step <log name> <command...>
    local name=$1; shift
    status "start $name: $*"
    "$@" > "$LOG/$name.log" 2>&1
    local code=$?
    status "end $name: exit $code"
    return $code
}

status "queue started (corpus $VESUVIUS_CORPUS, cue_dropout epochs $VESUVIUS_CUE_EPOCHS)"
if ! step 00_check python3 -m utils.mix6; then
    status "ABORT: the corpus check failed (see $LOG/00_check.log)"
    exit 1
fi

step 01_reader_v3 python3 finetune_readerv3.py --corpus orig7
step 02_cue_dropout python3 train.py
step 03_v8in python3 finetune_v8in.py --corpus orig7

READER=reader-v3/runs/orig7_3000/ckpt_003000.pth
CUE_DIR=$(ls -dt models/runs/cue_dropout_labelsv6_orig7* 2>/dev/null | head -1)  # a rerun lands in _2, _3 ...
CUE=$CUE_DIR/final.pth
V8IN=nader-fork/runs/v8in_orig7_3ep/model
missing=""
[ -f "$READER" ] || missing="$missing $READER"
[ -n "$CUE_DIR" ] && [ -f "$CUE" ] || missing="$missing models/runs/cue_dropout_labelsv6_orig7*/final.pth"
[ -f "$V8IN/model.safetensors" ] || missing="$missing $V8IN"
if [ -n "$missing" ]; then
    status "ABORT: the trainings left no result at:$missing"
    exit 1
fi

ENSEMBLE_SUBDIR=orig7 \
ENSEMBLE_READER=/vesuvius/$READER ENSEMBLE_READER_LABEL="step 3000" ENSEMBLE_READER_CORPUS=orig7 \
ENSEMBLE_CUE=/vesuvius/$CUE ENSEMBLE_CUE_LABEL="epoch 3" ENSEMBLE_CUE_CORPUS=orig7 \
ENSEMBLE_V8IN=/vesuvius/$V8IN ENSEMBLE_V8IN_LABEL="epoch 3" ENSEMBLE_V8IN_CORPUS=orig7 \
    step 04_ensemble python3 utils/run_notebook.py ensemble_eval.ipynb

VESUVIUS_CHECKPOINT=/vesuvius/$CUE \
    step 05_0846A_eval_cue_dropout python3 utils/run_notebook.py 0846A_eval.ipynb --keep-going --skip '^quick_patch_outputs ='
status "queue finished"
