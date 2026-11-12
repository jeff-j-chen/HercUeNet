#!/usr/bin/env bash
# the mix28 evaluation (called at the end of utils/queue_mix28.sh, or alone once the trainings have run); logs in logs_mix28/.
#   1-3. ensemble_eval.ipynb, three 3x2 ensembles on the prize sheet (reader-v3 step / HercUeNet epoch / v8-in epoch):
#        4.1  4000 / 4 / 4  -> output/ensemble/mix28_e4/<prize sheet>/
#        4.2  3000 / 3 / 3  -> output/ensemble/mix28_e3/<prize sheet>/
#        4.3  2000 / 2 / 2  -> output/ensemble/mix28_e2/<prize sheet>/
#   4.   0846A_eval.ipynb with the HercUeNet epoch 4: the prize sheet, the 14 trained 0846A sheets, the 14 donor patches
#        (3 new in mix28), the held-out patches (out_4 / out_5 and the 3 quick patches) -> output/0846A/cue_dropout_labelsv6_mix28_*_epoch_4/
# v8-in epoch N is the Lightning checkpoint trained-epoch<N-1>.ckpt (0-based), exported on demand.
set -u
cd /vesuvius
LOG=logs_mix28
mkdir -p "$LOG"
status() { echo "[eval $(date '+%F %T')] $*" | tee -a "$LOG/queue.log"; }

step() {  # step <log name> <command...>
    local name=$1; shift
    status "start $name: $*"
    "$@" > "$LOG/$name.log" 2>&1
    local code=$?
    status "end $name: exit $code"
    return $code
}

CUE_DIR=models/runs/cue_dropout_labelsv6_mix28
READER_DIR=reader-v3/runs/mix28
V8IN_RUN=nader-fork/runs/v8in_mix28_4ep

ensemble() {  # ensemble <n> <reader step> : cue epoch n, v8-in epoch n
    local n=$1 step=$2
    local cue="$CUE_DIR/epoch_$n.pth"
    local reader="$READER_DIR/ckpt_$(printf '%06d' "$step").pth"
    local v8in
    v8in=$(ls -1 "$V8IN_RUN"/runs/*/checkpoints/trained-epoch$((n - 1)).ckpt 2>/dev/null | tail -1)
    local missing=""
    [ -f "$cue" ] || missing="$missing $cue"
    [ -f "$reader" ] || missing="$missing $reader"
    [ -n "$v8in" ] || missing="$missing $V8IN_RUN/runs/*/checkpoints/trained-epoch$((n - 1)).ckpt"
    if [ -n "$missing" ]; then
        status "SKIP ensemble epoch $n: missing$missing"
        return 1
    fi
    ENSEMBLE_CUE="$cue" ENSEMBLE_CUE_LABEL="epoch $n" ENSEMBLE_CUE_CORPUS=mix28 \
    ENSEMBLE_READER="$reader" ENSEMBLE_READER_LABEL="step $step" ENSEMBLE_READER_CORPUS=mix28 \
    ENSEMBLE_V8IN="$v8in" ENSEMBLE_V8IN_LABEL="epoch $n" ENSEMBLE_V8IN_CORPUS=mix28 \
    ENSEMBLE_SUBDIR="mix28_e$n" \
        step "05_ensemble_e$n" python3 utils/run_notebook.py ensemble_eval.ipynb
}

status "eval started"
ensemble 4 4000
ensemble 3 3000
ensemble 2 2000

if [ -f "$CUE_DIR/epoch_4.pth" ]; then
    VESUVIUS_CHECKPOINT="$CUE_DIR/epoch_4.pth" \
        step 06_0846A_eval_cue python3 utils/run_notebook.py 0846A_eval.ipynb --keep-going --skip '^quick_patch_outputs ='
else
    status "SKIP 0846A_eval: missing $CUE_DIR/epoch_4.pth"
fi
status "eval finished"
