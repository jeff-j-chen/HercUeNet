#!/usr/bin/env bash
# The mix28 trainings against the current prize sheet (utils/mix6.py RUN_TAG _ps3), then three ensembles and the training-fragment figures.
# Strictly sequential (one GPU job at a time), logs in logs_ps3/. Waits for the cue build (tmux prep_ps3: python3 -m utils.mix6 --build).
#   0. check       python3 -m utils.mix6              read-only corpus check; cue_dropout only trains when it passes
#   1. cue_dropout python3 train.py                   4 epochs -> models/runs/cue_dropout_labelsv6_mix28_ps3/ (epoch_1..4.pth)
#   2. reader-v3   python3 finetune_readerv3.py       4000-step schedule, stopped once ckpt_002000.pth is written -> reader-v3/runs/mix28_ps3/ (ckpt_001000.pth, ckpt_002000.pth)
#   3. v8-in       python3 finetune_v8in.py           4-epoch cosine schedule, stopped once trained-epoch1.ckpt is written -> nader-fork/runs/v8in_mix28_4ep_ps3/
#                                                     (trained-epoch0.ckpt = epoch 1, trained-epoch1.ckpt = epoch 2)
#   4. ensembles   ensemble_eval.ipynb on the prize sheet (3x2 figure + product_ct_2x2.png each) -> output/ensemble/<subdir>/<prize sheet>/
#        1  cue epoch 2, reader step 1000, v8-in epoch 1  mix28_ps3_c2_r1000_v1
#        2  cue epoch 4, reader step 2000, v8-in epoch 2  mix28_ps3_c4_r2000_v2
#        3  cue epoch 4, reader step 2000, v8-in epoch 1  mix28_ps3_c4_r2000_v1
#   5. fragments   ensemble 3 on every sheet cue_dropout trained on (ENSEMBLE_SHEET=train0846a), then utils/label_overlay_figures.py:
#                  ensemble | v8-in epoch 1, and both with the red inklabel overlay -> output/label_figures/<sheet>.png
# Stopping a schedule early keeps exactly the weights the full run has at that point (the lr schedule does not look ahead).
# A failed step does not stop the next one; an ensemble is skipped when one of its checkpoints is missing.
#   tmux new -d -s queue_ps3 'bash /vesuvius/utils/queue_ps3.sh'
set -u
cd /vesuvius
LOG=logs_ps3
mkdir -p "$LOG"
CUE_DIR=/vesuvius/models/runs/cue_dropout_labelsv6_mix28_ps3
READER_DIR=/vesuvius/reader-v3/runs/mix28_ps3
V8IN_RUN=/vesuvius/nader-fork/runs/v8in_mix28_4ep_ps3
status() { echo "[queue_ps3 $(date '+%F %T')] $*" | tee -a "$LOG/queue.log"; }

step() {  # step <log name> <command...>
    local name=$1; shift
    status "start $name: $*"
    "$@" > "$LOG/$name.log" 2>&1
    local code=$?
    status "end $name: exit $code"
    return $code
}

until_checkpoint() {  # until_checkpoint <log name> <file glob> <command...>: stop the training once the file stopped growing
    local name=$1 pattern=$2; shift 2
    status "start $name (stops once $pattern is written): $*"
    setsid "$@" > "$LOG/$name.log" 2>&1 &
    local pid=$! previous="" found=""
    while kill -0 "$pid" 2>/dev/null; do
        sleep 30
        local file size
        file=$(ls -1 $pattern 2>/dev/null | tail -1)
        [ -n "$file" ] || continue
        size=$(stat -c %s "$file")
        if [ "$size" = "$previous" ]; then found=$file; break; fi
        previous=$size
    done
    if [ -n "$found" ]; then
        kill -TERM -- "-$pid" 2>/dev/null
        for _ in $(seq 12); do kill -0 "$pid" 2>/dev/null || break; sleep 5; done
        kill -KILL -- "-$pid" 2>/dev/null
        status "end $name: stopped at $found"
        return 0
    fi
    wait "$pid"; local code=$?
    status "end $name: exit $code before $pattern was written"
    return 1
}

v8in_checkpoint() {  # v8in_checkpoint <epoch>: the Lightning checkpoint after that epoch (0-based file name)
    ls -1 "$V8IN_RUN"/runs/*/checkpoints/trained-epoch$(($1 - 1)).ckpt 2>/dev/null | tail -1
}

ensemble() {  # ensemble <subdir> <cue epoch> <reader step> <v8-in epoch> <log name> [VAR=value ...]
    local subdir=$1 cue_epoch=$2 reader_step=$3 v8in_epoch=$4 log=$5; shift 5
    local cue="$CUE_DIR/epoch_$cue_epoch.pth"
    local reader="$READER_DIR/ckpt_$(printf '%06d' "$reader_step").pth"
    local v8in
    v8in=$(v8in_checkpoint "$v8in_epoch")
    local missing=""
    [ -f "$cue" ] || missing="$missing $cue"
    [ -f "$reader" ] || missing="$missing $reader"
    [ -n "$v8in" ] || missing="$missing v8-in epoch $v8in_epoch"
    if [ -n "$missing" ]; then
        status "SKIP $log: missing$missing"
        return 1
    fi
    status "start $log: cue epoch $cue_epoch, reader step $reader_step, v8-in epoch $v8in_epoch -> output/ensemble/$subdir"
    env "$@" \
        ENSEMBLE_CUE="$cue" ENSEMBLE_CUE_LABEL="epoch $cue_epoch" ENSEMBLE_CUE_CORPUS=mix28 \
        ENSEMBLE_READER="$reader" ENSEMBLE_READER_LABEL="step $reader_step" ENSEMBLE_READER_CORPUS=mix28 \
        ENSEMBLE_V8IN="$v8in" ENSEMBLE_V8IN_LABEL="epoch $v8in_epoch" ENSEMBLE_V8IN_CORPUS=mix28 \
        ENSEMBLE_SUBDIR="$subdir" \
        python3 utils/run_notebook.py ensemble_eval.ipynb > "$LOG/$log.log" 2>&1
    local code=$?
    status "end $log: exit $code"
    return $code
}

status "queue started"
while tmux has-session -t prep_ps3 2>/dev/null; do sleep 60; done
status "cue build finished"
if step 00_check python3 -m utils.mix6; then
    step 01_cue_dropout python3 train.py
else
    status "SKIP cue_dropout: the corpus check failed (see $LOG/00_check.log)"
fi
until_checkpoint 02_reader_v3 "$READER_DIR/ckpt_002000.pth" python3 finetune_readerv3.py
until_checkpoint 03_v8in "$V8IN_RUN/runs/*/checkpoints/trained-epoch1.ckpt" python3 finetune_v8in.py

ensemble mix28_ps3_c2_r1000_v1 2 1000 1 04_ensemble_1_c2_r1000_v1
ensemble mix28_ps3_c4_r2000_v2 4 2000 2 05_ensemble_2_c4_r2000_v2
ensemble mix28_ps3_c4_r2000_v1 4 2000 1 06_ensemble_3_c4_r2000_v1

# ensemble 3 on the training sheets, then the 2x2 label figures from its outputs
if ensemble mix28_ps3_c4_r2000_v1 4 2000 1 07_train_fragments_ensemble_3 ENSEMBLE_SHEET=train0846a; then
    step 08_label_figures python3 -m utils.label_overlay_figures
else
    status "SKIP 08_label_figures: the ensemble on the training sheets failed"
fi
status "queue finished"
