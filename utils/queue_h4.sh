#!/usr/bin/env bash
# "mix28_h4_r2000_v1" retrained against the prize sheet PRIZESHEET_NEW (utils/mix6.py RUN_TAG _ps2), strictly sequential (one GPU job at a time),
# logs in logs_h4_ps2/. Needs `python3 assemble_training_segments.py --force --only PRIZESHEET_NEW` and `python3 -m utils.mix6 --build` first
# (the sheet, the exclusion masks of the trained 0846A sheets and the cues built without the new zone).
#   0. check       python3 -m utils.mix6              read-only corpus check once the cue build has ended; cue_dropout only trains when it passes (reader-v3 and v8-in start at once, they need no cues)
#   1. reader-v3   python3 finetune_readerv3.py       4000-step schedule, stopped once ckpt_002000.pth is written (the model of the ensemble) -> reader-v3/runs/mix28_ps2/
#   2. v8-in       python3 finetune_v8in.py           4-epoch cosine schedule, stopped once trained-epoch0.ckpt (epoch 1) is written -> nader-fork/runs/v8in_mix28_4ep_ps2/
#   3. cue_dropout python3 train.py                   4 epochs -> models/runs/cue_dropout_labelsv6_mix28_ps2/ (epoch_4.pth = final.pth)
#   4. ensemble    ensemble_eval.ipynb                the 3x2 and 2x2 figures of the three models above on PRIZESHEET_NEW -> output/ensemble/mix28_h4_r2000_v1_ps2/
#   5. fragments   utils/render_trained_fragments.py  cue_dropout epoch 4 on every fragment it trained on -> output/h4_ps2/trained_fragments/
# Stopping a schedule early keeps exactly the weights the full run has at that point (the lr schedule does not look ahead). A failed step does not stop the next one;
# the ensemble and the fragments are skipped when their checkpoint is missing.
#   tmux new -d -s h4_ps2 'bash /vesuvius/utils/queue_h4.sh'
set -u
cd /vesuvius
LOG=logs_h4_ps2
mkdir -p "$LOG"
READER_CKPT=reader-v3/runs/mix28_ps2/ckpt_002000.pth
V8IN_GLOB='nader-fork/runs/v8in_mix28_4ep_ps2/runs/*/checkpoints/trained-epoch0.ckpt'
CUE_CKPT=models/runs/cue_dropout_labelsv6_mix28_ps2/epoch_4.pth
status() { echo "[queue_h4 $(date '+%F %T')] $*" | tee -a "$LOG/queue.log"; }

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

status "queue started"
until_checkpoint 01_reader_v3 "$READER_CKPT" python3 finetune_readerv3.py
until_checkpoint 02_v8in "$V8IN_GLOB" python3 finetune_v8in.py

# cue_dropout needs every asf cue map: wait for the cue build (tmux prep_ps2: python3 -m utils.mix6 --build), then check the whole corpus
while tmux has-session -t prep_ps2 2>/dev/null; do sleep 60; done
if step 00_check python3 -m utils.mix6; then
    step 03_cue_dropout python3 train.py
else
    status "SKIP cue_dropout: the corpus check failed (see $LOG/00_check.log)"
fi

if [ -f "$READER_CKPT" ] && [ -f "$CUE_CKPT" ] && ls $V8IN_GLOB >/dev/null 2>&1; then
    step 04_ensemble python3 utils/run_notebook.py ensemble_eval.ipynb
else
    status "SKIP ensemble: a checkpoint is missing ($READER_CKPT, $V8IN_GLOB, $CUE_CKPT)"
fi
if [ -f "$CUE_CKPT" ]; then
    VESUVIUS_CHECKPOINT="/vesuvius/$CUE_CKPT" VESUVIUS_OUT_DIR=/vesuvius/output/h4_ps2/trained_fragments \
        step 05_trained_fragments python3 -u utils/render_trained_fragments.py
else
    status "SKIP trained fragments: missing $CUE_CKPT"
fi
status "queue finished"
