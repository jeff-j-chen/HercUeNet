#!/usr/bin/env bash
# the mix28 training queue, strictly sequential (one GPU job at a time); logs in logs_mix28/. All three models train on the same corpus.
# The assembly (assemble_training_segments.py --mix28) and the prize-sheet exclusions (utils/holdout_exclusion.py) must be done.
#   0. check       python3 -m utils.mix6          read-only corpus check (cue_dropout only trains when it passes)
#   1. cue_dropout python3 train.py               4 epochs, checkpoint every epoch -> models/runs/cue_dropout_labelsv6_mix28/epoch_{1..4}.pth, final.pth
#   2. reader-v3   python3 finetune_readerv3.py   4000 steps, checkpoint every 1000 -> reader-v3/runs/mix28/ckpt_00{1..4}000.pth
#   3. v8-in       python3 finetune_v8in.py       4 epochs, checkpoint every epoch -> nader-fork/runs/v8in_mix28_4ep/{model,runs/*/checkpoints/trained-epoch{0..3}.ckpt}
#   4. utils/queue_mix28_eval.sh                  three ensembles (step 4000 / epoch 4 / epoch 4, then 3000 / 3 / 3, then 2000 / 2 / 2)
#                                                 and 0846A_eval.ipynb with the cue_dropout epoch 4
# A failed step does not stop the next one; the eval script skips what lacks a checkpoint.
#   tmux new -d -s mix28 'bash /vesuvius/utils/queue_mix28.sh'
set -u
cd /vesuvius
LOG=logs_mix28
mkdir -p "$LOG"
status() { echo "[queue $(date '+%F %T')] $*" | tee -a "$LOG/queue.log"; }

step() {  # step <log name> <command...>
    local name=$1; shift
    status "start $name: $*"
    "$@" > "$LOG/$name.log" 2>&1
    local code=$?
    status "end $name: exit $code"
    return $code
}

status "queue started"
if step 00_check python3 -m utils.mix6; then
    step 01_cue_dropout python3 train.py
else
    status "SKIP cue_dropout: the corpus check failed (see $LOG/00_check.log)"
fi
step 02_reader_v3 python3 finetune_readerv3.py
step 03_v8in python3 finetune_v8in.py
bash utils/queue_mix28_eval.sh
status "queue finished"
