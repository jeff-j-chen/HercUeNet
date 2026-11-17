#!/usr/bin/env bash
# The _ps5 test trainings on the 0846A-only corpus: the 14 PHerc0846A sheets + PHerc0009B patch 487 (utils/mix6.py CORPORA "0846a_487",
# VESUVIUS_CORPUS=0846a_487), same prize sheet, exclusions, seeds and schedules as utils/queue_ps5.sh: HercUeNet 6 epochs, reader-v3 6000
# steps, v8-in 6 epochs, a checkpoint after every epoch / 1000 steps, each model twice (s1, s2). Then every checkpoint on the test sheet:
# output/ps5_0846a_487/prize/{hercuenet,reader_v3,v8in}.png. Logs in logs_ps5_487/. Run after queue_ps5.sh (utils/queue_ps5_all.sh does).
#   outputs: models/runs/cue_dropout_labelsv6_0846a_487_ps5_<seed>/  reader-v3/runs/0846a_487_ps5_<seed>/  nader-fork/runs/v8in_0846a_487_6ep_ps5_<seed>/
set -u
cd /vesuvius
export VESUVIUS_PS5=1 VESUVIUS_CORPUS=0846a_487
LOG=${LOG:-logs_ps5_487}
mkdir -p "$LOG"
status() { echo "[queue_ps5_487 $(date '+%F %T')] $*" | tee -a "$LOG/queue.log"; }

step() {  # step <log name> <command...>
    local name=$1; shift
    status "start $name: $*"
    "$@" > "$LOG/$name.log" 2>&1
    local code=$?
    status "end $name: exit $code"
    return $code
}

train() {  # train <log name> <final checkpoint glob> <command...>: skipped when the final checkpoint exists
    local name=$1 pattern=$2; shift 2
    if ls $pattern > /dev/null 2>&1; then
        status "skip $name: $(ls -1 $pattern | tail -1) exists"
        return 0
    fi
    rm -rf /dev/shm/ves_zarr_cache/* 2>/dev/null
    step "$name" "$@"
    status "oom kills so far: $(grep -a oom_kill /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null | cut -d' ' -f2)"
}

status "queue started"
if ! step 01_check python3 -m utils.mix6; then
    status "ABORT: the corpus check failed (see $LOG/01_check.log)"
    exit 1
fi

for seed in s1 s2; do
    export VESUVIUS_SEED_TAG=$seed
    train "${seed}_02_cue_dropout" "models/runs/cue_dropout_labelsv6_0846a_487_ps5_$seed/epoch_6.pth" python3 train.py
    train "${seed}_03_reader_v3" "reader-v3/runs/0846a_487_ps5_$seed/ckpt_006000.pth" python3 finetune_readerv3.py
    V8IN_DIR="nader-fork/runs/v8in_0846a_487_6ep_ps5_$seed"
    FINAL="$V8IN_DIR/runs/*/checkpoints/trained-epoch5.ckpt"
    RESUME=$(ls -1 "$V8IN_DIR"/runs/*/checkpoints/epochepoch=*.ckpt 2>/dev/null | sort -t= -k2 -n | tail -1)
    if [ -n "$RESUME" ] && ! ls $FINAL > /dev/null 2>&1; then
        status "v8-in $seed: resuming from $RESUME"
        train "${seed}_04_v8in" "$FINAL" python3 finetune_v8in.py train --resume "$PWD/$RESUME"
    else
        train "${seed}_04_v8in" "$FINAL" python3 finetune_v8in.py
    fi
done
unset VESUVIUS_SEED_TAG

step 05_prize python3 -u utils/ps5_eval.py prize
status "queue finished"
