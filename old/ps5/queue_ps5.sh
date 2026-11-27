#!/usr/bin/env bash
# The _ps5 test trainings (utils/mix6.py PS5 = VESUVIUS_PS5=1): exclusions and trainings against the test prize sheet
# PRIZESHEET_11_11_flatboi_flatboi_abf_copy_copy_abf_copy_abf_flatboi_flatboi (not canonical), every model twice (seeds s1 / s2),
# slightly longer schedules, a checkpoint after every epoch / 1000 steps. Strictly sequential, logs in logs_ps5/.
#   0. build     python3 -m utils.mix6 --build   prize-sheet exclusions (overwrites holdout_exclusion*/), asf cues rebuilt where the zone changed
#   1. check     python3 -m utils.mix6           read-only corpus check
#   per seed:    HercUeNet 6 epochs (epoch_1..6.pth) -> models/runs/cue_dropout_labelsv6_mix28_ps5_<seed>/
#                reader-v3 6000 steps (ckpt_001000..006000.pth) -> reader-v3/runs/mix28_ps5_<seed>/
#                v8-in 6 epochs (trained-epoch0..5.ckpt) -> nader-fork/runs/v8in_mix28_6ep_ps5_<seed>/ (a run with epoch checkpoints is resumed)
#   2. prize     python3 utils/ps5_eval.py prize       every checkpoint on the test sheet -> output/ps5/prize/{hercuenet,reader_v3,v8in}.png
#   3. fragments python3 utils/ps5_eval.py fragments   v8-in epoch 3 + reader-v3 step 2000, both seeds, on every training sheet -> output/ps5/fragments/
# A failed step does not stop the next one; a step that needs a missing checkpoint says so.
#   tmux new -d -s queue_ps5 'bash /vesuvius/utils/queue_ps5.sh'
set -u
cd /vesuvius
export VESUVIUS_PS5=1
LOG=${LOG:-logs_ps5}
mkdir -p "$LOG"
status() { echo "[queue_ps5 $(date '+%F %T')] $*" | tee -a "$LOG/queue.log"; }

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
    # the inference copies of zarrs in /dev/shm count against the 62 GB container memory (oom_kill ended three trainings)
    rm -rf /dev/shm/ves_zarr_cache/* 2>/dev/null
    step "$name" "$@"
    status "oom kills so far: $(grep -a oom_kill /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null | cut -d' ' -f2)"
}

status "queue started"
step 00_build python3 -m utils.mix6 --build
if ! step 01_check python3 -m utils.mix6; then
    status "ABORT: the corpus check failed (see $LOG/01_check.log)"
    exit 1
fi

for seed in s1 s2; do
    export VESUVIUS_SEED_TAG=$seed
    train "${seed}_02_cue_dropout" "models/runs/cue_dropout_labelsv6_mix28_ps5_$seed/epoch_6.pth" python3 train.py
    train "${seed}_03_reader_v3" "reader-v3/runs/mix28_ps5_$seed/ckpt_006000.pth" python3 finetune_readerv3.py
    V8IN_DIR="nader-fork/runs/v8in_mix28_6ep_ps5_$seed"
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
step 06_fragments python3 -u utils/ps5_eval.py fragments
status "queue finished"
