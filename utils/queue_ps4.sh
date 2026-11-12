#!/usr/bin/env bash
# The mix28 trainings against the current prize sheet (utils/mix6.py RUN_TAG _ps4), each made twice with a different fixed seed
# (utils/mix6.py SEEDS: s1, s2), then the seed-averaged ensemble and the training-fragment figures.
# Strictly sequential (one GPU job at a time), logs in logs_ps4/. Waits for a running cue build (tmux prep_ps4: python3 -m utils.mix6 --build).
#   0. build       python3 -m utils.mix6 --build     missing derived artifacts (prize-sheet exclusions, anchors, 3 depths of asf cues)
#   1. check       python3 -m utils.mix6             read-only corpus check; nothing trains when it fails
#   per seed (VESUVIUS_SEED_TAG=s1, then s2); every training runs its FULL schedule (the learning rate progresses as scheduled); the
#   checkpoints the ensemble uses (utils/mix6.py default_models) are early ones, saved under their normal names on the way:
#   2. HercUeNet   python3 train.py               4-epoch schedule; epoch_3.pth is used
#                                                  -> models/runs/cue_dropout_labelsv6_mix28_ps4_<seed>/
#   3. reader-v3   python3 finetune_readerv3.py   4000-step schedule; ckpt_001000.pth is used
#                                                  -> reader-v3/runs/mix28_ps4_<seed>/   (background weight 0.8)
#   4. v8-in       python3 finetune_v8in.py       4-epoch cosine schedule; trained-epoch2.ckpt (epoch 3) is used
#                                                  -> nader-fork/runs/v8in_mix28_4ep_ps4_<seed>/   (NEG_WEIGHT 20)
#      A training whose used checkpoint exists already is skipped (the early checkpoints of a run stopped there are the full run's own
#      weights at that point: the schedule does not look ahead); a v8-in run with epoch checkpoints but without epoch 3 is resumed.
#      EARLY_STOP=1 stops each training once its used checkpoint is written instead of finishing the schedule.
#   5. ensemble    python3 utils/seed_ensemble.py   HercUeNet epoch 3 and reader-v3 step 1000 as the max over depths -2..0, v8-in epoch 3 as
#                                                  the max over -2..-1, each averaged over both seeds
#                                                  -> output/ensemble/mix28_ps4_c3_r1000_v3_seedavg/<prize sheet>/
#   6. fragments   seed s1 only (depth 0): ensemble_eval.ipynb on every sheet HercUeNet trained on (ENSEMBLE_SHEET=train0846a) ->
#                  output/ensemble/mix28_ps4_s1_c3_r1000_v3/, then utils/label_overlay_figures.py: 2x2 figures (z-score composite | v8-in,
#                  both with the red inklabel overlay underneath) -> output/label_figures_ps4/<sheet>.png
# A failed step does not stop the next one; a later step that needs a missing checkpoint is skipped and says so.
#   tmux new -d -s queue_ps4 'bash /vesuvius/utils/queue_ps4.sh'
set -u
cd /vesuvius
LOG=${LOG:-logs_ps4}
mkdir -p "$LOG"
status() { echo "[queue_ps4 $(date '+%F %T')] $*" | tee -a "$LOG/queue.log"; }

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
        # nothing of the stopped run may keep the GPU for the next one
        sleep 5
        status "end $name: stopped at $found"
        return 0
    fi
    wait "$pid"; local code=$?
    status "end $name: exit $code before $pattern was written"
    return 1
}

EARLY_STOP=${EARLY_STOP:-0}

train() {  # train <log name> <used checkpoint glob> <command...>: skip when the used checkpoint exists, else run (stop early with EARLY_STOP=1)
    local name=$1 pattern=$2; shift 2
    if ls $pattern > /dev/null 2>&1; then
        status "skip $name: $(ls -1 $pattern | tail -1) exists"
        return 0
    fi
    if [ "$EARLY_STOP" = 1 ]; then
        until_checkpoint "$name" "$pattern" "$@"
    else
        step "$name" "$@"
    fi
}

status "queue started"
while tmux has-session -t prep_ps4 2>/dev/null; do sleep 60; done
status "cue build finished"
step 00_build python3 -m utils.mix6 --build
if ! step 01_check python3 -m utils.mix6; then
    status "ABORT: the corpus check failed (see $LOG/01_check.log)"
    exit 1
fi

for seed in s1 s2; do
    export VESUVIUS_SEED_TAG=$seed
    train "${seed}_02_cue_dropout" "models/runs/cue_dropout_labelsv6_mix28_ps4_$seed/epoch_3.pth" python3 train.py
    train "${seed}_03_reader_v3" "reader-v3/runs/mix28_ps4_$seed/ckpt_001000.pth" python3 finetune_readerv3.py
    V8IN_DIR="nader-fork/runs/v8in_mix28_4ep_ps4_$seed"
    RESUME=$(ls -1 "$V8IN_DIR"/runs/*/checkpoints/epochepoch=*.ckpt 2>/dev/null | sort -t= -k2 -n | tail -1)
    if [ -n "$RESUME" ] && ! ls "$V8IN_DIR"/runs/*/checkpoints/trained-epoch2.ckpt > /dev/null 2>&1; then
        status "v8-in $seed: resuming from $RESUME"
        train "${seed}_04_v8in" "$V8IN_DIR/runs/*/checkpoints/trained-epoch2.ckpt" python3 finetune_v8in.py train --resume "$PWD/$RESUME"
    else
        train "${seed}_04_v8in" "$V8IN_DIR/runs/*/checkpoints/trained-epoch2.ckpt" python3 finetune_v8in.py
    fi
done
unset VESUVIUS_SEED_TAG

step 05_seed_ensemble python3 utils/seed_ensemble.py

# seed s1 on every training sheet (depth 0), then the 2x2 label figures from its outputs
V8IN_S1=$(ls -1 nader-fork/runs/v8in_mix28_4ep_ps4_s1/runs/*/checkpoints/trained-epoch2.ckpt 2>/dev/null | tail -1)
if [ -f models/runs/cue_dropout_labelsv6_mix28_ps4_s1/epoch_3.pth ] && [ -f reader-v3/runs/mix28_ps4_s1/ckpt_001000.pth ] && [ -n "$V8IN_S1" ]; then
    status "start 06_train_fragments (seed s1)"
    ENSEMBLE_SHEET=train0846a ENSEMBLE_SUBDIR=mix28_ps4_s1_c3_r1000_v3 ENSEMBLE_CUE_DEPTHS=0 ENSEMBLE_READER_DEPTHS=0 ENSEMBLE_V8IN_DEPTHS=0 \
        ENSEMBLE_CUE="$PWD/models/runs/cue_dropout_labelsv6_mix28_ps4_s1/epoch_3.pth" ENSEMBLE_CUE_LABEL="epoch 3" ENSEMBLE_CUE_CORPUS=mix28 \
        ENSEMBLE_READER="$PWD/reader-v3/runs/mix28_ps4_s1/ckpt_001000.pth" ENSEMBLE_READER_LABEL="step 1000" ENSEMBLE_READER_CORPUS=mix28 \
        ENSEMBLE_V8IN="$PWD/$V8IN_S1" ENSEMBLE_V8IN_LABEL="epoch 3" ENSEMBLE_V8IN_CORPUS=mix28 \
        python3 utils/run_notebook.py ensemble_eval.ipynb > "$LOG/06_train_fragments.log" 2>&1
    code=$?
    status "end 06_train_fragments: exit $code"
    if [ "$code" = 0 ]; then
        ENSEMBLE_SUBDIR=mix28_ps4_s1_c3_r1000_v3 step 07_label_figures python3 -m utils.label_overlay_figures
    else
        status "SKIP 07_label_figures: the ensemble on the training sheets failed"
    fi
else
    status "SKIP 06/07: a seed-s1 checkpoint is missing"
fi
status "queue finished"
