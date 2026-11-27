#!/bin/bash
# The mix28x queue against the prize sheet PRIZESHEET_11_19_flatboi (tag _ps7): the three trainings with the DEFAULTS of their scripts, one step at a time on the GPU,
# nothing else heavy next to it (the container has 62 GiB of RAM). Each script keeps the learning-rate schedule of the full run and stops itself at the checkpoint
# the ensemble uses (utils/mix6.py: Studio step 500 of 2000, reader-v3 step 1000 of 4000, v8-in epoch 1 of 4); HercUeNet is retired.
#   1. python3 finetune_studio.py    released Studio -> models/studio/mix28x_ps7/step_000500.safetensors   (setup, stage, train; jitter 40, soft nnPU, background 1)
#   2. python3 finetune_readerv3.py  from reader-v3/runs/mix28_ps6_s1/ckpt_002000.pth -> reader-v3/runs/mix28x_ps7_s1/ckpt_001000.pth
#   3. python3 finetune_v8in.py      from the mix28 trained-epoch2 -> nader-fork/runs/v8in_mix28x_4ep_ps7_s1/runs/*/checkpoints/trained-epoch0.ckpt (background weight 10)
# corpus mix28x: hand-grown 0846A/B sheets 4 : HerculeX x sheets 2 : each donor scroll 1 (0846A/B are 43% of the draws; v8-in: groups 8 + 8 : 121 : 1 each).
# After each training every saved checkpoint is run on the prize sheet (python3 -m utils.viz_checkpoints studio|reader|v8in): maps are cached in
# output/prize_PRIZESHEET_11_19_flatboi/cache/<model>/ and the images are output/prize_PRIZESHEET_11_19_flatboi/{studio,reader,v8in}_checkpoints.png. The ensemble
# figures are made by ensemble_eval.ipynb (python3 utils/run_notebook.py ensemble_eval.ipynb: 3x2 and 2x2 figures).
# A step that fails is retried once (a v8-in run resumes from its last checkpoint); if it fails again it is written to logs_herculex/queue_failures.log and the
# queue goes on with the next model. Progress: logs_herculex/queue.log.
#   tmux new -d -s queue_mix28x 'bash /vesuvius/utils/queue_mix28x.sh'
export PATH=/root/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /vesuvius
L=logs_herculex
mkdir -p $L
rm -f $L/queue_failures.log $L/queue_done
log() { echo "$(date +%F_%T) $*" | tee -a $L/queue.log; }
fail() { log "FAILED: $1"; echo "$1" >> $L/queue_failures.log; }
clean_shm() { rm -rf /dev/shm/ves_zarr_cache; }
archive() {  # archive <dir>: keep an earlier run of the same name out of the way
    [ -e "$1" ] && mv "$1" "${1}_before_$(date +%m%d_%H%M)"
    return 0
}
viz() {  # viz <studio|reader|v8in>: every saved checkpoint on the prize sheet, retried once
    for attempt in 1 2; do python3 -m utils.viz_checkpoints $1 > $L/viz_$1_$attempt.log 2>&1 && break; clean_shm; done || fail "$1 visualization"
    clean_shm
}

log "queue start"
python3 -m utils.mix6 > $L/mix6_check.log 2>&1
clean_shm

# ---------------------------------------------------------------- 1. Studio (setup -> stage -> train, stops itself at step 500)
SD=models/studio/mix28x_ps7
studio_train() { archive $SD; python3 finetune_studio.py > $L/studio_train_$1.log 2>&1 && [ -f $SD/step_000500.safetensors ]; }
log "studio: setup, staging, training"
[ -f $SD/step_000500.safetensors ] || studio_train 1 || { log "studio attempt 1 failed; retrying"; clean_shm; studio_train 2; } || fail "studio training"
log "studio: visualizing every checkpoint"
viz studio

# ---------------------------------------------------------------- 2. reader-v3 (prep -> train, stops itself at step 1000)
RD=reader-v3/runs/mix28x_ps7_s1
reader_train() { archive $RD; python3 finetune_readerv3.py > $L/reader_train_$1.log 2>&1 && [ -f $RD/ckpt_001000.pth ]; }
log "reader-v3: prep + training"
[ -f $RD/ckpt_001000.pth ] || reader_train 1 || { log "reader-v3 attempt 1 failed; retrying"; clean_shm; reader_train 2; } || fail "reader-v3 training"
log "reader-v3: visualizing every checkpoint"
viz reader

# ---------------------------------------------------------------- 3. v8-in (stage -> train, stops itself after epoch 1)
VD=nader-fork/runs/v8in_mix28x_4ep_ps7_s1
v8_done() { ls $VD/runs/*/checkpoints/trained-epoch0.ckpt > /dev/null 2>&1; }
v8_train() {  # v8_train <attempt>: attempt 2 resumes from the last checkpoint of attempt 1 when there is one
    local resume=""
    if [ "$1" = 2 ]; then
        local last=$(readlink -f $(ls -t $VD/runs/*/checkpoints/last.ckpt 2>/dev/null | head -1) 2>/dev/null)
        [ -n "$last" ] && resume="--resume $last"
        [ -z "$resume" ] && archive $VD
    fi
    python3 finetune_v8in.py $resume > $L/v8in_train_$1.log 2>&1
    v8_done
}
log "v8-in: staging + training"
v8_done || v8_train 1 || { log "v8-in attempt 1 failed; retrying"; clean_shm; v8_train 2; } || fail "v8-in training"
log "v8-in: visualizing every checkpoint"
viz v8in

log "queue finished" ; echo done > $L/queue_done
[ -f $L/queue_failures.log ] && log "failures: $(tr '\n' ';' < $L/queue_failures.log)"
