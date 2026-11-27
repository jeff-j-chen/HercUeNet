#!/bin/bash
# The training queue of the corpus mix28x_new (the regenerated x export), every model from its RELEASED checkpoint with the defaults of its script, one step at a time on the GPU:
#   1. reader-v3  4000-step schedule stopped at step 1000   -> reader-v3/runs/mix28x_new_ps7_s1/ckpt_001000.pth      -> maps on the evaluation sheets (cached)
#   2. Studio     2000-step schedule stopped at step 500    -> models/studio/mix28x_new_ps7/step_000500.safetensors  -> maps on the evaluation sheets (cached)
#   3. v8-in      4-epoch schedule stopped after epoch 2   -> nader-fork/runs/v8in_mix28x_new_4ep_ps7_s1/runs/*/checkpoints/trained-epoch{0,1}.ckpt; each epoch is visualized as soon as it is
#                 saved (utils/watch_v8in_epochs.sh)
#   4. the released Studio on the evaluation sheets, then two ensembles per sheet from the cached maps (no inference): Studio + reader-v3 + v8-in epoch 1, and the same with epoch 2
# Evaluation sheets: PRIZESHEET_11_19_flatboi and those listed in utils/local_eval_sheets.txt. Maps: output/mix28x_new_ps7/prize_<sheet>/cache/<model>/, ensembles
# output/ensemble/mix28x_new_ps7_s1_v8in_epoch{1,2}/<sheet>/. Waits for the running v8-in one-stage run, the staging (utils/stage_mix28x_new.sh) and the smoke test flag.
# A step that fails is retried once, then written to logs_herculex/new_queue/queue_failures.log; the queue goes on. Progress: logs_herculex/new_queue/queue.log.
#   tmux new -d -s queue_new 'bash /vesuvius/utils/queue_mix28x_new.sh'
export PATH=/root/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /vesuvius; L=logs_herculex/new_queue; mkdir -p $L
OUT=output/mix28x_new_ps7
RD=reader-v3/runs/mix28x_new_ps7_s1; SD=models/studio/mix28x_new_ps7; VD=nader-fork/runs/v8in_mix28x_new_4ep_ps7_s1
SHEETS="PRIZESHEET_11_19_flatboi $(cat utils/local_eval_sheets.txt 2>/dev/null | tr '\n' ' ')"
RELEASED_STUDIO=models/studio/studio-softloss-20261023-ema.safetensors
log() { echo "$(date +%F_%T) $*" | tee -a $L/queue.log; }
fail() { log "FAILED: $1"; echo "$1" >> $L/queue_failures.log; }
clean_shm() { rm -rf /dev/shm/ves_zarr_cache; }
need_space() {  # need_space <GB>: the pod's disk quota is not visible in df; a test write of that size must work, else stop (a full quota killed earlier jobs silently)
    if dd if=/dev/zero of=_ves_tmp/.quota_probe bs=1M count=$(( $1 * 1024 )) 2>/dev/null; then rm -f _ves_tmp/.quota_probe; else rm -f _ves_tmp/.quota_probe; log "STOP: less than $1 GB of quota left"; echo "quota: less than $1 GB left" >> $L/queue_failures.log; exit 1; fi
}
archive() { [ -e "$1" ] && mv "$1" "${1}_before_$(date +%m%d_%H%M)"; return 0; }
viz() {  # viz <model> : every saved checkpoint on every evaluation sheet, retried once
    for S in $SHEETS; do
        for attempt in 1 2; do python3 -m utils.viz_checkpoints $1 --sheet $S --out $OUT/prize_$S > $L/viz_${1}_${S}_${attempt}.log 2>&1 && break; clean_shm; done || fail "$1 visualization on $S"
    done
    clean_shm
}

rm -f $L/queue_failures.log $L/queue_done
log "waiting for: one-stage run, staging, smoke test"
while [ ! -f logs_herculex/onestage/queue_done ] || [ ! -f logs_herculex/stage_new/stage_ok ] || [ ! -f $L/smoke_ok ]; do sleep 60; done
log "queue start (sheets: $(echo $SHEETS | wc -w))"
python3 -m utils.mix6 > $L/mix6_check.log 2>&1
clean_shm

# ---------------------------------------------------------------- 1. reader-v3
need_space 10
log "reader-v3: prep + training"
[ -f $RD/ckpt_001000.pth ] || python3 finetune_readerv3.py > $L/reader_train_1.log 2>&1 || { log "reader-v3 attempt 1 failed; retrying"; clean_shm; archive $RD; python3 finetune_readerv3.py > $L/reader_train_2.log 2>&1; } 
[ -f $RD/ckpt_001000.pth ] || fail "reader-v3 training"
clean_shm
log "reader-v3: visualizing step 1000"
viz reader

# ---------------------------------------------------------------- 2. Studio
need_space 10
log "studio: setup, stage, training"
[ -f $SD/step_000500.safetensors ] || python3 finetune_studio.py > $L/studio_train_1.log 2>&1 || { log "studio attempt 1 failed; retrying"; clean_shm; archive $SD; python3 finetune_studio.py > $L/studio_train_2.log 2>&1; }
[ -f $SD/step_000500.safetensors ] || fail "studio training"
clean_shm
log "studio: visualizing step 500"
viz studio

# ---------------------------------------------------------------- 3. v8-in
need_space 16
log "v8-in: stage + training (2 epochs); each epoch is visualized when it is saved"
rm -f $L/v8in_watch_done
tmux new -d -s v8watch_new 'bash /vesuvius/utils/watch_v8in_epochs.sh'
if ! ls $VD/runs/*/checkpoints/trained-epoch1.ckpt > /dev/null 2>&1; then
    python3 finetune_v8in.py > $L/v8in_train_1.log 2>&1 || {
        log "v8-in attempt 1 failed; retrying from the last checkpoint"; clean_shm
        last=$(readlink -f $(ls -t $VD/runs/*/checkpoints/last.ckpt 2>/dev/null | head -1) 2>/dev/null)
        python3 finetune_v8in.py ${last:+--resume $last} > $L/v8in_train_2.log 2>&1; }
fi
ls $VD/runs/*/checkpoints/trained-epoch1.ckpt > /dev/null 2>&1 || fail "v8-in training"
while [ ! -f $L/v8in_watch_done ]; do sleep 30; done
clean_shm
log "v8-in: final visualization pass (cached maps are reused)"
viz v8in

# ---------------------------------------------------------------- 4. base Studio and the ensembles
for S in $SHEETS; do
    log "released Studio on $S"
    python3 -m utils.viz_checkpoints studio --sheet $S --ckpt $RELEASED_STUDIO --out $OUT/base_$S > $L/viz_base_$S.log 2>&1 || fail "released Studio on $S"
done
clean_shm
need_space 6
for e in 1 2; do
    for S in $SHEETS; do
        log "ensemble with v8-in epoch $e on $S"
        base=$(ls $OUT/base_$S/cache/studio/*.npy 2>/dev/null | head -1)
        python3 -m utils.ensemble_from_maps --sheet $S --subdir mix28x_new_ps7_s1_v8in_epoch$e \
            --studio $OUT/prize_$S/cache/studio/step_500.npy --reader $OUT/prize_$S/cache/reader/step_1000.npy \
            --v8in $OUT/prize_$S/cache/v8in/epoch_${e}_trained_epoch$((e-1)).npy --base "$base" \
            --studio-ckpt $SD/step_000500.safetensors --reader-ckpt $RD/ckpt_001000.pth \
            --v8in-ckpt $(ls $VD/runs/*/checkpoints/trained-epoch$((e-1)).ckpt | head -1) --v8in-label "epoch $e" > $L/ensemble_e${e}_$S.log 2>&1 || fail "ensemble epoch $e on $S"
    done
done
log "queue finished"; echo done > $L/queue_done
[ -f $L/queue_failures.log ] && log "failures: $(tr '\n' ';' < $L/queue_failures.log)"
