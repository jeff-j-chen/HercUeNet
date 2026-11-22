#!/bin/bash
# Smoke test of the mix28x_new queue (utils/queue_mix28x_new.sh), run once the running training and the staging are done: every trainer for a few steps on the real new corpus with its
# real stopping code, the visualization of the smoke checkpoints on every evaluation sheet, and the ensemble step from cached maps. Writes logs_herculex/new_queue/smoke_ok when all passed
# (the queue waits for it); the smoke runs are deleted afterwards.
export PATH=/root/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /vesuvius; L=logs_herculex/new_queue; mkdir -p $L/smoke; rm -f $L/smoke_ok $L/smoke_failed
SHEETS="PRIZESHEET_11_19_flatboi $(cat utils/local_eval_sheets.txt 2>/dev/null | tr '\n' ' ')"
OUT=output/mix28x_new_smoke
log() { echo "$(date +%F_%T) $*" | tee -a $L/smoke/summary.txt; }
bad=0
check() { if [ "$1" = 0 ]; then log "PASS $2"; else log "FAIL $2"; bad=1; fi; }
clean_shm() { rm -rf /dev/shm/ves_zarr_cache; }
log "waiting for the running training and the staging"
while [ ! -f logs_herculex/onestage/queue_done ] || [ ! -f logs_herculex/stage_new/stage_done ]; do sleep 60; done
[ -f logs_herculex/stage_new/stage_ok ] || { log "FAIL staging did not finish ok"; touch $L/smoke_failed; exit 1; }
clean_shm
log "reader-v3: real corpus, stop at step 10"
python3 finetune_readerv3.py --run-name smoke_new --stop-step 10 --save-every 10 > $L/smoke/reader.log 2>&1; check $? "reader-v3 trains and stops"
ls reader-v3/runs/smoke_new/ckpt_000010.pth > /dev/null 2>&1; check $? "reader-v3 checkpoint written"
clean_shm
log "studio: real corpus, stop at step 20"
.venv-studio/bin/python finetune_studio.py train --stop-step 20 --save-every 20 --workers 3 --out smoke_new > $L/smoke/studio.log 2>&1; check $? "studio trains and stops"
ls models/studio/smoke_new/step_000020.safetensors > /dev/null 2>&1; check $? "studio checkpoint written"
clean_shm
log "v8-in: real corpus, 6 batches per epoch, 2 of 4 epochs"
python3 finetune_v8in.py --run-name smoke_new --limit-batches 6 > $L/smoke/v8in.log 2>&1; check $? "v8-in trains and stops"
ls nader-fork/runs/smoke_new/runs/*/checkpoints/trained-epoch0.ckpt nader-fork/runs/smoke_new/runs/*/checkpoints/trained-epoch1.ckpt > /dev/null 2>&1; check $? "v8-in epoch 1 and 2 checkpoints written"
clean_shm
for S in $SHEETS; do
    log "visualizing the smoke checkpoints on $S"
    python3 -m utils.viz_checkpoints reader --run smoke_new --sheet $S --out $OUT/prize_$S > $L/smoke/viz_reader_$S.log 2>&1; check $? "reader viz on $S"
    python3 -m utils.viz_checkpoints studio --run smoke_new --sheet $S --out $OUT/prize_$S > $L/smoke/viz_studio_$S.log 2>&1; check $? "studio viz on $S"
    python3 -m utils.viz_checkpoints v8in --run smoke_new --sheet $S --out $OUT/prize_$S > $L/smoke/viz_v8in_$S.log 2>&1; check $? "v8-in viz on $S"
    python3 -m utils.viz_checkpoints studio --sheet $S --ckpt models/studio/studio-softloss-20261023-ema.safetensors --out $OUT/base_$S > $L/smoke/viz_base_$S.log 2>&1; check $? "released Studio on $S"
    clean_shm
    base=$(ls $OUT/base_$S/cache/studio/*.npy 2>/dev/null | head -1)
    python3 -m utils.ensemble_from_maps --sheet $S --subdir smoke_new --studio $OUT/prize_$S/cache/studio/step_20.npy --reader $OUT/prize_$S/cache/reader/step_10.npy \
        --v8in $OUT/prize_$S/cache/v8in/epoch_2_trained_epoch1.npy --base "$base" --studio-ckpt models/studio/smoke_new/step_000020.safetensors \
        --reader-ckpt reader-v3/runs/smoke_new/ckpt_000010.pth --v8in-ckpt $(ls nader-fork/runs/smoke_new/runs/*/checkpoints/trained-epoch1.ckpt | head -1) \
        --studio-label "step 20" --reader-label "step 10" --v8in-label "epoch 2" > $L/smoke/ensemble_$S.log 2>&1; check $? "ensemble figures on $S"
    ls output/ensemble/smoke_new/$S/ensemble_3x2.png output/ensemble/smoke_new/$S/zscore_product_ct_2x2.png > /dev/null 2>&1; check $? "ensemble figures exist on $S"
done
log "removing the smoke runs"
rm -rf reader-v3/runs/smoke_new nader-fork/runs/smoke_new models/studio/smoke_new output/mix28x_new_smoke output/ensemble/smoke_new
rm -rf _ves_tmp/v8in_exports/*smoke_new* 
clean_shm
if [ $bad = 0 ]; then touch $L/smoke_ok; log "SMOKE TEST PASSED"; else touch $L/smoke_failed; log "SMOKE TEST FAILED"; fi
