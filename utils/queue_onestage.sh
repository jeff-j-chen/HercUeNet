#!/bin/bash
# One-stage fine-tunes from the RELEASED reader-v3 / v8-in (not from our mix28 fine-tunes), corpus mix28x, schedules of the full runs stopped early:
#   reader-v3: 4000-step schedule, stopped at step 1000      -> reader-v3/runs/mix28x_ps7_s1_onestage/ckpt_001000.pth
#   v8-in:     4-epoch schedule, stopped after epoch 2       -> nader-fork/runs/v8in_mix28x_4ep_ps7_s1_onestage/runs/*/checkpoints/trained-epoch{0,1}.ckpt
# every saved checkpoint is run on the prize sheet (maps in output/onestage_PRIZESHEET_11_19_flatboi/), then compared with the two-stage maps.
export PATH=/root/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /vesuvius; L=logs_herculex/onestage; mkdir -p $L
OUT=output/onestage_PRIZESHEET_11_19_flatboi
log() { echo "$(date +%F_%T) $*" | tee -a $L/queue.log; }
clean_shm() { rm -rf /dev/shm/ves_zarr_cache; }
RD=reader-v3/runs/mix28x_ps7_s1_onestage
VD=nader-fork/runs/v8in_mix28x_4ep_ps7_s1_onestage
clean_shm
log "reader-v3 one-stage: prep + train (released init, stop at 1000)"
if [ ! -f $RD/ckpt_001000.pth ]; then python3 finetune_readerv3.py --init released --run-name mix28x_ps7_s1_onestage > $L/reader_train.log 2>&1; fi
[ -f $RD/ckpt_001000.pth ] || { log "FAILED reader-v3 training"; exit 1; }
clean_shm
log "reader-v3 one-stage: visualize"
python3 -m utils.viz_checkpoints reader --run mix28x_ps7_s1_onestage --out $OUT > $L/viz_reader.log 2>&1 || log "FAILED reader viz"
clean_shm
python3 -m utils.compare_onestage > $L/compare_reader.log 2>&1
log "v8-in one-stage: stage + train (released init, stop after epoch 2)"
if ! ls $VD/runs/*/checkpoints/trained-epoch1.ckpt > /dev/null 2>&1; then
    python3 finetune_v8in.py --init released --run-name v8in_mix28x_4ep_ps7_s1_onestage --stop-epoch 2 > $L/v8in_train.log 2>&1
fi
ls $VD/runs/*/checkpoints/trained-epoch1.ckpt > /dev/null 2>&1 || { log "FAILED v8-in training"; exit 1; }
clean_shm
log "v8-in one-stage: visualize both epochs"
python3 -m utils.viz_checkpoints v8in --run v8in_mix28x_4ep_ps7_s1_onestage --out $OUT > $L/viz_v8in.log 2>&1 || log "FAILED v8-in viz"
clean_shm
python3 -m utils.compare_onestage > $L/compare_all.log 2>&1
log "queue finished"; echo done > $L/queue_done
