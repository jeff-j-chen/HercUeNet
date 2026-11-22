#!/bin/bash
# Visualizes each one-stage v8-in epoch on the prize sheet as soon as its trained-epoch<N>.ckpt is written (stable size), next to the running training
# (micro-batch 2 keeps the GPU memory small; /dev/shm is not touched). Maps: output/onestage_PRIZESHEET_11_19_flatboi/cache/v8in/.
export PATH=/root/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VIZ_BATCH=2
cd /vesuvius; L=logs_herculex/onestage; VD=nader-fork/runs/v8in_mix28x_4ep_ps7_s1_onestage; OUT=output/onestage_PRIZESHEET_11_19_flatboi
for n in 0 1; do
    size=-1
    while true; do
        f=$(ls $VD/runs/*/checkpoints/trained-epoch$n.ckpt 2>/dev/null | head -1)
        if [ -n "$f" ]; then now=$(stat -c %s "$f"); [ "$now" = "$size" ] && break; size=$now; fi
        sleep 30
    done
    echo "$(date +%F_%T) epoch $((n+1)) (trained-epoch$n) saved: visualizing" >> $L/queue.log
    python3 -m utils.viz_checkpoints v8in --run v8in_mix28x_4ep_ps7_s1_onestage --out $OUT > $L/viz_v8in_epoch$n.log 2>&1 && echo "$(date +%F_%T) epoch $((n+1)) visualized" >> $L/queue.log || echo "$(date +%F_%T) FAILED epoch $((n+1)) viz" >> $L/queue.log
done
python3 -m utils.compare_onestage > $L/compare_watch.log 2>&1
