#!/bin/bash
# Visualizes each v8-in epoch (trained-epoch0, trained-epoch1) of the mix28x_new run on every evaluation sheet as soon as its checkpoint is written, next to the running training
# (micro-batch 2 keeps the GPU memory small; /dev/shm is not touched). Sheets: the prize sheet and the ones in utils/local_eval_sheets.txt (not tracked).
export PATH=/root/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VIZ_BATCH=2
cd /vesuvius; L=logs_herculex/new_queue; VD=nader-fork/runs/v8in_mix28x_new_4ep_ps7_s1; OUT=output/mix28x_new_ps7
SHEETS="PRIZESHEET_11_19_flatboi $(cat utils/local_eval_sheets.txt 2>/dev/null | tr '\n' ' ')"
for n in 0 1; do
    size=-1
    while true; do
        f=$(ls $VD/runs/*/checkpoints/trained-epoch$n.ckpt 2>/dev/null | head -1)
        if [ -n "$f" ]; then now=$(stat -c %s "$f"); [ "$now" = "$size" ] && break; size=$now; fi
        sleep 30
    done
    for S in $SHEETS; do
        echo "$(date +%F_%T) v8-in epoch $((n+1)) saved: visualizing on $S" >> $L/queue.log
        python3 -m utils.viz_checkpoints v8in --sheet $S --out $OUT/prize_$S > $L/viz_v8in_e${n}_$S.log 2>&1 && echo "$(date +%F_%T) epoch $((n+1)) visualized on $S" >> $L/queue.log || echo "$(date +%F_%T) FAILED epoch $((n+1)) viz on $S" >> $L/queue.log
    done
done
echo done > $L/v8in_watch_done
