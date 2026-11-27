#!/bin/bash
# v8-in epoch 2 (the second epoch of the 4-epoch schedule), started from the saved epoch-0 weights because the epoch-0 Lightning checkpoints (optimizer, scheduler) were lost.
# The cosine schedule is continued approximately: the original schedule (4 x 5224 optimizer steps, 15% warmup from 1e-6, cosine to 1e-6) runs from 9.70e-6 to 6.73e-6
# over epoch 2 (mean 8.43e-6); a 1-epoch cosine without warmup from 9.645e-6 to 7.205e-6 fits it to 1.3% mean / 7% max (at the end). Fresh AdamW moments, a different seed.
export PATH=/root/.local/bin:$PATH VESUVIUS_CORPUS=mix28x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /vesuvius; L=logs_herculex; RUN=v8in_mix28x_ps7_epoch2
INIT=$(readlink -f _ves_tmp/v8in_exports/er_fork_runs_v8in_mix28x_4ep_ps7_s1_runs_v8in_mix28x_4ep_ps7_s1_20261120_045916_8a634fc4_checkpoints_trained_epoch0_ckpt/model.safetensors)
rm -rf /dev/shm/ves_zarr_cache
echo "$(date +%T) v8-in epoch 2 start from $INIT" >> $L/queue.log
python3 finetune_v8in.py stage > $L/v8in_stage_r.log 2>&1
setsid python3 finetune_v8in.py train --run-name $RUN --epochs 1 --init "$INIT" --neg-weight 10 --lr 9.645e-6 --min-lr 7.205e-6 --warmup-pct 0 --seed 271828 > $L/v8in_train_r.log 2>&1 &
pid=$!; size=-1
while kill -0 $pid 2>/dev/null; do
  f=$(ls nader-fork/runs/$RUN/runs/*/checkpoints/trained-epoch0.ckpt 2>/dev/null | head -1)
  if [ -n "$f" ]; then now=$(stat -c %s "$f"); if [ "$now" = "$size" ]; then sleep 30; kill -TERM -- -$pid; break; fi; size=$now; fi
  sleep 20
done
wait $pid
f=$(ls $PWD/nader-fork/runs/$RUN/runs/*/checkpoints/trained-epoch0.ckpt | head -1)
[ -n "$f" ] || { echo "$(date +%T) v8in epoch 2 FAILED" >> $L/queue.log; exit 1; }
mkdir -p models/v8in_epoch2; cp "$f" models/v8in_epoch2/epoch2.ckpt
rm -rf /dev/shm/ves_zarr_cache
python3 -m utils.viz_checkpoints v8in --ckpt models/v8in_epoch2/epoch2.ckpt > $L/viz_v8in_r.log 2>&1
echo "$(date +%T) v8in epoch 2 + viz done" >> $L/queue.log; echo done > $L/v8in_epoch2_done
