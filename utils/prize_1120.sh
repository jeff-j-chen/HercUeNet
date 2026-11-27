#!/bin/bash
# Re-render PRIZESHEET_11_20 (its tifxyz changed) and run reader step 1000, studio step 500, v8-in epoch 0 on it, one at a time with small batches
# (v8-in training shares the GPU: ~10 GiB of 20), then compare the z-score ensemble with the 11_19_flatboi one.
export PATH=/root/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /vesuvius; L=logs_herculex
python3 - > $L/render_1120.log 2>&1 <<'P'
import sys; sys.path.insert(0, '/vesuvius')
import assemble_test_segments as t
from utils import mix6
from utils.assemble_herculex import VOLUMES
print(t.render_fragment('PRIZESHEET_11_20', 'PRIZESHEET_11_20', *VOLUMES['A'], 8, mix6.zarr_dir().as_posix(), t.SCRIPT_DIR, t.DEFAULT_CHUNK_DEPTH, t.DEFAULT_CHUNK_Y, t.DEFAULT_CHUNK_X, force=True, mask_dir=t.EXTRAS_MASK_DIR, output_dtype=t.EXTRAS_DTYPE, drop_cache=True, crop_valid=False))
P
echo "render rc=$?" >> $L/render_1120.log
export VIZ_BATCH=2
rm -rf /dev/shm/ves_zarr_cache
S="--sheet PRIZESHEET_11_20"
python3 -m utils.viz_checkpoints studio $S --ckpt models/studio/mix28x_ps7/step_000500.safetensors > $L/viz1120_studio.log 2>&1
python3 -m utils.viz_checkpoints reader $S --ckpt reader-v3/runs/mix28x_ps7_s1/ckpt_001000.pth > $L/viz1120_reader.log 2>&1
python3 -m utils.viz_checkpoints v8in $S --ckpt models/v8in_epoch0 > $L/viz1120_v8in.log 2>&1
rm -rf /dev/shm/ves_zarr_cache
python3 -m utils.compare_1120 > $L/compare1120.log 2>&1
echo finished >> $L/compare1120.log
