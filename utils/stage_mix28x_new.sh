#!/bin/bash
# Stage the corpus mix28x_new for the three trainers (CPU and disk only, no GPU), 2 jobs each. Staged folders are shared with older corpora; only new aliases are added.
#   reader-v3 prep and Studio stage first (next to a running v8-in training); v8-in stage waits for that training to end (logs_herculex/onestage/queue_done), deletes the staged copies of the
#   superseded first x export (HERCULEX_SHEETS) to make room, then stages. Writes logs_herculex/stage_new/stage_ok when all three worked.
export PATH=/root/.local/bin:$PATH VES_STAGE_JOBS=2
cd /vesuvius; L=logs_herculex/stage_new; mkdir -p $L; rm -f $L/stage_done $L/stage_ok
log() { echo "$(date +%F_%T) $*" >> $L/stage.log; }
rc=0
log "reader-v3 prep"; nice -n 10 python3 finetune_readerv3.py prep > $L/reader_prep.log 2>&1; r1=$?; log "reader prep rc=$r1"
log "studio stage";   nice -n 10 python3 finetune_studio.py stage --jobs 2 > $L/studio_stage.log 2>&1; r2=$?; log "studio stage rc=$r2"
log "waiting for the running v8-in training"; while [ ! -f logs_herculex/onestage/queue_done ]; do sleep 60; done
python3 - >> $L/stage.log 2>&1 <<'P'
import sys, shutil, os
sys.path.insert(0, '/vesuvius')
from utils import mix6
root = '/vesuvius/nader-fork/runs/v8in_mix25_data/data'
n = 0
for alias in mix6.HERCULEX_SHEETS:
    p = f'{root}/{alias}'
    if os.path.lexists(p):
        shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p); n += 1
print('deleted the staged v8-in copies of the first x export:', n)
P
log "v8-in stage"; nice -n 10 python3 finetune_v8in.py stage > $L/v8in_stage.log 2>&1; r3=$?; log "v8-in stage rc=$r3"
echo done > $L/stage_done
[ $r1 = 0 ] && [ $r2 = 0 ] && [ $r3 = 0 ] && echo ok > $L/stage_ok
