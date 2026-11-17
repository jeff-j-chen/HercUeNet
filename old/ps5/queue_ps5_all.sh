#!/usr/bin/env bash
# Everything pending, strictly one job at a time: the unfinished depth -5 ensemble of PRIZESHEET_11_12_NTcopy_back, the failed _ps5 trainings
# and evaluations (utils/queue_ps5.sh skips what exists and resumes v8-in), then the 0846A + patch 487 trainings (utils/queue_ps5_487.sh).
#   tmux new -d -s queue_all 'bash /vesuvius/utils/queue_ps5_all.sh'
cd /vesuvius
echo "[queue_ps5_all $(date '+%F %T')] depth -5 ensemble of PRIZESHEET_11_12_NTcopy_back" | tee -a logs_ps5/queue.log
rm -rf /dev/shm/ves_zarr_cache/* 2>/dev/null
S=PRIZESHEET_11_12_NTcopy_back
ENSEMBLE_SHEET=$S ENSEMBLE_SUBDIR=mix28_ps3_c3_r1000_v3_depth_m5 ENSEMBLE_CUE_DEPTHS=-5 ENSEMBLE_READER_DEPTHS=-5 ENSEMBLE_V8IN_DEPTHS=-5 \
    python3 utils/run_notebook.py ensemble_eval.ipynb > logs_ens_nt_back_m5.log 2>&1
echo "[queue_ps5_all $(date '+%F %T')] depth -5 ensemble: exit $?" | tee -a logs_ps5/queue.log
bash utils/queue_ps5.sh
bash utils/queue_ps5_487.sh
