#!/bin/bash
# While staging next to a running training: stop the staging when the container's anonymous memory passes 50 GiB (of 57.7) or when 5 GB can no longer be written (the disk quota is near;
# the training needs room for its ~1 GB checkpoints).
LOG=/vesuvius/logs_herculex/stage_new/watchdog.log
while [ ! -f /vesuvius/logs_herculex/stage_new/stage_done ]; do
    rss=$(awk '/^total_rss /{printf "%d", $2/2^30}' /sys/fs/cgroup/memory/memory.stat)
    if dd if=/dev/zero of=/vesuvius/_ves_tmp/.quota_probe bs=1M count=5120 2>/dev/null; then room=ok; else room=LOW; fi
    rm -f /vesuvius/_ves_tmp/.quota_probe
    echo "$(date +%T) rss ${rss} GiB, 5 GB probe $room" >> $LOG
    if [ "$rss" -gt 50 ] || [ "$room" = LOW ]; then
        echo "$(date +%T) LIMIT: stopping the staging" >> $LOG
        for pid in $(pgrep -f "stage_mix28x_new.sh"); do kill $pid; done
        for pid in $(pgrep -f "finetune_(readerv3|v8in|studio).py (prep|stage)"); do kill $pid; done
        break
    fi
    sleep 120
done
