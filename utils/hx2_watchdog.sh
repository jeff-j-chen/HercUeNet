#!/bin/bash
# Protects the running training while utils/assemble_herculex2.py builds the new x sheets: every 2 minutes, if the container's anonymous memory (rss) goes above
# 50 GiB of its 57.7 GiB limit, or the new renders' temporary chunk caches pass 25 GiB, the assembler (not the training) is stopped; it resumes where it was when started again.
LOG=/vesuvius/logs_herculex/hx2/watchdog.log
while pgrep -f "utils.assemble_herculex2" > /dev/null; do
    rss=$(awk '/^total_rss /{printf "%d", $2/2^30}' /sys/fs/cgroup/memory/memory.stat)
    tmp=$(du -s --block-size=1G /vesuvius/_ves_tmp/render_hx2_* 2>/dev/null | awk '{s+=$1} END {print s+0}')
    echo "$(date +%T) rss ${rss} GiB, chunk caches ${tmp} GiB" >> $LOG
    if [ "$rss" -gt 50 ] || [ "$tmp" -gt 25 ]; then
        echo "$(date +%T) LIMIT: stopping the assembler" >> $LOG
        for pid in $(pgrep -f "utils.assemble_herculex2"); do kill $pid; done
        break
    fi
    sleep 120
done
