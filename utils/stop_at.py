"""Run a training command and stop it once the checkpoint it should stop at is written (and its size is stable).

The fine-tunes keep the learning-rate schedule of their full run (Studio 2000 steps, reader-v3 4000 steps, v8-in 4 epochs) and are stopped early at
the checkpoint the ensemble uses (step 500, step 1000, epoch 1), so the learning rate at that checkpoint is the one of the full run.
"""
from __future__ import annotations

import glob
import os
import signal
import subprocess
import time


def run_until(command, pattern: str, cwd=None, env=None, poll: float = 20.0, settle: float = 30.0) -> int:
    """run `command` in its own process group; once a file matching `pattern` exists with a size unchanged between two polls, wait `settle` seconds
    (the trainer may still be flushing) and stop the group. Returns 0 on a stop at the checkpoint, else the command's return code (non-zero = failed
    before writing it; raises CalledProcessError)."""
    process = subprocess.Popen([str(part) for part in command], cwd=cwd, env=env, start_new_session=True)
    size = -1
    try:
        while process.poll() is None:
            found = sorted(glob.glob(pattern))
            if found:
                now = os.path.getsize(found[-1])
                if now == size:
                    time.sleep(settle)
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    return 0
                size = now
            time.sleep(poll)
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        raise
    if glob.glob(pattern):  # finished by itself with the checkpoint written (a schedule that ends at the stop)
        return 0
    raise subprocess.CalledProcessError(process.returncode, command)
