"""expand_corr.py -- add new mix25 fragments to an already-finished runs_corr study.

New training fragments (500p2-1 and p841_w00 already completed):
  500P2_m2    20250825181917  PHerc0500P2
  p343_4b2    20250902170441  PHerc0343P
  p9b_592     20250919132115  PHerc0009B
  seg0814_459 20250925222237  PHerc0814
  title       20260422000000  PHerc0139

Final sanity-check batch -- the 4 fiber_only_20261029d_flatboi neighbor sheets
(assembled by step 00_build AFTER the original plan was frozen, so absent from
patches.csv even though they are eligible pherc0846a training sources):
  fiber_only_20261029d_flatboi_in_1   20261029120001  pherc0846a
  fiber_only_20261029d_flatboi_in_2   20261029120003  pherc0846a
  fiber_only_20261029d_flatboi_out_1  20261029120002  pherc0846a
  fiber_only_20261029d_flatboi_out_2  20261029120004  pherc0846a

If the labels are consistent, the 11 trained 0846A sheets should cluster together
and show similar alignment to every other fragment.

Steps executed by this script:
  1. Verify each new patch is eligible (all inputs ready).
  2. Train each new patch's single-patch source model (3 epochs, same recipe as the study).
  3. Insert the new patches into plan.json (patches[] + sources[]).
  4. Reset daemon_state.json so the daemon re-runs probe + gram + report for the expanded plan.

The daemon (with plan_frozen=True already) will:
  - Probe every new target under all sources (incremental; _pair_ok guards existing pairs).
  - Probe every old target under every new source (same incremental guard).
  - Re-run gram_worker for all sources (their done.json "targets" list changed).
  - Rebuild the report.

Usage:
  python3 expand_corr.py            # full run
  python3 expand_corr.py --dry-run  # eligibility check only, no training
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
for _p in (str(REPO), str(REPO / "old")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)

# New patches that need source-model training + probing.
# Group 1: new training fragments (500P2_m1 and p841_w00 already in the study).
# Group 2: fiber_only_20261029d neighbours -- assembled by step 00_build AFTER the
#           original plan was frozen (Oct 29), so absent from patches.csv despite
#           being fully eligible pherc0846a training sources.
NEW_NAMES = frozenset([
    "500P2_m2", "p343_4b2", "p9b_592", "seg0814_459", "title",
    "fiber_only_20261029d_flatboi_in_1",
    "fiber_only_20261029d_flatboi_in_2",
    "fiber_only_20261029d_flatboi_out_1",
    "fiber_only_20261029d_flatboi_out_2",
])

# Match the hyperparameters of the existing study (from done.json of any finished run).
STUDY_ARGS = SimpleNamespace(
    epochs=3,
    lr=1e-4,
    batches=48,
    eval_batches=16,
    probe_workers=8,
    max_samples=0,
    root="runs_corr",
)


def _log(msg: str) -> None:
    print(f"[expand {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run_child_sync(run: str, mode: str) -> int:
    """run one corr_runs worker in a subprocess, log to runs_corr/logs/, return exit code."""
    import corr_runs

    corr_runs.LOGS.mkdir(parents=True, exist_ok=True)
    log_path = corr_runs.LOGS / f"{mode}_{run}.log"
    cmd = corr_runs._worker_command(run, mode, STUDY_ARGS)
    _log(f"  $ {' '.join(cmd)}")
    with open(log_path, "a") as handle:
        handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {mode} {run} =====\n")
        handle.flush()
        code = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT).returncode
    _log(f"  exit {code} -> {log_path}")
    return code


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="eligibility check only; no training or file changes")
    args = ap.parse_args()

    import corr_runs

    corr_runs.set_root(STUDY_ARGS.root)
    corr_runs.MAX_SAMPLES = None

    # ------------------------------------------------------------------ #
    # 1. Identify and verify new patches                                  #
    # ------------------------------------------------------------------ #
    all_p = corr_runs.all_patches()
    new_candidates = [p for p in all_p if p.name in NEW_NAMES]
    if not new_candidates:
        _log("ERROR: none of the new names appear in all_patches(); check assemble_training_segments.SEGMENTS")
        sys.exit(1)

    config = corr_runs.base_config()
    eligible: list = []
    skipped: dict = {}
    for p in new_candidates:
        probs = corr_runs.basic_problems(p, config)
        if not probs:
            probs = corr_runs.norm_problems(p, config)
        if probs:
            skipped[str(p.pid)] = "; ".join(probs)
            _log(f"SKIP {p.name} ({p.pid}): {skipped[str(p.pid)]}")
        else:
            eligible.append(p)

    _log(f"New patches: {len(eligible)} eligible, {len(skipped)} skipped")
    for p in eligible:
        _log(f"  OK  {p.label:<20} pid={p.pid}  scroll={p.scroll}  source_um={p.source_um}")
    for pid, reason in skipped.items():
        _log(f"  SKIP {pid}: {reason}")

    if args.dry_run:
        _log("--dry-run: stopping before any changes.")
        return

    if not eligible:
        _log("No eligible new patches; nothing to do.")
        return

    # ------------------------------------------------------------------ #
    # 2. Train each new source model                                      #
    # ------------------------------------------------------------------ #
    _log("=== Phase 1: train new source models ===")
    for patch in eligible:
        if corr_runs._trained_ok(patch.label, STUDY_ARGS):
            _log(f"{patch.label}: trained.pt already matches study params, skipping")
            continue
        _log(f"Training {patch.label} (3 epochs, lr=1e-4)...")
        t0 = time.time()
        code = _run_child_sync(patch.label, "train")
        elapsed = (time.time() - t0) / 60
        if code != 0:
            _log(f"ERROR: training {patch.label} failed (exit {code}); aborting")
            sys.exit(1)
        if not corr_runs._trained_ok(patch.label, STUDY_ARGS):
            _log(f"ERROR: {patch.label} trained.pt not found or params mismatch after training; aborting")
            sys.exit(1)
        _log(f"{patch.label}: trained in {elapsed:.1f} min")

    # ------------------------------------------------------------------ #
    # 3. Expand plan.json                                                 #
    # ------------------------------------------------------------------ #
    _log("=== Phase 2: expand plan.json ===")
    plan = corr_runs.load_plan()
    existing_pids = {p["pid"] for p in plan["patches"]}
    existing_sources = set(plan["sources"])
    added_patches, added_sources = [], []

    for p in eligible:
        if p.pid not in existing_pids:
            plan["patches"].append(dataclasses.asdict(p))
            added_patches.append(p.label)
        if p.label not in existing_sources:
            plan["sources"].append(p.label)
            added_sources.append(p.label)

    if not added_patches and not added_sources:
        _log("All eligible patches already present in plan.json; nothing to update")
    else:
        corr_runs._atomic_json(plan, corr_runs.PLAN)
        _log(f"plan.json: added {len(added_patches)} patches {added_patches}, "
             f"{len(added_sources)} sources {added_sources}")

    # ------------------------------------------------------------------ #
    # 4. Reset daemon state                                               #
    # ------------------------------------------------------------------ #
    _log("=== Phase 3: reset daemon state ===")
    state_path = corr_runs.ROOT / "daemon_state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    state.setdefault("failures", {})
    state.setdefault("errors", {})

    # New probe targets: clear any stale failure counts so daemon probes them.
    for p in eligible:
        state["failures"].pop(f"probe:{p.label}", None)
        state["failures"].pop(f"gram:{p.label}", None)

    # Existing gram runs are invalidated by the expanded target list; clear their failures.
    for source in plan["sources"]:
        state["failures"].pop(f"gram:{source}", None)

    state["failures"].pop("report:all", None)
    state["reported"] = False
    state["plan_frozen"] = True  # daemon skips the training phase and goes straight to probe

    corr_runs._atomic_json(state, state_path)
    _log("daemon_state.json: reported=False, plan_frozen=True, failure counts cleared for new/updated jobs")

    # ------------------------------------------------------------------ #
    # 5. Build cue map for eval-only sheets not in mix6.all_ids()        #
    # ------------------------------------------------------------------ #
    # fiber_only_20261030_flatboi alias 20261030120000: eval-only, not in
    # mix6.all_ids(), so the corr daemon won't build its cue automatically.
    _EVAL_ONLY_ALIASES = [20261030120000]
    for alias in _EVAL_ONLY_ALIASES:
        cue_path = corr_runs.REPO / "cue_asf_morph" / f"{alias}.npy"
        if cue_path.is_file():
            _log(f"cue for {alias} already exists, skipping")
            continue
        zarr_path = corr_runs.REPO / "ves_zarrs2" / f"{alias}.zarr"
        if not zarr_path.exists():
            _log(f"WARNING: zarr for alias {alias} not found at {zarr_path}; skipping cue build")
            continue
        _log(f"Building cue map for eval-only alias {alias}...")
        import subprocess as _sp
        code = _sp.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, '/vesuvius'); sys.path.insert(0, '/vesuvius/old'); "
             f"from utils import mix6; mix6._build_cue(({alias}, 0, True))"],
            cwd=str(corr_runs.REPO),
        ).returncode
        _log(f"cue build for {alias}: exit {code}")

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    new_targets = [p.label for p in eligible]
    all_sources = plan["sources"]
    n_new_pairs = len(new_targets) * len(all_sources) + (len(plan["patches"]) - len(new_targets)) * len(added_sources)
    _log(f"Ready. New probe pairs to compute: ~{n_new_pairs} "
         f"({len(new_targets)} new targets x {len(all_sources)} sources "
         f"+ {len(plan['patches']) - len(new_targets)} old targets x {len(added_sources)} new sources)")
    _log(f"Gram jobs to re-run: {len(all_sources)} (all sources; target list changed)")
    _log("Start the daemon with:")
    _log("  nohup python3 corr_runs.py --no-wait >> runs_corr/daemon.log 2>&1 &")


if __name__ == "__main__":
    main()
