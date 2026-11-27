"""ONE-TIME: back up the mix28 ink labels, then erode their white ink in place.

  python3 utils/erode_inklabels_once.py                 # dry run: prints the plan, writes nothing
  python3 utils/erode_inklabels_once.py --yes           # back up, verify the backup, erode
  python3 utils/erode_inklabels_once.py --restore DIR   # copy a backup back over dilated_inklabels/

Files: dilated_inklabels/<alias>.png of every segment of the mix28 corpus (utils/mix6.py corpus_segments("mix28")). Many aliases are symlinks to
a named file (fiber_only_..._flatboi_out_2.png, 20260921094413.png, ...): each REAL file is backed up and eroded once, whichever aliases point at it.

Backup: backups/dilated_inklabels_mix28_<UTC time>/ next to dilated_inklabels/ (outside it, so the trainers never read it): the real files under their
own names, `manifest.json` (alias -> real file, SHA-256 of each) and nothing else. The backup is re-read and checked against the hashes before anything
is eroded; the script refuses to run again on labels it already eroded (dilated_inklabels/.eroded_once.json), and --restore removes that mark.

Erosion: --iterations (default 5) of cv2.erode with a --kernel x --kernel (default 3 x 3) rectangle on the WHITE ink only. "White" is the regular ink
exactly as the trainers read it (utils/dataloader.split_extra_labels: value >= 128 and not a solid half-brightness stroke). Pixels the erosion removes
become black (0); every other pixel keeps its value, so half-brightness strokes (the assumed letters / the excluded region of a label) are untouched.
Staged data and cached figures made from the old labels (nader-fork/runs/v8in_mix25_data, reader-v3/data, train masks, ...) are NOT rebuilt here.
"""
import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

# utils/platform.py would shadow the stdlib module when this file is run as a script
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import cv2
import numpy as np

LABELS = REPO / "dilated_inklabels"
BACKUPS = REPO / "backups"
MARK = LABELS / ".eroded_once.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_extra_labels(raw):
    """utils/dataloader.split_extra_labels: (regular ink, assumed half-brightness letters)."""
    mid = ((raw >= 48) & (raw <= 207)).astype(np.uint8)
    extra = np.zeros(raw.shape, dtype=bool)
    if mid.any():
        grown = cv2.morphologyEx(mid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        if grown.any():
            step = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            for _ in range(3):
                grown = cv2.dilate(grown, step) & mid
            extra = grown > 0
    return (raw >= 128) & ~extra, extra


def erode_white(raw, kernel, iterations):
    white, _ = split_extra_labels(raw)
    kept = cv2.erode(white.astype(np.uint8), np.ones((kernel, kernel), np.uint8), iterations=iterations) > 0
    out = raw.copy()
    out[white & ~kept] = 0
    return out, int(white.sum()), int(kept.sum())


def mix28_files():
    """alias -> real file of dilated_inklabels for every mix28 segment that has one."""
    from utils import mix6

    found, missing = {}, []
    for alias in mix6.corpus_segments("mix28", labelled_only=False):
        path = LABELS / f"{alias}.png"
        (found.__setitem__(alias, path.resolve()) if path.exists() else missing.append(alias))
    return found, missing


def restore(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise SystemExit(f"backup file {name} does not match its hash; nothing restored")
    for name in manifest["files"]:
        shutil.copy2(directory / name, LABELS / name)
    MARK.unlink(missing_ok=True)
    print(f"restored {len(manifest['files'])} files from {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="really do it (default: print the plan)")
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--kernel", type=int, default=3, help="side of the square kernel")
    parser.add_argument("--restore", metavar="BACKUP_DIR")
    args = parser.parse_args()
    if args.restore:
        return restore(args.restore)
    if MARK.exists():
        raise SystemExit(f"{MARK} exists: these labels were already eroded once (use --restore BACKUP_DIR to undo)")

    files, missing = mix28_files()
    real = sorted({path for path in files.values()})
    print(f"{len(files)} mix28 aliases have a label ({len(missing)} without: {', '.join(missing) or 'none'}); {len(real)} real files:")
    for path in real:
        aliases = [alias for alias, target in files.items() if target == path]
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        _, white, kept = erode_white(raw, args.kernel, args.iterations)
        print(f"  {path.name:55s} aliases {','.join(aliases):28s} white {white:>10,d} -> {kept:>10,d} ({100 * kept / max(white, 1):.0f}% left)")
    if not args.yes:
        print(f"dry run: nothing written (--yes backs up to {BACKUPS}/dilated_inklabels_mix28_<time>/ and erodes {args.iterations} x {args.kernel}x{args.kernel})")
        return

    backup = BACKUPS / f"dilated_inklabels_mix28_{datetime.datetime.now(datetime.timezone.utc):%Y%m%d_%H%M%S}"
    backup.mkdir(parents=True)
    hashes = {}
    for path in real:
        shutil.copy2(path, backup / path.name)
        hashes[path.name] = sha256(path)
    for name, digest in hashes.items():
        if sha256(backup / name) != digest:
            raise SystemExit(f"backup of {name} failed verification; nothing eroded (backup left in {backup})")
    (backup / "manifest.json").write_text(json.dumps({"aliases": {a: p.name for a, p in files.items()}, "files": hashes, "created": backup.name}, indent=1))
    print(f"backed up and verified {len(real)} files in {backup}")

    for path in real:
        out, _, _ = erode_white(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), args.kernel, args.iterations)
        cv2.imwrite(str(path), out)
    MARK.write_text(json.dumps({"iterations": args.iterations, "kernel": args.kernel, "backup": str(backup), "files": [p.name for p in real]}, indent=1))
    print(f"eroded {len(real)} files ({args.iterations} x {args.kernel}x{args.kernel}, white only); mark {MARK}")


if __name__ == "__main__":
    main()
