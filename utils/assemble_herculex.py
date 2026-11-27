"""Assemble the HerculeX marks export (mix28x): 54 PHerc0846A patches + 17 PHerc0846B patches (utils/mix6.py HERCULEX_SHEETS).

  python3 -m utils.assemble_herculex            # labels, renders, flat aliases, anchors, asf cues (everything that is missing)
  python3 -m utils.assemble_herculex --check    # report only
  python3 -m utils.assemble_herculex --only 20261118010001 --force

Per sheet (tifxyz/HerculeX_marks_export_0846{A,B}/):
  tifxyz/<name>/        rendered from the raw 0846A volume (the 0846B sheets from the 0846B volume) like the fibre extras: uint8, zarr id =
                        mesh name, mask in masks/extras linked into masks/, the WHOLE mesh grid kept (the labels cover the padded grid).
                        Meshes listed in flipped_normals.txt are in utils/depth_reversed.json and rendered with reversed depth.
  inklabels/<name>.png  -> dilated_inklabels/<alias>.png (0 / 255), train_masks/<alias>.png = all 255 (like every 0846A sheet)
  flat depth-13 alias   zarr / mask / label links, surface_labels/<alias>, norm and surface anchors (assemble_test_segments.make_flat_alias)
  asf cues              cue_asf_morph, _dm1, _dp1 (utils/mix6.py ensure_cues)
No exclusion masks: the meshes are already cut away from the prize sheet.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from utils import mix6  # noqa: E402

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
VOLUMES = {
    "A": (f"{BUCKET}/PHerc0846A/volumes/20250728152254-9.362um-1.2m-113keV-masked.zarr/0", "14019,7726,7726"),
    "B": (f"{BUCKET}/PHerc0846B/volumes/20250804142305-9.362um-1.2m-113keV-masked.zarr/0", "13926,7859,7859"),
}


SHEETS = {**mix6.HERCULEX_SHEETS, **mix6.NEW_HAND_SHEETS}  # the two hand-labelled 0846B prize sheets are built the same way


def part(alias) -> str:
    """the scroll half (raw volume) of a sheet: A or B (the blank export is 0846B too)."""
    return "A" if SHEETS[int(alias)][0].startswith("PHerc0846A") else "B"


def label_source(alias) -> Path:
    name = SHEETS[int(alias)][0]
    if int(alias) in mix6.NEW_HAND_SHEETS:  # the label was copied to dilated_inklabels/<name>.png
        return REPO / "dilated_inklabels" / f"{name}.png"
    return REPO / "tifxyz" / mix6.HERCULEX_EXPORTS[mix6.export_key(name)][0] / "inklabels" / f"{name}.png"


def herculex_mesh(alias) -> str:
    return mix6.herculex_mesh(alias) if int(alias) in mix6.HERCULEX_SHEETS else SHEETS[int(alias)][0]


def render_one(job) -> tuple:
    """one sheet's render (a worker process): downloads its raw chunks, writes the zarr + mask, frees the chunk cache."""
    alias, workers, force = job
    import assemble_test_segments as t

    render, _name = SHEETS[alias]
    os.makedirs(t.EXTRAS_MASK_DIR, exist_ok=True)
    zid, status = t.render_fragment(
        render, herculex_mesh(alias), *VOLUMES[part(alias)], workers, mix6.zarr_dir().as_posix(), t.SCRIPT_DIR,
        t.DEFAULT_CHUNK_DEPTH, t.DEFAULT_CHUNK_Y, t.DEFAULT_CHUNK_X, force=force, mask_dir=t.EXTRAS_MASK_DIR,
        output_dtype=t.EXTRAS_DTYPE, drop_cache=True, crop_valid=False)
    return alias, zid, status


def write_labels(alias) -> str:
    """dilated_inklabels/<alias>.png from the export, train_masks/<alias>.png all 255 (only when the mask shape matches)."""
    render = SHEETS[alias][0]
    mask = cv2.imread(str(REPO / "masks" / "extras" / f"{render}.png"), cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(str(label_source(alias)), cv2.IMREAD_GRAYSCALE)
    if mask is None or label is None:
        return "missing mask or label"
    if mask.shape != label.shape:
        return f"mask {mask.shape} != label {label.shape}"
    if not (label >= 48).any():
        return "label without ink"
    shutil.copyfile(label_source(alias), REPO / "dilated_inklabels" / f"{alias}.png")
    cv2.imwrite(str(REPO / "train_masks" / f"{alias}.png"), np.full(mask.shape, 255, np.uint8))
    return "ok"


def alias_one(alias) -> str:
    import assemble_test_segments as t

    render = SHEETS[alias][0]
    link = Path(t.MASK_DIR) / f"{render}.png"
    if not link.is_symlink() and not link.exists():
        os.symlink(os.path.join(t.EXTRAS_SUBDIR, f"{render}.png"), link)
    problem = write_labels(alias)
    if problem != "ok":
        return problem
    return t.make_flat_alias(render, str(alias), mix6.zarr_dir().as_posix())


def sealed_dirs(aliases) -> list[str]:
    """mesh folders (under tifxyz/) of the sheets that are only in the passphrase-protected archive utils/tifxyz_0846A.tar.enc: the export folders
    (meshes + export labels + manifests) of the HerculeX sheets, and the meshes of the two hand-labelled 0846B prize sheets."""
    found = {mix6.HERCULEX_EXPORTS[mix6.export_key(SHEETS[int(a)][0])][0] for a in aliases if int(a) in mix6.HERCULEX_SHEETS}
    found |= {SHEETS[int(a)][0] for a in aliases if int(a) in mix6.NEW_HAND_SHEETS}
    return sorted(found)


def unlock_missing(aliases) -> None:
    """extract the archive when a mesh of the wanted sheets is not in tifxyz/ (TIFXYZ_PASSPHRASE, else a prompt)."""
    import getpass

    from utils import unlock_tifxyz_0846A as vault

    absent = [d for d in sealed_dirs(aliases) if not (REPO / "tifxyz" / d).is_dir()]
    if not absent:
        return
    print(f"[herculex] {len(absent)} folder(s) are only in {vault.ARCHIVE}: {absent}", flush=True)
    passphrase = os.getenv("TIFXYZ_PASSPHRASE") or (getpass.getpass("[herculex] tifxyz passphrase: ") if sys.stdin.isatty() else "")
    if not passphrase:
        raise SystemExit("set TIFXYZ_PASSPHRASE (or run from a terminal) to unlock the sealed meshes")
    vault.unlock(str(REPO), passphrase)


def missing(aliases) -> list[str]:
    return mix6.status(aliases, derived=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report what is missing, change nothing")
    parser.add_argument("--only", nargs="*", type=int, help="aliases to build (default all 71)")
    parser.add_argument("--force", action="store_true", help="re-render")
    parser.add_argument("--jobs", type=int, default=3, help="sheets rendered in parallel")
    parser.add_argument("--workers", type=int, default=24, help="download threads per sheet")
    parser.add_argument("--cue-workers", type=int, default=6)
    parser.add_argument("--no-cues", action="store_true", help="stop after the aliases, anchors and norms")
    args = parser.parse_args()
    aliases = args.only or list(SHEETS)
    if not args.check:
        unlock_missing(aliases)
    if args.check:
        problems = missing(aliases)
        print("\n".join(problems[:50]), f"\n{len(problems)} problem(s) for {len(aliases)} sheet(s)")
        return
    from utils.depth_flip import reversed_ids

    flipped = sum(SHEETS[a][0] in reversed_ids() for a in aliases)
    print(f"[herculex] {len(aliases)} sheets ({flipped} depth-reversed)", flush=True)
    todo = [a for a in aliases if args.force or not (mix6.zarr_dir() / f"{SHEETS[a][0]}.zarr").is_dir()
            or not (REPO / "masks" / "extras" / f"{SHEETS[a][0]}.png").is_file()]
    if todo:
        import multiprocessing

        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            for alias, zid, status in pool.map(render_one, [(a, args.workers, args.force) for a in todo]):
                print(f"[herculex] render {alias} {zid}: {status}", flush=True)
                if not status.startswith("OK"):
                    raise SystemExit(f"render of {zid} failed: {status}")
    for alias in aliases:
        print(f"[herculex] alias {alias} {SHEETS[alias][0]}: {alias_one(alias)}", flush=True)
    mix6.ensure_anchors(aliases)
    if args.no_cues:
        return
    mix6.ensure_cues(aliases, workers=args.cue_workers)
    problems = missing(aliases)
    print("\n".join(problems[:50]), f"\n[herculex] done: {len(problems)} problem(s)", flush=True)


if __name__ == "__main__":
    main()
