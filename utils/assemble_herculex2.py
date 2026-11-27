"""Assemble the regenerated HerculeX x export of 2026-11-21 (tifxyz/new_x.zip): 45 PHerc0846A + 214 PHerc0846B sheets (utils/mix6.py herculex2_sheets).

  python3 -m utils.assemble_herculex2            # unzip, register the reversed meshes, render, labels, flat aliases, norms, anchors (what is missing)
  python3 -m utils.assemble_herculex2 --check    # report only
  python3 -m utils.assemble_herculex2 --only 20261121010001 --force

The old x sheets (utils/assemble_herculex.py, PHerc0846A_marks_NNN ...) stay untouched: these are built under new names, zarr id hx2_<mesh name> and aliases
202611210100NN (A) / 202611210200NN (B), because a training may still be reading the old ones. Nothing here is part of a corpus yet.
Per sheet (tifxyz/HerculeX_labels_0846{A,B}/): tifxyz/<name>/ rendered from the raw volume (A or B), uint8, the whole mesh grid (the labels cover the padded grid), depth reversed for
the meshes in flipped_normals.txt (utils/depth_reversed.json gets their hx2_ ids); inklabels/<name>.png -> dilated_inklabels/<alias>.png and an all-255 train_masks/<alias>.png;
the flat depth-13 alias (zarr / mask / label links, surface_labels/<alias>, norm and surface anchors). The rendered shape is checked against manifest.json (expected_zarr_shape).
No asf cues (HercUeNet is retired) and no exclusion masks (the meshes are cut away from the prize sheet already; earmarked.txt lists the ones within 56 voxels of it).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from utils import mix6  # noqa: E402
from utils.assemble_herculex import VOLUMES  # noqa: E402

ZIP = REPO / "tifxyz" / "new_x.zip"
LIST = REPO / "utils" / "depth_reversed.json"
GUARD_GB = 60  # stop when the new zarrs exceed this (the manifests expect ~23 GB uncompressed)


def unzip() -> None:
    """the export folders under tifxyz/: from new_x.zip when it is there, else from the passphrase-protected archive utils/tifxyz_0846A.tar.enc
    (TIFXYZ_PASSPHRASE, else a prompt)."""
    if all((REPO / "tifxyz" / folder).is_dir() for folder, _p, _b in mix6.HERCULEX2_EXPORTS.values()):
        return
    if ZIP.is_file():
        with zipfile.ZipFile(ZIP) as archive:
            archive.extractall(REPO / "tifxyz")
        print(f"[hx2] extracted {ZIP}", flush=True)
        return
    import getpass

    from utils import unlock_tifxyz_0846A as vault

    passphrase = os.getenv("TIFXYZ_PASSPHRASE") or (getpass.getpass("[hx2] tifxyz passphrase: ") if sys.stdin.isatty() else "")
    if not passphrase:
        raise SystemExit("set TIFXYZ_PASSPHRASE (or run from a terminal) to unlock the sealed meshes")
    vault.unlock(str(REPO), passphrase)


def sheets() -> dict:
    return mix6.herculex2_sheets()


def export_of(name: str) -> str:
    return "A" if name.startswith("PHerc0846A") else "B"


def folder_of(name: str) -> Path:
    return REPO / "tifxyz" / mix6.HERCULEX2_EXPORTS[export_of(name)][0]


def manifest_of(name: str) -> dict:
    return next(row for row in json.loads((folder_of(name) / "manifest.json").read_text()) if row["name"] == name)


def register_reversed() -> int:
    """add the hx2_ ids of the meshes in flipped_normals.txt to utils/depth_reversed.json (only adds)."""
    wanted = set()
    for folder, _prefix, _base in mix6.HERCULEX2_EXPORTS.values():
        for line in (REPO / "tifxyz" / folder / "flipped_normals.txt").read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                wanted.add(f"hx2_{line.strip()}")
    data = json.loads(LIST.read_text(encoding="utf-8"))
    new = sorted(wanted - set(data["zids"]))
    if new:
        data["zids"] = sorted(set(data["zids"]) | wanted)
        temporary = LIST.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        os.replace(temporary, LIST)
    return len(new)


def render_one(job) -> tuple:
    alias, workers, force = job
    import assemble_test_segments as t

    render, name = sheets()[alias]
    os.makedirs(t.EXTRAS_MASK_DIR, exist_ok=True)
    folder = mix6.HERCULEX2_EXPORTS[export_of(name)][0]
    zid, status = t.render_fragment(
        render, f"{folder}/tifxyz/{name}", *VOLUMES[export_of(name)], workers, mix6.zarr_dir().as_posix(), t.SCRIPT_DIR,
        t.DEFAULT_CHUNK_DEPTH, t.DEFAULT_CHUNK_Y, t.DEFAULT_CHUNK_X, force=force, mask_dir=t.EXTRAS_MASK_DIR,
        output_dtype=t.EXTRAS_DTYPE, drop_cache=True, crop_valid=False)
    return alias, zid, status


def check_shape(alias) -> str:
    """'' when the render, its mask and its label have the manifest's shapes, else the problem."""
    import zarr

    render, name = sheets()[alias]
    expected = tuple(manifest_of(name)["expected_zarr_shape"])
    shape = tuple(zarr.open(str(mix6.zarr_dir() / f"{render}.zarr"), mode="r").shape)
    if shape != expected:
        return f"zarr {shape} != manifest {expected}"
    mask = cv2.imread(str(REPO / "masks" / "extras" / f"{render}.png"), cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(str(folder_of(name) / "inklabels" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None or label is None or mask.shape != label.shape or mask.shape != expected[1:]:
        return f"mask {None if mask is None else mask.shape} / label {None if label is None else label.shape} / manifest {expected[1:]}"
    return ""


def alias_one(alias) -> str:
    import assemble_test_segments as t

    render, name = sheets()[alias]
    link = Path(t.MASK_DIR) / f"{render}.png"
    if not link.is_symlink() and not link.exists():
        os.symlink(os.path.join(t.EXTRAS_SUBDIR, f"{render}.png"), link)
    problem = check_shape(alias)
    if problem:
        return f"SHAPE {problem}"
    label_path = folder_of(name) / "inklabels" / f"{name}.png"
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if not (label >= 48).any():
        return "label without ink"
    shutil.copyfile(label_path, REPO / "dilated_inklabels" / f"{alias}.png")
    cv2.imwrite(str(REPO / "train_masks" / f"{alias}.png"), np.full(label.shape, 255, np.uint8))
    return t.make_flat_alias(render, str(alias), mix6.zarr_dir().as_posix())


def new_size_gb() -> float:
    total = 0
    for render, _name in sheets().values():
        path = mix6.zarr_dir() / f"{render}.zarr"
        if path.is_dir():
            out = subprocess.run(["du", "-s", "--block-size=1M", str(path)], capture_output=True, text=True).stdout.split()
            total += int(out[0]) if out else 0
    return total / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report what is missing, change nothing")
    parser.add_argument("--only", nargs="*", type=int, help="aliases to build (default all)")
    parser.add_argument("--force", action="store_true", help="re-render")
    parser.add_argument("--jobs", type=int, default=2, help="sheets rendered in parallel")
    parser.add_argument("--workers", type=int, default=12, help="download threads per sheet")
    args = parser.parse_args()
    if not args.check:
        unzip()
    table = sheets()
    aliases = args.only or list(table)
    if args.check:
        problems = [f"{a}: missing {mix6.zarr_dir() / (table[a][0] + '.zarr')}" for a in aliases if not (mix6.zarr_dir() / f"{table[a][0]}.zarr").is_dir()]
        problems += [f"{a}: missing {p}" for a in aliases for p in (REPO / "dilated_inklabels" / f"{a}.png", REPO / "surface_labels" / str(a) / "depth.npy") if not p.exists()]
        print("\n".join(problems[:50]), f"\n{len(problems)} problem(s) for {len(aliases)} sheet(s)")
        return
    added = register_reversed()
    reversed_now = sum(table[a][0] in json.loads(LIST.read_text())["zids"] for a in aliases)
    print(f"[hx2] {len(aliases)} sheets ({reversed_now} depth-reversed; {added} ids newly registered)", flush=True)
    todo = [a for a in aliases if args.force or not (mix6.zarr_dir() / f"{table[a][0]}.zarr").is_dir()
            or not (REPO / "masks" / "extras" / f"{table[a][0]}.png").is_file()]
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for batch in range(0, len(todo), 16):
            for alias, zid, status in pool.map(render_one, [(a, args.workers, args.force) for a in todo[batch:batch + 16]]):
                print(f"[hx2] render {alias} {zid}: {status}", flush=True)
                if not status.startswith("OK"):
                    raise SystemExit(f"render of {zid} failed: {status}")
            done += len(todo[batch:batch + 16])
            size = new_size_gb()
            print(f"[hx2] {done}/{len(todo)} rendered, new zarrs {size:.1f} GB", flush=True)
            if size > GUARD_GB:
                raise SystemExit(f"[hx2] stop: the new zarrs take {size:.0f} GB (> {GUARD_GB}); check the disk before going on")
    bad = []
    for alias in aliases:
        status = alias_one(alias)
        print(f"[hx2] alias {alias} {table[alias][0]}: {status}", flush=True)
        if status != "OK":
            bad.append((alias, status))
    print(f"[hx2] done: {len(aliases) - len(bad)} ok, {len(bad)} problem(s): {bad[:10]}", flush=True)


if __name__ == "__main__":
    main()
