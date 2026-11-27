"""Add / remove paths of the passphrase-protected tifxyz archive (utils/tifxyz_0846A.tar.enc) and re-encrypt it.

  TIFXYZ_PASSPHRASE=... python3 -m utils.seal_tifxyz --add tifxyz/<dir> [tifxyz/<dir> ...] [--remove tifxyz/<dir> ...] [--list] [--verify]

The archive is openssl aes-256-cbc / pbkdf2 (600000 iterations) over a tar of tifxyz/<dir> paths (utils/unlock_tifxyz_0846A.py extracts it).
--add replaces any copy of that directory already in the archive; --remove drops one. A backup is written next to the archive as
.tar.enc.bak until --verify (a fresh decryption compared byte for byte with the working copies of everything that was added) passed.
The passphrase is read from TIFXYZ_PASSPHRASE or --pass-file; it is never printed.
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARCHIVE = REPO / "utils" / "tifxyz_0846A.tar.enc"
OPENSSL = ["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-iter", "600000", "-pass", "env:TIFXYZ_PW"]


def passphrase(args) -> str:
    if args.pass_file:
        return Path(args.pass_file).read_text().rstrip("\n")
    value = os.environ.get("TIFXYZ_PASSPHRASE")
    if not value:
        raise SystemExit("set TIFXYZ_PASSPHRASE or --pass-file")
    return value


def decrypt(archive: Path, dest: Path, pw: str) -> None:
    env = {**os.environ, "TIFXYZ_PW": pw}
    with open(archive, "rb") as source:
        openssl = subprocess.Popen([*OPENSSL[:2], "-d", *OPENSSL[2:]], stdin=source, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        tar = subprocess.Popen(["tar", "-xf", "-", "-C", str(dest), "--no-same-owner"], stdin=openssl.stdout, stderr=subprocess.DEVNULL)
        openssl.stdout.close()
        tar_rc, openssl_rc = tar.wait(), openssl.wait()
    if tar_rc or openssl_rc:
        raise SystemExit("decryption failed (wrong passphrase?)")


def encrypt(source_dir: Path, target: Path, pw: str) -> None:
    env = {**os.environ, "TIFXYZ_PW": pw}
    tar = subprocess.Popen(["tar", "-cf", "-", "-C", str(source_dir), "tifxyz"], stdout=subprocess.PIPE)
    with open(target, "wb") as out:
        openssl = subprocess.Popen(OPENSSL, stdin=tar.stdout, stdout=out, stderr=subprocess.DEVNULL, env=env)
        tar.stdout.close()
        openssl_rc, tar_rc = openssl.wait(), tar.wait()
    if tar_rc or openssl_rc:
        raise SystemExit("encryption failed")


def tree(path: Path):
    return sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--add", nargs="*", default=[])
    parser.add_argument("--remove", nargs="*", default=[])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verify", action="store_true", help="only decrypt and compare the --add paths with the working copies")
    parser.add_argument("--pass-file")
    args = parser.parse_args()
    pw = passphrase(args)
    with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
        work = Path(tmp)
        decrypt(ARCHIVE, work, pw)
        if args.list:
            print("\n".join(sorted(p.name for p in (work / "tifxyz").iterdir())))
            return
        if args.verify:
            bad = 0
            for rel in args.add:
                comparison = filecmp.dircmp(REPO / rel, work / rel)
                files = [Path(r) / f for r, _, fs in os.walk(REPO / rel) for f in fs]
                for f in files:
                    other = work / f.relative_to(REPO)
                    if not other.is_file() or not filecmp.cmp(f, other, shallow=False):
                        bad += 1
                        print("DIFFERS", f)
                print(f"{rel}: {len(files)} files, {'ok' if not bad else 'MISMATCH'}")
            raise SystemExit(1 if bad else 0)
        for rel in [*args.remove, *args.add]:
            shutil.rmtree(work / rel, ignore_errors=True)
        for rel in args.add:
            if not (REPO / rel).is_dir():
                raise SystemExit(f"{rel} is not a directory in the working tree")
            shutil.copytree(REPO / rel, work / rel)
        shutil.copyfile(ARCHIVE, ARCHIVE.with_suffix(".enc.bak"))
        new = ARCHIVE.with_suffix(".enc.new")
        encrypt(work, new, pw)
        os.replace(new, ARCHIVE)
    print(f"sealed {len(args.add)} path(s), removed {len(args.remove)}; archive {ARCHIVE.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
