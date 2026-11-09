"""Decrypt the passphrase-protected PHerc0846A tifxyz archive into tifxyz/.

  python3 -m utils.unlock_tifxyz_0846A              # prompts for the passphrase (or set TIFXYZ_PASSPHRASE)
  python3 -m utils.unlock_tifxyz_0846A --dest DIR   # extract under DIR instead of the repo root

utils/tifxyz_0846A.tar.enc is openssl aes-256-cbc / pbkdf2 (600000 iterations) over a tar of tifxyz/<dir> paths.
"""
import argparse, getpass, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(HERE, "tifxyz_0846A.tar.enc")


def unlock(dest, passphrase):
    os.makedirs(dest, exist_ok=True)
    env = {**os.environ, "TIFXYZ_PW": passphrase}
    with open(ARCHIVE, "rb") as source:
        openssl = subprocess.Popen(["openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-iter", "600000", "-pass", "env:TIFXYZ_PW"],
                                   stdin=source, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        tar = subprocess.Popen(["tar", "-xf", "-", "-C", dest, "--no-same-owner"], stdin=openssl.stdout, stderr=subprocess.DEVNULL)
        openssl.stdout.close()
        tar_rc, openssl_rc = tar.wait(), openssl.wait()
    if tar_rc or openssl_rc:
        raise ValueError("decryption failed (wrong passphrase?)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default=os.path.dirname(HERE))
    args = parser.parse_args()
    try:
        unlock(args.dest, os.getenv("TIFXYZ_PASSPHRASE") or getpass.getpass("tifxyz passphrase: "))
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(f"extracted into {os.path.join(args.dest, 'tifxyz')}")
