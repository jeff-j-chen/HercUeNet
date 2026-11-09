"""compute every algorithm map on the three sites -> maps.npz (box crops only)."""
import time
from multiprocessing import Pool

import numpy as np

from algos import ALGOS, composite, run_algo
from common import SITES, box

_c = {}


def job(a):
    nm, algo = a
    if nm not in _c:
        _c[nm] = composite(nm)
    z, m, off = _c[nm]
    t = time.time()
    try:
        return nm, algo, box(run_algo(algo, z, m), off), time.time() - t
    except Exception as e:  # noqa: BLE001
        return nm, algo, repr(e), 0


if __name__ == "__main__":
    t0 = time.time()
    with Pool(24) as p:
        res = p.map(job, [(nm, a) for nm in SITES for a in ALGOS], chunksize=1)
    out = {}
    for nm, a, r, dt in res:
        if isinstance(r, str):
            print("FAIL", nm, a, r)
        else:
            out[f"{nm}|{a}"] = r
    comp = {}
    for nm in SITES:
        z, m, off = composite(nm)
        comp[f"{nm}|composite"] = box(z, off)
    np.savez_compressed("maps.npz", **out, **comp)
    print(len(out), "maps", time.time() - t0, "s; slowest", sorted(((dt, a) for _, a, _, dt in res), reverse=True)[:4])
