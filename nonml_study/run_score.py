"""every algorithm on 3 sites (as given + 180-degree-rotated frame) and on random null boxes; per-map letter-fit and blob metrics."""
import json, time
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image

import common
from algos import ALGOS, composite_xy, run_algo
from common import BOX, SITES, box
from metrics import fast_fit, unsupervised

H, W = 2861, 4621


def boxes():
    t = {}
    for nm, (x, y, L) in SITES.items():
        t[nm] = (x, y, L)
        t[nm + "_rot"] = (W - x - BOX, H - y - BOX, L)
    mask = np.array(Image.open(common.MASK).convert("L")) > 0
    rng = np.random.default_rng(7)
    null = {}
    while len(null) < 14:
        x, y = int(rng.integers(0, W - BOX)), int(rng.integers(0, H - BOX))
        if mask[y:y + BOX, x:x + BOX].mean() < 1.0:
            continue
        if any(abs(x - tx) < 600 and abs(y - ty) < 600 for tx, ty, _ in t.values()):
            continue
        if any(abs(x - nx) < 520 and abs(y - ny) < 520 for nx, ny, _ in null.values()):
            continue
        null[f"null{len(null):02d}"] = (x, y, "-")
    return t, null


def job(item):
    name, (x, y, L), keep = item
    cv2.setNumThreads(1)
    z, m, off = composite_xy(x, y)
    res, maps = {}, {}
    for a in ALGOS:
        try:
            mp = box(run_algo(a, z, m), off)
        except Exception as e:  # noqa: BLE001
            res[a] = {"error": repr(e)}
            continue
        res[a] = {"fit": {l: fast_fit(mp, l) for l in "NOM"}, "unsup": unsupervised(mp)}
        if keep:
            maps[a] = mp.astype(np.float16)
    maps["composite"] = box(z, off).astype(np.float16)
    common._cache.clear()
    return name, res, maps


if __name__ == "__main__":
    t0 = time.time()
    tg, nl = boxes()
    items = [(k, v, True) for k, v in tg.items()] + [(k, v, False) for k, v in nl.items()]
    with Pool(20) as p:
        out = p.map(job, items, chunksize=1)
    json.dump({"targets": tg, "null": nl, "scores": {n: r for n, r, _ in out}}, open("scores.json", "w"))
    np.savez_compressed("maps_targets.npz", **{f"{n}|{a}": v for n, _, m in out for a, v in m.items()})
    print("done", time.time() - t0)
