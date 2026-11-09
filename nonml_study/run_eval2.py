"""score the second algorithm family on the three sites against the hand-drawn masks; keeps the maps."""
import json, sys, time
from multiprocessing import Pool
import cv2, numpy as np
from PIL import Image
from scipy import ndimage as ndi
from algos import composite
from algos2 import ALGOS2, clip
from common import SITES, box
from metrics import auc, unsupervised

GTF = {"s1_easy": "n", "s2_medium": "o", "s3_hard": "m"}
GT = {k: np.array(Image.open(f"/vesuvius/{f}.jpg").convert("L")) > 127 for k, f in GTF.items()}
COMP = {k: composite(k) for k in SITES}


def dice_cov(x, g, tol=6):
    """dice between the top-|g| pixels of the map and the drawn letter (dilated by tol px for hand-drawing slack)."""
    k = int(g.sum())
    thr = np.partition(x.ravel(), -k)[-k]
    b = x >= thr
    gd = ndi.binary_dilation(g, iterations=tol)
    return float(2 * (b & gd).sum() / (b.sum() + gd.sum()))


def score(x, g):
    pos, neg = ndi.binary_erosion(g, iterations=5), ndi.distance_transform_edt(~g) > 25
    z = (x - x[neg].mean()) / (x[neg].std() + 1e-9)
    return {"auc": auc(x, pos, neg), "d": float(z[pos].mean()), "dice": dice_cov(x, g), "unsup": unsupervised(x)}


def job(a):
    s, name = a
    cv2.setNumThreads(1)
    z, m, off = COMP[s]
    try:
        x = box(ALGOS2[name](clip(z)), off).astype(np.float32)
    except Exception as e:  # noqa: BLE001
        return s, name, repr(e), None
    return s, name, score(x, GT[s]), x.astype(np.float16)


if __name__ == "__main__":
    t0 = time.time()
    names = sys.argv[1:] or list(ALGOS2)
    with Pool(24) as p:
        res = p.map(job, [(s, a) for s in SITES for a in names], chunksize=1)
    out, maps = {}, {}
    for s, a, r, x in res:
        if isinstance(r, str):
            print("FAIL", s, a, r); continue
        out.setdefault(a, {})[s] = r
        maps[f"{s}|{a}"] = x
    np.savez_compressed("maps2.npz", **maps)
    json.dump(out, open("eval2.json", "w"))
    rows = sorted(out.items(), key=lambda kv: -np.mean([kv[1][s]["auc"] for s in SITES]))
    print("%-22s" % "algo", "   ".join("%-8s auc  dice" % s[:8] for s in SITES), "  mean_auc mean_dice")
    for a, r in rows:
        print("%-22s" % a, "   ".join("%.3f %.3f " % (r[s]["auc"], r[s]["dice"]) for s in SITES), "  %.3f %.3f" % (np.mean([r[s]["auc"] for s in SITES]), np.mean([r[s]["dice"] for s in SITES])))
    print("t", time.time() - t0)
