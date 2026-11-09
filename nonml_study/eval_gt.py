"""score every algorithm map against the hand-drawn letter masks (m.jpg / n.jpg / o.jpg)."""
import json
import cv2, numpy as np
from PIL import Image
from scipy import ndimage as ndi
from algos import ALGOS
from metrics import auc, unsupervised

GTF = {"s1_easy": "n", "s2_medium": "o", "s3_hard": "m"}
D = np.load("maps.npz")
GT = {k: np.array(Image.open(f"/vesuvius/{f}.jpg").convert("L")) > 127 for k, f in GTF.items()}


def regions(g):
    pos = ndi.binary_erosion(g, iterations=5)
    neg = ndi.distance_transform_edt(~g) > 25
    return pos, neg


def shift_tol(x, g, pos, neg, r=12):
    """best d' over shifts of the map within +-r px: tolerates a slightly off hand drawing."""
    best = -9
    for dy in range(-r, r + 1, 4):
        for dx in range(-r, r + 1, 4):
            xs = np.roll(x, (dy, dx), (0, 1))
            z = (xs - xs[neg].mean()) / (xs[neg].std() + 1e-9)
            best = max(best, float(z[pos].mean()))
    return best


rows = []
for a in ALGOS + ["composite"]:
    r = {"algo": a}
    for s, g in GT.items():
        x = D[f"{s}|{a}"].astype(np.float32)
        pos, neg = regions(g)
        z = (x - x[neg].mean()) / (x[neg].std() + 1e-9)
        r[s] = {"auc": auc(x, pos, neg), "d": float(z[pos].mean()), "dshift": shift_tol(x, g, pos, neg),
                "unsup": unsupervised(x)}
    rows.append(r)
json.dump(rows, open("eval_gt.json", "w"))
rows.sort(key=lambda r: -np.mean([r[s]["auc"] for s in GT]))
print("%-18s" % "algo", " ".join("%-19s" % s for s in GT), "mean")
for r in rows:
    print("%-18s" % r["algo"], " ".join("auc %.3f d %5.2f   " % (r[s]["auc"], r[s]["d"]) for s in GT), "%.3f" % np.mean([r[s]["auc"] for s in GT]))
