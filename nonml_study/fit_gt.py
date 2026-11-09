"""pseudo ground truth: fit the known letter to the rank-consensus of all maps."""
import cv2, numpy as np
from algos import ALGOS, n
from common import SITES
from metrics import fit_letter, rank01

D = np.load("maps.npz")
out, rows, gts = {}, [], {}
for nm, (_, _, L) in SITES.items():
    cons = np.mean([rank01(D[f"{nm}|{a}"]) for a in ALGOS], 0)
    score, p, gt = fit_letter(n.gblur(cons, 6), L)
    print(nm, L, "zncc %.3f" % score, "params(h/2,aspect,stroke,rot)", p)
    gts[nm] = gt
    def u8(x):
        a, b = np.percentile(x, [1, 99]); return np.clip((x - a) / (b - a) * 255, 0, 255).astype(np.uint8)
    tiles = []
    for img in (cons, D[f"{nm}|composite"], D[f"{nm}|none+asf"]):
        t = cv2.cvtColor(u8(n.gblur(img, 3)), cv2.COLOR_GRAY2BGR)
        cs, _ = cv2.findContours((gt > 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(t, cs, -1, (0, 0, 255), 1)
        tiles.append(t)
    rows.append(np.hstack(tiles))
np.savez("gt_consensus.npz", **gts)
cv2.imwrite("/tmp/gt_fit.jpg", cv2.resize(np.vstack(rows), None, fx=0.65, fy=0.65, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
