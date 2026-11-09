"""letter templates, pseudo ground-truth fits and the scoring of an ink map inside a 520 px box."""
import itertools

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata

from algos import n

BOX = 520


def letter_mask(letter, h, aspect, stroke, rot=0.0, size=BOX, scale=1.0):
    """binary stroke mask of a block capital centred in a size x size canvas (supersampled x2, then area-reduced)."""
    S = size * 2
    w = h * aspect
    cx = cy = S / 2
    L, R, T, B = cx - w, cx + w, cy - h, cy + h  # w, h here are half extents (x2 supersampling applied by caller)
    img = np.zeros((S, S), np.uint8)
    t = int(round(stroke * 2))
    pts = {
        "N": [[(L, B), (L, T), (R, B), (R, T)]],
        "M": [[(L, B), (L, T), (cx, cy + 0.35 * h), (R, T), (R, B)]],
    }
    if letter == "O":
        cv2.ellipse(img, (int(cx), int(cy)), (int(w), int(h)), 0, 0, 360, 255, t, cv2.LINE_AA)
    else:
        for poly in pts[letter]:
            cv2.polylines(img, [np.round(np.array(poly)).astype(np.int32)], False, 255, t, cv2.LINE_AA)
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D((cx, cy), rot, 1.0), (S, S))
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def make_params(letter):
    hs = np.arange(75, 215, 20)  # half heights: letter height 150..420
    asp = [0.5, 0.7, 0.9] if letter != "O" else [0.55, 0.75, 1.0]
    st = [20, 30, 40]
    rots = [-8, 0, 8]
    return list(itertools.product(hs, asp, st, rots))


def fit_letter(x, letter, params=None, margin=30):
    """best ZNCC of a letter template over (size, aspect, stroke, rotation, position) in the box map x.
    returns (score, params, placed stroke mask)."""
    x = ((x - x.mean()) / (x.std() + 1e-6)).astype(np.float32)
    best = (-2.0, None, None)
    for p in params or make_params(letter):
        tpl = letter_mask(letter, p[0], p[1], p[2], p[3], size=BOX)
        ys, xs = np.nonzero(tpl > 0.05)
        y0, y1, x0, x1 = max(0, ys.min() - 20), min(BOX, ys.max() + 21), max(0, xs.min() - 20), min(BOX, xs.max() + 21)
        r = cv2.matchTemplate(x, tpl[y0:y1, x0:x1], cv2.TM_CCOEFF_NORMED)
        # the stroke bbox must stay `margin` px inside the box
        ok_y = np.arange(r.shape[0]); ok_y = (ok_y - y0 + ys.min() >= margin) & (ok_y - y0 + ys.max() <= BOX - 1 - margin)
        ok_x = np.arange(r.shape[1]); ok_x = (ok_x - x0 + xs.min() >= margin) & (ok_x - x0 + xs.max() <= BOX - 1 - margin)
        r = np.where(ok_y[:, None] & ok_x[None, :], r, -2)
        i = np.unravel_index(np.argmax(r), r.shape)
        if r[i] > best[0]:
            best = (float(r[i]), p, (int(i[0]) - y0, int(i[1]) - x0, tpl))
    score, p, (dy, dx, tpl) = best
    return score, p, cv2.warpAffine(tpl, np.float32([[1, 0, dx], [0, 1, dy]]), (BOX, BOX))


def rank01(x):
    return (rankdata(x.ravel()).reshape(x.shape) / x.size).astype(np.float32)


def auc(x, pos, neg):
    r = rankdata(np.concatenate([x[pos], x[neg]]))
    npos = pos.sum()
    return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * neg.sum()))


def gt_masks(gt, stroke_core=0.5, bg_gap=20):
    pos = gt > stroke_core
    neg = ~(ndi.binary_dilation(gt > 0.05, iterations=1) | (ndi.distance_transform_edt(gt <= 0.05) < bg_gap))
    return pos, neg


def supervised(x, pos, neg):
    z = (x - x[neg].mean()) / (x[neg].std() + 1e-6)
    return {"auc": auc(x, pos, neg), "dprime": float(z[pos].mean())}


def unsupervised(x, q=85, min_area=150):
    """threshold at the q-th percentile of the box; describe the resulting blobs."""
    b = x > np.percentile(x, q)
    lab, k = ndi.label(b)
    if k == 0:
        return {"largest": 0.0, "leak": 1.0, "ncomp": 0, "solid": 0.0, "mean_area": 0.0}
    areas = ndi.sum(b, lab, range(1, k + 1))
    keep = [i + 1 for i, a in enumerate(areas) if a >= min_area]
    if not keep:
        return {"largest": 0.0, "leak": 1.0, "ncomp": 0, "solid": 0.0, "mean_area": 0.0}
    sl = ndi.find_objects(lab)
    fg = sum(areas[i - 1] for i in keep)
    leak = 0.0
    solid = 0.0
    for i in keep:
        hh, ww = (s.stop - s.start for s in sl[i - 1])
        a = areas[i - 1]
        if max(hh, ww) / max(1, min(hh, ww)) > 4 and a / (hh * ww) < 0.35:
            leak += a
        cnt = cv2.findContours((lab[sl[i - 1]] == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        hull = cv2.contourArea(cv2.convexHull(max(cnt, key=cv2.contourArea))) if cnt else a
        solid += a * min(1.0, a / max(hull, 1))
    return {"largest": float(max(areas[i - 1] for i in keep) / fg), "leak": float(leak / fg), "ncomp": len(keep),
            "solid": float(solid / fg), "mean_area": float(fg / len(keep))}


_TPL = {}


def fast_fit(x, letter, margin=15):
    """best ZNCC over a coarse template grid on the half-resolution map; letter height 200-400 px."""
    h = cv2.resize(x, (BOX // 2, BOX // 2), interpolation=cv2.INTER_AREA)
    h = ((h - h.mean()) / (h.std() + 1e-6)).astype(np.float32)
    if letter not in _TPL:
        asp = [0.5, 0.75] if letter != "O" else [0.6, 0.85]
        lst = []
        for hh, a, st, rot in itertools.product((100, 125, 150, 175, 200), asp, (25, 40), (-6, 0, 6)):
            t = cv2.resize(letter_mask(letter, hh, a, st, rot, size=BOX), (BOX // 2, BOX // 2), interpolation=cv2.INTER_AREA)
            ys, xs = np.nonzero(t > 0.05)
            y0, y1, x0, x1 = max(0, ys.min() - 10), min(BOX // 2, ys.max() + 11), max(0, xs.min() - 10), min(BOX // 2, xs.max() + 11)
            lst.append((t[y0:y1, x0:x1], (y0, x0, ys.min(), ys.max(), xs.min(), xs.max()), (hh, a, st, rot)))
        _TPL[letter] = lst
    best = -2.0
    hb = BOX // 2
    for crop, (y0, x0, ymin, ymax, xmin, xmax), _ in _TPL[letter]:
        r = cv2.matchTemplate(h, crop, cv2.TM_CCOEFF_NORMED)
        iy, ix = np.arange(r.shape[0]), np.arange(r.shape[1])
        ok = ((iy - y0 + ymin >= margin) & (iy - y0 + ymax <= hb - 1 - margin))[:, None] & \
             ((ix - x0 + xmin >= margin) & (ix - x0 + xmax <= hb - 1 - margin))[None, :]
        if ok.any():
            best = max(best, float(r[ok].max()))
    return best
