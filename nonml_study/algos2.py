"""second family of non-ML maps, all fed by the same 8-layer max composite (clipped robust z)."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

import algos as A
from algos import n, u8


def clip(z):
    return np.clip(z, -3, 3).astype(np.float32)


def asf_r(x, radii, pre=4, post=10):
    g = n.gblur(x, pre) if pre else x
    return n.gblur(n.asf(g, radii), post) if post else n.asf(g, radii)


def recon_asf(x, radii=(4, 8, 12, 16), pre=4, post=8):
    g = n.gblur(x, pre)
    for r in radii:
        k = n.disk(r)
        g = cv2.erode(g, k)
        g = morphology.reconstruction(g, n.gblur(x, pre) if False else g_dil(g, k, x, pre), method="dilation").astype(np.float32)
    return n.gblur(g, post)


def g_dil(g, k, x, pre):
    return np.minimum(cv2.dilate(g, k), n.gblur(x, pre)).astype(np.float32)


def lcn(x, sigma):
    mu = n.gblur(x, sigma)
    sd = np.sqrt(np.maximum(n.gblur((x - mu) ** 2, sigma), 1e-4))
    return ((x - mu) / sd).astype(np.float32)


def median(x, size, post=6):
    return n.gblur(cv2.medianBlur(u8(n.gblur(x, 3)), size).astype(np.float32), post)


def oriented(x, sl=30, ss=7, mode="max", nang=12):
    """elongated gaussian line detector over orientations: coherent strokes win, isotropic specks do not."""
    S = int(sl * 3) | 1
    u, v = np.meshgrid(np.arange(S) - S // 2, np.arange(S) - S // 2)
    rs = []
    for a in np.arange(nang) * np.pi / nang:
        uu, vv = u * np.cos(a) + v * np.sin(a), -u * np.sin(a) + v * np.cos(a)
        k = np.exp(-(uu**2) / (2 * sl**2) - (vv**2) / (2 * ss**2)).astype(np.float32)
        rs.append(cv2.filter2D(x, -1, k / k.sum(), borderType=cv2.BORDER_REFLECT))
    rs = np.array(rs)
    return rs.max(0) if mode == "max" else rs.max(0) - rs.mean(0)


def bilateral(x, it=3, sc=0.5, ss=12):
    g = n.gblur(x, 2)
    for _ in range(it):
        g = cv2.bilateralFilter(g, 31, sc, ss)
    return g


def areaopen(x, area, pre=4, post=6):
    r = morphology.area_closing(morphology.area_opening(u8(n.gblur(x, pre)), area), area)
    return n.gblur(r.astype(np.float32), post)


def single_open(x, r, pre=4, post=8):
    g = n.gblur(x, pre)
    k = n.disk(r)
    return n.gblur(cv2.morphologyEx(g, cv2.MORPH_OPEN, k), post)


R6, R12, R16, R20 = (2, 4, 6), (2, 4, 6, 8, 10, 12), (3, 6, 9, 12, 15, 18), (4, 8, 12, 16, 20, 24)
ALGOS2 = {
    # asf scale / smoothing sweep
    "asf_r6": lambda x: asf_r(x, R6),
    "asf_r12": lambda x: asf_r(x, R12),
    "asf_r16": lambda x: asf_r(x, R16),
    "asf_r20": lambda x: asf_r(x, R20),
    "asf_r12_pre8": lambda x: asf_r(x, R12, pre=8),
    "asf_r12_pre2": lambda x: asf_r(x, R12, pre=2),
    "asf_r12_post20": lambda x: asf_r(x, R12, post=20),
    "asf_r16_post16": lambda x: asf_r(x, R16, post=16),
    "asf_r12_nopost": lambda x: asf_r(x, R12, post=0),
    "recon_asf": lambda x: recon_asf(x),
    "open_r8": lambda x: single_open(x, 8),
    "open_r12": lambda x: single_open(x, 12),
    "open_r16": lambda x: single_open(x, 16),
    # local contrast normalisation first (bright regions)
    "lcn50+asf_r12": lambda x: asf_r(lcn(x, 50), R12),
    "lcn100+asf_r12": lambda x: asf_r(lcn(x, 100), R12),
    "lcn100+asf_r16": lambda x: asf_r(lcn(x, 100), R16),
    "lcn100+blur": lambda x: n.gblur(lcn(x, 100), 8),
    # median family
    "median15": lambda x: median(x, 15),
    "median25": lambda x: median(x, 25),
    "median35": lambda x: median(x, 35),
    "median51": lambda x: median(x, 51),
    # oriented coherence
    "orient_l20": lambda x: oriented(n.gblur(x, 3), 20, 7),
    "orient_l35": lambda x: oriented(n.gblur(x, 3), 35, 7),
    "orient_l35_ridge": lambda x: oriented(n.gblur(x, 3), 35, 7, "ridge"),
    "asf+orient_l20": lambda x: oriented(asf_r(x, R12, post=0), 20, 7),
    "asf+orient_l35": lambda x: oriented(asf_r(x, R12, post=0), 35, 7),
    "asf+orient_l35_ridge": lambda x: oriented(asf_r(x, R12, post=0), 35, 7, "ridge"),
    "med25+orient_l30": lambda x: oriented(median(x, 25, 0), 30, 7),
    # others
    "bilateral": lambda x: bilateral(x),
    "areaopen400": lambda x: areaopen(x, 400),
    "areaopen2000": lambda x: areaopen(x, 2000),
    "areaopen4000": lambda x: areaopen(x, 4000),
    "asf+areaopen2000": lambda x: areaopen(asf_r(x, R12, post=0), 2000),
    "median25+asf": lambda x: asf_r(n.gblur(cv2.medianBlur(u8(x), 25).astype(np.float32), 0), R12),
}
