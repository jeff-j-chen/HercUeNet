"""non-ML ink maps from ONE input: the 8-layer max composite (slices 10-17, local-z normalised, blur 4 px).

each algorithm = fibre-removal stage (pre) + stroke-enhancing stage (post); all lengths are native px.
"""
import sys

sys.path[:0] = ["/vesuvius", "/vesuvius/old", "/vesuvius/nonml_study"]
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.signal.windows import tukey
from skimage import feature, filters, morphology, restoration

import nonml_ink as n
from common import SITES, load_xy

n.F = 1
MIP = range(10, 18)


def composite(name):
    return composite_xy(*SITES[name][:2])


def composite_xy(x, y):
    """robust-z 8-layer max composite of the padded crop; returns (z, mask, box offset)."""
    v, m, off = load_xy(x, y)
    stack = [n.gblur(n.local_z(n.fill(v[d], m), m, 250), 4) for d in MIP]
    c = n.fill(np.max(stack, 0), m)
    return n.robust_z(c, m).astype(np.float32), m, off


def u8(x, lo=0.5, hi=99.5):
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / max(b - a, 1e-6) * 255, 0, 255).astype(np.uint8)


def line_kernel(length, angle):
    k = np.zeros((length, length), np.float32)
    c = length // 2
    dx, dy = np.cos(np.deg2rad(angle)) * c, np.sin(np.deg2rad(angle)) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    return k / k.sum()


# ---------------------------------------------------------------- fibre removal (z, mask) -> z
def pre_none(z, m):
    return z


def pre_clip(z, m):
    return np.clip(z, -3, 3)


def pre_rowcol(z, m):
    """subtract the per-row then per-column median: long axis-aligned fibres are constant along their line."""
    x = np.clip(z, -3, 3).astype(np.float32)
    for _ in range(2):
        x = x - np.nan_to_num(np.nanmedian(np.where(m, x, np.nan), axis=1, keepdims=True))
        x = x - np.nan_to_num(np.nanmedian(np.where(m, x, np.nan), axis=0, keepdims=True))
    return x


def pre_fft(z, m, k0=20, w=1):
    """zero the spectral axes beyond k0 bins: infinite horizontal / vertical lines live there, letter strokes (finite) do not."""
    x = np.clip(z, -3, 3)
    H, W = x.shape
    win = np.outer(tukey(H, 0.2), tukey(W, 0.2)).astype(np.float32)
    F = np.fft.fft2((x - x.mean()) * win)
    KY, KX = np.meshgrid(np.fft.fftfreq(H) * H, np.fft.fftfreq(W) * W, indexing="ij")
    F[((np.abs(KX) <= w) & (np.abs(KY) > k0)) | ((np.abs(KY) <= w) & (np.abs(KX) > k0))] = 0
    return np.real(np.fft.ifft2(F)).astype(np.float32)


def pre_defib(z, m):
    """detect thin bright horizontal / vertical lines (elongated mexican-hat) and fill them from their surroundings."""
    x = np.clip(z, -3, 3).astype(np.float32)
    S, sl, sv = 121, 25.0, 1.5
    u, v = np.meshgrid(np.arange(S) - S // 2, np.arange(S) - S // 2)
    prof = (1 - v**2 / sv**2) * np.exp(-(v**2) / (2 * sv**2))
    k = (np.exp(-(u**2) / (2 * sl**2)) * prof).astype(np.float32)
    k -= k.mean()
    k /= np.abs(k).sum()
    r = np.maximum(cv2.filter2D(x, -1, k), cv2.filter2D(x, -1, k.T))
    fib = cv2.dilate((n.robust_z(r, m) > 2.5).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return np.where(fib, n.nblur(x, m & ~fib, 8), x).astype(np.float32)


# ---------------------------------------------------------------- stroke enhancement x -> map
def post_blur(x):
    return n.gblur(x, 8)


def post_asf(x):
    return n.gblur(n.asf(n.gblur(x, 4)), 10)


def post_areaopen(x):
    r = morphology.area_closing(morphology.area_opening(u8(n.gblur(x, 4)), 800), 800)
    return n.gblur(r.astype(np.float32), 6)


def post_mindir(x):
    """a pixel is stroke-like only if line averages in all four orientations are high: thin lines fail the crossing ones."""
    g = n.gblur(x, 2)
    return n.gblur(np.min([cv2.filter2D(g, -1, line_kernel(41, a)) for a in (0, 45, 90, 135)], 0), 5)


def post_dog(x):
    return n.gblur(x, 10) - n.gblur(x, 40)


def post_sato(x):
    return filters.sato(n.gblur(x, 3), sigmas=[6, 9, 12], black_ridges=False).astype(np.float32)


def post_tv(x):
    g = n.gblur(x, 3)
    return n.gblur(restoration.denoise_tv_chambolle(g / g.std(), weight=2.0).astype(np.float32), 3)


def post_tophat(x):
    g = n.gblur(x, 3)
    return n.gblur(g - cv2.morphologyEx(g, cv2.MORPH_OPEN, n.disk(16)), 6)


def post_nlm(x):
    return n.gblur(cv2.fastNlMeansDenoising(u8(x), None, 25, 7, 35).astype(np.float32), 4)


def post_hdome(x):
    g = n.gblur(x, 6)
    return g - morphology.reconstruction(g - 1.0, g, method="dilation")


def post_log(x):
    return np.max([-(s**2) * ndi.gaussian_laplace(x, s) for s in (10, 14, 18)], 0)


def post_hess(x):
    return -feature.hessian_matrix_eigvals(feature.hessian_matrix(x, sigma=9, order="rc", use_gaussian_derivatives=True))[1]


PRE = {"none": pre_none, "clip": pre_clip, "rowcol": pre_rowcol, "fft": pre_fft, "defib": pre_defib}
POST = {"blur": post_blur, "asf": post_asf, "areaopen": post_areaopen, "mindir": post_mindir, "dog": post_dog,
        "sato": post_sato, "tv": post_tv, "tophat": post_tophat, "nlm": post_nlm, "hdome": post_hdome,
        "log": post_log, "hess": post_hess}
ALGOS = [f"{a}+{b}" for a in PRE for b in POST]


def run_algo(name, z, m):
    pre, post = name.split("+")
    return POST[post](PRE[pre](z, m)).astype(np.float32)
