"""3-model ensembles on the prize sheet -> output/prize_<sheet>/ens3_methods/ (two-column figure + maps).
Ensemble 1: reader + studio plain stretch (p1->0), v8-in p50->0.  Ensemble 2: all three p50->0.   python3 -m utils.ensemble3_figure"""
import os, sys
import cv2, numpy as np
from PIL import Image, ImageDraw
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import mix6
from utils.fast_figure import colorize, orient_display, _font
Image.MAX_IMAGE_PIXELS = None
S = mix6.PRIZE_SHEET; c = f'/vesuvius/output/prize_{S}/cache/'
m = cv2.imread(f'/vesuvius/masks/extras/{S}.png', 0) > 0; H, W = m.shape
ld = lambda p: np.load(p)[:H, :W].astype(np.float32)
raw = {'reader': ld(c + 'reader/step_1000.npy'), 'studio': ld(c + 'studio/step_500.npy'),
       'v8in': ld(c + 'v8in/' + os.environ.get('V8IN_MAP', 'epoch_1_of_2_trained_epoch0.npy')), 'base': ld(c + 'studio_base/released.npy')}
def st(a, lo):
    l, h = np.percentile(a[m], (lo, 99.5)); return np.clip((a - l) / (h - l), 0, 1)
def comb(ms):
    zs = np.mean([(x - x[m].mean()) / x[m].std() for x in ms], 0)
    gm = np.exp(np.mean([np.log(np.clip(x, 1e-3, 1)) for x in ms], 0))
    rk = np.zeros((H, W), np.float32); rk[m] = rankdata(np.mean([x[m] for x in ms], 0)) / m.sum()
    return {'mean': np.mean(ms, 0), 'geometric mean': gm, 'z-score mean': zs, 'mean rank': rk}
r1, s1, v50 = st(raw['reader'], 1), st(raw['studio'], 1), st(raw['v8in'], 50)
E1 = comb([r1, s1, v50]); E2 = comb([st(raw['reader'], 50), st(raw['studio'], 50), v50])
out = f'/vesuvius/output/prize_{S}/ens3_methods' + os.environ.get('V8IN_OUT', ''); os.makedirs(out, exist_ok=True)
w = 1300
def tile(a, l, norm):
    if norm: lo, hi = np.percentile(a[m], (1, 99.5)); a = np.clip((a - lo) / (hi - lo), 0, 1)
    rgb = colorize(orient_display(np.where(m, a, np.nan), S), 'gray_r', (0.78, 0.86, 0.95))
    im = Image.fromarray(np.ascontiguousarray(rgb)); im = im.resize((w, int(im.height * w / im.width)), Image.Resampling.LANCZOS)
    t = Image.new('RGB', (w, im.height + 40), 'white'); t.paste(im, (0, 40)); ImageDraw.Draw(t).text((8, 5), l, fill='black', font=_font(24)); return t
TAG = os.environ.get('V8IN_TAG', 'epoch 1 of 2')
n1 = 'Ensemble 1: reader p1->0 + studio p1->0 + v8-in p50->0'; n2 = 'Ensemble 2: reader + studio + v8-in, all p50->0'
tiles = [(st(raw['base'], 1), 'base studio (released model), plain stretch', 0), (s1, 'studio step 500, plain stretch (p1->0)', 0),
         (r1, 'reader-v3 step 1000, plain stretch (p1->0)', 0), (v50, f'v8-in ({TAG}), p50->0', 0)]
for k in E1:
    tiles += [(E1[k], f'{n1} | {k}', 1), (E2[k], f'{n2} | {k}', 1)]
    np.save(f'{out}/ens1_{k.replace(" ", "_")}.npy', E1[k]); np.save(f'{out}/ens2_{k.replace(" ", "_")}.npy', E2[k])
ts = [tile(*t) for t in tiles]; rh = [max(ts[i].height, ts[i + 1].height) for i in range(0, len(ts), 2)]
g = Image.new('RGB', (2 * w + 10, sum(rh) + 10 * len(rh)), 'white'); y = 0
for j, h in enumerate(rh): g.paste(ts[2 * j], (0, y)); g.paste(ts[2 * j + 1], (w + 10, y)); y += h + 10
g.save(f'{out}/ensembles_3model_2col.png'); g.resize((g.width // 2, g.height // 2)).save('/tmp/e3.png'); print(g.size)
