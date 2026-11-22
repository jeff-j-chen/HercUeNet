"""Ensemble (reader step 1000 p1->0, studio step 500 p1->0, v8-in epoch 0 p50->0; z-score mean) on PRIZESHEET_11_20 vs the same on 11_19_flatboi.
python3 -m utils.compare_1120"""
import glob, os, sys
import cv2, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fast_figure import colorize, orient_display, _font
Image.MAX_IMAGE_PIXELS = None
A, B = 'PRIZESHEET_11_19_flatboi', 'PRIZESHEET_11_20'
def maps(s):
    c = f'/vesuvius/output/prize_{s}/cache/'
    one = lambda pat: np.load(sorted(glob.glob(c + pat))[-1]).astype(np.float32)
    return {'reader': one('reader/*1000.npy'), 'studio': one('studio/*500.npy'),
            'v8in': one('v8in/epoch_1_of_2_trained_epoch0.npy' if s == A else 'v8in/*epoch0*.npy')}
def ens(s):
    m = cv2.imread(f'/vesuvius/masks/extras/{s}.png', 0) > 0; H, W = m.shape; mp = {k: v[:H, :W] for k, v in maps(s).items()}
    st = lambda a, lo: np.clip((a - np.percentile(a[m], lo)) / (np.percentile(a[m], 99.5) - np.percentile(a[m], lo)), 0, 1)
    ms = [st(mp['reader'], 1), st(mp['studio'], 1), st(mp['v8in'], 50)]
    z = np.mean([(x - x[m].mean()) / x[m].std() for x in ms], 0)
    return m, z, mp, ms
ma, za, pa, sa = ens(A); mb, zb, pb, sb = ens(B)
print('masks differ on', int((ma ^ mb).sum()), 'px')
m = ma & mb
print('z-score ensemble: Pearson r %.4f | mean|diff| %.4f z (std of ensemble %.3f) | 99th pct |diff| %.3f' % (
    np.corrcoef(za[m], zb[m])[0, 1], np.abs(za - zb)[m].mean(), za[m].std(), np.percentile(np.abs(za - zb)[m], 99)))
for i, k in enumerate(('reader', 'studio', 'v8in')):
    print('  %-6s r %.4f  mean|diff| %.4f (p-stretched, 0-1 scale)' % (k, np.corrcoef(sa[i][m], sb[i][m])[0, 1], np.abs(sa[i] - sb[i])[m].mean()))
d = zb - za; lim = float(np.percentile(np.abs(d[m]), 99.5))
t = np.clip(d / lim, -1, 1); pos = np.clip(t, 0, 1)[..., None]; neg = np.clip(-t, 0, 1)[..., None]
rgbd = (255 - pos * np.array([0, 255, 255]) * .9 - neg * np.array([255, 255, 0]) * .9).astype(np.uint8); rgbd[~m] = (200, 220, 245)
def gray(z, mm):
    lo, hi = np.percentile(z[mm], (1, 99.5)); return colorize(orient_display(np.where(mm, np.clip((z - lo) / (hi - lo), 0, 1), np.nan), A), 'gray_r', (0.78, 0.86, 0.95))
tiles = [(gray(za, ma), f'{A}: z-score ensemble'), (gray(zb, mb), f'{B}: z-score ensemble'),
         (orient_display(rgbd, A), f'difference 11_20 - 11_19 (red: 11_20 higher, blue: 11_19 higher; +-{lim:.2f} z saturates)')]
w = 1500; ims = []
for rgb, l in tiles:
    im = Image.fromarray(np.ascontiguousarray(rgb)).resize((w, int(rgb.shape[0] * w / rgb.shape[1])), Image.Resampling.LANCZOS)
    t_ = Image.new('RGB', (w, im.height + 44), 'white'); t_.paste(im, (0, 44)); ImageDraw.Draw(t_).text((8, 6), l, fill='black', font=_font(26)); ims.append(t_)
g = Image.new('RGB', (w, sum(i.height for i in ims) + 20), 'white'); y = 0
for i in ims: g.paste(i, (0, y)); y += i.height + 10
os.makedirs('/vesuvius/output/prize_PRIZESHEET_11_20', exist_ok=True)
g.save('/vesuvius/output/prize_PRIZESHEET_11_20/ensemble_compare_with_11_19.png'); g.resize((g.width // 2, g.height // 2)).save('/tmp/cmp_ens.png'); print(g.size)
