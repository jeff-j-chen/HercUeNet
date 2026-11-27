"""One-stage fine-tunes (reader-v3 / v8-in from the released models) vs the earlier two-stage ones (from the mix28 fine-tunes), on the prize sheet.
Maps: output/prize_<sheet>/cache (two-stage, Studio) and output/onestage_<sheet>/cache (one-stage); one row per comparison, only what exists.
-> output/onestage_<sheet>/compare_onestage_vs_twostage.png and the statistics on stdout.   python3 -m utils.compare_onestage"""
import glob, os, sys
import cv2, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import mix6
from utils.fast_figure import colorize, orient_display, _font
Image.MAX_IMAGE_PIXELS = None
S = mix6.PRIZE_SHEET
OLD, NEW = f'/vesuvius/output/prize_{S}/cache/', f'/vesuvius/output/onestage_{S}/cache/'
m = cv2.imread(f'/vesuvius/masks/extras/{S}.png', 0) > 0; H, W = m.shape
def load(pattern):
    found = sorted(glob.glob(pattern))
    return np.load(found[0])[:H, :W].astype(np.float32) if found else None
def show(a, lo):
    l, h = np.percentile(a[m], (lo, 99.5)); p = np.where(m, np.clip((a - l) / (h - l), 0, 1), np.nan)
    return colorize(orient_display(p, S), 'gray_r', (0.78, 0.86, 0.95))
rows = [  # (title left, map left, stretch, title right, map right)
    ('Studio step 500 (single stage: unchanged)', OLD + 'studio/step_500.npy', 1, 'Studio step 500 (same file)', OLD + 'studio/step_500.npy'),
    ('reader-v3 step 1000, TWO-stage (mix28 -> mix28x)', OLD + 'reader/step_1000.npy', 1, 'reader-v3 step 1000, ONE-stage (released -> mix28x)', NEW + 'reader/step_1000.npy'),
    ('v8-in epoch 1, TWO-stage (trained-epoch0)', OLD + 'v8in/epoch_1_of_2_trained_epoch0.npy', 50, 'v8-in epoch 1, ONE-stage (trained-epoch0)', NEW + 'v8in/epoch_1_trained_epoch0.npy'),
    ('v8-in epoch 2, two-stage continuation (trained-epoch0 of the epoch-2 run)', OLD + 'v8in/v8in_epoch2_epoch2.npy', 50, 'v8-in epoch 2, ONE-stage (trained-epoch1)', NEW + 'v8in/epoch_2_trained_epoch1.npy'),
]
w = 1300; tiles = []
for tl, pl, lo, tr, pr in rows:
    a, b = load(pl), load(pr)
    if a is None or b is None:
        continue
    r = np.corrcoef(a[m], b[m])[0, 1]
    print(f'{tr}: vs {tl}: Pearson r {r:.4f} | mean {a[m].mean():.3f} -> {b[m].mean():.3f} | >0.5 {100*(a[m]>.5).mean():.1f}% -> {100*(b[m]>.5).mean():.1f}% | '
          f'p99 {np.percentile(a[m],99):.3f} -> {np.percentile(b[m],99):.3f}')
    for arr, title in ((a, tl), (b, f'{tr} | r={r:.3f} vs left')):
        im = Image.fromarray(np.ascontiguousarray(show(arr, lo))); im = im.resize((w, int(im.height * w / im.width)), Image.Resampling.LANCZOS)
        t = Image.new('RGB', (w, im.height + 40), 'white'); t.paste(im, (0, 40)); ImageDraw.Draw(t).text((8, 5), f'{title} (p{lo}->0)', fill='black', font=_font(22)); tiles.append(t)
if tiles:
    hs = [max(tiles[i].height, tiles[i + 1].height) + 10 for i in range(0, len(tiles), 2)]
    g = Image.new('RGB', (2 * w + 10, sum(hs)), 'white'); y = 0
    for j, h in enumerate(hs):
        g.paste(tiles[2 * j], (0, y)); g.paste(tiles[2 * j + 1], (w + 10, y)); y += h
    out = f'/vesuvius/output/onestage_{S}/compare_onestage_vs_twostage.png'; os.makedirs(os.path.dirname(out), exist_ok=True); g.save(out); print('saved', out, g.size)
