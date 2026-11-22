"""Scores and figures of the Studio fine-tune experiments on PRIZESHEET_11_17_2 (predictions are stride-64 maps in output/studio_ft_exps/ and /tmp).

  python3 -m utils.studio_exp_report            # whatever exists: seed test, nnPU + background test, the six corpora master plot
"""
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output" / "studio_ft_exps"
OUT.mkdir(parents=True, exist_ok=True)
MASK = cv2.imread(str(REPO / "masks/extras/PRIZESHEET_11_17_2.png"), cv2.IMREAD_GRAYSCALE) > 0
H, W = MASK.shape
CACHE = REPO / "output/ensemble/mix28_ps6_s1_c3_r1000_v3_studio/PRIZESHEET_11_17_2/cache"
BASE = REPO / "output/ensemble/mix28_ps6_s1_c3_r1000_v8ep2_studio/PRIZESHEET_11_17_2/cache/studio_base.npy"


def load(path):
    path = Path(path)
    return np.load(path)[:H, :W].astype(np.float32) if path.is_file() else None


def stats(a):
    v = a[MASK]
    res = a - cv2.GaussianBlur(a, (0, 0), 24)
    phase = (np.arange(H)[:, None] % 64) * np.ones((1, W), int)
    profile = np.array([res[(phase == i) & MASK].mean() for i in range(64)])
    return dict(mean=float(v.mean()), c50=float((v > .5).mean()), c80=float((v > .8).mean()), p99=float(np.percentile(v, 99)),
                grid=float(profile.max() - profile.min()))


def row(name, a):
    s = stats(a)
    return f"{name:44s} {s['mean']:.3f} {100 * s['c50']:5.1f}% {100 * s['c80']:5.1f}%  {s['p99']:.3f} {s['grid']:.3f}"


HEADER = f"{'map':44s} mean   >0.5   >0.8    p99   grid"


def picture(a, label, width=1000):
    v = a[MASK]
    lo, hi = np.percentile(v, (1, 99.5))
    img = (255 * (1 - np.clip((a - lo) / (hi - lo), 0, 1))).astype(np.uint8)
    img[~MASK] = 255
    rgb = np.stack([img] * 3, -1)
    rgb[~MASK] = (200, 220, 245)
    tile = Image.fromarray(rgb).resize((width, int(H * width / W)))
    ImageDraw.Draw(tile).text((6, 4), label, fill="black")
    return tile


def difference(a, b):
    """what the two maps disagree on: Pearson r over the sheet, mean |difference| and its share of the mean probability"""
    x, y = a[MASK], b[MASK]
    return float(np.corrcoef(x, y)[0, 1]), float(np.abs(x - y).mean()), float(np.abs(x - y).mean() / max(0.5 * (x.mean() + y.mean()), 1e-9))


def main():
    lines = []
    ref = {"base Studio": load(BASE), "A bg10": load("/tmp/early_exp_A_neg10.npy"), "B nnPU": load("/tmp/early_exp_B_nnpu.npy"),
           "D seed 1 (jitter, bg 0.8)": load("/tmp/exp_D_jitter.npy"), "E bg6": load("/tmp/exp_E_neg6.npy")}
    d2, f = load(OUT / "exp_D2_seed2.npy"), load(OUT / "exp_F_neg4_nnpu.npy")
    lines += ["== 1. seed test: D (seed 1) vs D2 (seed 2), same recipe, same 117 sheets, 500 steps", HEADER]
    if d2 is not None:
        lines += [row("D seed 1", ref["D seed 1 (jitter, bg 0.8)"]), row("D2 seed 2", d2)]
        r, mad, rel = difference(ref["D seed 1 (jitter, bg 0.8)"], d2)
        lines.append(f"D vs D2: Pearson r {r:.4f}, mean |diff| {mad:.4f} ({100 * rel:.1f}% of the mean probability)")
        lines.append("for scale, other pairs (same sheet, different recipe):")
        for a, b in (("D seed 1 (jitter, bg 0.8)", "E bg6"), ("D seed 1 (jitter, bg 0.8)", "B nnPU"), ("A bg10", "E bg6"), ("base Studio", "D seed 1 (jitter, bg 0.8)")):
            r, mad, rel = difference(ref[a], ref[b])
            lines.append(f"  {a} vs {b}: r {r:.4f}, mean |diff| {mad:.4f} ({100 * rel:.1f}%)")
        diff = np.abs(ref["D seed 1 (jitter, bg 0.8)"] - d2)
        img = (255 * (1 - np.clip(diff / 0.5, 0, 1))).astype(np.uint8)
        img[~MASK] = 255
        pair = Image.new("RGB", (2010, picture(d2, "").height))
        pair.paste(picture(ref["D seed 1 (jitter, bg 0.8)"], "D (seed 1)"), (0, 0))
        pair.paste(picture(d2, "D2 (seed 2)"), (1010, 0))
        pair.save(OUT / "seed_test_D_vs_D2.png")
        Image.fromarray(img).resize((1500, int(H * 1500 / W))).save(OUT / "seed_test_absdiff.png")
    else:
        lines.append("(D2 not finished)")
    lines += ["", "== 2. weaker nnPU (weight 0.5, prior 0.1) + background weight 4 (F), against its neighbours", HEADER]
    for k in ("base Studio", "D seed 1 (jitter, bg 0.8)", "B nnPU", "E bg6", "A bg10"):
        lines.append(row(k, ref[k]))
    if f is not None:
        lines.append(row("F: bg 4 + nnPU(0.5, prior 0.1)", f))
        tiles = [picture(ref["base Studio"], "base Studio"), picture(ref["B nnPU"], "B: nnPU (1.0, prior 0.2)"), picture(ref["E bg6"], "E: bg 6"),
                 picture(f, "F: bg 4 + weaker nnPU")]
        grid = Image.new("RGB", (2010, 2 * (tiles[0].height + 8)), "white")
        for i, t in enumerate(tiles):
            grid.paste(t, ((i % 2) * 1010, (i // 2) * (t.height + 8)))
        grid.save(OUT / "nnpu_bg4_compare.png")
    lines += ["", "== 3. corpora, 800 steps each, jitter + background weight 6 (the E recipe)", HEADER]
    corpora = [("a_hand", "3.1 0846A hand-grown only (14)"), ("ab_hand", "3.2 0846A + 0846B hand-grown (16)"),
               ("a_all", "3.3 0846A hand + x (14 + 54)"), ("ab_all", "3.4 0846A + 0846B, hand + all x (16 + 121)"),
               ("mix28", "3.5 mix28 (14 + donors)"), ("mix28x", "3.6 mix28x (hand + x + donors)")]
    tiles = []
    for key, label in corpora:
        a = load(OUT / f"ds_{key}.npy")
        if a is None:
            lines.append(f"{label}: not finished")
            tiles.append(Image.new("RGB", (1000, picture(ref['E bg6'], '').height), "white"))
            continue
        lines.append(row(label, a))
        tiles.append(picture(a, label))
    if any(load(OUT / f"ds_{k}.npy") is not None for k, _ in corpora):
        grid = Image.new("RGB", (2010, 3 * (tiles[0].height + 8)), "white")
        for i, t in enumerate(tiles):
            grid.paste(t, ((i % 2) * 1010, (i // 2) * (t.height + 8)))
        grid.save(OUT / "corpora_master_3x2.png")
        done = {k: load(OUT / f"ds_{k}.npy") for k, _ in corpora}
        done = {k: v for k, v in done.items() if v is not None}
        keys = list(done)
        lines += ["", "pairwise Pearson r between the corpus maps:", "          " + " ".join(f"{k:>8s}" for k in keys)]
        for a in keys:
            lines.append(f"{a:>9s} " + " ".join(f"{difference(done[a], done[b])[0]:8.3f}" for b in keys))
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
