"""summary_figures.py -- human-readable figures and statistics from a runs_corr report.

  python3 runs_corr/summary_figures.py                                   # report/ -> report/figures
  python3 runs_corr/summary_figures.py --report runs_corr/report_v1 --keep-0846a 0846a_out_1 --out runs_corr/figures_0846a_out_1
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--report", default=str(ROOT / "report"))
parser.add_argument("--out", default="", help="figure folder (default <report>/figures)")
parser.add_argument("--keep-0846a", default="", help="comma-separated 0846A labels to keep; the other sheets are dropped as targets and models")
args = parser.parse_args()
REPORT = Path(args.report).resolve()
OUT = Path(args.out).resolve() if args.out else REPORT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SHORT = {
    "pherc0009b": "0009B", "pherc0139": "0139", "pherc0172": "0172", "pherc0343": "0343", "pherc0343p": "0343P", "pherc0500p2": "0500P2",
    "pherc0814": "0814", "pherc0841": "0841", "pherc0846a": "0846A", "pherc1447": "1447", "pherc1667": "1667",
    "pherc51cr4_fr8": "51Cr4 Fr8", "phercparis1_fr34": "Paris1 Fr34", "phercparis2_fr143": "Paris2", "phercparis4": "Paris4",
}
FRAGMENTS = {"cr1fr3", "scroll6_fr8", "paris1_fr34", "paris2_fr143", "paris2_fr47"}
FRAGMENTS |= {f"{name}_flat" for name in FRAGMENTS}

targets = pd.read_csv(REPORT / "all_targets.csv")
consensus = pd.read_csv(REPORT / "consensus_cos_cross.csv", index_col=0)
init = pd.read_csv(REPORT / "init_cos_cross.csv", index_col=0)
labels = list(consensus.index)
if args.keep_0846a:
    keep = {name.strip() for name in args.keep_0846a.split(",")}
    labels = [l for l in labels if not l.startswith("0846a") or l in keep]
    targets = targets[targets.target_label.isin(labels) & (targets.source_label.isin(labels) | (targets.run == "init"))]
    # consensus over the kept models only
    stack = []
    for source in labels:
        pairs = pd.read_csv(ROOT / "runs" / source / "pairs.csv")
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
        for t, u, v in zip(pairs.t_label, pairs.u_label, pairs.cos_cross):
            if t in matrix.index and u in matrix.index:
                matrix.loc[t, u] = matrix.loc[u, t] = v
        stack.append(matrix.to_numpy())
    consensus = pd.DataFrame(np.nanmean(stack, axis=0), index=labels, columns=labels)
    init = init.loc[labels, labels]
scroll = {label: SHORT.get(s, s) for label, s in targets.drop_duplicates("target_label").set_index("target_label")["target_scroll"].items()}
scrolls = sorted(set(scroll.values()), key=lambda s: [scroll[l] for l in labels].index(s))
palette = dict(zip(scrolls, plt.cm.tab20(np.linspace(0, 1, len(scrolls)))))
n = len(labels)
iu = np.triu_indices(n, 1)

base = targets[targets.run == "init"].set_index("target_label")
cross = targets[(targets.run != "init") & (targets.same_patch == 0)].copy()
for metric in ("eval_char_ap", "eval_pr_auc", "eval_loss"):
    cross["d_" + metric] = cross[metric].to_numpy() - base.loc[cross.target_label, metric].to_numpy()
transfer = cross.pivot(index="source_label", columns="target_label", values="d_eval_char_ap").reindex(index=labels, columns=labels)
conflict = cross.pivot(index="source_label", columns="target_label", values="cos_src_cross").reindex(index=labels, columns=labels)

C = consensus.to_numpy()
distance = np.clip((1 - 0.5 * (C + C.T)) / 2, 0, 1)
np.fill_diagonal(distance, 0)
tree = linkage(squareform(distance, checks=False), "average")
order = list(leaves_list(tree))
ordered = [labels[i] for i in order]


def scroll_strip(ax, names, axis):
    colors = np.array([palette[scroll[name]] for name in names])[None] if axis == "x" else \
        np.array([palette[scroll[name]] for name in names])[:, None]
    ax.imshow(colors, aspect="auto")
    ax.set_xticks([]), ax.set_yticks([])


def heatmap(matrix, rows, cols, path, title, cmap="RdBu_r", center=0.0, span=None, label=""):
    fig = plt.figure(figsize=(13, 11.5))
    grid = fig.add_gridspec(2, 3, width_ratios=[0.025, 1, 0.03], height_ratios=[0.025, 1], wspace=0.02, hspace=0.02)
    ax = fig.add_subplot(grid[1, 1])
    span = span or np.nanmax(np.abs(matrix - center))
    image = ax.imshow(matrix, cmap=cmap, vmin=center - span, vmax=center + span, aspect="auto")
    ax.set_xticks(range(len(cols)), [f"{c} ({scroll[c]})" for c in cols], rotation=90, fontsize=7)
    ax.set_yticks([]), ax.set_yticklabels([])
    left = fig.add_subplot(grid[1, 0])
    scroll_strip(left, rows, "y")
    left.set_yticks(range(len(rows)), [f"{r} ({scroll[r]})" for r in rows], fontsize=7)
    top = fig.add_subplot(grid[0, 1])
    scroll_strip(top, cols, "x")
    top.set_title(title, fontsize=11)
    fig.colorbar(image, cax=fig.add_subplot(grid[1, 2]), label=label)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# 1. patch x patch affinity (consensus over all trained models), clustered
M = consensus.loc[ordered, ordered].to_numpy().copy()
np.fill_diagonal(M, np.nan)
heatmap(M, ordered, ordered, OUT / "01_patch_affinity.png",
        f"Patch affinity: do two patches pull the model the same way?\n(cross-half gradient cosine, averaged over all {n} trained models; clustered)",
        cmap="viridis", center=0.4, span=0.45, label="gradient cosine")

# 2. scroll x scroll affinity (mean over patch pairs; diagonal = within-scroll pairs only)
S = pd.DataFrame(np.nan, index=scrolls, columns=scrolls)
counts = S.copy()
for a in scrolls:
    for b in scrolls:
        ia = [i for i, l in enumerate(labels) if scroll[l] == a]
        ib = [i for i, l in enumerate(labels) if scroll[l] == b]
        values = [C[i, j] for i in ia for j in ib if i != j]
        if values:
            S.loc[a, b], counts.loc[a, b] = np.mean(values), len(values)
scroll_order = list(dict.fromkeys(scroll[l] for l in ordered))
S = S.loc[scroll_order, scroll_order]
fig, ax = plt.subplots(figsize=(9, 8))
image = ax.imshow(S.to_numpy(dtype=float), cmap="viridis", vmin=0.1, vmax=0.8)
for i in range(len(scroll_order)):
    for j in range(len(scroll_order)):
        value = S.iloc[i, j]
        if np.isfinite(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="w" if value < 0.5 else "k")
ax.set_xticks(range(len(scroll_order)), scroll_order, rotation=60, ha="right")
ax.set_yticks(range(len(scroll_order)), scroll_order)
ax.set_title("Scroll-level affinity (mean patch-pair gradient cosine)\ndiagonal: patches of the same scroll; blank = scroll has one patch", fontsize=10)
fig.colorbar(image, ax=ax, shrink=0.8, label="gradient cosine")
fig.savefig(OUT / "02_scroll_affinity.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# 3. 2D map (classical MDS of the affinity distance)
D2 = distance ** 2
J = np.eye(n) - 1.0 / n
B = -0.5 * J @ D2 @ J
values, vectors = np.linalg.eigh(B)
xy = vectors[:, -2:][:, ::-1] * np.sqrt(np.maximum(values[-2:][::-1], 0))
explained = values[-2:][::-1] / values[values > 0].sum()
fig, ax = plt.subplots(figsize=(11, 9))
for i, label in enumerate(labels):
    ax.scatter(*xy[i], s=140, color=palette[scroll[label]], edgecolor="k", marker="s" if label in FRAGMENTS else "o", zorder=3)
    ax.annotate(label, xy[i], fontsize=7, xytext=(5, 4), textcoords="offset points")
for s in scrolls:
    ax.scatter([], [], color=palette[s], label=s, s=60)
ax.legend(fontsize=7, ncol=2, loc="best", title="scroll (squares = detached fragments)")
ax.set_title(f"Map of patches: close = similar gradients (MDS of 1 - affinity; axes explain "
             f"{explained[0]:.0%} / {explained[1]:.0%})", fontsize=10)
ax.set_xticks([]), ax.set_yticks([])
fig.savefig(OUT / "03_patch_map.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# 4. dendrogram
fig, ax = plt.subplots(figsize=(14, 5))
dendrogram(tree, labels=[f"{l} ({scroll[l]})" for l in labels], leaf_rotation=90, leaf_font_size=7, ax=ax,
           color_threshold=0.30)
ax.set_ylabel("distance = (1 - cosine) / 2")
ax.set_title("Hierarchical clustering of patches (average linkage on the consensus affinity)")
fig.savefig(OUT / "04_dendrogram.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# 5. transfer: model trained on row patch, evaluated on column patch, minus the untrained model
heatmap(transfer.loc[ordered, ordered].to_numpy(), ordered, ordered, OUT / "05_transfer_char_ap.png",
        "Transfer: character AP on the column patch of a model trained on the row patch,\nminus the untrained model's (red = helps, blue = hurts)",
        cmap="RdBu_r", span=0.3, label="delta character AP")
heatmap(conflict.loc[ordered, ordered].to_numpy(), ordered, ordered, OUT / "06_conflict_now.png",
        "Gradient agreement after training on the row patch: would continuing on it also\nreduce the column patch's loss? (red = yes, blue = conflict)",
        span=0.9, label="cos(G_column, G_row) at the row model")

# 7. help / hurt per source and per target
source = cross.groupby("source_label").d_eval_char_ap.agg(["mean", lambda x: (x < -0.1).mean(), lambda x: (x > 0.1).mean()])
source.columns = ["mean", "hurt", "help"]
target = cross.groupby("target_label").d_eval_char_ap.agg(["mean", lambda x: (x < -0.1).mean(), lambda x: (x > 0.1).mean()])
target.columns = ["mean", "hurt", "help"]
fig, axes = plt.subplots(1, 2, figsize=(15, 9))
for ax, frame, title in ((axes[0], source, f"as SOURCE: average effect of training on this patch\non the {n - 1} other patches"),
                         (axes[1], target, f"as TARGET: average effect of the {n - 1} other models\non this patch")):
    frame = frame.sort_values("mean")
    y = np.arange(len(frame))
    ax.barh(y, frame["mean"], color=[palette[scroll[l]] for l in frame.index], edgecolor="k")
    for i, (label, row) in enumerate(frame.iterrows()):
        ax.text(max(row["mean"], 0) + 0.004, i, f"help {row['help']:.0%} / hurt {row['hurt']:.0%}", va="center", fontsize=6)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y, [f"{l} ({scroll[l]})" for l in frame.index], fontsize=7)
    ax.set_xlabel("mean delta character AP vs untrained model (help/hurt = share of pairs beyond +/-0.10)")
    ax.set_title(title, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "07_help_hurt.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# 8. distributions: transfer, same vs other scroll; calibration (loss) vs ranking (AP)
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
same = cross.source_scroll == cross.target_scroll
bins = np.linspace(-0.25, 0.45, 36)
axes[0].hist(cross.loc[~same, "d_eval_char_ap"], bins=bins, alpha=0.6, label=f"different scroll (n={(~same).sum()})", density=True)
axes[0].hist(cross.loc[same, "d_eval_char_ap"], bins=bins, alpha=0.6, label=f"same scroll (n={same.sum()})", density=True)
for edge in (-0.1, 0.1):
    axes[0].axvline(edge, color="k", ls="--", lw=0.8)
axes[0].set_xlabel("delta character AP vs untrained model")
axes[0].set_title("Transfer to other patches", fontsize=10)
axes[0].legend(fontsize=8)
axes[1].scatter(cross.d_eval_char_ap, cross.d_eval_loss, s=6, alpha=0.4)
axes[1].axhline(0, color="k", lw=0.8), axes[1].axvline(0, color="k", lw=0.8)
axes[1].set_xlabel("delta character AP (ranking; right = better)")
axes[1].set_ylabel("delta eval loss (calibration; up = worse)")
axes[1].set_title("Ranking improves while calibration degrades", fontsize=10)
axes[2].scatter(cross.cos_src_cross, cross.d_eval_char_ap, s=6, alpha=0.4)
rho = stats.spearmanr(cross.cos_src_cross, cross.d_eval_char_ap)[0]
axes[2].set_xlabel("gradient agreement after training (cos_src_cross)")
axes[2].set_ylabel("delta character AP")
axes[2].set_title(f"End-of-training gradient agreement barely predicts transfer (Spearman {rho:.2f})", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "08_distributions.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# 9. untrained vs trained affinity
fig, ax = plt.subplots(figsize=(7, 6))
I = init.loc[labels, labels].to_numpy()
ax.scatter(I[iu], C[iu], s=8, alpha=0.5)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("affinity under the UNTRAINED model")
ax.set_ylabel("affinity averaged over TRAINED models")
ax.set_title(f"Before training every patch wants the same thing (mean {I[iu].mean():.2f});\n"
             f"after training patches diverge (mean {C[iu].mean():.2f})", fontsize=10)
fig.savefig(OUT / "09_init_vs_trained.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# 10. layers
groups = pd.concat([pd.read_csv(ROOT / "runs" / s / "target_groups.csv") for s in labels], ignore_index=True)
groups = groups.merge(targets[["run", "target", "same_patch", "same_scroll"]], on=["run", "target"])
groups = groups[groups.target_label.isin(labels)]
groups["relation"] = np.where(groups.same_patch == 1, "self", np.where(groups.same_scroll == 1, "same_scroll", "other_scroll"))
layer = groups.groupby(["group", "relation"]).cos_src_cross.mean().unstack()
layer["grad_share"] = groups.groupby("group").grad_share.mean()
layer = layer.reset_index().sort_values("grad_share", ascending=False).head(14)
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(layer))
for k, (column, color) in enumerate((("self", "tab:green"), ("same_scroll", "tab:blue"), ("other_scroll", "tab:red"))):
    ax.bar(x + (k - 1) * 0.27, layer[column], 0.27, label=column.replace("_", " "), color=color)
ax.set_xticks(x, [f"{g}\n{s:.0%} of grad" for g, s in zip(layer.group, layer.grad_share)], rotation=60, ha="right", fontsize=7)
ax.set_ylabel("mean cross-half cosine to the source patch")
ax.set_title("Where do patches agree? Agreement by network part (sorted by share of the gradient)")
ax.legend()
fig.savefig(OUT / "10_layers.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# statistics used in the summary
rng = np.random.default_rng(0)


def partition_stats(groups):
    groups = np.asarray(groups)
    same_mask = groups[iu[0]] == groups[iu[1]]
    observed = C[iu][same_mask].mean() - C[iu][~same_mask].mean()
    null = []
    for _ in range(2000):
        g = rng.permutation(groups)
        m = g[iu[0]] == g[iu[1]]
        null.append(C[iu][m].mean() - C[iu][~m].mean())
    return C[iu][same_mask].mean(), C[iu][~same_mask].mean(), observed, (1 + np.sum(np.array(null) >= observed)) / 2001


print("== partitions (within / between affinity, contrast, permutation p) ==")
partitions = {
    "by scroll": [scroll[l] for l in labels],
    "average linkage k=2": fcluster(tree, 2, "maxclust"),
    "average linkage k=4": fcluster(tree, 4, "maxclust"),
    "average linkage k=8": fcluster(tree, 8, "maxclust"),
}
for name, groups in partitions.items():
    within, between, contrast, p = partition_stats(groups)
    print(f"{name:32s} within {within:.3f} between {between:.3f} contrast {contrast:.3f} p {p:.4f}")

print("\n== per-scroll coherence (within-scroll vs that scroll to others) ==")
for s in scrolls:
    idx = [i for i, l in enumerate(labels) if scroll[l] == s]
    if len(idx) < 2:
        continue
    within = [C[i, j] for i in idx for j in idx if i < j]
    outside = [C[i, j] for i in idx for j in range(n) if j not in idx]
    print(f"{s:12s} n={len(idx)} within {np.mean(within):.3f} to-others {np.mean(outside):.3f} "
          f"MWU p={stats.mannwhitneyu(within, outside, alternative='greater').pvalue:.3g}")

print("\n== transfer ==")
d = cross.d_eval_char_ap
print(f"mean {d.mean():.3f}, median {d.median():.3f}; helps (>+0.10) {(d > 0.1).mean():.1%}, hurts (<-0.10) {(d < -0.1).mean():.1%}, "
      f"within +/-0.05 {(d.abs() < 0.05).mean():.1%}; eval loss worse than untrained {(cross.d_eval_loss > 0).mean():.1%}")
print(f"same-scroll mean {d[same].mean():.3f} vs other {d[~same].mean():.3f}, MWU p={stats.mannwhitneyu(d[same], d[~same], alternative='greater').pvalue:.3g}")
frag_src = cross.source_label.isin(FRAGMENTS)
frag_tgt = cross.target_label.isin(FRAGMENTS)
for name, mask in (("fragment -> fragment", frag_src & frag_tgt), ("fragment -> segment", frag_src & ~frag_tgt),
                   ("segment -> fragment", ~frag_src & frag_tgt), ("segment -> segment", ~frag_src & ~frag_tgt)):
    print(f"{name:22s} n={mask.sum():4d} mean dAP {d[mask].mean():+.3f} hurt {(d[mask] < -0.1).mean():.1%} help {(d[mask] > 0.1).mean():.1%}")
print(f"\nfigures written to {OUT}")
