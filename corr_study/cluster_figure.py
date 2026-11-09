"""cluster_figure.py -- Ward k=5 clusters of the consensus gradient affinity, with each patch's help for 0846A and the next-run corpus.

  python3 runs_corr/cluster_figure.py        # report/ -> report/figures/12_clusters_ward_k5.png, report/cluster_ward_k5.csv
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

REPORT = Path(__file__).resolve().parent / "report"
K = 5
PROPOSED = {"p9b_487", "p9b_592", "w041", "ph1447_w060", "500P2_m1", "500P2_m2", "p343_letters", "w087", "w013", "p841", "p841_w00"}
THRESHOLD = 0.04
NAMES = "ABCDE"

consensus = pd.read_csv(REPORT / "consensus_cos_cross.csv", index_col=0)
delta_ap = pd.read_csv(REPORT / "matrix_delta_eval_char_ap.csv", index_col=0)
labels = [label for label in consensus.index if not label.endswith("_flat")]
sheets = [label for label in labels if label.startswith(("0846a", "fiber_only"))]
matrix = consensus.loc[labels, labels].to_numpy()
cluster = fcluster(linkage(matrix, "ward"), K, "maxclust")


def help_for_0846a(label):
    return delta_ap.loc[label, [s for s in sheets if s != label]].mean()


help_ = pd.Series({label: help_for_0846a(label) for label in labels})
order = sorted(set(cluster), key=lambda c: -help_[[l for l, x in zip(labels, cluster) if x == c]].mean())
name_of = {c: NAMES[i] for i, c in enumerate(order)}
frame = pd.DataFrame({"patch": labels, "cluster": [name_of[c] for c in cluster], "dAP_on_0846a": help_.values})
frame["proposed"] = frame.patch.isin(PROPOSED) | frame.patch.isin(sheets)
frame = frame.sort_values(["cluster", "dAP_on_0846a"], ascending=[True, False]).reset_index(drop=True)
frame.to_csv(REPORT / "cluster_ward_k5.csv", index=False)

ordered = list(frame.patch)
heat = consensus.loc[ordered, ordered].to_numpy()
palette = dict(zip(NAMES, plt.cm.Set2(np.arange(K))))
fig, (ax, bar) = plt.subplots(1, 2, figsize=(17, 9), gridspec_kw={"width_ratios": [1.15, 1]})
image = ax.imshow(heat, cmap="viridis", vmin=0, vmax=0.9)
ax.set_xticks(range(len(ordered)), ordered, rotation=90, fontsize=7)
ax.set_yticks(range(len(ordered)), ordered, fontsize=7)
edges = np.cumsum([0] + [int((frame.cluster == c).sum()) for c in NAMES])
for edge in edges[1:-1]:
    ax.axhline(edge - 0.5, color="w", lw=1.5)
    ax.axvline(edge - 0.5, color="w", lw=1.5)
for c, (lo, hi) in zip(NAMES, zip(edges[:-1], edges[1:])):
    ax.text((lo + hi - 1) / 2, -1.2, c, ha="center", va="bottom", fontsize=16, fontweight="bold", color=palette[c], clip_on=False)
fig.colorbar(image, ax=ax, fraction=0.04, label="consensus gradient cosine")
ax.set_title("Consensus affinity, 44 patches (flat twins excluded), ordered by Ward k=5 clusters", fontsize=10, pad=28)

y = np.arange(len(ordered))[::-1]
bar.barh(y, frame.dAP_on_0846a, color=[palette[c] for c in frame.cluster], edgecolor=["k" if p else "none" for p in frame.proposed],
         linewidth=[1.4 if p else 0 for p in frame.proposed])
bar.set_yticks(y, [f"{p}{'  *' if q else ''}" for p, q in zip(frame.patch, frame.proposed)], fontsize=7)
bar.axvline(0, color="k", lw=0.8)
bar.axvline(THRESHOLD, color="r", ls="--", lw=1, label=f"selection threshold +{THRESHOLD}")
for c, (lo, hi) in zip(NAMES, zip(edges[:-1], edges[1:])):
    sub = frame.iloc[lo:hi]
    left = c in "ABC"
    bar.text(-0.135 if left else 0.155, (len(ordered) - 1 - (lo + hi - 1) / 2), f"{c}: mean {sub.dAP_on_0846a.mean():+.3f}\n{len(sub)} patches",
             va="center", ha="left" if left else "right", fontsize=9, color="k", bbox=dict(boxstyle="round", fc=palette[c], alpha=0.7))
bar.set_xlim(-0.14, 0.16)
bar.set_ylim(-0.7, len(ordered) - 0.3)
bar.set_xlabel("character-AP change on the 0846A sheets (vs untrained model)")
bar.set_title("Help for 0846A per patch (* = in the proposed corpus, 0846A sheets all kept)", fontsize=10)
bar.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(REPORT / "figures" / "12_clusters_ward_k5.png", dpi=130, bbox_inches="tight")
print(frame.groupby("cluster").agg(n=("patch", "size"), mean_dAP=("dAP_on_0846a", "mean")).round(3).to_string())
print("wrote", REPORT / "figures" / "12_clusters_ward_k5.png")
