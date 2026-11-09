"""flat_comparison.py -- detected-surface vs flat-surface (intensity-peak layer) runs of the detached fragments.

python3 runs_corr/flat_comparison.py   (after the scheduler's report; writes report/flat_comparison.{csv,png})
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report"
FRAGMENTS = ("cr1fr3", "scroll6_fr8", "paris1_fr34", "paris2_fr143", "paris2_fr47")
FLAT = tuple(f"{name}_flat" for name in FRAGMENTS)

targets = pd.read_csv(REPORT / "all_targets.csv")
consensus = pd.read_csv(REPORT / "consensus_cos_cross.csv", index_col=0)
labels = list(consensus.index)
scroll = targets.drop_duplicates("target_label").set_index("target_label")["target_scroll"].to_dict()
segments = [l for l in labels if l not in FRAGMENTS and l not in FLAT]
sheets_0846a = [l for l in labels if l.startswith("0846a")]
base = targets[targets.run == "init"].set_index("target_label")
cross = targets[(targets.run != "init") & (targets.same_patch == 0)].copy()
cross["dAP"] = cross.eval_char_ap.to_numpy() - base.loc[cross.target_label, "eval_char_ap"].to_numpy()
own = targets[targets.same_patch == 1].set_index("source_label")


def mean_affinity(label, others):
    return float(np.nanmean([consensus.loc[label, o] for o in others if o != label]))


def train_metric(label, column):
    path = ROOT / "runs" / label / "train_epochs.csv"
    return float(pd.read_csv(path).iloc[-1][column]) if path.is_file() else np.nan


rows = []
for name in FRAGMENTS:
    for variant, label in (("detected", name), ("flat", f"{name}_flat")):
        if label not in labels:
            continue
        group = FRAGMENTS if variant == "detected" else FLAT
        as_source = cross[cross.source_label == label]
        as_target = cross[cross.target_label == label]
        rows.append({
            "fragment": name, "surface": variant,
            "train_pr_auc_ep3": train_metric(label, "pr_auc"),
            "own_eval_char_ap": float(own.loc[label, "eval_char_ap"]) if label in own.index else np.nan,
            "own_gain_vs_init": float(own.loc[label, "eval_char_ap"] - base.loc[label, "eval_char_ap"]) if label in own.index else np.nan,
            "reliability_full": float(targets[targets.target_label == label].reliability_full.median()),
            "affinity_to_segments": mean_affinity(label, segments),
            "affinity_to_0846a": mean_affinity(label, sheets_0846a),
            "affinity_to_same_variant_fragments": mean_affinity(label, [g for g in group if g in labels]),
            "affinity_to_twin": float(consensus.loc[name, f"{name}_flat"]) if f"{name}_flat" in labels else np.nan,
            "source_dAP_on_segments": as_source[as_source.target_label.isin(segments)].dAP.mean(),
            "source_dAP_on_0846a": as_source[as_source.target_label.isin(sheets_0846a)].dAP.mean(),
            "source_hurt_rate_segments": (as_source[as_source.target_label.isin(segments)].dAP < -0.1).mean(),
            "target_dAP_from_segments": as_target[as_target.source_label.isin(segments)].dAP.mean(),
        })
table = pd.DataFrame(rows)
table.to_csv(REPORT / "flat_comparison.csv", index=False)
pd.set_option("display.width", 250)
print(table.round(3).to_string(index=False))

columns = ["affinity_to_segments", "affinity_to_0846a", "affinity_to_same_variant_fragments",
           "source_dAP_on_segments", "source_dAP_on_0846a", "target_dAP_from_segments", "own_gain_vs_init"]
fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 4.5), sharey=True)
y = np.arange(len(FRAGMENTS))
for ax, column in zip(axes, columns):
    for k, (variant, color) in enumerate((("detected", "tab:gray"), ("flat", "tab:orange"))):
        values = [table[(table.fragment == f) & (table.surface == variant)][column].mean() for f in FRAGMENTS]
        ax.barh(y + (k - 0.5) * 0.38, values, 0.38, color=color, label=f"{variant} surface")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(column.replace("_", " "), fontsize=9)
axes[0].set_yticks(y, FRAGMENTS)
axes[0].legend(fontsize=8)
fig.suptitle("Detached fragments: detected surface vs flat surface at the intensity-peak layer")
fig.tight_layout()
fig.savefig(REPORT / "figures" / "11_flat_vs_detected.png", dpi=130, bbox_inches="tight")
print(f"wrote {REPORT / 'flat_comparison.csv'} and figures/11_flat_vs_detected.png")
