"""The report's result figures, from <drive>/evals_summary.csv (scripts/summarize_runs.py).

  fig_compass.png    Recall@1 on 08 against the compass error sigma
  fig_rotation.png   Recall@1 (08 and the mean of the sequences) against the query rotation alpha
  fig_tnet.png       self_top1 against the rotation of the scan (NDT-Transformer runs, tnet evaluation)
  fig_sparsity.png   mean Recall@1 with 64 beams and with 16 beams (both / query modes)
A figure whose evaluation is missing is skipped. Runs are picked by the last 6 characters of their
folder name (the hash), see RUNS.

  python scripts/report_figures.py --drive /content/drive/MyDrive/lpr --out_dir <drive>/report_figures
"""
import argparse
import os

import numpy as np
import pandas as pd

# hash -> (label, colour, line style)
RUNS = {
    "5c90e2": ("NDT, no aug", "#c0392b", "--"),
    "f60015": ("NDT + rot_aug", "#c0392b", "-"),
    "e39427": ("NDT + rot_points", "#e67e22", "-"),
    "dbc4f4": ("NDT + rot_aug + beam_aug", "#7f8c8d", "-"),
    "283ebc": ("own NDT, official arch.", "#d35400", ":"),
    "3fe7a5": ("own NDT, REIN", "#8e44ad", "-"),
    "b14094": ("BEV plain, no aug", "#2471a3", "--"),
    "216fa5": ("BEV plain + rot_aug", "#2471a3", "-"),
    "26962e": ("BEV plain + rot + beam_aug", "#5dade2", "-"),
    "4ac7d1": ("BEV REIN (ours)", "#17a589", "-"),
    "7711ab": ("BEV REIN (authors)", "#117a65", "-"),
}


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def curve(plt, e, etype, key, runs, seq, xlabel, ylabel, path, ax=None, title=None):
    d = e[e.eval_type == etype]
    if seq != "mean":
        d = d[d.seq == seq]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
    drawn = 0
    for h in runs:
        g = d[d.r == h]
        if g.empty:
            continue
        t = g.groupby(key)["R@1"].mean()
        label, col, ls = RUNS[h]
        ax.plot(t.index, t.values, ls, color=col, marker="o", ms=3.5, lw=1.8, label=label)
        drawn += 1
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_ylim(0, 100); ax.grid(alpha=.25)
    ax.set_xticks(sorted(d[key].dropna().unique()))
    if title:
        ax.set_title(title)
    if own:
        if not drawn:
            plt.close(fig)
            return False
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return drawn > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True, help="folder with evals_summary.csv")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    plt = setup()
    e = pd.read_csv(os.path.join(a.drive, "evals_summary.csv"), dtype={"seq": str}, low_memory=False)
    e["r"] = e.run.astype(str).str[-6:]
    out = lambda n: os.path.join(a.out_dir, n)
    made = []

    if curve(plt, e, "compass", "sigma_deg", ["5c90e2", "b14094", "f60015", "3fe7a5", "216fa5", "7711ab"],
             "08", "compass noise σ (deg)", "Recall@1 on seq 08 (%)", out("fig_compass.png")):
        made.append("fig_compass.png")

    rot_runs = ["5c90e2", "f60015", "e39427", "283ebc", "3fe7a5", "b14094", "216fa5", "4ac7d1", "7711ab"]
    if (e.eval_type == "rotation").any():
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        curve(plt, e, "rotation", "alpha_deg", rot_runs, "08", "query rotation amplitude α (deg)",
              "Recall@1 (%)", None, ax=axes[0], title="KITTI 08")
        curve(plt, e, "rotation", "alpha_deg", rot_runs, "mean", "query rotation amplitude α (deg)",
              "Recall@1 (%)", None, ax=axes[1], title="mean of 00, 02, 05, 06, 08")
        axes[1].legend(fontsize=7.5, frameon=False, loc="lower left")
        fig.tight_layout(); fig.savefig(out("fig_rotation.png"), dpi=200); plt.close(fig)
        made.append("fig_rotation.png")

    t = e[e.eval_type == "tnet"]
    if not t.empty:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for h in ["5c90e2", "f60015", "e39427", "dbc4f4"]:
            g = t[t.r == h].sort_values("angle")
            if g.empty:
                continue
            label, col, ls = RUNS[h]
            ax.plot(g.angle, 100 * g.self_top1, ls, color=col, marker="o", ms=3.5, lw=1.8, label=label)
        ax.set_xlabel("rotation of the scan about z (deg)"); ax.set_ylabel("self_top1 (%)")
        ax.set_xticks([0, 15, 30, 45, 60, 90, 120, 150, 180]); ax.set_ylim(0, 102); ax.grid(alpha=.25)
        ax.legend(fontsize=8, frameon=False, loc="center right")
        fig.tight_layout(); fig.savefig(out("fig_tnet.png"), dpi=200); plt.close(fig)
        made.append("fig_tnet.png")

    s = e[e.eval_type == "sparsity"].copy()
    if not s.empty:
        s["k"] = s.beams.astype(int).astype(str) + "_" + s["mode"]
        m = s.pivot_table(index="r", columns="k", values="R@1", aggfunc="mean")
        runs = [h for h in ["f60015", "dbc4f4", "3fe7a5", "216fa5", "26962e", "4ac7d1", "7711ab"] if h in m.index]
        cols = [("64_both", "64 beams", "#bdc3c7"), ("16_both", "16 both", "#5d6d7e"), ("16_query", "16 query", "#c0392b")]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        x = np.arange(len(runs)); w = 0.26
        for i, (k, label, col) in enumerate(cols):
            ax.bar(x + (i - 1) * w, m.loc[runs, k], w, color=col, label=label)
        ax.set_xticks(x); ax.set_xticklabels([RUNS[h][0] for h in runs], rotation=22, ha="right", fontsize=8)
        ax.set_ylabel("mean Recall@1, 5 seqs (%)"); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=.25)
        ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.12))
        fig.tight_layout(); fig.savefig(out("fig_sparsity.png"), dpi=200); plt.close(fig)
        made.append("fig_sparsity.png")
    print("->", a.out_dir, made)


if __name__ == "__main__":
    main()
