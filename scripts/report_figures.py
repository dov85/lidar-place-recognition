"""Every figure of the report, into one folder (the figures/ folder of the LaTeX project).

  python scripts/report_figures.py --summary /content/drive/MyDrive/lpr/evals_summary.csv --out figures
  python scripts/report_figures.py --summary ... --out figures --kitti_root /content/kitti   # + Figures 1, 2, 7

From evals_summary.csv (vector PDF, plus a PNG preview of each):
  fig_rotation.pdf   Figure 3: rotation curve, Recall@1 vs the query rotation amplitude alpha,
                     on 08 and the mean of 00, 02, 05, 06
  fig_compass.pdf    Figure 4: Recall@1 on 08 vs the compass noise sigma
  fig_tnet.pdf       Figure 5: self_top1 vs the rotation of the scan (T-Net check, 200 scans of 08)
  fig_sparsity.pdf   Figure 6: Recall@1 with 64, 32 and 16 beams, both and query modes,
                     on 08 and the mean of 00, 02, 05, 06

With --kitti_root also the figures that need raw scans (the two scans are downloaded if missing):
  fig_method.png     Figure 1: one scan as points, NDT cells and BEV image (08/000030, figure_method.py)
  fig_networks.png   Figure 2: the three networks (figure_networks.py; needs no data)
  ground_preview.png Figure 7: cells with and without ground removal (08/000657, ground_coverage.py)

evals_summary.csv is written by scripts/summarize_runs.py (notebook 3, section 5). The runs are found by
the hash at the end of their folder name, so the same file works for every copy of the runs; the
evaluation of the last checkpoint of a run is used.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OTHER = ["00", "02", "05", "06"]
RED, VIOLET, BLUE, TEAL = "#a31a1a", "#7a1fa2", "#1f3fa3", "#138585"

# report label -> (hash of the run folder, line style)
RUNS = {
    "NDT-orig": ("5c90e2", dict(color=RED, ls="--")),
    "NDT-rot": ("f60015", dict(color=RED, ls="-")),
    "NDT-REIN": ("3fe7a5", dict(color=VIOLET, ls="-")),
    "BEV-plain": ("b14094", dict(color=BLUE, ls="--")),
    "BEV-plain-rot": ("216fa5", dict(color=BLUE, ls="-")),
    "BEV-REIN-ours": ("4ac7d1", dict(color=TEAL, ls="-")),
}
FIG_RUNS = {
    "rotation": ["NDT-orig", "NDT-rot", "NDT-REIN", "BEV-plain", "BEV-plain-rot", "BEV-REIN-ours"],
    "compass": ["NDT-orig", "NDT-rot", "NDT-REIN", "BEV-plain", "BEV-plain-rot"],
    "tnet": ["NDT-orig", "NDT-rot"],
    "sparsity": ["NDT-orig", "NDT-rot", "NDT-REIN", "BEV-plain", "BEV-plain-rot", "BEV-REIN-ours"],
}
# sparsity bars: (mode, beams, label, colour)
BARS = [("both", 64, "64 beams", "#c8c8c8"), ("both", 32, "32, both", "#9a9a9a"),
        ("both", 16, "16, both", "#525252"), ("query", 32, "32, query", "#f0a6a6"),
        ("query", 16, "16, query", "#a31a1a")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": "#e3e3e3",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
MARK = dict(marker="o", markersize=3.2, linewidth=1.4)


def ckpt_epoch(c):
    m = re.search(r"(\d+)", str(c))
    return int(m.group(1)) if m else -1


def select(E, eval_type, label):
    """Rows of one evaluation type of one run, last checkpoint only."""
    h = RUNS[label][0]
    d = E[(E.eval_type == eval_type) & E.run.str.endswith("_" + h)]
    if d.empty:
        raise SystemExit(f"{eval_type}: no evaluation of {label} (run *_{h}) in the summary")
    last = max(d.checkpoint.unique(), key=ckpt_epoch)
    return d[d.checkpoint == last]


def curve(d, key):
    """(x, R@1 on 08, mean R@1 over the other four sequences) of a curve evaluation."""
    s08 = d[d.seq == "08"].groupby(key)["R@1"].mean()
    oth = d[d.seq.isin(OTHER)].groupby(key)["R@1"].mean()
    return s08.index.values, s08.values, oth.reindex(s08.index).values


def fig_rotation(E, out):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    for label in FIG_RUNS["rotation"]:
        x, y08, yo = curve(select(E, "rotation", label), "alpha_deg")
        ax[0].plot(x, y08, label=label, **RUNS[label][1], **MARK)
        ax[1].plot(x, yo, **RUNS[label][1], **MARK)
    for a, title in zip(ax, ["KITTI 08", "mean of 00, 02, 05, 06"]):
        a.set(title=title, xlabel=r"query rotation amplitude $\alpha$ (deg)", ylim=(0, 100),
              xticks=[0, 15, 45, 90, 180])
    ax[0].set_ylabel("Recall@1 (%)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.5, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save(fig, out, "fig_rotation")


def fig_compass(E, out):
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    for label in FIG_RUNS["compass"]:
        d = select(E, "compass", label)
        s = d[d.seq == "08"].groupby("sigma_deg")["R@1"].mean()
        ax.plot(s.index.values, s.values, label=label, **RUNS[label][1], **MARK)
    ax.set(xlabel=r"compass noise $\sigma$ (deg)", ylabel="Recall@1 on seq 08 (%)", ylim=(0, 100),
           xticks=[0, 5, 10, 20, 45])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, handlelength=2.2)
    save(fig, out, "fig_compass")


def fig_tnet(E, out):
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    for label in FIG_RUNS["tnet"]:
        d = select(E, "tnet", label).groupby("angle")["self_top1"].mean() * 100
        ax.plot(d.index.values, d.values, label=label, **RUNS[label][1], **MARK)
    ax.set(xlabel="rotation of the scan about z (deg)", ylabel="self_top1 (%)", ylim=(0, 102),
           xticks=[0, 15, 30, 45, 60, 90, 120, 150, 180])
    ax.legend(loc="lower left", frameon=False, handlelength=2.2)
    save(fig, out, "fig_tnet")


def fig_sparsity(E, out):
    labels = FIG_RUNS["sparsity"]
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 4.6))
    width = 0.15
    for a, seqs, title in [(ax[0], ["08"], "KITTI 08"), (ax[1], OTHER, "mean of 00, 02, 05, 06")]:
        for j, (mode, beams, name, colour) in enumerate(BARS):
            vals = []
            for label in labels:
                d = select(E, "sparsity", label)
                d = d[(d["mode"] == mode) & (d.beams == beams) & d.seq.isin(seqs)]
                vals.append(d.groupby("seq")["R@1"].mean().mean())
            xs = [i + (j - 2) * width for i in range(len(labels))]
            a.bar(xs, vals, width * 0.92, color=colour, label=name)
        a.set(title=title, ylabel="Recall@1 (%)", ylim=(0, 100), xticks=range(len(labels)))
        a.set_xticklabels(labels)
        a.grid(axis="x", visible=False)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, "fig_sparsity")


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("  ->", os.path.join(out, name + ".pdf"))


FIGURES = {"rotation": fig_rotation, "compass": fig_compass, "tnet": fig_tnet, "sparsity": fig_sparsity}
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
METHOD_SCAN, GROUND_SCAN = ("08", 30), ("08", 657)


def py(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


def scan_figures(kitti_root, out):
    """Figures 1, 2 and 7, which are drawn from raw KITTI scans (Figure 2 needs none)."""
    py(os.path.join(SCRIPTS, "figure_networks.py"), "--out", os.path.join(out, "fig_networks.png"))
    for seq, idx in (METHOD_SCAN, GROUND_SCAN):
        if not os.path.exists(os.path.join(kitti_root, "sequences", seq, "velodyne", f"{idx:06d}.bin")):
            py(os.path.join(SCRIPTS, "download_kitti.py"), "--out", kitti_root, "--seqs", seq, "--idx", idx)
    seq, idx = METHOD_SCAN
    py(os.path.join(SCRIPTS, "figure_method.py"), "--kitti_root", kitti_root, "--seq", seq, "--idx", idx,
       "--out", os.path.join(out, "fig_method.png"))
    seq, idx = GROUND_SCAN
    tmp = os.path.join(out, "_ground_check")
    py(os.path.join(SCRIPTS, "ground_coverage.py"), "--kitti_root", kitti_root, "--seq", seq,
       "--scans", idx, "--workers", 1, "--out_dir", tmp)
    shutil.copy(glob.glob(os.path.join(tmp, f"ground_previews_{seq}", f"{idx:06d}.png"))[0],
                os.path.join(out, "ground_preview.png"))
    shutil.rmtree(tmp)
    print("  ->", os.path.join(out, "fig_method.png, fig_networks.png, ground_preview.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="evals_summary.csv")
    ap.add_argument("--out", required=True, help="folder for the figures (e.g. the LaTeX figures/)")
    ap.add_argument("--only", nargs="+", choices=list(FIGURES), default=list(FIGURES))
    ap.add_argument("--kitti_root", default=None, help="also Figures 1, 2 and 7 (raw scans)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    E = pd.read_csv(a.summary, dtype={"seq": str, "run": str, "checkpoint": str})
    for name in a.only:
        FIGURES[name](E, a.out)
    if a.kitti_root:
        scan_figures(a.kitti_root, a.out)


if __name__ == "__main__":
    main()
