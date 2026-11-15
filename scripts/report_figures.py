"""Every figure of the report, into one folder (the figures/ folder of the LaTeX project).

  python scripts/report_figures.py --summary /content/drive/MyDrive/lpr/evals_summary.csv --out figures
  python scripts/report_figures.py --summary ... --out figures --kitti_root /content/kitti   # + Figures 1, 7
  python scripts/report_figures.py --out figures --only networks                             # one figure

From evals_summary.csv (--summary; vector PDF, plus a PNG preview of each):
  fig_rotation.pdf   Figure 3: rotation curve, Recall@1 vs the query rotation amplitude alpha,
                     on 08 and the mean of 00, 02, 05, 06
  fig_compass.pdf    Figure 4: Recall@1 on 08 vs the compass noise sigma
  fig_tnet.pdf       Figure 5: self_top1 vs the rotation of the scan (T-Net check, 200 scans of 08)
  fig_sparsity.pdf   Figure 6: Recall@1 with 64, 32 and 16 beams, both and query modes,
                     on 08 and the mean of 00, 02, 05, 06
Needs no data (always drawn unless --only says otherwise):
  fig_networks.png   Figure 2: the three networks with the output shape of every stage
With --kitti_root, from raw scans (downloaded if missing):
  fig_method.png     Figure 1: one scan as points, NDT cells and BEV image (--method_scan, default 08 30)
  ground_preview.png Figure 7: cells with and without ground removal (08/000657, scripts/ground_coverage.py)

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
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lpr.evals import canonical  # noqa: E402

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
    "compass": ["NDT-orig", "NDT-rot", "NDT-REIN", "BEV-plain", "BEV-plain-rot", "BEV-REIN-ours"],
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
        x, y08, yo = curve(select(E, "rotation_curve", label), "alpha_deg")
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


# ---- Figure 1: one scan in the two representations

def default_style(f):
    """Figures 1 and 2 are drawn with matplotlib's default style, not the one of the result plots."""
    def wrapped(*args, **kw):
        with plt.style.context("default"):
            return f(*args, **kw)
    wrapped.__doc__ = f.__doc__
    return wrapped


@default_style
def fig_method(kitti_root, seq, idx, out, zoom=40.0):
    """Figure 1: one KITTI scan in the two representations, with the preprocessing of the
    experiments (NDT_DEFAULTS, BEV_DEFAULTS): (a) the cropped scan from above, coloured by height;
    (b) the 2000 NDT cells after ground removal (2-sigma ellipses of the x-y marginals);
    (c) the BEVPlace++ density image of the same scan. `zoom`: half size of the shown area (m)."""
    from lpr import kitti, ndt_check
    from lpr.bev import make_bev
    from lpr.experiments import BEV_DEFAULTS, NDT_DEFAULTS
    from lpr.ndt_g import ndt_cells
    from lpr.preprocess import crop, preprocess

    P = NDT_DEFAULTS
    scan = kitti.load_scan(kitti.scan_path(kitti_root, seq, idx))
    cropped = crop(scan, P["crop"], P["half_size"])
    pts = preprocess(scan, P["ground"], P["half_size"], P["crop"])
    means, covs, _ = ndt_cells(pts, **{k: P[k] for k in ("k", "over", "radius", "min_pts", "knn", "cov_eps")})
    bev_args = {k: BEV_DEFAULTS[k] for k in ("half_size", "z_abs", "voxel", "res", "max_count", "ground")
                if k in BEV_DEFAULTS}
    img, _ = make_bev(scan, **bev_args)

    z = zoom
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    sc = axes[0].scatter(cropped[:, 0], cropped[:, 1], s=0.05, c=cropped[:, 2], cmap="viridis",
                         vmin=-2.5, vmax=2.0, rasterized=True)
    fig.colorbar(sc, ax=axes[0], fraction=0.046, label="z (m)")
    axes[0].set_title(f"(a) point cloud, {len(cropped):,} points")
    axes[1].scatter(pts[:, 0], pts[:, 1], s=0.02, c="0.75", rasterized=True)
    ndt_check._ellipses(axes[1], means, covs, "tab:red")
    axes[1].set_title(f"(b) {len(means)} NDT cells (ground removed)")
    for ax in axes[:2]:
        ax.set_xlim(-z, z); ax.set_ylim(-z, z); ax.set_aspect("equal")
        ax.set_xlabel("x (m, forward)"); ax.set_ylabel("y (m, left)")
    h = BEV_DEFAULTS["half_size"]
    # image rows = -x, columns = -y (src/lpr/bev.py): transposed and flipped so that, as in
    # (a) and (b), x points right and y up
    axes[2].imshow(img.T[:, ::-1], cmap="gray_r", extent=[-h, h, -h, h])
    axes[2].set_xlim(-z, z); axes[2].set_ylim(-z, z)
    axes[2].set_title(f"(c) BEV density image, {img.shape[0]}x{img.shape[1]} px, 0.4 m/px")
    axes[2].set_xlabel("x (m, forward)"); axes[2].set_ylabel("y (m, left)")
    fig.suptitle(f"KITTI {seq}/{idx:06d}")
    fig.tight_layout()
    path = os.path.join(out, "fig_method.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("  ->", path)


# ---- Figure 2: the three networks

C_NDT, C_OWN, C_BEV, C_HEAD, C_IO = "#fde2e1", "#ece2f5", "#dde9f7", "#e3f2e6", "#f2f2f2"


def box(ax, x, y, w, h, title, body, shape, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=color, ec="#555555", lw=0.9))
    ax.text(x + w / 2, y + h - 0.016, title, ha="center", va="top", fontsize=8.6, weight="bold")
    if body:
        ax.text(x + w / 2, y + h / 2 - 0.004, body, ha="center", va="center", fontsize=7.2,
                color="#333333", linespacing=1.25)
    ax.text(x + w / 2, y + 0.012, shape, ha="center", va="bottom", fontsize=8,
            color="#b03a2e", family="monospace")


def row(ax, y, h, label, stages, gap=0.012, x0=0.01, x1=0.99):
    ax.text(x0, y + h + 0.012, label, fontsize=10, weight="bold", va="bottom")
    total = sum(s[0] for s in stages)
    avail = x1 - x0 - gap * (len(stages) - 1)
    x = x0
    for i, (wr, title, body, shape, color) in enumerate(stages):
        w = avail * wr / total
        box(ax, x, y, w, h, title, body, shape, color)
        if i < len(stages) - 1:
            ax.annotate("", (x + w + gap, y + h / 2), (x + w, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=1, color="#444444"))
        x += w + gap


@default_style
def fig_networks(out):
    """Figure 2: the three networks with the output shape of every stage, and the inside of the
    Transformer encoder layer, the T-Net and the ResNet34 part of BEVPlace++. Shapes are per scan
    (batch dimension left out); N = 2000 NDT cells, R = 8 rotations. Needs no data."""
    fig = plt.figure(figsize=(17, 11.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    h = 0.115

    row(ax, 0.835, h, "(a) NDT-Transformer, official architecture [1]", [
        (1.0, "NDT cells", "mean (3) +\ncovariance (9)", "2000×12", C_IO),
        (1.2, "T-Net (STN3d)", "3×3 matrix T\nmeans m → m T\n(see (e))", "2000×12", C_NDT),
        (1.1, "conv1 1×1", "12 → 256\nBatchNorm, ReLU", "x1: 2000×256", C_NDT),
        (1.35, "Transformer encoder", "3 layers, d_model 256,\n4 heads, FFN 1024\n(see (d))", "2000×256", C_NDT),
        (1.1, "concat", "[x1, Transformer(x1)]", "2000×512", C_NDT),
        (1.1, "conv2 1×1", "512 → 1024\nBatchNorm, ReLU", "2000×1024", C_NDT),
        (1.45, "NetVLADLoupe", "K = 64 clusters:\n64×1024 = 65,536\n→ FC 256, BatchNorm,\ncontext gating", "256", C_HEAD),
        (0.9, "descriptor", "", "256", C_IO),
    ])
    row(ax, 0.645, h, "(b) Own NDT network, REIN-like rotations + Loupe head (3fe7a5)", [
        (1.0, "NDT cells", "mean (3) +\ncovariance (9)", "2000×12", C_IO),
        (1.25, "rotate cells about z", "θ = 0°, 45°, …, 315°\nμ → Rμ,  Σ → RΣRᵀ", "8×2000×12", C_OWN),
        (2.1, "shared encoder g", "conv1 1×1 12→256 → Transformer ×3 (2000×256)\n→ concat (2000×512) → conv2 1×1 512→1024\nsame weights for the 8 copies, no T-Net",
         "8×2000×1024", C_OWN),
        (1.35, "max over the 8 copies", "for every cell i and\nchannel c separately;\nno warp back (cells keep\ntheir index)", "2000×1024", C_OWN),
        (1.45, "NetVLADLoupe", "K = 64: 64×1024\n→ FC 256, BatchNorm,\ngating, L2 norm", "256", C_HEAD),
        (0.9, "descriptor", "", "256", C_IO),
    ])
    row(ax, 0.455, h, "(c) BEVPlace++ with REIN [2]", [
        (1.05, "BEV image", "density, 0.4 m/px,\ngray copied to 3 ch.", "3×201×201", C_IO),
        (1.15, "rotate image", "θ = 0°, 45°, …, 315°\n(bicubic grid_sample)", "8×3×201×201", C_BEV),
        (1.45, "f: ResNet34 conv1…layer2", "shared weights,\nImageNet-pretrained\n(see (f))", "8×128×26×26", C_BEV),
        (1.15, "rotate back by −θ", "every channel resampled\nas an image, so pixel\n(u,v) = same place", "8×128×26×26", C_BEV),
        (1.2, "max over the 8 maps", "for every channel c and\npixel (u, v) separately", "128×26×26", C_BEV),
        (1.15, "upsample + L2 norm", "bicubic to H/4 × W/4;\n2500 local features", "128×50×50", C_BEV),
        (1.3, "NetVLAD", "K = 64 clusters,\nk-means init,\nintra-norm + L2", "64×128", C_HEAD),
        (0.9, "descriptor", "", "8192", C_IO),
    ])

    # details
    hd = 0.085
    ax.text(0.01, 0.395, "(d) One Transformer encoder layer (×3; post-norm, dropout 0.1, no positional encoding)",
            fontsize=10, weight="bold")
    x = 0.01
    for wr, title, body, shape in [
        (1.0, "input", "", "2000×256"),
        (1.9, "multi-head self-attention", "4 heads × 64 dims: Q, K, V 2000×64 each;\nsoftmax(QKᵀ/√64) 2000×2000 per head", "2000×256"),
        (1.0, "add & LayerNorm", "", "2000×256"),
        (1.5, "feed-forward", "Linear 256→1024, ReLU,\nLinear 1024→256", "2000×1024 → 2000×256"),
        (1.0, "add & LayerNorm", "", "2000×256"),
    ]:
        w = 0.98 * wr / 6.4 - 0.012
        box(ax, x, 0.295, w, hd, title, body, shape, C_NDT)
        if title != "add & LayerNorm" or x < 0.6:
            ax.annotate("", (x + w + 0.012, 0.295 + hd / 2), (x + w, 0.295 + hd / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=1, color="#444444"))
        x += w + 0.012

    ax.text(0.01, 0.25, "(e) T-Net (STN3d) of NDT-Transformer", fontsize=10, weight="bold")
    x = 0.01
    stn = [("means m", "", "2000×3"), ("conv 1×3, ReLU", "", "2000×64"), ("conv 1×1, ReLU", "", "2000×128"),
           ("conv 1×1, ReLU", "", "2000×1024"), ("max over cells", "", "1024"),
           ("FC, ReLU", "", "512"), ("FC, ReLU", "", "256"), ("FC + identity", "", "9 → T: 3×3")]
    for i, (title, body, shape) in enumerate(stn):
        w = 0.98 / len(stn) - 0.012
        box(ax, x, 0.16, w, 0.075, title, body, shape, C_NDT)
        if i < len(stn) - 1:
            ax.annotate("", (x + w + 0.012, 0.1975), (x + w, 0.1975),
                        arrowprops=dict(arrowstyle="-|>", lw=1, color="#444444"))
        x += w + 0.012

    ax.text(0.01, 0.115, "(f) f in BEVPlace++ = the first layers of ResNet34", fontsize=10, weight="bold")
    x = 0.01
    res = [("input", "3×201×201"), ("conv 7×7 /2, BN, ReLU", "64×101×101"), ("max-pool 3×3 /2", "64×51×51"),
           ("layer1: 3 basic blocks", "64×51×51"), ("layer2: 4 basic blocks, /2", "128×26×26")]
    for i, (title, shape) in enumerate(res):
        w = 0.98 / len(res) - 0.012
        box(ax, x, 0.02, w, 0.08, title, "", shape, C_BEV)
        if i < len(res) - 1:
            ax.annotate("", (x + w + 0.012, 0.06), (x + w, 0.06),
                        arrowprops=dict(arrowstyle="-|>", lw=1, color="#444444"))
        x += w + 0.012

    path = os.path.join(out, "fig_networks.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("  ->", path)


# ---- command line

SUMMARY_FIGURES = {"rotation": fig_rotation, "compass": fig_compass, "tnet": fig_tnet,
                   "sparsity": fig_sparsity}
SCAN_FIGURES = ["method", "ground"]          # need --kitti_root
ALL = list(SUMMARY_FIGURES) + ["networks"] + SCAN_FIGURES
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
GROUND_SCAN = ("08", 657)


def py(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


def need_scan(kitti_root, seq, idx):
    if not os.path.exists(os.path.join(kitti_root, "sequences", seq, "velodyne", f"{idx:06d}.bin")):
        py(os.path.join(SCRIPTS, "download_kitti.py"), "--out", kitti_root, "--seqs", seq, "--idx", idx)


def fig_ground(kitti_root, out):
    """Figure 7: the cells of one scan with and without ground removal (scripts/ground_coverage.py)."""
    seq, idx = GROUND_SCAN
    need_scan(kitti_root, seq, idx)
    tmp = os.path.join(out, "_ground_check")
    py(os.path.join(SCRIPTS, "ground_coverage.py"), "--kitti_root", kitti_root, "--seq", seq,
       "--scans", idx, "--workers", 1, "--out_dir", tmp)
    shutil.copy(glob.glob(os.path.join(tmp, f"ground_previews_{seq}", f"{idx:06d}.png"))[0],
                os.path.join(out, "ground_preview.png"))
    shutil.rmtree(tmp)
    print("  ->", os.path.join(out, "ground_preview.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=None, help="evals_summary.csv (Figures 3-6)")
    ap.add_argument("--out", required=True, help="folder for the figures (e.g. the LaTeX figures/)")
    ap.add_argument("--only", nargs="+", choices=ALL, default=None, help="default: every figure the inputs allow")
    ap.add_argument("--kitti_root", default=None, help="raw KITTI scans (Figures 1 and 7)")
    ap.add_argument("--method_scan", nargs=2, default=["08", "30"], metavar=("SEQ", "IDX"),
                    help="the scan of Figure 1")
    a = ap.parse_args()
    only = a.only or ((list(SUMMARY_FIGURES) if a.summary else []) + ["networks"]
                      + (SCAN_FIGURES if a.kitti_root else []))
    if a.summary is None and set(only) & set(SUMMARY_FIGURES):
        ap.error("--summary is needed for " + ", ".join(sorted(set(only) & set(SUMMARY_FIGURES))))
    if a.kitti_root is None and set(only) & set(SCAN_FIGURES):
        ap.error("--kitti_root is needed for " + ", ".join(sorted(set(only) & set(SCAN_FIGURES))))
    os.makedirs(a.out, exist_ok=True)
    if set(only) & set(SUMMARY_FIGURES):
        E = pd.read_csv(a.summary, dtype={"seq": str, "run": str, "checkpoint": str})
        E["eval_type"] = E.eval_type.map(canonical)   # summaries written with the earlier names
    for name in only:
        if name in SUMMARY_FIGURES:
            SUMMARY_FIGURES[name](E, a.out)
        elif name == "networks":
            fig_networks(a.out)
        elif name == "method":
            seq, idx = a.method_scan[0], int(a.method_scan[1])
            need_scan(a.kitti_root, seq, idx)
            fig_method(a.kitti_root, seq, idx, a.out)
        elif name == "ground":
            fig_ground(a.kitti_root, a.out)


if __name__ == "__main__":
    main()
