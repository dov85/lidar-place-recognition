"""Architecture figure of the report: the three networks with the output shape of every stage,
and the inside of the Transformer encoder layer, the T-Net and the ResNet34 part of BEVPlace++.
Shapes are per scan (batch dimension left out); N = 2000 NDT cells, R = 8 rotations.

  python scripts/figure_networks.py --out fig_networks.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
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

    fig.savefig(a.out, dpi=180)
    print("->", a.out)


if __name__ == "__main__":
    main()
