"""
Architecture diagrams for the MLP and TrajectoryTransformer noise-prediction
networks used in the diffusion policy.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

# Output dir is anchored to this script's location so it works from any cwd.
IMAGES_DIR = Path(__file__).resolve().parent / "images"


# ── colour palette ────────────────────────────────────────────────────────────
C_INPUT   = "#AED6F1"   # light blue   – raw inputs
C_EMBED   = "#A9DFBF"   # light green  – embedding blocks
C_MAIN    = "#F9E79F"   # light yellow – main network / transformer layers
C_OUT     = "#F1948A"   # light red    – output
C_TOKEN   = "#D2B4DE"   # lavender     – token-level ops
C_CONCAT  = "#FAD7A0"   # peach        – concat / reshape

FONT = dict(fontsize=8, ha="center", va="center", fontfamily="monospace")


# ── helpers ───────────────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, label, color, fontsize=8, bold=False):
    rect = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.03", linewidth=0.8,
        edgecolor="#555", facecolor=color, zorder=3,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, label, fontsize=fontsize, ha="center", va="center",
            fontfamily="monospace", fontweight=weight, zorder=4,
            wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#444"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9),
        zorder=2,
    )


def label(ax, x, y, text, fontsize=7, color="#333"):
    ax.text(x, y, text, fontsize=fontsize, ha="center", va="center",
            color=color, fontfamily="monospace", zorder=5)


# ── MLP diagram ───────────────────────────────────────────────────────────────
def draw_mlp(ax, td=24, ced=64, hidden=256):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_title("MLP Noise-Prediction Network", fontsize=11, fontweight="bold", pad=6)

    bw, bh = 2.4, 0.55   # default box width / height

    # ── row 1: inputs ──────────────────────────────────────────────────────
    y_in = 13.0
    box(ax, 2.0, y_in, bw, bh, f"a_noisy\n(B, H, nu)", C_INPUT, 7)
    box(ax, 5.0, y_in, bw, bh, f"t\n(B,)",             C_INPUT, 7)
    box(ax, 8.0, y_in, bw, bh, f"cond\n(B, cond_dim)", C_INPUT, 7)

    # ── row 2: first transforms ────────────────────────────────────────────
    y2 = 11.6
    box(ax, 2.0, y2, bw, bh, "Flatten\n→ (B, H·nu)", C_CONCAT, 7)
    box(ax, 5.0, y2, bw, bh, f"Linear(1→{td})\nSiLU", C_EMBED, 7)
    box(ax, 8.0, y2, bw, bh, f"Linear(cd→{ced})\nSiLU", C_EMBED, 7)

    # ── row 3: second embed layers ─────────────────────────────────────────
    y3 = 10.2
    box(ax, 5.0, y3, bw, bh, f"Linear({td}→{td})\n= t_embed", C_EMBED, 7)
    box(ax, 8.0, y3, bw, bh, f"Linear({ced}→{ced})\n= c_embed", C_EMBED, 7)

    # ── row 4: concat ──────────────────────────────────────────────────────
    y4 = 8.7
    box(ax, 5.0, y4, 5.2, bh, "Concat  [a_flat ‖ t_embed ‖ c_embed]", C_CONCAT, 8)

    # ── row 5: MLP hidden layers ───────────────────────────────────────────
    y5, y6, y7 = 7.3, 5.9, 4.5
    box(ax, 5.0, y5, bw, bh, f"Linear(H·nu+{td}+{ced}→{hidden})\nSiLU", C_MAIN, 7)
    box(ax, 5.0, y6, bw, bh, f"Linear({hidden}→{hidden})\nSiLU",         C_MAIN, 7)
    box(ax, 5.0, y7, bw, bh, f"Linear({hidden}→{hidden})\nSiLU",         C_MAIN, 7)

    # ── row 6: output linear + reshape ─────────────────────────────────────
    y8 = 3.1
    box(ax, 5.0, y8, bw, bh, f"Linear({hidden}→H·nu)", C_MAIN, 7)
    y9 = 1.8
    box(ax, 5.0, y9, bw, bh, "Reshape\n→ (B, H, nu)",  C_OUT,   7)

    # ── arrows ─────────────────────────────────────────────────────────────
    dh = bh / 2
    # inputs → transforms
    arrow(ax, 2.0, y_in - dh, 2.0, y2 + dh)
    arrow(ax, 5.0, y_in - dh, 5.0, y2 + dh)
    arrow(ax, 8.0, y_in - dh, 8.0, y2 + dh)
    # flatten stays in col 2, but needs to merge down to concat
    arrow(ax, 5.0, y2 - dh, 5.0, y3 + dh)
    arrow(ax, 8.0, y2 - dh, 8.0, y3 + dh)
    # all three → concat
    arrow(ax, 2.0, y2 - dh, 5.0 - 2.0, y4 + dh)   # a_flat
    arrow(ax, 5.0, y3 - dh, 5.0,        y4 + dh)   # t_embed
    arrow(ax, 8.0, y3 - dh, 5.0 + 2.0,  y4 + dh)   # c_embed
    # MLP chain
    for ya, yb in [(y4, y5), (y5, y6), (y6, y7), (y7, y8), (y8, y9)]:
        arrow(ax, 5.0, ya - dh, 5.0, yb + dh)

    # ── dim annotations ────────────────────────────────────────────────────
    label(ax, 2.85, (y_in + y2) / 2, f"(B, H·nu)", 6.5)
    label(ax, 5.7,  (y2  + y3) / 2, f"(B, {td})",  6.5)
    label(ax, 8.7,  (y2  + y3) / 2, f"(B, {ced})", 6.5)


# ── Transformer diagram ───────────────────────────────────────────────────────
def draw_transformer(ax, d=128, n_heads=4, n_layers=4, ff=256):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 16)
    ax.axis("off")
    ax.set_title("TrajectoryTransformer Noise-Prediction Network",
                 fontsize=11, fontweight="bold", pad=6)

    bw, bh = 2.8, 0.6
    dh = bh / 2

    # ── inputs (row 1) ────────────────────────────────────────────────────
    y_in = 15.2
    box(ax, 2.0,  y_in, bw, bh, "a_noisy\n(B, H, nu)",   C_INPUT, 7)
    box(ax, 6.0,  y_in, bw, bh, "t  (B,)",               C_INPUT, 7)
    box(ax, 10.0, y_in, bw, bh, "cond\n(B, cond_dim)",   C_INPUT, 7)

    # ── action branch ─────────────────────────────────────────────────────
    y_a1 = 13.6
    box(ax, 2.0, y_a1, bw, bh, f"Linear(nu→{d})\naction_in", C_EMBED, 7)
    y_a2 = 12.2
    box(ax, 2.0, y_a2, bw, bh, f"+ pos_embed\n(1, H, {d})",  C_TOKEN, 7)
    y_a3 = 10.8
    box(ax, 2.0, y_a3, bw, bh, f"a_tok\n(B, H, {d})",        C_TOKEN, 7, bold=True)

    arrow(ax, 2.0, y_in  - dh, 2.0, y_a1 + dh)
    arrow(ax, 2.0, y_a1  - dh, 2.0, y_a2 + dh)
    arrow(ax, 2.0, y_a2  - dh, 2.0, y_a3 + dh)

    # ── timestep branch ───────────────────────────────────────────────────
    y_t1 = 13.6
    box(ax, 6.0, y_t1, bw, bh, f"SinusoidalEmbed\n→ ({d},)",  C_EMBED, 7)
    y_t2 = 12.2
    box(ax, 6.0, y_t2, bw, bh, f"Linear({d}→{d})\nSiLU",      C_EMBED, 7)
    y_t3 = 10.8
    box(ax, 6.0, y_t3, bw, bh, f"Linear({d}→{d})\n= t_tok (B,1,{d})", C_TOKEN, 7, bold=True)

    arrow(ax, 6.0, y_in - dh, 6.0, y_t1 + dh)
    arrow(ax, 6.0, y_t1 - dh, 6.0, y_t2 + dh)
    arrow(ax, 6.0, y_t2 - dh, 6.0, y_t3 + dh)

    # ── conditioning branch ────────────────────────────────────────────────
    y_c1 = 13.6
    box(ax, 10.0, y_c1, bw, bh, f"Linear(cd→{d})\nSiLU",      C_EMBED, 7)
    y_c2 = 12.2
    box(ax, 10.0, y_c2, bw, bh, f"Linear({d}→{d})\n= c_tok (B,1,{d})", C_TOKEN, 7, bold=True)

    arrow(ax, 10.0, y_in - dh, 10.0, y_c1 + dh)
    arrow(ax, 10.0, y_c1 - dh, 10.0, y_c2 + dh)

    # ── concat all tokens ─────────────────────────────────────────────────
    y_cat = 9.3
    box(ax, 6.0, y_cat, 8.0, bh,
        f"Concat  [t_tok ‖ c_tok ‖ a_tok]\n→ tokens  (B, H+2, {d})", C_CONCAT, 8)

    arrow(ax, 2.0,  y_a3  - dh, 2.2,   y_cat + dh)
    arrow(ax, 6.0,  y_t3  - dh, 6.0,   y_cat + dh)
    arrow(ax, 10.0, y_c2  - dh, 9.8,   y_cat + dh)

    # ── transformer encoder block ─────────────────────────────────────────
    y_enc = 7.7
    enc_h = 1.5
    rect = mpatches.FancyBboxPatch((3.3, y_enc - enc_h / 2), 5.4, enc_h,
                                   boxstyle="round,pad=0.05", linewidth=1.2,
                                   edgecolor="#888", facecolor=C_MAIN, zorder=3)
    ax.add_patch(rect)
    ax.text(6.0, y_enc + 0.28,
            f"TransformerEncoder  ×{n_layers} layers",
            fontsize=8.5, ha="center", va="center", fontweight="bold",
            fontfamily="monospace", zorder=4)
    ax.text(6.0, y_enc - 0.22,
            f"Pre-LN  |  MHSA ({n_heads} heads)  |  FFN(dim={ff})  |  GELU  |  dropout=0",
            fontsize=7, ha="center", va="center", color="#333",
            fontfamily="monospace", zorder=4)

    arrow(ax, 6.0, y_cat - dh, 6.0, y_enc + enc_h / 2)

    # ── slice action positions ─────────────────────────────────────────────
    y_sl = 5.9
    box(ax, 6.0, y_sl, bw + 0.4, bh, f"Slice  [:, 2:]  →  (B, H, {d})", C_TOKEN, 8)
    arrow(ax, 6.0, y_enc - enc_h / 2, 6.0, y_sl + dh)

    # ── layer norm + output proj ──────────────────────────────────────────
    y_ln  = 4.5
    y_out = 3.1
    y_fin = 1.8
    box(ax, 6.0, y_ln,  bw, bh, f"LayerNorm({d})", C_MAIN, 8)
    box(ax, 6.0, y_out, bw, bh, f"Linear({d}→nu)\naction_out", C_MAIN, 8)
    box(ax, 6.0, y_fin, bw, bh, "output\n(B, H, nu)", C_OUT, 8, bold=True)

    arrow(ax, 6.0, y_sl  - dh, 6.0, y_ln  + dh)
    arrow(ax, 6.0, y_ln  - dh, 6.0, y_out + dh)
    arrow(ax, 6.0, y_out - dh, 6.0, y_fin + dh)

    # ── dim annotation ────────────────────────────────────────────────────
    label(ax, 7.1, (y_cat + y_enc) / 2 + 0.2,
          f"(B, H+2, {d})", 6.5)


# ── legend ────────────────────────────────────────────────────────────────────
def add_legend(fig):
    legend_items = [
        mpatches.Patch(facecolor=C_INPUT,  edgecolor="#555", label="Input tensor"),
        mpatches.Patch(facecolor=C_EMBED,  edgecolor="#555", label="Embedding layer"),
        mpatches.Patch(facecolor=C_TOKEN,  edgecolor="#555", label="Token operation"),
        mpatches.Patch(facecolor=C_CONCAT, edgecolor="#555", label="Concat / reshape"),
        mpatches.Patch(facecolor=C_MAIN,   edgecolor="#555", label="Core network layer"),
        mpatches.Patch(facecolor=C_OUT,    edgecolor="#555", label="Output"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=6,
               fontsize=7.5, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    # MLP
    fig_mlp, ax_mlp = plt.subplots(figsize=(8, 10))
    fig_mlp.suptitle("Diffusion Policy — MLP Noise-Prediction Network",
                     fontsize=12, fontweight="bold", y=0.99)
    draw_mlp(ax_mlp)
    add_legend(fig_mlp)
    plt.figure(fig_mlp.number)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    out_mlp = IMAGES_DIR / "architecture_mlp.png"
    fig_mlp.savefig(out_mlp, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_mlp}")

    # Transformer
    fig_tr, ax_tr = plt.subplots(figsize=(8, 10))
    fig_tr.suptitle("Diffusion Policy — TrajectoryTransformer Noise-Prediction Network",
                    fontsize=12, fontweight="bold", y=0.99)
    draw_transformer(ax_tr)
    add_legend(fig_tr)
    plt.figure(fig_tr.number)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    out_tr = IMAGES_DIR / "architecture_transformer.png"
    fig_tr.savefig(out_tr, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_tr}")


if __name__ == "__main__":
    main()
