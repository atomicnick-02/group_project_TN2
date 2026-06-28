"""
Render block diagrams of the Diffusion Policy noise-prediction networks.

The DiffusionPolicy (diffusion_models/diffusion_policy.py) is backbone-agnostic:
both backbones share the SAME contract  forward(a_noisy, t, cond) -> eps_hat,
so swapping them is a one-line change. The two backbones differ only in how they
mix the three inputs:

  * MLP                 -- flattens the H-step action chunk and ignores its
                           temporal structure (one big concat -> dense stack).
  * TrajectoryTransformer -- treats the H actions as a SEQUENCE of tokens and
                           attends across them; t and cond are prepended as two
                           extra context tokens so every action attends to them.

This script draws one block diagram per backbone and writes two PNGs. The layer
shapes/sizes are taken from the training config (train_diffusion_policy.py) so
the diagrams match the deployed networks. Pure matplotlib (no torch needed).

Run from the repo root:
    python double_pendulum/visualize_architecture.py
    python double_pendulum/visualize_architecture.py --out-dir some/dir
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from matplotlib.path import Path as MplPath

# ── Architecture hyper-parameters (kept in sync with train_diffusion_policy.py) ──
# Conditioning vector: k state-history frames (NX_FEAT each) + a goal frame.
K, NX_FEAT, USE_GOAL = 4, 6, True
COND_DIM = K * NX_FEAT + (NX_FEAT if USE_GOAL else 0)   # = 30
H, NU    = 10, 2                                         # horizon, action dim
FLAT_A   = H * NU                                        # flattened action chunk = 20

# MLP backbone
MLP_HIDDEN, MLP_TIME_DIM, MLP_COND_EMB = 512, 24, 64

# Transformer backbone
TF_D_MODEL, TF_HEADS, TF_LAYERS, TF_FF = 128, 4, 2, 256

# ── Colour palette (one colour role per block type) ─────────────────────────────
C_INPUT  = "#cfe8ff"   # raw inputs
C_EMBED  = "#d6f5d6"   # per-input embeddings / projections
C_MERGE  = "#e0e0e0"   # concat / token-assembly bars
C_CORE   = "#ffe0b3"   # the backbone core (dense stack / encoder)
C_HEAD   = "#f7c6c6"   # output head
EDGE     = "#444444"


class Diagram:
    """Minimal box-and-arrow canvas on a 0..100 square in both axes."""

    def __init__(self, title, subtitle, figsize=(11, 13), ax=None, fig=None):
        # Standalone figure by default; pass (fig, ax) to draw into a panel of a
        # multi-axes figure (used to place the encoder-layer detail next to the
        # transformer architecture).
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig, self.ax = fig, ax
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.axis("off")
        self.boxes = {}
        if title:
            self.ax.text(50, 98.5, title, ha="center", va="top",
                         fontsize=15, fontweight="bold")
        if subtitle:
            self.ax.text(50, 95.0, subtitle, ha="center", va="top",
                         fontsize=10.0, color="#555555", style="italic")

    def box(self, name, cx, cy, w, h, text, fc, fontsize=9.5, bold=False):
        patch = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            linewidth=1.4, edgecolor=EDGE, facecolor=fc, zorder=2,
        )
        self.ax.add_patch(patch)
        self.ax.text(cx, cy, text, ha="center", va="center", zorder=3,
                     fontsize=fontsize, fontweight="bold" if bold else "normal")
        self.boxes[name] = (cx, cy, w, h)
        return name

    def _anchor(self, name, side):
        cx, cy, w, h = self.boxes[name]
        return {
            "top":    (cx, cy + h / 2),
            "bottom": (cx, cy - h / 2),
            "left":   (cx - w / 2, cy),
            "right":  (cx + w / 2, cy),
        }[side]

    def arrow(self, a, b, a_side="bottom", b_side="top", label=None, color=EDGE):
        x0, y0 = self._anchor(a, a_side)
        x1, y1 = self._anchor(b, b_side)
        self.ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
            linewidth=1.6, color=color, shrinkA=0, shrinkB=0, zorder=1,
        ))
        if label:
            self.ax.text((x0 + x1) / 2 + 1.5, (y0 + y1) / 2, label,
                         ha="left", va="center", fontsize=8, color="#333333")

    def add_node(self, name, cx, cy, r=2.3):
        """A small ⊕ circle used for residual (skip-connection) additions."""
        self.ax.add_patch(plt.Circle((cx, cy), r, facecolor=C_MERGE,
                                      edgecolor=EDGE, linewidth=1.4, zorder=3))
        self.ax.text(cx, cy, "+", ha="center", va="center",
                     fontsize=13, fontweight="bold", zorder=4)
        self.boxes[name] = (cx, cy, 2 * r, 2 * r)
        return name

    def skip(self, a, b, rail=16, label="residual"):
        """Dashed residual elbow: leave a's left, run down a left rail, enter b."""
        ax_, ay, aw, _ = self.boxes[a]
        bx_, by, bw, _ = self.boxes[b]
        verts = [(ax_ - aw / 2, ay), (rail, ay), (rail, by), (bx_ - bw / 2, by)]
        codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO]
        self.ax.add_patch(FancyArrowPatch(
            path=MplPath(verts, codes), arrowstyle="-|>", mutation_scale=13,
            linewidth=1.5, color="#999999", linestyle="--", zorder=1,
        ))
        if label:
            self.ax.text(rail - 1.4, (ay + by) / 2, label, ha="right",
                         va="center", fontsize=7.5, color="#777777", rotation=90)

    def frame(self, x, y, w, h, label):
        """Dashed enclosure marking 'one layer, repeated × N'."""
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=2",
            linewidth=1.6, edgecolor="#c8923a", facecolor="#fff6e9",
            linestyle=(0, (6, 4)), zorder=0,
        ))
        self.ax.text(x + w - 2, y + h - 1.6, label, ha="right", va="top",
                     fontsize=10, color="#a86b1f", fontweight="bold")

    def save(self, path):
        self.fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(self.fig)


def _legend(d):
    """Small colour key in the bottom-left corner."""
    items = [("input", C_INPUT), ("embedding / projection", C_EMBED),
             ("concat / tokens", C_MERGE), ("backbone core", C_CORE),
             ("output head", C_HEAD)]
    for i, (lbl, col) in enumerate(items):
        y = 12.0 - i * 2.4
        d.ax.add_patch(FancyBboxPatch((3, y - 0.8), 2.4, 1.6,
                                      boxstyle="round,pad=0.1", linewidth=1.0,
                                      edgecolor=EDGE, facecolor=col, zorder=2))
        d.ax.text(6.2, y, lbl, ha="left", va="center", fontsize=8)


def draw_mlp(out_path):
    d = Diagram(
        "Diffusion Policy  ·  MLP backbone",
        "noise-prediction net  eps_hat = f(a_noisy, t, cond)   "
        "— flattens the action chunk; ignores temporal structure",
    )
    # Three input lanes: actions (left), timestep (centre), conditioning (right)
    d.box("a_in", 20, 88, 30, 8,
          f"Noisy action chunk\n$a_t$  (B, H={H}, nu={NU})", C_INPUT, bold=True)
    d.box("t_in", 50, 88, 22, 8, "Diffusion step\n$t$  (B,)", C_INPUT, bold=True)
    d.box("c_in", 80, 88, 30, 8,
          f"Conditioning\nstate hist + goal  (B, {COND_DIM})", C_INPUT, bold=True)

    # Per-lane preprocessing / embedding
    d.box("a_flat", 20, 72, 30, 8, f"Flatten\n(B, H·nu = {FLAT_A})", C_EMBED)
    d.box("t_emb", 50, 72, 24, 11,
          f"Time MLP\nLinear(1→{MLP_TIME_DIM})  SiLU\n"
          f"Linear({MLP_TIME_DIM}→{MLP_TIME_DIM})", C_EMBED)
    d.box("c_emb", 80, 72, 30, 11,
          f"Cond MLP\nLinear({COND_DIM}→{MLP_COND_EMB})  SiLU\n"
          f"Linear({MLP_COND_EMB}→{MLP_COND_EMB})", C_EMBED)

    d.arrow("a_in", "a_flat"); d.arrow("t_in", "t_emb"); d.arrow("c_in", "c_emb")

    # Concatenate the three feature vectors
    total = FLAT_A + MLP_TIME_DIM + MLP_COND_EMB
    d.box("concat", 50, 55, 78, 7,
          f"Concatenate   [ $a_{{flat}}$  ‖  $t_{{emb}}$  ‖  $c_{{emb}}$ ]"
          f"   → (B, {total})", C_MERGE, bold=True)
    d.arrow("a_flat", "concat", b_side="left", label=f"{FLAT_A}")
    d.arrow("t_emb", "concat", label=f"{MLP_TIME_DIM}")
    d.arrow("c_emb", "concat", b_side="right", label=f"{MLP_COND_EMB}")

    # Dense backbone
    d.box("mlp", 50, 38, 56, 15,
          f"MLP\n"
          f"Linear({total}→{MLP_HIDDEN})  SiLU\n"
          f"Linear({MLP_HIDDEN}→{MLP_HIDDEN})  SiLU\n"
          f"Linear({MLP_HIDDEN}→{MLP_HIDDEN})  SiLU\n"
          f"Linear({MLP_HIDDEN}→{FLAT_A})", C_CORE, bold=True)
    d.arrow("concat", "mlp")

    # Output head
    d.box("reshape", 50, 21, 40, 7, f"Reshape → (B, H={H}, nu={NU})", C_HEAD)
    d.arrow("mlp", "reshape")
    d.box("out", 50, 9.5, 46, 7,
          "Predicted noise  $\\hat{\\epsilon}$  (B, H, nu)", C_HEAD, bold=True)
    d.arrow("reshape", "out")

    _legend(d)
    d.save(out_path)
    return out_path


def _draw_transformer_arch(d):
    """Left panel: end-to-end transformer backbone. The encoder core is kept
    as ONE compact block — its internals live in the companion detail panel."""
    # Three input lanes
    d.box("a_in", 20, 88, 30, 8,
          f"Noisy action chunk\n$a_t$  (B, H={H}, nu={NU})", C_INPUT, bold=True)
    d.box("t_in", 50, 88, 22, 8, "Diffusion step\n$t$  (B,)", C_INPUT, bold=True)
    d.box("c_in", 80, 88, 30, 8,
          f"Conditioning\nstate hist + goal  (B, {COND_DIM})", C_INPUT, bold=True)

    # Tokenisation: each lane becomes d_model-wide token(s)
    d.box("a_tok", 20, 71, 30, 11,
          f"Linear(nu→{TF_D_MODEL})\n+ learned pos-embed\n"
          f"→ {H} action tokens", C_EMBED)
    d.box("t_tok", 50, 71, 24, 11,
          f"Sinusoidal embed\nLinear  SiLU  Linear\n→ 1 token", C_EMBED)
    d.box("c_tok", 80, 71, 30, 11,
          f"Linear({COND_DIM}→{TF_D_MODEL})\nSiLU  Linear\n→ 1 token", C_EMBED)

    d.arrow("a_in", "a_tok"); d.arrow("t_in", "t_tok"); d.arrow("c_in", "c_tok")

    # Assemble the token sequence  [t, c, a_1..a_H]
    d.box("seq", 50, 54, 80, 7,
          f"Token sequence   [ $t$,  $c$,  $a_1$ … $a_{{{H}}}$ ]"
          f"   → (B, H+2 = {H + 2}, {TF_D_MODEL})", C_MERGE, bold=True)
    d.arrow("a_tok", "seq", b_side="left", label=f"{H}×{TF_D_MODEL}")
    d.arrow("t_tok", "seq", label=f"1×{TF_D_MODEL}")
    d.arrow("c_tok", "seq", b_side="right", label=f"1×{TF_D_MODEL}")

    # Transformer encoder core — single block; details are in the right panel.
    d.box("enc", 50, 37, 48, 12,
          f"Transformer Encoder\n× {TF_LAYERS} layers\n(detail ▸)",
          C_CORE, fontsize=11.5, bold=True)
    d.arrow("seq", "enc")

    # Output head: drop context tokens, project action positions back to nu
    d.box("drop", 50, 21, 56, 7,
          f"Keep H action tokens (drop $t$, $c$)  ·  LayerNorm  ·  "
          f"Linear({TF_D_MODEL}→{NU})", C_HEAD)
    d.arrow("enc", "drop")
    d.box("out", 50, 9.5, 46, 7,
          "Predicted noise  $\\hat{\\epsilon}$  (B, H, nu)", C_HEAD, bold=True)
    d.arrow("drop", "out")

    _legend(d)


def _draw_encoder_detail(d):
    """Right panel: the internals of ONE pre-norm Transformer Encoder layer
    (norm_first=True), i.e. the orange core block expanded — two residual
    sub-blocks (multi-head self-attention, then feed-forward)."""
    d_head = TF_D_MODEL // TF_HEADS
    d.frame(9, 7, 82, 81, f"one layer  ·  repeated × {TF_LAYERS}")

    d.box("din", 50, 92, 58, 7,
          f"Token sequence  (B, H+2={H + 2}, d={TF_D_MODEL})", C_MERGE, bold=True)

    # ── Sub-block A — Multi-Head Self-Attention (pre-norm + residual) ──
    d.box("ln1", 50, 80, 30, 5.5, "LayerNorm", C_EMBED, fontsize=9)
    d.box("qkv", 50, 70.5, 60, 8,
          f"Q, K, V = Linear({TF_D_MODEL}→{TF_D_MODEL})  ×3\n"
          f"reshape into {TF_HEADS} heads  (d$_{{head}}$={d_head})",
          C_EMBED, fontsize=8.8)
    d.box("attn", 50, 60, 60, 8.5,
          f"Scaled dot-product attention  (per head)\n"
          f"softmax$\\left(QK^\\top/\\sqrt{{{d_head}}}\\right)\\,V$",
          C_CORE, fontsize=9.5, bold=True)
    d.box("oproj", 50, 50.5, 60, 6,
          f"concat heads → Linear({TF_D_MODEL}→{TF_D_MODEL})",
          C_EMBED, fontsize=8.8)
    d.add_node("add1", 50, 43)

    d.arrow("din", "ln1"); d.arrow("ln1", "qkv"); d.arrow("qkv", "attn")
    d.arrow("attn", "oproj"); d.arrow("oproj", "add1")
    d.skip("din", "add1")

    # ── Sub-block B — position-wise Feed-Forward (pre-norm + residual) ──
    d.box("ln2", 50, 35, 30, 5.5, "LayerNorm", C_EMBED, fontsize=9)
    d.box("ff1", 50, 27, 54, 6,
          f"Linear({TF_D_MODEL}→{TF_FF})  ·  GELU", C_CORE, fontsize=9.5, bold=True)
    d.box("ff2", 50, 19.5, 54, 5.5,
          f"Linear({TF_FF}→{TF_D_MODEL})", C_CORE, fontsize=9.5, bold=True)
    d.add_node("add2", 50, 12.5)

    d.arrow("add1", "ln2"); d.arrow("ln2", "ff1"); d.arrow("ff1", "ff2")
    d.arrow("ff2", "add2")
    d.skip("add1", "add2")

    d.box("dout", 50, 4, 58, 5.5,
          f"Updated tokens  (B, H+2, {TF_D_MODEL})", C_MERGE, bold=True, fontsize=9)
    d.arrow("add2", "dout")


def draw_transformer(out_path):
    """Transformer backbone (left) with the encoder-layer internals broken out
    into a companion panel on the right, linked by a dashed 'zoom' arrow."""
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(19, 12.5), gridspec_kw={"width_ratios": [1.05, 0.9]})
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03, wspace=0.04)

    dL = Diagram(
        "Diffusion Policy  ·  Transformer backbone",
        "noise-prediction net  eps_hat = f(a_noisy, t, cond)   "
        "— actions are tokens; self-attention preserves temporal structure",
        ax=axL, fig=fig)
    _draw_transformer_arch(dL)

    dR = Diagram(
        "Inside one Transformer Encoder layer",
        "pre-norm residual blocks   (norm_first=True, GELU activation)",
        ax=axR, fig=fig)
    _draw_encoder_detail(dR)

    # Dashed 'zoom-in' connector: the orange core block → the detail panel.
    ex, ey, ew, _ = dL.boxes["enc"]
    bx, by, bw, _ = dR.boxes["din"]
    fig.add_artist(ConnectionPatch(
        xyA=(ex + ew / 2, ey), coordsA=axL.transData,
        xyB=(bx - bw / 2, by), coordsB=axR.transData,
        arrowstyle="-|>", linestyle=(0, (6, 4)), color="#c8923a",
        linewidth=2.0, mutation_scale=20, zorder=5,
    ))

    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    default_out = Path(__file__).resolve().parent / "graphs" / "architecture"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(default_out),
                    help="directory to write the two PNGs into")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = draw_mlp(out_dir / "diffusion_mlp_architecture.png")
    p2 = draw_transformer(out_dir / "diffusion_transformer_architecture.png")
    print(f"Saved:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
