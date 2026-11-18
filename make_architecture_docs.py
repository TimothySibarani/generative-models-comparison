"""Membuat diagram skematik arsitektur (Bagian II laporan) dan tabel konfigurasi parameter.

    python make_architecture_docs.py

Output:
    outputs/figures/arch_<model>.png      diagram per model
    outputs/figures/arch_all.png          gabungan ketujuh model
    outputs/config_table.md / .csv        tabel konfigurasi (lapisan, laten, aktivasi, optimizer, lr, ...)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src import config as C
from src import models as Md

# warna per jenis blok
COL = {"in": "#E8E8E8", "enc": "#BFD7EA", "lat": "#FFD8A8", "dec": "#C8E6C9",
       "gen": "#D1C4E9", "disc": "#F8BBD0", "out": "#E8E8E8", "loss": "#FFF3B0", "unet": "#B2EBF2"}


def box(ax, x, y, w, h, text, kind, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=COL[kind], ec="#333333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, style="-|>", ls="-", color="#333333", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=10,
                                 lw=1, ls=ls, color=color, connectionstyle=f"arc3,rad={rad}"))


def chain(ax, blocks, y=0.0, h=0.9, w=1.35, gap=0.35, x0=0.0, fs=8):
    """blocks: list (teks, jenis). Menggambar blok berurutan kiri->kanan, mengembalikan posisi x tiap blok."""
    xs = []
    x = x0
    for i, (t, k) in enumerate(blocks):
        box(ax, x, y, w, h, t, k, fs)
        xs.append(x)
        if i < len(blocks) - 1:
            arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)
        x += w + gap
    return xs, x - gap


def finish(ax, title, xmax, ymin, ymax):
    ax.set_xlim(-0.3, xmax + 0.3)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")


# ---------------------------------------------------------------- per model
def draw_ae(ax):
    blocks = [("Input x\n1x28x28 → 784", "in"), ("Linear 784→256\nReLU", "enc"), ("Linear 256→64", "enc"),
              (f"z ∈ R^{C.LATENT_AE}", "lat"), ("Linear 64→256\nReLU", "dec"), ("Linear 256→784\nSigmoid", "dec"),
              ("x̂  1x28x28", "out")]
    xs, xe = chain(ax, blocks)
    box(ax, xe - 1.35, -1.3, 1.35, 0.8, "MSE(x, x̂)", "loss")
    arrow(ax, xe - 0.675, 0, xe - 0.675, -0.5)
    ax.text(xs[1], -0.9, "Encoder", fontsize=8, color="#1f4e79")
    ax.text(xs[4], -0.9, "Decoder", fontsize=8, color="#2e7d32")
    finish(ax, "(a) Vanilla Autoencoder", xe, -1.5, 1.2)


def draw_sae(ax):
    blocks = [("Input x\n784", "in"), (f"Linear 784→{C.SPARSE_HIDDEN}\nReLU = h", "enc"), ("Linear 256→64", "enc"),
              (f"z ∈ R^{C.LATENT_AE}", "lat"), ("Linear 64→256\nReLU", "dec"), ("Linear 256→784\nSigmoid", "dec"),
              ("x̂  1x28x28", "out")]
    xs, xe = chain(ax, blocks)
    box(ax, xs[1] - 0.2, -1.3, 1.75, 0.8, f"λ‖h‖₁  (λ={C.SPARSE_LAMBDA:g})", "loss")
    arrow(ax, xs[1] + 0.675, 0, xs[1] + 0.675, -0.5, ls="--")
    box(ax, xe - 1.35, -1.3, 1.35, 0.8, "MSE(x, x̂)", "loss")
    arrow(ax, xe - 0.675, 0, xe - 0.675, -0.5)
    ax.text(xs[3], -1.0, "L = MSE + λ‖h‖₁", fontsize=8)
    finish(ax, "(b) Sparse Autoencoder (penalti L1 pada hidden 256-d)", xe, -1.5, 1.2)


def draw_vae(ax):
    blocks = [("Input x\n784", "in"), ("Linear 784→256\nReLU", "enc"), ("Linear→μ (64)\nLinear→log σ² (64)", "enc"),
              ("z = μ + σ⊙ε\nε~N(0,I)", "lat"), ("Linear 64→256\nReLU", "dec"), ("Linear 256→784\nSigmoid", "dec"),
              ("x̂  1x28x28", "out")]
    xs, xe = chain(ax, blocks, w=1.45)
    box(ax, xs[2] - 0.1, -1.3, 1.65, 0.8, "KL(q(z|x) ‖ N(0,I))", "loss")
    arrow(ax, xs[2] + 0.72, 0, xs[2] + 0.72, -0.5, ls="--")
    box(ax, xe - 1.45, -1.3, 1.45, 0.8, "BCE(x, x̂)", "loss")
    arrow(ax, xe - 0.72, 0, xe - 0.72, -0.5)
    box(ax, xs[3], 1.3, 1.45, 0.7, "Sampling:\nz~N(0,I) → decoder", "lat", fs=7)
    arrow(ax, xs[3] + 0.72, 1.3, xs[3] + 0.72, 0.9, ls=":")
    finish(ax, "(c) Variational Autoencoder (reparameterization trick)", xe, -1.5, 2.2)


def draw_gan(ax, dc=False):
    if dc:
        g = [(f"z ∈ R^{C.Z_GAN}\n~N(0,I)", "lat"), ("ConvT 100→256\nk7 s1 → 7x7\nBN+ReLU", "gen"),
             ("ConvT 256→128\nk4 s2 → 14x14\nBN+ReLU", "gen"), ("ConvT 128→1\nk4 s2 → 28x28\nTanh", "gen"), ("G(z)\n1x28x28", "out")]
        d = [("x real / G(z)\n1x28x28", "in"), ("Conv 1→64\nk4 s2 → 14x14\nLeakyReLU 0.2", "disc"),
             ("Conv 64→128\nk4 s2 → 7x7\nBN+LeakyReLU", "disc"), ("Conv 128→1\nk7 s1 → 1x1", "disc"), ("logit D(x)", "out")]
    else:
        g = [(f"z ∈ R^{C.Z_GAN}\n~N(0,I)", "lat"), ("Linear 100→256\nLeakyReLU", "gen"), ("Linear 256→512\nLeakyReLU", "gen"),
             ("Linear 512→1024\nLeakyReLU", "gen"), ("Linear 1024→784\nTanh", "gen")]
        d = [("x real / G(z)\n784", "in"), ("Linear 784→512\nLeakyReLU\nDropout 0.3", "disc"),
             ("Linear 512→256\nLeakyReLU\nDropout 0.3", "disc"), ("Linear 256→1", "disc"), ("logit D(x)", "out")]
    h = 1.1
    _, xe1 = chain(ax, g, y=1.5, h=h, w=1.5, gap=0.35)
    _, xe2 = chain(ax, d, y=0.0, h=h, w=1.5, gap=0.35)
    ax.text(0.75, 1.3, "↓ input D: G(z) atau x real", fontsize=7, color="#6a1b9a", ha="center", va="center")
    ax.text(xe1 + 0.15, 2.0, "Generator G", fontsize=9, color="#4a148c")
    ax.text(xe2 + 0.15, 0.45, "Discriminator D", fontsize=9, color="#880e4f")
    box(ax, xe2 - 1.5, -1.5, 3.4, 0.9, "BCE: L_D = −[log D(x) + log(1−D(G(z)))]\nL_G = −log D(G(z))  (non-saturating)", "loss", fs=7)
    arrow(ax, xe2 - 0.75, 0, xe2 - 0.75, -0.6)
    finish(ax, "(e) DCGAN (konvolusional penuh, tanpa pooling/FC)" if dc else "(d) Vanilla GAN (fully connected)",
           xe2 + 2.0, -1.8, 2.9)


def draw_ddpm(ax):
    ch = C.DDPM_BASE_CH
    w, h = 1.45, 0.95
    # jalur turun (atas), tengah, jalur naik (atas kanan)
    pos = {
        "in": (0, 2.4, "x_t 1x28x28\n+ t", "in"),
        "c0": (1.8, 2.4, f"Conv3x3 1→{ch}\nResBlock {ch}\n28x28", "unet"),
        "d1": (3.6, 1.2, f"Down s2 → 14x14\nResBlock {ch}→{2 * ch}", "unet"),
        "m": (5.4, 0.0, f"Down s2 → 7x7\nResBlock {2 * ch}→{4 * ch}\nResBlock {4 * ch}", "unet"),
        "u1": (7.2, 1.2, f"Up ConvT → 14x14\n⊕skip, ResBlock→{2 * ch}", "unet"),
        "u0": (9.0, 2.4, f"Up ConvT → 28x28\n⊕skip, ResBlock→{ch}", "unet"),
        "out": (10.8, 2.4, "GN+SiLU+Conv\nε̂_θ(x_t,t)\n1x28x28", "out"),
    }
    for _, (x, y, t, k) in pos.items():
        box(ax, x, y, w, h, t, k, fs=7)
    order = ["in", "c0", "d1", "m", "u1", "u0", "out"]
    for a, b in zip(order, order[1:]):
        xa, ya = pos[a][0] + w, pos[a][1] + h / 2
        xb, yb = pos[b][0], pos[b][1] + h / 2
        arrow(ax, xa, ya, xb, yb)
    for a, b in (("c0", "u0"), ("d1", "u1")):   # skip connections
        arrow(ax, pos[a][0] + w, pos[a][1] + h * 0.85, pos[b][0], pos[b][1] + h * 0.85, ls="--", color="#00838f")
    ax.text(5.0, 3.6, "skip connections (concat)", fontsize=8, color="#00838f")
    box(ax, 0, 0.0, 3.1, 1.1, f"Time embedding:\nsinusoidal({ch}) → Linear→SiLU→Linear (256)\nditambahkan ke tiap ResBlock", "lat", fs=7)
    box(ax, 7.6, -1.6, 4.7, 1.2,
        f"Forward: x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε,  T={C.DDPM_T}, β {C.DDPM_BETA[0]:g}→{C.DDPM_BETA[1]:g}\n"
        "Loss: ‖ε − ε̂_θ(x_t, t)‖²   |   Sampling: T langkah reverse, EMA 0.999", "loss", fs=7)
    finish(ax, "(f) DDPM - U-Net denoiser ε_θ(x_t, t)", 12.4, -1.9, 4.0)


def draw_mlp(ax):
    blocks = [("Input x\n784", "in"), ("fc1: Linear 784→256\nReLU  (W1)", "enc"), ("fc2: Linear 256→64\nReLU  (W2)", "enc"),
              ("z_MLP ∈ R^64", "lat"), ("fc3: Linear 64→10\n(W3)", "enc"), ("softmax\nP(y|x)", "out")]
    xs, xe = chain(ax, blocks, w=1.55)
    box(ax, xe - 1.55, -1.3, 1.55, 0.8, "CrossEntropy(y, ŷ)", "loss")
    arrow(ax, xe - 0.77, 0, xe - 0.77, -0.5)
    box(ax, xs[3] - 0.2, 1.3, 2.4, 0.8, "Tahap 4: Frozen decoder\nLinear 64→256 ReLU→784 Sigmoid", "dec", fs=7)
    arrow(ax, xs[3] + 0.77, 0.9, xs[3] + 0.77, 1.3, ls="--")
    finish(ax, "(g) MLP Classifier pendamping (struktur = encoder AE)", xe, -1.5, 2.3)


DRAW = [("ae", draw_ae, 3.0), ("sae", draw_sae, 3.0), ("vae", draw_vae, 3.6), ("gan", lambda a: draw_gan(a, False), 4.6),
        ("dcgan", lambda a: draw_gan(a, True), 4.6), ("ddpm", draw_ddpm, 5.6), ("mlp", draw_mlp, 3.6)]


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def config_table():
    ddpm = Md.DDPM()
    rows = [
        ["Vanilla AE", "784-256-64-256-784 (FC)", "64", "ReLU; Sigmoid (output)", "MSE", "Adam", C.LR["ae"], C.EPOCHS["ae"], n_params(Md.VanillaAE())],
        ["Sparse AE", "784-256(sparse)-64-256-784 (FC)", f"64 (bottleneck) / {C.SPARSE_HIDDEN} (sparse)", "ReLU; Sigmoid (output)",
         f"MSE + λ‖h‖₁, λ={C.SPARSE_LAMBDA:g}", "Adam", C.LR["sae"], C.EPOCHS["sae"], n_params(Md.SparseAE())],
        ["VAE", "784-256-(μ,logσ²: 64)-256-784 (FC)", "64", "ReLU; Sigmoid (output)", "BCE + KL", "Adam", C.LR["vae"], C.EPOCHS["vae"], n_params(Md.VAE())],
        ["Vanilla GAN", "G: 100-256-512-1024-784; D: 784-512-256-1 (FC)", f"{C.Z_GAN} (prior)", "LeakyReLU 0.2; Tanh (G out); Dropout 0.3 (D)",
         "BCE minimax, G non-saturating", "Adam β=(0.5,0.999)", C.LR["gan"], C.EPOCHS["gan"],
         f"G {n_params(Md.MLPGenerator()):,} / D {n_params(Md.MLPDiscriminator()):,}"],
        ["DCGAN", "G: ConvT 7→14→28; D: Conv 28→14→7→1", f"{C.Z_GAN} (prior)", "G: BN+ReLU, Tanh; D: BN+LeakyReLU 0.2",
         "BCE minimax, G non-saturating, init N(0,0.02)", "Adam β=(0.5,0.999)", C.LR["dcgan"], C.EPOCHS["dcgan"],
         f"G {n_params(Md.DCGenerator()):,} / D {n_params(Md.DCDiscriminator()):,}"],
        ["DDPM", f"U-Net {C.DDPM_BASE_CH}-{2 * C.DDPM_BASE_CH}-{4 * C.DDPM_BASE_CH} ch, 28→14→7, ResBlock+GroupNorm",
         "28x28 (dimensi citra)", "SiLU; time emb sinusoidal",
         f"MSE prediksi noise, T={C.DDPM_T}, β {C.DDPM_BETA[0]:g}→{C.DDPM_BETA[1]:g}, EMA 0.999, AMP fp16", "Adam", C.LR["ddpm"],
         C.EPOCHS["ddpm"], n_params(ddpm.net)],
        ["MLP Classifier", "784-256-64-10 (FC)", "64 (z_MLP)", "ReLU", "Cross-Entropy", "Adam", C.LR["mlp"], C.EPOCHS["mlp"], n_params(Md.MLPClassifier())],
        ["Frozen Latent Decoder", "64-256-784 (FC), di atas z_MLP beku", "64", "ReLU; Sigmoid (output)", "MSE", "Adam",
         C.LR["frozen_dec"], C.EPOCHS["frozen_dec"], n_params(Md.FrozenLatentDecoder())],
    ]
    cols = ["Model", "Lapisan", "Dim Laten", "Aktivasi", "Loss", "Optimizer", "Learning Rate", "Epoch", "Jumlah Parameter"]
    df = pd.DataFrame(rows, columns=cols)
    df["Jumlah Parameter"] = df["Jumlah Parameter"].map(lambda v: f"{v:,}" if isinstance(v, int) else v)
    common = (f"\n\nPengaturan umum: dataset {C.DATASET}, split 80/10/10 (56.000/7.000/7.000), batch size {C.BATCH_SIZE}, "
              f"seed {C.SEED}, input dinormalisasi ke [0,1] (GAN/DCGAN/DDPM memakai [-1,1]).\n")
    df.to_csv(os.path.join(C.OUT_DIR, "config_table.csv"), index=False)
    with open(os.path.join(C.OUT_DIR, "config_table.md"), "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False) + common)
    print(df.to_string())


def main():
    C.make_dirs()
    for name, fn, hgt in DRAW:
        fig, ax = plt.subplots(figsize=(14, hgt))
        fn(ax)
        fig.savefig(os.path.join(C.FIG_DIR, f"arch_{name}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(len(DRAW), 1, figsize=(14, sum(h for _, _, h in DRAW)),
                             gridspec_kw={"height_ratios": [h for _, _, h in DRAW]})
    for ax, (_, fn, _) in zip(axes, DRAW):
        fn(ax)
    fig.suptitle("Diagram Skematik Arsitektur Model", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "arch_all.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Diagram tersimpan di", C.FIG_DIR)
    config_table()


if __name__ == "__main__":
    main()
