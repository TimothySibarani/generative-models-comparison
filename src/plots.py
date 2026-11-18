"""Pembuatan lima figur wajib (dan figur pendukung) untuk laporan. Semua disimpan PNG 200 dpi."""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from . import config as C

matplotlib.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 100})


def _np(grid):
    return grid.permute(1, 2, 0).cpu().numpy()


def _row(imgs, nrow=None):
    imgs = imgs.detach().float().cpu().clamp(0, 1)
    return _np(make_grid(imgs, nrow=nrow or len(imgs), padding=2, pad_value=1.0))


def _save(fig, name):
    path = os.path.join(C.FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("saved:", path)
    return path


def stacked_rows(rows, title, fname, label_width=0.18):
    """rows: list of (label, tensor_imgs) atau (label, tensor_imgs, nrow). Satu panel per label."""
    heights = [int(np.ceil(len(r[1]) / (r[2] if len(r) > 2 else len(r[1])))) for r in rows]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.45 * sum(heights) + 0.6),
                             gridspec_kw={"height_ratios": heights})
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, rows):
        label, imgs = r[0], r[1]
        ax.imshow(_row(imgs, nrow=r[2] if len(r) > 2 else None), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    return fig, _save(fig, fname)


# ---------------------------------------------------------------- Figur 1
def figure1_reconstruction(x, recons: dict):
    rows = [("Ground Truth", x)] + [(k, v) for k, v in recons.items()]
    return stacked_rows(rows, "Figur 1 - Rekonstruksi Test Set: Vanilla AE vs Sparse AE vs VAE",
                        "fig1_reconstruction.png")


# ---------------------------------------------------------------- Figur 2
def figure2_sampling(samples: dict):
    fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4.4))
    for ax, (k, v) in zip(axes, samples.items()):
        ax.imshow(_row(v[:16], nrow=4), cmap="gray")
        ax.set_title(k)
        ax.axis("off")
    fig.suptitle("Figur 2 - Random Generative Sampling (4x4) dari prior N(0, I)", fontsize=13)
    fig.tight_layout()
    return fig, _save(fig, "fig2_random_sampling.png")


# ---------------------------------------------------------------- Figur 3
def _minmax(t):
    lo, hi = t.amin(dim=(1, 2, 3), keepdim=True), t.amax(dim=(1, 2, 3), keepdim=True)
    return (t - lo) / (hi - lo + 1e-8)


def figure3_mlp_limits(x, wt_class, wt_latent, pinv_latent, frozen_rec, ae_rec, prior_samples, vae_samples,
                       wt_class_pre=None, wt_latent_pre=None):
    rows = [
        ("(1) Ground Truth", x),
        ("(2a) Sigmoid(W^T ... y_c)\none-hot kelas 0..9", wt_class),
    ]
    if wt_class_pre is not None:
        rows.append(("(2a') sama, kontras\ndinormalisasi min-max", _minmax(wt_class_pre)))
    rows.append(("(2b) Sigmoid(W^T ... z_MLP)\n(citra uji)", wt_latent))
    if wt_latent_pre is not None:
        rows.append(("(2b') sama, kontras\ndinormalisasi min-max", _minmax(wt_latent_pre)))
    rows += [
        ("(2c) pseudo-inverse W^+\n(pembanding)", pinv_latent),
        ("(3a) Frozen Latent Decoder\n(z_MLP)", frozen_rec),
        ("(3b) Vanilla AE (z_AE)", ae_rec),
        ("(4a) Sampling z~N(mu_z, s_z^2)\nruang laten MLP", prior_samples, 8),
        ("(4b) Pembanding: VAE\nz~N(0, I)", vae_samples, 8),
    ]
    return stacked_rows(rows, "Figur 3 - Bukti Empiris Batasan Bobot & Latent Features MLP Classifier",
                        "fig3_mlp_classifier_limits.png")


# ---------------------------------------------------------------- Figur 4
def figure4_interpolation(interps: dict, alphas):
    fig, axes = plt.subplots(len(interps), 1, figsize=(7.5, 1.45 * len(interps) + 0.9))
    for ax, (k, v) in zip(axes, interps.items()):
        ax.imshow(_row(v), cmap="gray")
        ax.set_yticks([])
        ax.set_ylabel(k, rotation=0, ha="right", va="center")
        ax.set_xticks([(i + 0.5) * 30 for i in range(len(alphas))])
        ax.set_xticklabels([f"a={a:.1f}" for a in alphas], fontsize=8)
    fig.suptitle("Figur 4 - Linear Latent Interpolation  z(a) = (1-a) z_A + a z_B", fontsize=13)
    fig.tight_layout()
    return fig, _save(fig, "fig4_latent_interpolation.png")


# ---------------------------------------------------------------- Figur 5
def figure5_losses_and_ddpm(H: dict, traj):
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9])

    def curve(ax, title, series, ylabel="loss", logy=False):
        for label, ys in series:
            ax.plot(np.arange(1, len(ys) + 1), ys, label=label, lw=1.6, marker=".", ms=4)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    curve(fig.add_subplot(gs[0, 0]), "Vanilla AE (MSE)", [("train", H["ae"]["train_loss"]), ("val", H["ae"]["val_loss"])])
    sae_rec = [l["recon"] for l in H["sae"]["logs"]]
    curve(fig.add_subplot(gs[0, 1]), "Sparse AE (MSE + lambda*L1)",
          [("total train", H["sae"]["train_loss"]), ("total val", H["sae"]["val_loss"]), ("MSE train", sae_rec)])
    curve(fig.add_subplot(gs[0, 2]), "VAE (BCE + KL, per citra)",
          [("total train", H["vae"]["train_loss"]), ("total val", H["vae"]["val_loss"]),
           ("KL train", [l["kl"] for l in H["vae"]["logs"]])])
    for j, (key, title) in enumerate([("gan", "Vanilla GAN (MLP)"), ("dcgan", "DCGAN")]):
        ax = fig.add_subplot(gs[1, j])
        it_g, it_d = np.array(H[key]["iter_g"]), np.array(H[key]["iter_d"])
        ipe = max(1, len(it_g) // max(1, len(H[key]["g_loss"])))
        xs = np.arange(len(it_g)) / ipe
        ax.plot(xs, it_g, alpha=0.25, lw=0.5, color="C0")
        ax.plot(xs, it_d, alpha=0.25, lw=0.5, color="C1")
        ep = np.arange(1, len(H[key]["g_loss"]) + 1)
        ax.plot(ep, H[key]["g_loss"], color="C0", lw=2, label="Generator (mean/epoch)")
        ax.plot(ep, H[key]["d_loss"], color="C1", lw=2, label="Discriminator (mean/epoch)")
        ax.set_title(f"{title}: loss G & D (tipis = per-iterasi)")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    curve(fig.add_subplot(gs[1, 2]), "DDPM (MSE prediksi noise)",
          [("train", H["ddpm"]["train_loss"]), ("val", H["ddpm"]["val_loss"])], logy=True)

    ax = fig.add_subplot(gs[2, :])
    steps = [t for t, _ in traj]
    cols = torch.stack([imgs for _, imgs in traj], dim=0)     # (S, n, 1, 28, 28)
    n = cols.size(1)
    grid = torch.cat([cols[:, i] for i in range(n)], dim=0)   # baris per sampel
    ax.imshow(_row(grid, nrow=len(steps)), cmap="gray")
    ax.set_yticks([])
    ax.set_xticks([(i + 0.5) * 30 for i in range(len(steps))])
    ax.set_xticklabels([f"t={t}" for t in steps], fontsize=8)
    ax.set_title("DDPM reverse diffusion trajectory  x_T -> x_0")
    fig.suptitle("Figur 5 - Training Loss Curves & DDPM Denoising Trajectory", fontsize=14)
    fig.tight_layout()
    return fig, _save(fig, "fig5_losses_ddpm_trajectory.png")


# ---------------------------------------------------------------- Figur pendukung
def figure_gan_diagnostics(H):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for key, ls in (("gan", "-"), ("dcgan", "--")):
        ep = np.arange(1, len(H[key]["d_real"]) + 1)
        axes[0].plot(ep, H[key]["d_real"], ls, label=f"{key}: D(x)")
        axes[0].plot(ep, H[key]["d_fake"], ls, label=f"{key}: D(G(z))")
        axes[1].plot(ep, H[key]["g_grad"], ls, label=key)
    axes[0].set_title("Output Discriminator (rata-rata)")
    axes[1].set_title("Norma gradien Generator  ||dL_G/dtheta_G||")
    for a in axes:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("Figur Pendukung - Diagnostik stabilitas GAN vs DCGAN")
    fig.tight_layout()
    return fig, _save(fig, "figS1_gan_diagnostics.png")


def figure_filters(ae, sae, n=64):
    """Bobot encoder lapisan pertama (784->256) divisualisasikan sebagai filter 28x28."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, (name, W) in zip(axes, (("Vanilla AE", ae.encoder[1].weight), ("Sparse AE", sae.enc1[1].weight))):
        w = W.detach().cpu()[:n].view(-1, 1, 28, 28)
        w = (w - w.amin(dim=(2, 3), keepdim=True)) / (w.amax(dim=(2, 3), keepdim=True) - w.amin(dim=(2, 3), keepdim=True) + 1e-8)
        ax.imshow(_row(w, nrow=8), cmap="gray")
        ax.set_title(f"{name}: 64 filter pertama W1")
        ax.axis("off")
    fig.suptitle("Figur Pendukung - Fitur hidden layer Vanilla AE vs Sparse AE")
    fig.tight_layout()
    return fig, _save(fig, "figS2_ae_vs_sae_filters.png")


def figure_sparsity_hist(h_ae, h_sae):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(0, max(h_ae.max(), h_sae.max()), 60)
    ax.hist(h_ae.flatten(), bins=bins, alpha=0.6, label="Vanilla AE", log=True)
    ax.hist(h_sae.flatten(), bins=bins, alpha=0.6, label="Sparse AE", log=True)
    ax.set_xlabel("nilai aktivasi hidden 256-d (test set)")
    ax.set_ylabel("frekuensi (log)")
    ax.legend()
    ax.set_title("Figur Pendukung - Distribusi aktivasi hidden layer")
    fig.tight_layout()
    return fig, _save(fig, "figS3_activation_hist.png")


def figure_class_hist(cov: dict):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    w = 0.8 / len(cov)
    for i, (k, (ent, hist)) in enumerate(cov.items()):
        ax.bar(np.arange(10) + i * w, hist, width=w, label=f"{k} (H={ent:.3f})")
    ax.set_xticks(np.arange(10) + 0.4 - w / 2)
    ax.set_xticklabels(C.CLASSES, rotation=20, fontsize=8)
    ax.set_ylabel("jumlah sampel")
    ax.legend(fontsize=8)
    ax.set_title("Figur Pendukung - Mode coverage: distribusi kelas sampel sintetis (diklasifikasi MLP)")
    fig.tight_layout()
    return fig, _save(fig, "figS4_mode_coverage.png")
