"""Orkestrasi eksperimen. Semua state disimpan dalam dict `S` agar mudah dipanggil per-tahap
dari notebook maupun dari run_all.py.
"""
import json
import os

import pandas as pd
import torch

from . import config as C
from . import metrics as M
from . import mlp_limits as L
from . import models as Md
from . import plots as P
from .data import get_loaders, get_splits
from .train import train_gan, train_generic


# ------------------------------------------------------------------ setup
def setup():
    C.set_seed()
    C.make_dirs()
    splits = get_splits()
    S = {"splits": splits, "loaders": get_loaders(splits), "H": {}, "models": {}}
    print(f"Device: {C.DEVICE} | QUICK={C.QUICK}")
    print({k: len(v[0]) for k, v in splits.items()})
    return S


# ------------------------------------------------------------------ Bagian 2 - Tahap 1
@torch.no_grad()
def accuracy(clf, loader):
    clf.eval()
    correct = sum((clf(x.to(C.DEVICE)).argmax(1).cpu() == y).sum().item() for x, y in loader)
    return correct / len(loader.dataset)


def train_classifier(S):
    C.set_seed()
    clf = Md.MLPClassifier()
    S["H"]["mlp"] = train_generic("mlp_classifier", clf, S["loaders"], C.EPOCHS["mlp"], C.LR["mlp"],
                                  loss_fn=lambda m, x, y: m.loss(x, y))
    acc = accuracy(clf, S["loaders"]["test"])
    S["models"]["mlp"] = clf.eval()
    S["mlp_test_acc"] = acc
    print(f"MLP Classifier test accuracy = {acc * 100:.2f}%  (syarat >= 88%: {'OK' if acc >= 0.88 else 'BELUM'})")
    return clf


# ------------------------------------------------------------------ Autoencoder family
def train_autoencoders(S):
    for key, cls in (("ae", Md.VanillaAE), ("sae", Md.SparseAE), ("vae", Md.VAE)):
        C.set_seed()
        m = cls()
        S["H"][key] = train_generic(key, m, S["loaders"], C.EPOCHS[key], C.LR[key])
        S["models"][key] = m.eval()


# ------------------------------------------------------------------ GAN family
def train_gans(S):
    C.set_seed()
    G, D = Md.MLPGenerator(), Md.MLPDiscriminator()
    S["H"]["gan"] = train_gan("gan", G, D, S["loaders"], C.EPOCHS["gan"], C.LR["gan"])
    S["models"]["gan"] = G.eval()

    C.set_seed()
    G, D = Md.DCGenerator(), Md.DCDiscriminator()
    G.apply(Md.dcgan_init)
    D.apply(Md.dcgan_init)
    S["H"]["dcgan"] = train_gan("dcgan", G, D, S["loaders"], C.EPOCHS["dcgan"], C.LR["dcgan"])
    S["models"]["dcgan"] = G.eval()


# ------------------------------------------------------------------ DDPM
def train_ddpm(S):
    C.set_seed()
    m = Md.DDPM()
    S["H"]["ddpm"] = train_generic("ddpm", m, S["loaders"], C.EPOCHS["ddpm"], C.LR["ddpm"], ema=True, amp=True)
    S["models"]["ddpm"] = m.eval()


def sample_fns(S):
    """Fungsi sampling k citra [0,1] dari prior N(0,I) untuk setiap model generatif."""
    Mo = S["models"]
    zg = lambda k: torch.randn(k, C.Z_GAN, device=C.DEVICE)
    return {
        "VAE": lambda k: Mo["vae"].sample(k),
        "Vanilla GAN": lambda k: (Mo["gan"](zg(k)) + 1) / 2,
        "DCGAN": lambda k: (Mo["dcgan"](zg(k)) + 1) / 2,
        "DDPM": lambda k: Mo["ddpm"].sample(k),
    }


# ------------------------------------------------------------------ Bagian 2 - Tahap 2..5
def mlp_limit_experiments(S):
    clf, ae, vae = S["models"]["mlp"], S["models"]["ae"], S["models"]["vae"]
    test_loader = S["loaders"]["test"]
    R = {"test_accuracy": S["mlp_test_acc"]}

    # Tahap 2-3 + diagnostik bobot
    R["weights"] = L.weight_diagnostics(clf, test_loader)
    x8 = _pick_examples(S, 8)
    wt_cls, wt_cls_pre = L.transpose_from_class(clf)
    wt_lat, wt_lat_pre = L.transpose_from_latent(clf, x8)
    pinv_lat = L.pinv_from_latent(clf, x8)
    R["transpose_latent_recon"] = M.recon_metrics(lambda x: L.transpose_from_latent(clf, x)[0], test_loader)
    R["pinv_latent_recon"] = M.recon_metrics(lambda x: L.pinv_from_latent(clf, x), test_loader)

    # Tahap 4: frozen latent decoder
    C.set_seed()
    pipe = L.FrozenPipe(clf, Md.FrozenLatentDecoder())
    S["H"]["frozen_dec"] = train_generic("frozen_decoder", pipe, S["loaders"], C.EPOCHS["frozen_dec"], C.LR["frozen_dec"])
    pipe.eval()
    S["models"]["frozen_pipe"] = pipe
    R["frozen_decoder_recon"] = M.recon_metrics(pipe.reconstruct, test_loader)
    R["vanilla_ae_recon"] = M.recon_metrics(ae.reconstruct, test_loader)

    # Variasi intra-kelas yang tersisa di ruang laten (z_MLP vs z_AE)
    z_mlp, y, mu, sd = L.latent_stats(clf, test_loader)
    z_ae = torch.cat([ae.encode(x.to(C.DEVICE)).cpu() for x, _ in test_loader])
    R["intra_class_var_ratio"] = {"z_MLP": L.intra_class_variance_ratio(z_mlp, y),
                                  "z_AE": L.intra_class_variance_ratio(z_ae, y)}
    R["z_MLP_fraction_zero"] = float((z_mlp == 0).float().mean())

    # Tahap 5: sampling di ruang laten MLP
    g = torch.Generator().manual_seed(C.SEED)
    prior16 = L.prior_sampling(pipe.decoder, mu, sd, 16, generator=g)
    fake = M.generate(lambda k: L.prior_sampling(pipe.decoder, mu, sd, k), C.FID_N)
    real = S["splits"]["test"][0][:C.FID_N]
    R["mlp_latent_sampling_fid"] = M.fid_score(real, fake)
    R["mlp_latent_sampling_coverage"] = M.mode_coverage(clf, fake)[0]
    S["mlp_fake"] = fake

    torch.manual_seed(C.SEED)
    vae16 = S["models"]["vae"].sample(16)
    S["fig3"] = P.figure3_mlp_limits(x8, wt_cls, wt_lat, pinv_lat, pipe.reconstruct(x8.to(C.DEVICE)),
                                     ae.reconstruct(x8.to(C.DEVICE)), prior16, vae16,
                                     wt_class_pre=wt_cls_pre, wt_latent_pre=wt_lat_pre)
    S["mlp_results"] = R
    with open(os.path.join(C.OUT_DIR, "mlp_limits_results.json"), "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))
    return R


def _pick_examples(S, n):
    """Satu contoh per kelas (kelas 0..n-1) dari test set -> grid yang beragam."""
    x, y = S["splits"]["test"]
    idx = [int((y == c).nonzero()[0]) for c in range(n)]
    return x[idx].to(C.DEVICE)


# ------------------------------------------------------------------ Evaluasi kuantitatif (Bagian IV)
@torch.no_grad()
def evaluate_all(S):
    Mo, H, test = S["models"], S["H"], S["loaders"]["test"]
    real = S["splits"]["test"][0][:C.FID_N]
    clf = Mo["mlp"]
    x64 = S["splits"]["test"][0][:C.LATENCY_BATCH].to(C.DEVICE)
    fns = sample_fns(S)
    rows, S["fake"], S["coverage"] = [], {}, {}

    def epoch_time(key):
        return sum(H[key]["epoch_time"]) / len(H[key]["epoch_time"])

    # Autoencoder deterministik + VAE (rekonstruksi)
    for key, name in (("ae", "Vanilla Autoencoder"), ("sae", "Sparse Autoencoder"), ("vae", "Variational Autoencoder (VAE)")):
        m = Mo[key]
        rec = M.recon_metrics(m.reconstruct, test)
        if key == "vae":
            torch.manual_seed(C.SEED)
            lat = M.latency_ms(lambda: fns["VAE"](C.LATENCY_BATCH))
            fake = M.generate(fns["VAE"], C.FID_N)
            fid = M.fid_score(real, fake)
            S["fake"]["VAE"] = fake
            S["coverage"]["VAE"] = M.mode_coverage(clf, fake)
            stab = "Stabil / Konvergen"
            dim = str(C.LATENT_AE)
            cov = S["coverage"]["VAE"][0]
        else:
            lat = M.latency_ms(lambda: m.reconstruct(x64))
            fid, cov = "N/A (Latent Voids)", "N/A"
            stab = "Stabil / Konvergen"
            dim = f"{C.LATENT_AE}" if key == "ae" else f"{C.LATENT_AE} (bottleneck) / {C.SPARSE_HIDDEN} (sparse)"
        rows.append({"Model": name, "Dim Laten": dim, "Train Time/Epoch (s)": epoch_time(key),
                     "Latency (ms/batch 64)": lat[0], "Test MSE": rec["mse"], "Test PSNR (dB)": rec["psnr"],
                     "Test SSIM": rec["ssim"], "FID": fid, "Mode Coverage (H)": cov,
                     "Loss CV (stabilitas)": M.stability_cv(H[key]["iter_loss"]), "Kestabilan Training": stab})

    # GAN, DCGAN, DDPM (generatif murni)
    for key, name, dim in (("gan", "Generative Adversarial Network (GAN)", f"{C.Z_GAN} (prior noise)"),
                           ("dcgan", "Deep Convolutional GAN (DCGAN)", f"{C.Z_GAN} (prior noise)"),
                           ("ddpm", "Denoising Diffusion (DDPM)", "28x28 (dimensi citra)")):
        label = {"gan": "Vanilla GAN", "dcgan": "DCGAN", "ddpm": "DDPM"}[key]
        torch.manual_seed(C.SEED)
        lat = M.latency_ms(lambda: fns[label](C.LATENCY_BATCH), repeats=3 if key == "ddpm" else 10,
                           warmup=1 if key == "ddpm" else 2)
        fake = M.generate(fns[label], C.FID_N)
        S["fake"][label] = fake
        S["coverage"][label] = M.mode_coverage(clf, fake)
        if key == "ddpm":
            cv, stab = M.stability_cv(H[key]["iter_loss"]), "Sangat Stabil"
        else:
            cv = M.stability_cv(H[key]["iter_g"])
            stab = "Rentan osilasi / mode collapse" if key == "gan" else "Cukup stabil (sensitif arsitektur)"
        rows.append({"Model": name, "Dim Laten": dim, "Train Time/Epoch (s)": epoch_time(key),
                     "Latency (ms/batch 64)": lat[0], "Test MSE": "N/A", "Test PSNR (dB)": "N/A",
                     "Test SSIM": "N/A", "FID": M.fid_score(real, fake),
                     "Mode Coverage (H)": S["coverage"][label][0], "Loss CV (stabilitas)": cv,
                     "Kestabilan Training": stab})

    # Referensi: FID antara dua himpunan citra riil (batas bawah praktis untuk N sampel yang sama)
    val_real = S["splits"]["val"][0][:C.FID_N]
    rows.append({"Model": "Referensi: Real (val) vs Real (test)", "Dim Laten": "-", "Train Time/Epoch (s)": "-",
                 "Latency (ms/batch 64)": "-", "Test MSE": "-", "Test PSNR (dB)": "-", "Test SSIM": "-",
                 "FID": M.fid_score(real, val_real), "Mode Coverage (H)": M.mode_coverage(clf, val_real)[0],
                 "Loss CV (stabilitas)": "-", "Kestabilan Training": "-"})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT_DIR, "metrics_table.csv"), index=False)
    with open(os.path.join(C.OUT_DIR, "metrics_table.md"), "w") as f:
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
    with open(os.path.join(C.OUT_DIR, "mode_coverage.json"), "w") as f:
        json.dump(S["coverage"], f, indent=2)
    S["table"] = df
    return df


# ------------------------------------------------------------------ Figur
def _encode_pair(S, model):
    x, y = S["splits"]["test"]
    a, b = C.INTERP_CLASSES
    xa = x[int((y == a).nonzero()[0])][None].to(C.DEVICE)
    xb = x[int((y == b).nonzero()[0])][None].to(C.DEVICE)
    return model.encode(xa), model.encode(xb)


@torch.no_grad()
def _gan_pair(S, G):
    """GAN tidak punya encoder: pilih z_A, z_B yang hasilnya diklasifikasi MLP sebagai dua kelas kontras."""
    clf = S["models"]["mlp"]
    gen = torch.Generator(device="cpu").manual_seed(C.SEED)
    z = torch.randn(2000, C.Z_GAN, generator=gen).to(C.DEVICE)
    prob = torch.softmax(clf((G(z) + 1) / 2), 1)
    a, b = C.INTERP_CLASSES
    return z[prob[:, a].argmax()][None], z[prob[:, b].argmax()][None]


@torch.no_grad()
def make_figures(S):
    Mo, H = S["models"], S["H"]
    x8 = _pick_examples(S, 8)
    S["fig1"] = P.figure1_reconstruction(x8, {"Vanilla AE": Mo["ae"].reconstruct(x8),
                                              "Sparse AE": Mo["sae"].reconstruct(x8),
                                              "VAE": Mo["vae"].reconstruct(x8)})
    S["fig2"] = P.figure2_sampling({
        "A: VAE": S["fake"]["VAE"], "B: Vanilla GAN": S["fake"]["Vanilla GAN"],
        "C: DCGAN": S["fake"]["DCGAN"], "D: DDPM": S["fake"]["DDPM"]})

    alphas = torch.tensor(C.INTERP_ALPHAS, device=C.DEVICE)[:, None]
    interps = {}
    for key, name in (("ae", "Vanilla AE"), ("sae", "Sparse AE"), ("vae", "VAE")):
        za, zb = _encode_pair(S, Mo[key])
        interps[name] = Mo[key].decode((1 - alphas) * za + alphas * zb)
    for key, name in (("gan", "Vanilla GAN"), ("dcgan", "DCGAN")):
        za, zb = _gan_pair(S, Mo[key])
        interps[name] = (Mo[key]((1 - alphas) * za + alphas * zb) + 1) / 2
    S["fig4"] = P.figure4_interpolation(interps, C.INTERP_ALPHAS)

    torch.manual_seed(C.SEED)
    steps = max(1, C.DDPM_T // 10)
    _, traj = Mo["ddpm"].sample(4, traj_every=steps)
    S["fig5"] = P.figure5_losses_and_ddpm(H, traj)

    # Figur pendukung
    P.figure_gan_diagnostics(H)
    P.figure_filters(Mo["ae"], Mo["sae"])
    xt = S["splits"]["test"][0][:2000].to(C.DEVICE)
    h_ae = Mo["ae"].encoder[2](Mo["ae"].encoder[1](xt.flatten(1))).cpu().numpy()
    h_sae = Mo["sae"].hidden(xt).cpu().numpy()
    S["sparsity"] = {"Vanilla AE": float((h_ae < 1e-3).mean()), "Sparse AE": float((h_sae < 1e-3).mean()),
                     "mean|h| Vanilla AE": float(abs(h_ae).sum(1).mean()), "mean|h| Sparse AE": float(abs(h_sae).sum(1).mean())}
    with open(os.path.join(C.OUT_DIR, "sparsity_stats.json"), "w") as f:
        json.dump(S["sparsity"], f, indent=2)
    print("Fraksi aktivasi hidden ~0:", S["sparsity"])
    P.figure_sparsity_hist(h_ae, h_sae)
    P.figure_class_hist(S["coverage"])
