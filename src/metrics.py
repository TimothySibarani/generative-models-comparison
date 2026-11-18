"""Metrik evaluasi: MSE, PSNR, SSIM (rekonstruksi), FID (sampel sintetis), latency, mode coverage."""
import time

import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

from . import config as C


@torch.no_grad()
def recon_metrics(recon_fn, loader):
    """recon_fn(x)->x_hat di [0,1]. Mengembalikan rata-rata per-citra MSE, PSNR(dB), SSIM pada loader."""
    mse_all, psnr_all, ssim_sum, n = [], [], 0.0, 0
    for x, _ in loader:
        x = x.to(C.DEVICE)
        x_hat = recon_fn(x).clamp(0, 1)
        mse = ((x_hat - x) ** 2).flatten(1).mean(1)
        mse_all.append(mse.cpu())
        psnr_all.append((10 * torch.log10(1.0 / mse.clamp_min(1e-10))).cpu())
        ssim_sum += ssim_fn(x_hat, x, data_range=1.0).item() * x.size(0)
        n += x.size(0)
    return {"mse": torch.cat(mse_all).mean().item(),
            "psnr": torch.cat(psnr_all).mean().item(),
            "ssim": ssim_sum / n}


@torch.no_grad()
def generate(sample_fn, n, batch=C.GEN_BATCH):
    """sample_fn(k)->k citra [0,1]. Digenerate bertahap agar hemat memori."""
    out = []
    while sum(o.size(0) for o in out) < n:
        k = min(batch, n - sum(o.size(0) for o in out))
        out.append(sample_fn(k).clamp(0, 1).cpu())
    return torch.cat(out)


@torch.no_grad()
def fid_score(real, fake, batch=C.FID_BATCH):
    """FID (InceptionV3 pool3, 2048-d). Citra grayscale direplikasi ke 3 kanal; resize ke 299 dilakukan internal."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(C.DEVICE)
    for imgs, is_real in ((real, True), (fake, False)):
        for i in range(0, len(imgs), batch):
            fid.update(imgs[i:i + batch].to(C.DEVICE).repeat(1, 3, 1, 1), real=is_real)
    return fid.compute().item()


@torch.no_grad()
def latency_ms(fn, repeats=10, warmup=2):
    """Rata-rata waktu (ms) satu pemanggilan fn() yang menghasilkan satu batch (64 citra)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


@torch.no_grad()
def mode_coverage(classifier, imgs):
    """Distribusi kelas (menurut MLP Classifier) dari sampel sintetis.
    Mengembalikan entropi ternormalisasi (1.0 = 10 kelas merata; rendah = indikasi mode collapse)
    dan histogram kelas."""
    classifier.eval()
    preds = []
    for i in range(0, len(imgs), 500):
        preds.append(classifier(imgs[i:i + 500].to(C.DEVICE)).argmax(1).cpu())
    hist = torch.bincount(torch.cat(preds), minlength=10).float()
    p = hist / hist.sum()
    ent = max(0.0, -(p[p > 0] * p[p > 0].log()).sum().item() / np.log(10))
    return ent, hist.long().tolist()


def stability_cv(iter_losses, tail_frac=0.5):
    """Koefisien variasi loss per-iterasi pada separuh akhir training (indikator osilasi)."""
    a = np.asarray(iter_losses[int(len(iter_losses) * (1 - tail_frac)):])
    return float(a.std() / (abs(a.mean()) + 1e-8))
