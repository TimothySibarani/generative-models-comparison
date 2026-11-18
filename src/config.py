"""Konfigurasi global eksperimen: seed, device, hyperparameter, dan path output.

Set environment variable QUICK=1 untuk smoke test cepat (subset data kecil, 1 epoch,
T DDPM kecil). Untuk hasil laporan, jalankan tanpa QUICK di GPU (Colab T4).
"""
import os
import random

import numpy as np
import torch

SEED = 42
QUICK = os.environ.get("QUICK", "0") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUT_DIR = os.environ.get("OUT_DIR", "./outputs")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# ---------------- Data ----------------
DATASET = "FashionMNIST"
BATCH_SIZE = 128
SPLIT = (0.8, 0.1, 0.1)                 # train / val / test dari 70.000 citra
QUICK_SUBSET = 2048                      # jumlah citra train saat QUICK=1

# ---------------- Arsitektur ----------------
LATENT_AE = 64                           # dim laten Vanilla AE, Sparse AE, VAE, MLP z
SPARSE_HIDDEN = 256                      # lapisan tersembunyi yang diberi penalti L1
SPARSE_LAMBDA = 1e-4                     # bobot penalti lambda * ||h||_1
Z_GAN = 100                              # dim prior noise Vanilla GAN & DCGAN
DDPM_T = 50 if QUICK else 1000           # jumlah langkah difusi
DDPM_BETA = (1e-4, 0.02)                 # linear beta schedule (Ho et al., 2020)
DDPM_BASE_CH = 64

# ---------------- Training ----------------
EPOCHS = dict(mlp=20, frozen_dec=20, ae=20, sae=20, vae=30, gan=50, dcgan=25, ddpm=30)
if QUICK:
    EPOCHS = {k: 1 for k in EPOCHS}
LR = dict(mlp=1e-3, frozen_dec=1e-3, ae=1e-3, sae=1e-3, vae=1e-3, gan=2e-4, dcgan=2e-4, ddpm=2e-4)

# ---------------- Evaluasi ----------------
FID_N = 200 if QUICK else 2000           # jumlah sampel sintetis & real untuk FID
GEN_BATCH = 250                          # ukuran potongan saat generate sampel (hemat VRAM, aman utk GPU 6GB)
FID_BATCH = 100                          # ukuran potongan saat ekstraksi fitur InceptionV3 (299x299)
LATENCY_BATCH = 64
INTERP_ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
INTERP_CLASSES = (0, 9)                  # T-shirt/top  vs  Ankle boot (kontras)

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def set_seed(seed: int = SEED) -> None:
    """Menetapkan seed agar seluruh eksperimen reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dirs() -> None:
    for d in (OUT_DIR, CKPT_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)
