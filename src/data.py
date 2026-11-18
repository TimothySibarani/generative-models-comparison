"""Pemuatan dataset Fashion-MNIST dengan split 80/10/10 yang seragam untuk semua model.

Train (60k) dan test (10k) bawaan torchvision digabung menjadi 70.000 citra, lalu dibagi
secara acak dengan generator ber-seed tetap -> 56.000 / 7.000 / 7.000.
Citra disimpan dalam rentang [0, 1]. Model GAN/DCGAN/DDPM mengubahnya ke [-1, 1] sendiri.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from . import config as C


def _load_tensors():
    tr = datasets.FashionMNIST(C.DATA_DIR, train=True, download=True)
    te = datasets.FashionMNIST(C.DATA_DIR, train=False, download=True)
    x = torch.cat([tr.data, te.data]).float().div(255.0).unsqueeze(1)   # (70000,1,28,28)
    y = torch.cat([tr.targets, te.targets]).long()
    return x, y


def get_splits():
    """Mengembalikan dict berisi tensor (x, y) untuk train/val/test."""
    x, y = _load_tensors()
    n = len(x)
    n_tr, n_va = int(C.SPLIT[0] * n), int(C.SPLIT[1] * n)
    g = torch.Generator().manual_seed(C.SEED)
    perm = torch.randperm(n, generator=g)
    idx = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    if C.QUICK:
        idx = {"train": idx["train"][:C.QUICK_SUBSET],
               "val": idx["val"][:C.QUICK_SUBSET // 4],
               "test": idx["test"][:max(C.QUICK_SUBSET // 4, C.FID_N)]}
    return {k: (x[v], y[v]) for k, v in idx.items()}


def get_loaders(splits, batch_size: int = C.BATCH_SIZE):
    g = torch.Generator().manual_seed(C.SEED)
    loaders = {}
    for name, (x, y) in splits.items():
        loaders[name] = DataLoader(TensorDataset(x, y), batch_size=batch_size,
                                   shuffle=(name == "train"), generator=g if name == "train" else None,
                                   drop_last=False)
    return loaders
