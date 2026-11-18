"""Bagian 2: pembuktian empiris bahwa bobot & fitur laten MLP Classifier tidak bisa dipakai untuk generasi citra.

Tahap 1 : latih MLP 784-256-64-10 (CE) -> akurasi test >= 88%          (experiment.py)
Tahap 2 : proyeksi balik transpos dari one-hot kelas   x = s(W1^T r(W2^T r(W3^T y_c)))
Tahap 3 : proyeksi balik transpos dari z_MLP citra uji x = s(W1^T r(W2^T z_MLP))
Tahap 4 : frozen latent decoder 64-256-784 (MSE) vs Vanilla AE z in R^64  (experiment.py)
Tahap 5 : sampling z ~ N(mu_z, diag(sigma_z^2)) -> decoder Tahap 4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class FrozenPipe(nn.Module):
    """Classifier beku (requires_grad=False) + decoder yang dilatih di atas z_MLP."""

    def __init__(self, classifier, decoder):
        super().__init__()
        self.classifier = classifier.requires_grad_(False)
        self.decoder = decoder

    def train(self, mode=True):
        super().train(mode)
        self.classifier.eval()          # classifier selalu mode eval
        return self

    def reconstruct(self, x):
        with torch.no_grad():
            z = self.classifier.features(x)
        return self.decoder(z)

    def loss(self, x):
        x_hat = self.reconstruct(x)
        mse = F.mse_loss(x_hat, x)
        return mse, {"recon": mse.item()}


def weights(clf):
    return clf.fc1.weight.detach(), clf.fc2.weight.detach(), clf.fc3.weight.detach()   # (256,784),(64,256),(10,64)


@torch.no_grad()
def transpose_from_class(clf):
    """Tahap 2: one-hot y_c (10 kelas) -> ruang piksel via transpos bobot (tanpa bias, tanpa training)."""
    W1, W2, W3 = weights(clf)
    y = torch.eye(10, device=W1.device)
    z = F.relu(y @ W3)          # (10,64)   = ReLU(W3^T y_c)
    h = F.relu(z @ W2)          # (10,256)  = ReLU(W2^T z)
    pre = h @ W1                # (10,784)  = W1^T h
    return torch.sigmoid(pre).view(-1, 1, 28, 28), pre.view(-1, 1, 28, 28)


@torch.no_grad()
def transpose_from_latent(clf, x):
    """Tahap 3: z_MLP citra uji -> ruang piksel via transpos bobot."""
    W1, W2, _ = weights(clf)
    z = clf.features(x)
    pre = F.relu(z @ W2) @ W1
    return torch.sigmoid(pre).view(-1, 1, 28, 28), pre.view(-1, 1, 28, 28)


@torch.no_grad()
def pinv_from_latent(clf, x):
    """Pembanding: invers semu Moore-Penrose (W^+) termasuk bias -- tetap gagal karena
    dimensi direduksi (784 -> 64) dan ReLU sudah membuang nilai negatif."""
    W1, W2, _ = weights(clf)
    b1, b2 = clf.fc1.bias.detach(), clf.fc2.bias.detach()
    z = clf.features(x)
    h = (z - b2) @ torch.linalg.pinv(W2).T
    xr = (h - b1) @ torch.linalg.pinv(W1).T
    return xr.clamp(0, 1).view(-1, 1, 28, 28)


@torch.no_grad()
def weight_diagnostics(clf, loader):
    """Bukti numerik: ||W^T W - I||_F, ||W^T - W^+||_F, rank, dan fraksi aktivasi ReLU = 0."""
    W1, W2, W3 = weights(clf)
    out = {}
    for name, W in (("W1", W1), ("W2", W2), ("W3", W3)):
        n_in = W.shape[1]
        WtW = W.T @ W
        out[name] = {
            "shape": list(W.shape),
            "rank": int(torch.linalg.matrix_rank(W).item()),
            "input_dim": n_in,
            "||W^T W - I||_F": float(torch.linalg.norm(WtW - torch.eye(n_in, device=W.device))),
            "||W^T - W^+||_F": float(torch.linalg.norm(W.T - torch.linalg.pinv(W))),
        }
    dead1, dead2, n = 0.0, 0.0, 0
    for x, _ in loader:
        x = x.to(C.DEVICE)
        h1 = F.relu(clf.fc1(x.flatten(1)))
        h2 = F.relu(clf.fc2(h1))
        dead1 += (h1 == 0).float().mean().item() * x.size(0)
        dead2 += (h2 == 0).float().mean().item() * x.size(0)
        n += x.size(0)
    out["relu_zero_frac_layer1"] = dead1 / n
    out["relu_zero_frac_layer2"] = dead2 / n
    return out


@torch.no_grad()
def latent_stats(clf, loader):
    zs, ys = [], []
    for x, y in loader:
        zs.append(clf.features(x.to(C.DEVICE)).cpu())
        ys.append(y)
    z, y = torch.cat(zs), torch.cat(ys)
    return z, y, z.mean(0), z.std(0)


@torch.no_grad()
def intra_class_variance_ratio(z, y):
    """Rasio varians intra-kelas / total pada ruang fitur. Nilai kecil = variasi intra-kelas dibuang."""
    total = z.var(0).sum()
    intra = sum(z[y == c].var(0).sum() * (y == c).sum() for c in range(10)) / len(y)
    return float(intra / total)


@torch.no_grad()
def prior_sampling(decoder, mu, sigma, n, generator=None):
    """Tahap 5: z_rand ~ N(mu_z, diag(sigma_z^2)) lalu didekode."""
    eps = torch.randn(n, mu.numel(), generator=generator)
    z = (mu + sigma * eps).to(C.DEVICE)
    return decoder(z)
