"""Definisi arsitektur: Vanilla AE, Sparse AE, VAE, Vanilla GAN (MLP), DCGAN, DDPM, MLP Classifier.

Setiap model keluarga autoencoder menyediakan:
  - loss(x)         -> (loss_scalar, dict_log)      dipakai oleh training loop generik
  - reconstruct(x)  -> citra rekonstruksi [0,1]
  - encode(x) / decode(z)
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


# =====================================================================================
# 1) Vanilla Autoencoder  : 784 -> 256 -> 64 -> 256 -> 784
#    L = || x - x_hat ||^2  (MSE)
# =====================================================================================
class VanillaAE(nn.Module):
    def __init__(self, latent: int = C.LATENT_AE, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU(),
                                     nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 784), nn.Sigmoid())

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def reconstruct(self, x):
        return self.forward(x)[0]

    def loss(self, x):
        x_hat, _ = self.forward(x)
        mse = F.mse_loss(x_hat, x)
        return mse, {"recon": mse.item()}


# =====================================================================================
# 2) Sparse Autoencoder : 784 -> 256 (sparse, penalti L1) -> 64 -> 256 -> 784
#    L = || x - x_hat ||^2 + lambda * || h ||_1
# =====================================================================================
class SparseAE(nn.Module):
    def __init__(self, latent: int = C.LATENT_AE, hidden: int = C.SPARSE_HIDDEN,
                 lam: float = C.SPARSE_LAMBDA):
        super().__init__()
        self.lam = lam
        self.enc1 = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU())
        self.enc2 = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 784), nn.Sigmoid())

    def hidden(self, x):
        return self.enc1(x)

    def encode(self, x):
        return self.enc2(self.enc1(x))

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        h = self.enc1(x)
        z = self.enc2(h)
        return self.decode(z), z, h

    def reconstruct(self, x):
        return self.forward(x)[0]

    def loss(self, x):
        x_hat, _, h = self.forward(x)
        mse = F.mse_loss(x_hat, x)
        l1 = h.abs().sum(dim=1).mean()
        return mse + self.lam * l1, {"recon": mse.item(), "l1": l1.item()}


# =====================================================================================
# 3) Variational Autoencoder : 784 -> 256 -> (mu, logvar) in R^64 -> 256 -> 784
#    L = E_q[-log p(x|z)] (BCE) + KL( q(z|x) || N(0, I) )
# =====================================================================================
class VAE(nn.Module):
    def __init__(self, latent: int = C.LATENT_AE, hidden: int = 256):
        super().__init__()
        self.latent = latent
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU())
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 784), nn.Sigmoid())

    def encode_dist(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def encode(self, x):          # dipakai untuk interpolasi -> mean posterior
        return self.encode_dist(x)[0]

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode_dist(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)   # reparameterization trick
        return self.decode(z), mu, logvar

    def reconstruct(self, x):
        return self.decode(self.encode(x))

    def sample(self, n, device=C.DEVICE):
        return self.decode(torch.randn(n, self.latent, device=device))

    def loss(self, x):
        x_hat, mu, logvar = self.forward(x)
        b = x.size(0)
        bce = F.binary_cross_entropy(x_hat, x, reduction="sum") / b
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
        return bce + kl, {"recon": bce.item(), "kl": kl.item(),
                          "mse": F.mse_loss(x_hat, x).item()}


# =====================================================================================
# 4) Vanilla GAN (fully connected)
#    min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]   (G memakai non-saturating loss)
# =====================================================================================
class MLPGenerator(nn.Module):
    def __init__(self, z_dim: int = C.Z_GAN):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), nn.Tanh())

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1))        # logit

    def forward(self, x):
        return self.net(x).view(-1)


# =====================================================================================
# 5) DCGAN (Radford et al., 2016) disesuaikan untuk 28x28
#    G: z(100) -> ConvT 7x7 -> 14x14 -> 28x28, BatchNorm + ReLU, output Tanh
#    D: Conv stride-2 28 -> 14 -> 7 -> 1, BatchNorm + LeakyReLU, tanpa pooling / FC
# =====================================================================================
class DCGenerator(nn.Module):
    def __init__(self, z_dim: int = C.Z_GAN, ngf: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 4, 7, 1, 0, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class DCDiscriminator(nn.Module):
    def __init__(self, ndf: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, 7, 1, 0, bias=False))

    def forward(self, x):
        return self.net(x).view(-1)


def dcgan_init(m):
    """Inisialisasi bobot N(0, 0.02) sesuai paper DCGAN."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# =====================================================================================
# 6) DDPM (Ho et al., 2020) dengan U-Net kecil
#    q(x_t|x_0) = N( sqrt(abar_t) x_0, (1-abar_t) I )
#    L_simple = E_{t,x0,eps} || eps - eps_theta(x_t, t) ||^2
# =====================================================================================
class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, 1, 1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, 1, 1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """U-Net 28 -> 14 -> 7 -> 14 -> 28 dengan skip connection dan time embedding."""

    def __init__(self, ch: int = C.DDPM_BASE_CH, tdim: int = 256):
        super().__init__()
        self.temb = nn.Sequential(SinusoidalEmb(ch), nn.Linear(ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inc = nn.Conv2d(1, ch, 3, 1, 1)
        self.d1 = ResBlock(ch, ch, tdim)
        self.ds1 = nn.Conv2d(ch, ch, 4, 2, 1)                      # 28 -> 14
        self.d2 = ResBlock(ch, ch * 2, tdim)
        self.ds2 = nn.Conv2d(ch * 2, ch * 2, 4, 2, 1)              # 14 -> 7
        self.m1 = ResBlock(ch * 2, ch * 4, tdim)
        self.m2 = ResBlock(ch * 4, ch * 4, tdim)
        self.us2 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1)     # 7 -> 14
        self.u2 = ResBlock(ch * 4, ch * 2, tdim)
        self.us1 = nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1)         # 14 -> 28
        self.u1 = ResBlock(ch * 2, ch, tdim)
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, 1, 1))

    def forward(self, x, t):
        te = self.temb(t)
        h1 = self.d1(self.inc(x), te)                 # 28x28
        h2 = self.d2(self.ds1(h1), te)                # 14x14
        m = self.m2(self.m1(self.ds2(h2), te), te)    # 7x7
        u = self.u2(torch.cat([self.us2(m), h2], 1), te)
        u = self.u1(torch.cat([self.us1(u), h1], 1), te)
        return self.out(u)


class DDPM(nn.Module):
    def __init__(self, T: int = C.DDPM_T, beta=C.DDPM_BETA, ema_decay: float = 0.999):
        super().__init__()
        self.T = T
        self.net = UNet()
        self.ema = copy.deepcopy(self.net).requires_grad_(False)   # bobot EMA untuk sampling
        self.ema_decay = ema_decay
        betas = torch.linspace(beta[0], beta[1], T)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)

    @torch.no_grad()
    def update_ema(self):
        for pe, p in zip(self.ema.parameters(), self.net.parameters()):
            pe.mul_(self.ema_decay).add_(p.detach(), alpha=1 - self.ema_decay)
        for be, b in zip(self.ema.buffers(), self.net.buffers()):
            be.copy_(b)

    def q_sample(self, x0, t, noise):
        ab = self.abar[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, x):
        x0 = x * 2 - 1                                   # [0,1] -> [-1,1]
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        pred = self.net(self.q_sample(x0, t, noise), t)
        mse = F.mse_loss(pred, noise)
        return mse, {"noise_mse": mse.item()}

    @torch.no_grad()
    def sample(self, n, device=C.DEVICE, traj_every: int = 0):
        """Reverse process x_T ~ N(0,I) -> x_0. Mengembalikan citra [0,1] (dan trajektori)."""
        net = self.ema
        x = torch.randn(n, 1, 28, 28, device=device)
        traj = []
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t, device=device, dtype=torch.long)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                eps = net(x, tt)
            eps = eps.float()
            a, ab, b = self.alphas[t], self.abar[t], self.betas[t]
            mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            x = mean + b.sqrt() * torch.randn_like(x) if t > 0 else mean
            if traj_every and (t % traj_every == 0 or t == self.T - 1):
                traj.append((t, ((x.clamp(-1, 1) + 1) / 2).cpu()))
        out = (x.clamp(-1, 1) + 1) / 2
        return (out, traj) if traj_every else out


# =====================================================================================
# 7) MLP Classifier pendamping : 784 -> 256 -> 64 -> 10   (struktur identik encoder AE)
#    L = CrossEntropy(y, softmax(f(x)))  -> memodelkan P(y|x)
# =====================================================================================
class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, C.LATENT_AE)
        self.fc3 = nn.Linear(C.LATENT_AE, 10)

    def features(self, x):
        """z_MLP in R^64: representasi tepat sebelum lapisan klasifikasi."""
        h = F.relu(self.fc1(x.flatten(1)))
        return F.relu(self.fc2(h))

    def forward(self, x):
        return self.fc3(self.features(x))

    def loss(self, x, y):
        logits = self.forward(x)
        ce = F.cross_entropy(logits, y)
        return ce, {"ce": ce.item(), "acc": (logits.argmax(1) == y).float().mean().item()}


class FrozenLatentDecoder(nn.Module):
    """Decoder MLP 64 -> 256 -> 784 untuk Tahap 4 (dilatih di atas z_MLP yang dibekukan)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(C.LATENT_AE, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)
