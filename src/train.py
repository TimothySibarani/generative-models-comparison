"""Training loop: generik (AE / Sparse AE / VAE / DDPM / MLP / Frozen decoder) dan adversarial (GAN / DCGAN).

Setiap fungsi mengembalikan `history` (dict of lists) yang berisi loss per-epoch,
loss per-iterasi (untuk analisis osilasi), dan waktu training per epoch.
Checkpoint + history disimpan sehingga training bisa dilewati bila sudah pernah dijalankan
(berguna saat runtime Colab terputus dan OUT_DIR berada di Google Drive).
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from . import config as C


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _paths(name):
    return os.path.join(C.CKPT_DIR, f"{name}.pt"), os.path.join(C.CKPT_DIR, f"{name}_history.json")


def try_load(name, modules: dict):
    ck, hp = _paths(name)
    if os.path.exists(ck) and os.path.exists(hp):
        state = torch.load(ck, map_location=C.DEVICE)
        for k, m in modules.items():
            m.load_state_dict(state[k])
        with open(hp) as f:
            print(f"[{name}] checkpoint ditemukan -> training dilewati")
            return json.load(f)
    return None


def save(name, modules: dict, history):
    ck, hp = _paths(name)
    torch.save({k: m.state_dict() for k, m in modules.items()}, ck)
    with open(hp, "w") as f:
        json.dump(history, f)


def train_generic(name, model, loaders, epochs, lr, loss_fn=None, ema=False, amp=False):
    """loss_fn(model, x, y) -> (loss, logs). Default: model.loss(x)."""
    model.to(C.DEVICE)
    hist = try_load(name, {"model": model})
    if hist is not None:
        return hist
    loss_fn = loss_fn or (lambda m, x, y: m.loss(x))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    use_amp = amp and C.DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hist = {"train_loss": [], "val_loss": [], "epoch_time": [], "iter_loss": [], "logs": []}
    for ep in range(epochs):
        model.train()
        _sync()
        t0 = time.time()
        losses, logs_acc = [], {}
        for x, y in loaders["train"]:
            x, y = x.to(C.DEVICE, non_blocking=True), y.to(C.DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss, logs = loss_fn(model, x, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema:
                model.update_ema()
            losses.append(loss.item())
            for k, v in logs.items():
                logs_acc.setdefault(k, []).append(v)
        _sync()
        hist["epoch_time"].append(time.time() - t0)
        hist["train_loss"].append(float(np.mean(losses)))
        hist["iter_loss"].extend(losses)
        hist["logs"].append({k: float(np.mean(v)) for k, v in logs_acc.items()})
        hist["val_loss"].append(evaluate_loss(model, loaders["val"], loss_fn))
        print(f"[{name}] epoch {ep + 1:3d}/{epochs}  train={hist['train_loss'][-1]:.4f}  "
              f"val={hist['val_loss'][-1]:.4f}  {hist['logs'][-1]}  ({hist['epoch_time'][-1]:.1f}s)")
    save(name, {"model": model}, hist)
    return hist


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn):
    model.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(C.DEVICE), y.to(C.DEVICE)
        loss, _ = loss_fn(model, x, y)
        tot += loss.item() * x.size(0)
        n += x.size(0)
    return tot / n


def _grad_norm(module):
    g = [p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
    return torch.stack(g).norm().item() if g else 0.0


def train_gan(name, G, D, loaders, epochs, lr, z_dim=C.Z_GAN):
    """Minimax GAN dengan BCE; generator memakai non-saturating loss  -log D(G(z))."""
    G.to(C.DEVICE)
    D.to(C.DEVICE)
    hist = try_load(name, {"G": G, "D": D})
    if hist is not None:
        return hist
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    hist = {k: [] for k in ["g_loss", "d_loss", "d_real", "d_fake", "g_grad", "epoch_time",
                            "iter_g", "iter_d"]}
    for ep in range(epochs):
        G.train()
        D.train()
        _sync()
        t0 = time.time()
        acc = {k: [] for k in ["g", "d", "dr", "df", "gg"]}
        for x, _ in loaders["train"]:
            x = x.to(C.DEVICE) * 2 - 1                    # [0,1] -> [-1,1]
            b = x.size(0)
            ones = torch.ones(b, device=C.DEVICE)
            zeros = torch.zeros(b, device=C.DEVICE)
            fake = G(torch.randn(b, z_dim, device=C.DEVICE))

            # ---- update Discriminator: max log D(x) + log(1 - D(G(z)))
            out_r, out_f = D(x), D(fake.detach())
            d_loss = bce(out_r, ones) + bce(out_f, zeros)
            optD.zero_grad(set_to_none=True)
            d_loss.backward()
            optD.step()

            # ---- update Generator: min -log D(G(z))
            out_g = D(fake)
            g_loss = bce(out_g, ones)
            optG.zero_grad(set_to_none=True)
            g_loss.backward()
            acc["gg"].append(_grad_norm(G))
            optG.step()

            acc["g"].append(g_loss.item())
            acc["d"].append(d_loss.item())
            acc["dr"].append(torch.sigmoid(out_r).mean().item())
            acc["df"].append(torch.sigmoid(out_f).mean().item())
        _sync()
        hist["epoch_time"].append(time.time() - t0)
        for hk, ak in [("g_loss", "g"), ("d_loss", "d"), ("d_real", "dr"), ("d_fake", "df"), ("g_grad", "gg")]:
            hist[hk].append(float(np.mean(acc[ak])))
        hist["iter_g"].extend(acc["g"])
        hist["iter_d"].extend(acc["d"])
        print(f"[{name}] epoch {ep + 1:3d}/{epochs}  G={hist['g_loss'][-1]:.3f}  D={hist['d_loss'][-1]:.3f}  "
              f"D(x)={hist['d_real'][-1]:.2f}  D(G(z))={hist['d_fake'][-1]:.2f}  "
              f"|gradG|={hist['g_grad'][-1]:.3f}  ({hist['epoch_time'][-1]:.1f}s)")
    save(name, {"G": G, "D": D}, hist)
    return hist
