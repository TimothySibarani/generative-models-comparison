# Studi Komparatif Model Generatif: AE, Sparse AE, VAE, GAN, DCGAN, DDPM

Tugas Take-Home Deep Learning (D6421) Sesi 07. Repositori ini berisi skrip PyTorch modular
untuk melatih dan membandingkan enam model generatif di **Fashion-MNIST**, ditambah eksperimen
yang membuktikan bahwa bobot dan fitur laten **MLP Classifier** tidak bisa dipakai untuk menghasilkan citra.

## Struktur

```
src/
  config.py      # seed (42), device, hyperparameter, jumlah epoch, path output
  data.py        # Fashion-MNIST 70k -> split 80/10/10 (seed tetap), DataLoader batch 128
  models.py      # VanillaAE, SparseAE, VAE, MLP GAN, DCGAN, DDPM (U-Net + EMA), MLPClassifier
  train.py       # training loop generik & adversarial, checkpoint/resume
  metrics.py     # MSE, PSNR, SSIM, FID, latency, mode coverage, stabilitas loss
  mlp_limits.py  # Bagian 2: transpos bobot, pseudo-inverse, frozen decoder, prior sampling
  plots.py       # Figur 1-5 + figur pendukung
  experiment.py  # orkestrasi per tahap
run_all.py       # menjalankan seluruh pipeline
notebook/Generative_Models_Comparison.ipynb   # versi Google Colab (self-contained)
```

## Cara menjalankan

**Google Colab (disarankan):** buka `notebook/Generative_Models_Comparison.ipynb`,
pilih *Runtime -> Change runtime type -> T4 GPU*, lalu *Run all*. Total waktu sekitar 60-90 menit di T4.
Aktifkan `USE_DRIVE = True` di sel kedua agar checkpoint tersimpan di Google Drive; jika runtime
terputus, jalankan ulang dan model yang sudah selesai tidak akan dilatih ulang.

**Lokal:**

```bash
pip install -r requirements.txt
python run_all.py
```

Smoke test cepat (subset kecil, 1 epoch): `QUICK=1 python run_all.py`.

## Output (`outputs/`)

| Berkas | Isi |
|---|---|
| `metrics_table.csv` / `.md` | Tabel Bagian IV: dimensi laten, waktu/epoch, latency per batch 64, MSE, PSNR, SSIM, FID, mode coverage, stabilitas |
| `mlp_limits_results.json` | Bagian III: akurasi classifier, ‖WᵀW − I‖, rank, fraksi ReLU nol, metrik transpos/pseudo-inverse/frozen decoder vs AE, FID sampling laten MLP |
| `figures/fig1..fig5*.png` | Lima figur wajib |
| `figures/figS*.png` | Figur pendukung: diagnostik GAN, filter AE vs Sparse AE, histogram aktivasi, mode coverage |
| `checkpoints/` | Bobot model & riwayat loss |

## Catatan metodologi

- Seed tetap `42` (`random`, `numpy`, `torch`), cuDNN deterministic. Seed di-reset sebelum inisialisasi tiap model.
- Semua model memakai split, batch size (128), dan test set yang sama.
- FID dihitung dengan InceptionV3 (torchmetrics) antara `FID_N=2000` sampel sintetis dan 2000 citra test.
  Sebagai acuan, tabel juga memuat FID *real (val) vs real (test)*.
- Mode coverage = entropi ternormalisasi dari distribusi kelas sampel sintetis menurut MLP Classifier
  (1.0 = semua kelas merata, rendah = indikasi mode collapse).
- Latency diukur sebagai rata-rata waktu menghasilkan satu batch 64 citra (`torch.cuda.synchronize`).
  Untuk Vanilla AE / Sparse AE yang diukur adalah encode+decode, karena keduanya tidak punya prior untuk sampling.
