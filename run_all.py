"""Menjalankan seluruh pipeline eksperimen secara berurutan.

    python run_all.py              # eksperimen penuh (disarankan GPU)
    QUICK=1 python run_all.py      # smoke test cepat
"""
from src import experiment as E


def main():
    S = E.setup()
    E.train_classifier(S)          # Bagian 2 - Tahap 1
    E.train_autoencoders(S)        # Vanilla AE, Sparse AE, VAE
    E.mlp_limit_experiments(S)     # Bagian 2 - Tahap 2..5 + Figur 3
    E.train_gans(S)                # Vanilla GAN, DCGAN
    E.train_ddpm(S)                # DDPM
    df = E.evaluate_all(S)         # Tabel metrik Bagian IV
    print(df.to_string())
    E.make_figures(S)              # Figur 1, 2, 4, 5 + figur pendukung


if __name__ == "__main__":
    main()
