import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from cnn_functions import build_fourier_decoder_conv1d


# --------------------------------------------------
# Dataset: Truncated Fourier → spatial field
# --------------------------------------------------

def load_fourier_dataset(npy_path, N_modes):
    """
    Load dataset of shape (Nx, Nsnap) and return
    (truncated Fourier coeffs → field).
    """

    u = np.load(npy_path)        # (Nx, Nsnap)
    u = u.T                      # (Nsnap, Nx)
    Nsnap, Nx = u.shape

    # FFT along spatial axis
    u_hat = np.fft.rfft(u, axis=1) / Nx

    # Truncate modes
    u_hat_trunc = u_hat[:, :N_modes]

    # Real + Imag parts
    X = np.concatenate([u_hat_trunc.real, u_hat_trunc.imag], axis=1)
    y = u

    # Simple normalization (stable, minimal)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)

    X = tf.convert_to_tensor(X.astype(np.float32))
    y = tf.convert_to_tensor(y[..., None].astype(np.float32))

    return tf.data.Dataset.from_tensor_slices((X, y))


def prepare_dataset(ds, batch_size=32, shuffle=True):
    if shuffle:
        ds = ds.shuffle(2048)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    # -------- Config path --------
    config_path = "configs/L22_nu1.0/fourier_20.json"

    with open(config_path) as f:
        p = json.load(f)

    Nx = int(p["N"])
    L = int(p["L"])
    nu = float(p["nu"])
    T = float(p["T"])
    N_train_k = int(p["N_train"])

    batch_size = int(p.get("batch_size", 16))
    epochs = int(p.get("epochs", 50))
    lr = float(p.get("lr", 7.5e-4))

    N_modes = int(p["N_modes"])
    filters = p.get("filters", [128, 64, 32])
    kernel_size = p.get("kernel_size", 5)

    # Safety check
    assert Nx % (2 ** len(filters)) == 0, \
        "Nx must be divisible by 2^len(filters)"

    # -------- Dataset paths --------
    dataset_dir = (
        f"../simulations/"
        f"KS_dataset_L{L}_nu{nu}_N{Nx}_T{T}_{N_train_k}k"
    )

    train_path = os.path.join(
        dataset_dir,
        f"u_train_L{L}_nu{nu}_N{Nx}.npy"
    )

    val_path = os.path.join(
        dataset_dir,
        f"u_test_L{L}_nu{nu}_N{Nx}.npy"
    )

    # -------- Build datasets --------
    train_ds = prepare_dataset(
        load_fourier_dataset(train_path, N_modes),
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = prepare_dataset(
        load_fourier_dataset(val_path, N_modes),
        batch_size=batch_size,
        shuffle=False
    )

    # -------- Build model --------
    model = build_fourier_decoder_conv1d(
        N_modes=N_modes,
        Nx=Nx,
        filters=filters,
        kernel_size=kernel_size,
    )

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse",
    )

    model.summary()

    # -------- Output directory --------
    run_dir = (
        f"fourier_L{L}_nu{nu}_"
        f"Nm{N_modes}"
    )
    os.makedirs(run_dir, exist_ok=True)

    # -------- Callbacks --------
    csv_logger = CSVLogger(
        os.path.join(run_dir, "training_log.csv")
    )

    ckpt = ModelCheckpoint(
        os.path.join(run_dir, "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.75,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )

    # -------- Train --------
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[csv_logger, ckpt, reduce_lr],
        verbose=1,
    )


if __name__ == "__main__":
    main()
