import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from cnn_functions import neural_configuration_1d, build_model_1d


# ----------------------- Dataset utils ----------------------- #
def load_npy_autoencoder_dataset(npy_file: str) -> tf.data.Dataset:
    """Loads (Nx, Nsnap) .npy and returns dataset of (X, X) with shape (Nsnap, Nx, 1)."""
    data = np.load(npy_file)
    if data.ndim != 2:
        raise ValueError(f"Expected (Nx, N), got {data.shape} in {npy_file}")

    x = data.T[..., None].astype(np.float32)  # (N, Nx, 1)
    ds = tf.data.Dataset.from_tensor_slices((x, x))
    return ds


def prepare_dataset(ds: tf.data.Dataset, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    if shuffle:
        ds = ds.shuffle(buffer_size=2048, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    # ------------------------------------------------------------------
    # Load run parameters (written/selected externally)
    # ------------------------------------------------------------------
    params_dir = "configs/L22_nu1.0"
    param_file = "7_1.json"

    with open(os.path.join(params_dir, param_file), "r") as f:
        p = json.load(f)

    # Core parameters
    Nx = int(p["N"])
    L = int(p.get("L", 22))
    nu = float(p.get("nu", 0.1))
    T = float(p.get("T", 1100.0))
    N_train_k = int(p.get("N_train", 100))  # interpreted as "k" in folder name

    batch_size = int(p.get("batch_size", 16))
    epochs = int(p.get("epochs", 100))
    lr = float(p.get("lr", 1e-4))

    # Model parameters
    dh = int(p["dh"])
    encoder_neurons = p["encoder"]
    kernel_size = p["kernel_size"]
    strides = p["strides"]

    # ------------------------------------------------------------------
    # Data paths
    # ------------------------------------------------------------------
    dataset_dir = f"../simulations/KS_dataset_L{L}_nu{nu}_N{Nx}_T{T}_{N_train_k}k"
    train_path = os.path.join(dataset_dir, f"u_train_L{L}_nu{nu}_N{Nx}.npy")
    val_path = os.path.join(dataset_dir, f"u_test_L{L}_nu{nu}_N{Nx}.npy")

    # ------------------------------------------------------------------
    # Output directory (one folder per JSON)
    # ------------------------------------------------------------------
    run_name = os.path.splitext(param_file)[0]
    out_dir = os.path.join(params_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------
    train_ds = prepare_dataset(load_npy_autoencoder_dataset(train_path), batch_size, shuffle=True)
    val_ds = prepare_dataset(load_npy_autoencoder_dataset(val_path), batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    enc_cfg, dec_cfg = neural_configuration_1d(encoder_neurons, kernel_size, strides)
    model = build_model_1d(Nx=Nx, encoder_layers=enc_cfg, decoder_layers=dec_cfg, dh=dh)

    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    model.summary()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    csv_logger = CSVLogger(os.path.join(out_dir, "training_log.csv"), append=True)

    ckpt = ModelCheckpoint(
        filepath=os.path.join(out_dir, "best_model.keras"),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.75,
        patience=int(p.get("patience", 2)),
        min_lr=1e-6,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[csv_logger, ckpt, reduce_lr],
        verbose=1,
    )


if __name__ == "__main__":
    main()
