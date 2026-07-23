# Kuramoto–Sivashinsky Inertial Manifold Dimension

Code accompanying the paper
“A prediction for the dimension of the inertial manifold of the Kuramoto–Sivashinsky equation.”

This repository contains the numerical experiments and dataset generation
pipeline used to study dimensional scaling in the 1D Kuramoto–Sivashinsky (KS) equation.

## Repository Structure
```
simulations/   → KS solver and dataset generation
cnn/           → neural network training (Autoencoder and Fourier Encoder)
```

## Installation

The code was run with Python 3.11. Dependencies are pinned in
`requirements.txt`:

```bash
pip install -r requirements.txt
```

## Dataset Generation

Datasets are generated using a pseudo-spectral solver ([`spooky` backend](https://github.com/PatricioClark/spooky)). They are created with:

```bash
python create_dataset.py
```

### Simulation Parameters

Below are the parameter values used to generate the main datasets
employed in the scaling study.

To reproduce a specific dataset, edit `params.py` accordingly
and run `create_dataset.py`.

#### Parameter Table

| L   | ν    | dt     | T      | Nx   | ostep  | N_total |
|-----|------|--------|--------|------|--------|---------|
| 22  | 0.01 | 1e-05  | 2000.0 | 512  | 10000  | 100k  |
| 22  | 0.1  | 1e-05  | 1100.0 | 256  | 10000  | 100k  |
| 22  | 1.0  | 1e-05  | 1000.0 | 256  | 10000  | 100k  |
| 44  | 0.01 | 1e-05  | 1000.0 | 1024 | 10000  | 100k  |
| 44  | 0.1  | 1e-05  | 1100.0 | 256  | 10000  | 100k  |
| 44  | 1.0  | 1e-06  | 1100.0 | 256  | 100000 | 100k  |
| 66  | 0.01 | 1e-05  | 1000.0 | 1024 | 10000  | 100k  |
| 66  | 0.1  | 1e-05  | 1100.0 | 256  | 10000  | 100k  |
| 66  | 1.0  | 1e-05  | 1000.0 | 256  | 10000  | 100k  |
| 100 | 0.01 | 1e-05  | 1100.0 | 1024 | 10000  | 100k  |
| 100 | 0.1  | 1e-05  | 1100.0 | 256  | 10000  | 100k  |
| 100 | 1.0  | 1e-05  | 1000.0 | 256  | 10000  | 100k  |
| 200 | 0.1  | 1e-05  | 1100.0 | 512  | 10000  | 100k  |
| 200 | 1.0  | 1e-05  | 1000.0 | 256  | 10000  | 100k  |

## Training configs (JSON)

Training runs are configured via small JSON files stored under

`cnn/configs/autoencoder/L{L}_nu{nu}/{dh}_{run}.json`

where `dh` is the bottleneck (latent) dimension and `run` identifies one of the
three repetitions described below. Configs are provided for **all 14 (L, ν)
cases** in the paper, at the bottleneck `dh = d_A` reported for that case.

Train a single config with:

```bash
cd cnn
python train_model.py configs/autoencoder/L44_nu0.1/60_0.json
```

Outputs (best checkpoint, `training_log.csv`) are written to a directory named
after the config, e.g. `configs/autoencoder/L44_nu0.1/60_0/`.

### The three runs

All three runs of a given case share the same dataset, the same train/test
split and the same optimisation settings. They differ only as follows:

| run | encoder filters        | seed | isolates            |
|-----|------------------------|------|---------------------|
| 0   | `[32, 64, 128, 256]`   | 0    | reference           |
| 1   | `[32, 64, 128, 256]`   | 1    | initialisation      |
| 2   | `[64, 128, 256, 512]`  | 0    | network capacity    |

Run 0 vs. run 1 measures how much `d_A` moves when the *same* network is
retrained from a different initialisation. Run 0 vs. run 2 checks that `d_A`
does not move when the network is made larger (≈3.2× the parameters at
`L = 44`, `ν = 0.1`), which is what one expects if `d_A` reflects the
dimension of the attractor rather than the capacity of the model.

The seed set in the config controls the weight initialisation, the `tf.data`
shuffling of the training set, and any numpy-side randomness, so runs are
reproducible.

### Reproducing a full MSE(d) curve

The estimate `d_A` is read off the knee of the reconstruction error as a
function of the bottleneck dimension, so reproducing a figure means training
the same case at several `dh`. Use `make_configs.py` to write those configs:

```bash
cd cnn
python make_configs.py 40 50 55 60 70 80    # every (L, nu), these bottlenecks
```

This regenerates `configs/autoencoder/L{L}_nu{nu}/{dh}_{run}.json` for each
requested `dh` and each of the three runs. Calling it with no arguments
restores the shipped set (one bottleneck per case, at `dh = d_A`).

To change anything else, edit the JSON directly:

- `dh` — latent dimension
- `encoder`, `kernel_size`, `strides` — architecture
- `lr`, `batch_size`, `patience`, `epochs`, `seed` — training parameters
- `L`, `nu`, `T`, `N_train` — dataset selection. These are used to locate the
  dataset directory `simulations/KS_dataset_L{L}_nu{nu}_N{Nx}_T{T}_{N_train}k`,
  so they must match the parameters the dataset was generated with.

### Training protocol

Networks are trained with Adam on a mean-squared reconstruction loss,
learning rate `7.5e-4`, batch size 16, with `ReduceLROnPlateau`
(factor 0.75, patience 2, minimum learning rate `1e-6`) and best-validation-loss
checkpointing. `epochs` is an upper bound: the validation loss plateaus well
before it, typically within ~50 epochs.

## Fourier Encoder

In addition to the convolutional autoencoder, we include a model that instead of the encoder does a Fourier Truncation (Fourier-Encoder in the paper).

The config files are in

`cnn/configs/fourier_encoder/L{L}_nu{nu}/{N_modes}_{run}.json`

and the model is trained the same way as the autoencoder:

```bash
cd cnn
python train_fourier_encoder.py configs/fourier_encoder/L44_nu0.1/30_0.json
```

The three runs follow the same convention as above: run 0 is the reference
network, run 1 repeats it with a different seed, and run 2 uses a wider
decoder (`[512, 256, 128, 64]` instead of `[256, 128, 64, 32]`).

Fourier-specific parameters in the JSON configuration include:
- `N_modes` — number of retained Fourier modes
- `filters`, `kernel_size` — decoder architecture

The encoder keeps `N_modes` complex modes, i.e. `2 * N_modes` real degrees of
freedom, so a Fourier-encoder config with `N_modes` modes is compared against
an autoencoder config with `dh = 2 * N_modes`. `make_configs.py` writes the
Fourier configs together with the autoencoder ones and applies this factor,
so the bottlenecks passed on the command line are in units of `dh`.

