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

## Dataset Generation

Datasets are generated using a pseudo-spectral solver (`spooky` backend). They are created with:

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

Training runs are configured via small JSON files stored under folders of the form:

`cnn/configs/autoencoder/L{L}_nu{nu}/`

In this repository we include **two example configs**:

- `L22_nu1.0/7_0.json` — *small network*
- `L22_nu1.0/7_1.json` — *large network*

### Naming convention

The JSON filenames follow the pattern:

`{dh}_{model_id}.json`

- The first number (`7`) corresponds to the latent dimension `dh`.
- The second number (`0`, `1`, …) distinguishes different architectural variants
  (e.g. different numbers of filters).

For example:

- `7_0.json` → latent dimension `dh = 7`, small architecture
- `7_1.json` → latent dimension `dh = 7`, larger architecture

These files are meant as **templates**. In our full set of experiments we used additional configurations
(e.g. different filter widths and latent sizes), but we do not track the entire sweep in the public repo.

To reproduce or extend experiments, copy one of the example JSONs and modify:

- `dh` — latent dimension
- `encoder`, `kernel_size`, `strides` — architecture
- `lr`, `batch_size`, `patience` — training parameters
- `L`, `nu`, `T`, `N_train` — dataset selection

Example:

```bash
python train_model.py
```

## Fourier Encoder

In addition to the convolutional autoencoder, we include a model that instead of the encoder does a Fourier Truncation (Fourier-Encoder in the paper).

The Fourier model is trained using:
```
python train_fourier_encoder.py
```

The config files are in:

`cnn/configs/fourier_encoder/L{L}_nu{nu}/`

Fourier-specific parameters in the JSON configuration include:
- `N_modes` — number of retained Fourier modes

