# Kuramoto–Sivashinsky Inertial Manifold Dimension

Code accompanying the paper “A prediction for the dimension of the inertial manifold of the Kuramoto–Sivashinsky equation.”

This repository contains the numerical experiments and dataset generation
pipeline used to study dimensional scaling in the 1D Kuramoto–Sivashinsky (KS) equation.

Datasets are generated using a pseudo-spectral solver (`spooky` backend),
and are used for downstream dimensionality-reduction

## Dataset Generation

Datasets are created with:

```bash
python create_dataset.py
```

Simulation parameters are defined in:

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
