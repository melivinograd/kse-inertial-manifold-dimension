"""
Generate the autoencoder training configs for the full (L, nu) sweep.

For every (L, nu) case in the paper three configs are written, all sharing the
same dataset and optimisation settings and differing only as follows:

    run 0   reference network,  seed 0
    run 1   reference network,  seed 1   -> isolates the initialisation
    run 2   wider network,      seed 0   -> isolates the network capacity

Comparing run 0 with run 1 measures how much d_A moves when the same network is
retrained; comparing run 0 with run 2 checks that d_A does not move when the
network is made larger.

By default the bottleneck of each config is the d_A reported in the paper. To
reproduce a whole MSE(d) curve, pass the bottlenecks you want:

    python3 make_configs.py                  # one config per case at d = d_A
    python3 make_configs.py 40 50 55 60 70   # these bottlenecks, every case

Configs are written to configs/autoencoder/L{L}_nu{nu}/{d}_{run}.json and are
trained with

    python3 train_model.py configs/autoencoder/L44_nu0.1/60_0.json
"""
import json
import os
import sys

# (L, nu) -> (Nx, T, N_train in thousands, d_A reported in the paper)
CASES = {
    (22,  0.01): (512,  2000.0, 100,  90),
    (22,  0.1):  (256,  1100.0, 100,  30),
    (22,  1.0):  (256,   110.0, 100,   8),
    (44,  0.01): (1024, 1000.0,  99, 180),
    (44,  0.1):  (256,  1100.0, 100,  60),
    (44,  1.0):  (256,   110.0, 100,  19),
    (66,  0.01): (1024, 1000.0, 100, 280),
    (66,  0.1):  (256,  1100.0, 100,  85),
    (66,  1.0):  (256,  1000.0,  90,  26),
    (100, 0.01): (1024, 1100.0, 100, 410),
    (100, 0.1):  (256,  1100.0, 100, 130),
    (100, 1.0):  (256,  1000.0, 100,  42),
    (200, 0.1):  (512,  1100.0, 100, 260),
    (200, 1.0):  (256,  1000.0, 100,  90),
}

# run -> (encoder filters, seed)
RUNS = {
    0: ([32, 64, 128, 256], 0),
    1: ([32, 64, 128, 256], 1),
    2: ([64, 128, 256, 512], 0),
}

KERNEL = 5
STRIDE = 2


def make_config(L, nu, Nx, T, N_train, dh, encoder, seed):
    n = len(encoder)
    return {
        "N": Nx,
        "dh": dh,

        "encoder": encoder,
        "kernel_size": [KERNEL] * n,
        "strides": [STRIDE] * n,

        "lr": 7.5e-4,
        "batch_size": 16,
        "patience": 2,
        "epochs": 100,
        "seed": seed,

        "L": L,
        "nu": nu,
        "T": T,
        "N_train": N_train,
    }


def main():
    bottlenecks = [int(a) for a in sys.argv[1:]]

    n_written = 0
    for (L, nu), (Nx, T, N_train, d_A) in sorted(CASES.items()):
        out_dir = os.path.join("configs", "autoencoder", f"L{L}_nu{nu}")
        os.makedirs(out_dir, exist_ok=True)

        for dh in (bottlenecks or [d_A]):
            for run, (encoder, seed) in RUNS.items():
                cfg = make_config(L, nu, Nx, T, N_train, dh, encoder, seed)
                path = os.path.join(out_dir, f"{dh}_{run}.json")
                with open(path, "w") as f:
                    json.dump(cfg, f, indent=4)
                    f.write("\n")
                n_written += 1

    print(f"wrote {n_written} configs for {len(CASES)} (L, nu) cases")


if __name__ == "__main__":
    main()
