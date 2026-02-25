# create_dataset.py
import os
import glob
import numpy as np

import spooky as sp
from spooky.solvers import KuramotoSivashinsky

import params as pm


# --------------------------- USER PARAMETERS --------------------------- #
N_total = 100_000
realizations = 10
train_fraction = 0.8

transient_cut_steps = 1_000_000  # timesteps
seed = 0

save_dir = f"KS_dataset_L{pm.Lx}_nu{getattr(pm,'nu','NA')}_N{pm.Nx}_T{pm.Tevolve}_N{N_total}"
os.makedirs(save_dir, exist_ok=True)


def random_ic(x, Lx, rng, n_terms=(3, 6), max_freq=6, amp_range=(0.3, 1.0)):
    """Random cosine-mixture initial condition."""
    n = rng.integers(n_terms[0], n_terms[1])
    freqs = rng.integers(1, max_freq, size=n)
    amps = rng.uniform(amp_range[0], amp_range[1], size=n)
    phases = rng.uniform(0, 2 * np.pi, size=n)

    u0 = np.zeros_like(x, dtype=float)
    for a, f, ph in zip(amps, freqs, phases):
        u0 += a * np.cos(2 * np.pi * f * x / Lx + ph)
    return u0


def load_uu_stack(pattern="uu.*.npy"):
    """Load uu.XXXX.npy into array (n_snaps, Nx)."""
    files = sorted(glob.glob(pattern))
    return np.array([np.load(f) for f in files])


def cleanup_uu(pattern="uu.*.npy"):
    for f in glob.glob(pattern):
        os.remove(f)


def main():
    rng = np.random.default_rng(seed)

    snaps_per_real = int(np.ceil(N_total / realizations))
    cut_index = int(transient_cut_steps / pm.ostep)

    chunks = []

    for r in range(realizations):
        grid = sp.Grid1D(pm.Lx, pm.Nx, pm.dt)
        solver = KuramotoSivashinsky(grid)

        u0 = random_ic(grid.xx, pm.Lx, rng)

        # evolves and writes uu.XXXX.npy
        solver.evolve([u0], T=pm.Tevolve, bstep=pm.bstep, ostep=pm.ostep)

        fields = load_uu_stack("uu.*.npy")  # (n_snaps, Nx)
        cleanup_uu("uu.*.npy")

        # minimal sanity check
        if not np.all(np.isfinite(fields)):
            raise RuntimeError(f"Non-finite values in realization {r+1}. Try smaller dt / different IC.")

        fields = fields[cut_index:]  # drop transient

        if fields.shape[0] > snaps_per_real:
            idx = np.linspace(0, fields.shape[0] - 1, snaps_per_real, dtype=int)
            fields = fields[idx]

        chunks.append(fields.T)  # (Nx, n_snaps)

        print(f"Realization {r+1}/{realizations}: kept {fields.shape[0]} snapshots")

    dataset = np.hstack(chunks)       # (Nx, N_total_approx)
    dataset = dataset[:, :N_total]    # enforce exact N_total

    N_train = int(train_fraction * dataset.shape[1])
    u_train = dataset[:, :N_train]
    u_test = dataset[:, N_train:]

    max_train = np.max(np.abs(u_train))
    u_train /= max_train
    u_test /= max_train

    nu_val = getattr(pm, "nu", None)
    nu_tag = "NA" if nu_val is None else str(nu_val)

    np.save(os.path.join(save_dir, f"u_train_L{pm.Lx}_nu{nu_tag}_N{pm.Nx}.npy"), u_train)
    np.save(os.path.join(save_dir, f"u_test_L{pm.Lx}_nu{nu_tag}_N{pm.Nx}.npy"), u_test)

    np.savez(
        os.path.join(save_dir, "data_meta.npz"),
        L=pm.Lx, Nx=pm.Nx, nu=nu_val,
        Tevolve=pm.Tevolve, dt=pm.dt, ostep=pm.ostep, bstep=pm.bstep, rkord=getattr(pm, "rkord", None),
        transient_cut_steps=transient_cut_steps,
        cut_index=cut_index,
        realizations=realizations,
        N_total_effective=dataset.shape[1],
        train_fraction=train_fraction,
        max_train=max_train,
        seed=seed,
    )

    print(f"\nSaved dataset in: {save_dir}")
    print(f"Train shape: {u_train.shape} | Test shape: {u_test.shape}")
    print(f"Normalization max_train: {max_train:.6g}")


if __name__ == "__main__":
    main()
