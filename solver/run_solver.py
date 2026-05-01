"""Run 2D turbulence simulation using vendored py2d routines.

Generates an HDF5 file containing vorticity (Omega) snapshots directly
into --output_dir/output.h5 in the gust common format.

Uses an Adam-Bashforth 2nd order / Crank-Nicolson time-stepping scheme
with 3/2 dealiasing, matching the upstream py2d Py2D_solver.

Example:
    python -m solvers.py2d_turbulence.run_solver --output_dir ./output --Re 1000 --NX 512 --tTotal 20.0
    python -m solvers.py2d_turbulence.run_solver --output_dir ./output --Re 1000 --NX 64 --ic_file ~/data/10000.mat
"""

import argparse
import os
import time as timer_mod
from pathlib import Path

import h5py
import numpy as np
import jax.numpy as jnp
from scipy.io import loadmat

from . import py2d_core


def check_stability(Re, fkx, fky, NX, dt, Lx=2*np.pi):
    """Check CFL, viscous, and forcing stability conditions."""
    dx = Lx / NX
    U_max = 1.0
    nu = 1.0 / Re

    cfl = U_max * dt / dx
    viscous_limit = dx**2 / (4 * nu)
    viscous_ok = dt < viscous_limit

    kf = np.sqrt(fkx**2 + fky**2)
    if kf > 0:
        forcing_limit = 1.0 / (kf * U_max)
        forcing_ok = dt < forcing_limit
    else:
        forcing_limit = None
        forcing_ok = True

    print(f"Stability checks: Re={Re}, fkx={fkx}, fky={fky}, NX={NX}, dt={dt}")
    print(f"  CFL: {cfl:.3e} (OK if < 1.0)")
    print(f"  Viscous: dt={dt:.2e} < {viscous_limit:.2e} (OK: {viscous_ok})")
    if forcing_limit is not None:
        print(f"  Forcing: dt={dt:.2e} < {forcing_limit:.2e} (OK: {forcing_ok})")

    if not (cfl < 1.0 and viscous_ok and forcing_ok):
        print("  WARNING: One or more stability conditions violated!")
        return False
    print("  All stability conditions satisfied.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run 2D turbulence simulation (vendored py2d)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save HDF5 output')
    parser.add_argument('--Re', type=float, default=1000,
                        help='Reynolds number')
    parser.add_argument('--NX', type=int, default=512,
                        help='Grid points in x and y (square domain)')
    parser.add_argument('--fkx', type=int, default=4,
                        help='Forcing wavenumber in x')
    parser.add_argument('--fky', type=int, default=4,
                        help='Forcing wavenumber in y')
    parser.add_argument('--dt', type=float, default=1e-4,
                        help='Timestep')
    parser.add_argument('--tTotal', type=float, default=20.0,
                        help='Total simulation time')
    parser.add_argument('--tSave', type=float, default=1e-2,
                        help='Time interval for saving snapshots')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='Coriolis parameter (beta-plane)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Rayleigh drag coefficient')
    parser.add_argument('--ICnum', type=int, default=1,
                        help='Initial condition number (1-20)')
    parser.add_argument('--ic_file', type=str, default=None,
                        help='Path to a .mat file with Omega field to use as IC '
                             '(overrides --ICnum)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (unused by Py2D, but sets ICnum context)')
    parser.add_argument('--skip_stability_check', action='store_true',
                        help='Skip stability check before running')
    args = parser.parse_args()

    if not args.skip_stability_check:
        stable = check_stability(args.Re, args.fkx, args.fky, args.NX, args.dt)
        if not stable:
            print("Aborting due to stability violation. Use --skip_stability_check to override.")
            return

    output_dir = Path(args.output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)

    NX = args.NX
    Lx = 2 * np.pi
    Ly = 2 * np.pi
    nu = 1.0 / args.Re
    dt = args.dt
    alpha = args.alpha
    beta = args.beta
    maxit = int(args.tTotal / dt)
    NSAVE = int(args.tSave / dt)

    # --- Grid and wavenumbers ---
    _, _, X, Y, dx, dy = py2d_core.gridgen(Lx, Ly, NX, NX)
    Kx, Ky, Kabs, Ksq, invKsq = py2d_core.initialize_wavenumbers_rfft2(
        NX, NX, Lx, Ly)

    # Convert wavenumber arrays to JAX
    Kx = jnp.array(Kx)
    Ky = jnp.array(Ky)
    Ksq = jnp.array(Ksq)
    invKsq = jnp.array(invKsq)

    # --- Deterministic forcing ---
    fkx, fky = args.fkx, args.fky
    Fk = fky * np.cos(fky * Y) + fkx * np.cos(fkx * X)
    Fk_hat = jnp.array(np.fft.rfft2(Fk))

    # --- Initial conditions ---
    if args.ic_file is not None:
        ic_path = Path(args.ic_file).expanduser().resolve()
        if not ic_path.exists():
            raise FileNotFoundError(f"IC file not found: {ic_path}")
        mat = loadmat(str(ic_path))
        Omega0 = mat['Omega'].astype(np.float64)
        if Omega0.shape[0] != NX:
            # Simple regrid via spectral interpolation
            Omega0_hat_full = np.fft.rfft2(Omega0)
            N_old = Omega0.shape[0]
            Omega0_hat_new = np.zeros((NX, NX // 2 + 1), dtype=complex)
            kn_old = min(N_old // 2, NX // 2)
            Omega0_hat_new[:kn_old, :kn_old + 1] = Omega0_hat_full[:kn_old, :kn_old + 1]
            Omega0_hat_new[NX - kn_old + 1:, :kn_old + 1] = Omega0_hat_full[N_old - kn_old + 1:, :kn_old + 1]
            Omega0_hat_new *= (NX / N_old) ** 2
            Omega0 = np.fft.irfft2(Omega0_hat_new, s=[NX, NX])
        Omega1_hat = jnp.array(np.fft.rfft2(Omega0))
    else:
        # Load from bundled IC data (look relative to this file, then fallback)
        ic_search_dirs = [
            Path(__file__).parent / 'data' / 'ICs' / f'NX{NX}',
        ]
        ic_path = None
        for d in ic_search_dirs:
            candidate = d / f'{args.ICnum}.mat'
            if candidate.exists():
                ic_path = candidate
                break
        if ic_path is None:
            raise FileNotFoundError(
                f"No IC file found for NX={NX}, ICnum={args.ICnum}. "
                f"Use --ic_file to provide one.")
        mat = loadmat(str(ic_path))
        Omega0 = mat['Omega'].astype(np.float64)
        Omega1_hat = jnp.array(np.fft.rfft2(Omega0))

    Omega0_hat = Omega1_hat
    Psi1_hat = py2d_core.Omega2Psi_spectral(Omega1_hat, invKsq)
    Psi0_hat = Psi1_hat

    # --- Open HDF5 output file ---
    h5_path = output_dir / 'output.h5'
    h5f = h5py.File(h5_path, 'w')
    omega_ds = h5f.create_dataset(
        'fields/omega', shape=(0, NX, NX), maxshape=(None, NX, NX),
        dtype='float32', chunks=(1, NX, NX), compression='gzip')

    # --- Time-stepping ---
    time = 0.0
    save_count = 0
    t_start = timer_mod.time()

    convec_func = py2d_core.convection_conserved_dealias

    print(f"Running simulation: Re={args.Re}, NX={NX}, dt={dt}, "
          f"tTotal={args.tTotal}, tSave={args.tSave}")
    print(f"  maxit={maxit}, NSAVE={NSAVE}")

    for it in range(maxit):
        U1_hat, V1_hat = py2d_core.Psi2UV_spectral(Psi1_hat, Kx, Ky)
        convec1_hat = convec_func(Omega1_hat, U1_hat, V1_hat, Kx, Ky)

        if it == 0:
            U0_hat, V0_hat = py2d_core.Psi2UV_spectral(Psi0_hat, Kx, Ky)
            convec0_hat = convec_func(Omega0_hat, U0_hat, V0_hat, Kx, Ky)

        # AB2 convection
        convec_hat = 1.5 * convec1_hat - 0.5 * convec0_hat

        # Diffusion
        diffu_hat = -Ksq * Omega1_hat

        # CN + AB2 time advance (NoSGS: eddyViscosity = 0)
        RHS = (Omega1_hat
               - dt * convec_hat
               + dt * 0.5 * nu * diffu_hat
               - dt * Fk_hat
               + dt * beta * V1_hat)
        Omega_hat_new = RHS / (1.0 + dt * alpha + 0.5 * dt * nu * Ksq)

        # Shift state
        Omega0_hat = Omega1_hat
        convec0_hat = convec1_hat
        Psi0_hat = Psi1_hat

        Omega1_hat = Omega_hat_new
        Psi1_hat = py2d_core.Omega2Psi_spectral(Omega1_hat, invKsq)
        time += dt

        # Save snapshot
        if (it + 1) % NSAVE == 0:
            Omega = np.array(jnp.fft.irfft2(Omega1_hat, s=[NX, NX]))
            enstrophy = 0.5 * np.mean(Omega ** 2)

            if np.isnan(enstrophy):
                print(f"NaN detected at it={it+1}, time={time:.4f}. Aborting.")
                break

            omega_ds.resize(save_count + 1, axis=0)
            omega_ds[save_count] = Omega.astype(np.float32)
            save_count += 1

            if save_count % 100 == 0 or it + 1 == maxit:
                elapsed = timer_mod.time() - t_start
                print(f"  it={it+1}/{maxit}, time={time:.4f}, "
                      f"enstrophy={enstrophy:.6e}, saved={save_count}, "
                      f"elapsed={elapsed:.1f}s")

    # --- Write coordinates and metadata, close ---
    dx_grid = Lx / NX
    dy_grid = Ly / NX
    coords_grp = h5f.create_group('coordinates')
    coords_grp.create_dataset('x', data=np.arange(NX, dtype=np.float32) * dx_grid)
    coords_grp.create_dataset('y', data=np.arange(NX, dtype=np.float32) * dy_grid)

    meta_grp = h5f.create_group('metadata')
    meta_grp.attrs['solver'] = 'py2d'
    meta_grp.attrs['field'] = 'omega'
    meta_grp.attrs['n_samples'] = save_count
    meta_grp.attrs['H'] = NX
    meta_grp.attrs['W'] = NX
    meta_grp.attrs['Re'] = args.Re
    meta_grp.attrs['NX'] = NX

    h5f.close()

    elapsed = timer_mod.time() - t_start
    print(f"Simulation complete. {save_count} snapshots saved to {h5_path}")
    print(f"Total wall time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
