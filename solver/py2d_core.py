"""Vendored subset of py2d (MIT license) for 2D turbulence simulation.

Contains only the functions needed by the gust solver, extracted from
https://github.com/envfluids/py2d to remove the external dependency
(which hardcodes output paths that conflict with gust's --output_dir).

Original authors: envfluids group.
License: MIT (same as upstream).
"""

import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit


# ---------------------------------------------------------------------------
# From py2d/util.py
# ---------------------------------------------------------------------------

def fft2_to_rfft2(a_hat_fft):
    if a_hat_fft.shape[0] % 2 == 0:
        return a_hat_fft[:, :a_hat_fft.shape[1] // 2 + 1]
    else:
        return a_hat_fft[:, :(a_hat_fft.shape[1] - 1) // 2 + 1]


# ---------------------------------------------------------------------------
# From py2d/initialize.py
# ---------------------------------------------------------------------------

def gridgen(Lx, Ly, Nx, Ny, INDEXING='ij'):
    dx = Lx / Nx
    dy = Ly / Ny
    x = np.linspace(0, Lx - dx, num=Nx)
    y = np.linspace(0, Lx - dx, num=Ny)
    X, Y = np.meshgrid(x, y, indexing=INDEXING)
    return Lx, Ly, X, Y, dx, dy


def initialize_wavenumbers_fft2(nx, ny, Lx, Ly, INDEXING='ij'):
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
    (Kx, Ky) = np.meshgrid(kx, ky, indexing=INDEXING)
    Ksq = Kx ** 2 + Ky ** 2
    Kabs = np.sqrt(Ksq)
    Ksq[0, 0] = 1e16
    invKsq = 1.0 / Ksq
    invKsq[0, 0] = 0.0
    Ksq[0, 0] = 0.0
    return Kx, Ky, Kabs, Ksq, invKsq


def initialize_wavenumbers_rfft2(nx, ny, Lx, Ly, INDEXING='ij'):
    Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_fft2(
        nx, ny, Lx, Ly, INDEXING=INDEXING)
    return (fft2_to_rfft2(Kx), fft2_to_rfft2(Ky), fft2_to_rfft2(Kabs),
            fft2_to_rfft2(Ksq), fft2_to_rfft2(invKsq))


# ---------------------------------------------------------------------------
# From py2d/convert.py
# ---------------------------------------------------------------------------

def Omega2Psi_spectral(Omega_hat, invKsq):
    return -(-Omega_hat) * invKsq  # = Omega_hat * invKsq


def Psi2UV_spectral(Psi_hat, Kx, Ky):
    U_hat = (1.j) * Ky * Psi_hat
    V_hat = -(1.j) * Kx * Psi_hat
    return U_hat, V_hat


# ---------------------------------------------------------------------------
# From py2d/filter.py
# ---------------------------------------------------------------------------

@partial(jit, static_argnums=(1,))
def coarse_spectral_filter_square_jit(a_hat, NCoarse):
    N = a_hat.shape[0]
    kn_fine = N // 2
    kn_coarse = NCoarse // 2

    u_hat_coarse = jnp.zeros((NCoarse, NCoarse // 2 + 1), dtype=complex)
    u_hat_coarse = u_hat_coarse.at[:kn_coarse + 1, :].set(
        a_hat[:kn_coarse + 1, :kn_coarse + 1])
    u_hat_coarse = u_hat_coarse.at[kn_coarse + 1:, :].set(
        a_hat[kn_fine + 1 + (kn_fine - kn_coarse):, :kn_coarse + 1])
    u_hat_coarse = u_hat_coarse * ((NCoarse / N) ** 2)

    if NCoarse % 2 == 0:
        u_hat_coarse = u_hat_coarse.at[kn_coarse, :].set(0)
        u_hat_coarse = u_hat_coarse.at[:, kn_coarse].set(0)
    else:
        u_hat_coarse = u_hat_coarse.at[kn_coarse, :].set(0.0)
        u_hat_coarse = u_hat_coarse.at[kn_coarse + 1, :].set(0.0)
        u_hat_coarse = u_hat_coarse.at[:, kn_coarse].set(0.0)

    return u_hat_coarse


# ---------------------------------------------------------------------------
# From py2d/dealias.py
# ---------------------------------------------------------------------------

def padding_for_dealias_spectral_jit(u_hat, K=3 / 2):
    N_coarse = u_hat.shape[0]
    N_pad = int(K * N_coarse)
    kn_pad = N_pad // 2
    kn_coarse = N_coarse // 2

    u_hat_scaled = (N_pad / N_coarse) ** 2 * u_hat
    u_hat_pad = jnp.zeros((N_pad, kn_pad + 1), dtype=complex)

    u_hat_pad = u_hat_pad.at[:kn_coarse + 1, :kn_coarse + 1].set(
        u_hat_scaled[:kn_coarse + 1, :kn_coarse + 1])

    if N_pad % 2 == 0:
        u_hat_pad = u_hat_pad.at[N_pad - kn_coarse + 1:, :kn_coarse + 1].set(
            u_hat_scaled[N_coarse - kn_coarse + 1:, :kn_coarse + 1])
    else:
        u_hat_pad = u_hat_pad.at[N_pad - kn_coarse:, :kn_coarse + 1].set(
            u_hat_scaled[N_coarse - kn_coarse:, :kn_coarse + 1])

    if N_pad % 2 == 0:
        u_hat_pad = u_hat_pad.at[kn_coarse, :].set(0)
        u_hat_pad = u_hat_pad.at[:, kn_coarse].set(0)
    else:
        u_hat_pad = u_hat_pad.at[kn_coarse, :].set(0)
        u_hat_pad = u_hat_pad.at[kn_pad + (kn_pad - kn_coarse) + 1, :].set(0)
        u_hat_pad = u_hat_pad.at[:, kn_coarse].set(0)

    return u_hat_pad


@jit
def multiply_dealias_spectral_jit(a_hat, b_hat):
    Ncoarse = a_hat.shape[0]

    a_dealias_hat = padding_for_dealias_spectral_jit(a_hat)
    b_dealias_hat = padding_for_dealias_spectral_jit(b_hat)

    Nfine = a_dealias_hat.shape[0]

    a_dealias = jnp.fft.irfft2(a_dealias_hat, s=[Nfine, Nfine])
    b_dealias = jnp.fft.irfft2(b_dealias_hat, s=[Nfine, Nfine])

    a_dealias_b_dealias_hat = jnp.fft.rfft2(a_dealias * b_dealias)

    ab_dealias_hat = coarse_spectral_filter_square_jit(
        a_dealias_b_dealias_hat, Ncoarse)

    return ab_dealias_hat


# ---------------------------------------------------------------------------
# From py2d/convection_conserved.py
# ---------------------------------------------------------------------------

@jit
def convection_conserved_dealias(Omega1_hat, U1_hat, V1_hat, Kx, Ky):
    # Conservative form (dealiased)
    U1Omega1_hat = multiply_dealias_spectral_jit(U1_hat, Omega1_hat)
    V1Omega1_hat = multiply_dealias_spectral_jit(V1_hat, Omega1_hat)

    conu1 = (1.j) * Kx * U1Omega1_hat
    conv1 = (1.j) * Ky * V1Omega1_hat
    convec_hat = conu1 + conv1

    # Non-conservative form (dealiased)
    Omega1x_hat = (1.j) * Kx * Omega1_hat
    Omega1y_hat = (1.j) * Ky * Omega1_hat

    U1Omega1x_hat = multiply_dealias_spectral_jit(U1_hat, Omega1x_hat)
    V1Omega1y_hat = multiply_dealias_spectral_jit(V1_hat, Omega1y_hat)

    convecN_hat = U1Omega1x_hat + V1Omega1y_hat

    return 0.5 * (convec_hat + convecN_hat)


@jit
def convection_conserved(Omega1_hat, U1_hat, V1_hat, Kx, Ky):
    N = Omega1_hat.shape[0]

    U1 = jnp.fft.irfft2(U1_hat, s=[N, N])
    V1 = jnp.fft.irfft2(V1_hat, s=[N, N])
    Omega1 = jnp.fft.irfft2(Omega1_hat, s=[N, N])

    # Conservative form
    conu1 = (1.j) * Kx * jnp.fft.rfft2(U1 * Omega1)
    conv1 = (1.j) * Ky * jnp.fft.rfft2(V1 * Omega1)
    convec_hat = conu1 + conv1

    # Non-conservative form
    Omega1x = jnp.fft.irfft2((1.j) * Kx * Omega1_hat, s=[N, N])
    Omega1y = jnp.fft.irfft2((1.j) * Ky * Omega1_hat, s=[N, N])

    convecN_hat = jnp.fft.rfft2(U1 * Omega1x) + jnp.fft.rfft2(V1 * Omega1y)

    return 0.5 * (convec_hat + convecN_hat)
