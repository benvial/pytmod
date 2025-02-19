#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3


import numpy as np
import scipy.special as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
from pytmod.eig import nonlinear_eigensolver
from pytmod.helpers import move_last_two_axes_to_beginning

np.random.seed(12345)


plt.ion()
plt.close("all")

# --- Problem Parameters ---
rho = 1.0  # Mass density
h = 0.1  # Plate thickness
E = 1.0  # Young's modulus
nu = 0.3  # Poisson ratio
N = 10  # Number of resonators
positions = np.random.rand(N, 2)  # Randomly placed resonators

# Resonator parameters
m_R = np.ones(N)  # Resonator masses
k_R = np.ones(N)  # Spring constants
omega_R = np.sqrt(k_R / m_R)  # Resonant frequencies


# Plate bending stiffness
D = E * h**3 / (12 * (1 - nu**2))

EPS = 1e-12


def j0(kr):
    return sp.jn(0, kr)


def y0(kr):
    return sp.yv(0, kr)


def k0(kr):
    return sp.kv(0, kr)


def gfreal(kr):
    return np.where(np.abs(np.array(kr)) < EPS, 1.0, j0(kr))


def gfimag(kr):
    return np.where(
        np.abs(np.array(kr)) < EPS,
        0.0,
        y0(kr) + 2 / np.pi * k0(kr),
    )


def _norma_gf(k):
    return 1j / (8 * k**2)


# --- Green's Function for the Plate ---
def greens_function(r, k):
    """Green's function for a thin plate (Kirchhoff-Love theory)."""
    kr = k * r
    return _norma_gf(k) * (gfreal(kr) + 1j * gfimag(kr))


# --- Interaction Matrix ---
def t_alpha(omega, i):
    """Resonator impedance function t_alpha(omega)."""
    return (m_R[i] / D) * (omega_R[i] ** 2 * omega**2) / (omega_R[i] ** 2 - omega**2)


def M_matrix(omega):
    """Construct the interaction matrix M(omega)."""
    k = omega**0.5 * (rho * h / D) ** (1 / 4)  # Plate wavenumber
    M = np.zeros((N, N) + omega.shape, dtype=complex)
    for i in range(N):
        for j in range(N):
            delta = 1 if i == j else 0
            r = np.linalg.norm(positions[i] - positions[j])
            M[i, j] = delta / t_alpha(omega, j) - greens_function(r, k)
    return M


def two_sided_arnoldi(M, b, c, m):
    """Compute left and right Krylov subspaces using Arnoldi iteration."""
    n = M.shape[0]

    # Right Arnoldi basis
    V = np.zeros((n, m + 1), dtype=np.complex128)
    H = np.zeros((m + 1, m), dtype=np.complex128)
    V[:, 0] = b / la.norm(b)

    # Left Arnoldi basis
    W = np.zeros((n, m + 1), dtype=np.complex128)
    G = np.zeros((m + 1, m), dtype=np.complex128)
    W[:, 0] = c / la.norm(c)

    for j in range(m):
        # Right subspace
        w = M @ V[:, j]
        for i in range(j + 1):
            H[i, j] = np.vdot(V[:, i], w)
            w -= H[i, j] * V[:, i]
        H[j + 1, j] = la.norm(w)
        if H[j + 1, j] < 1e-12:
            break
        V[:, j + 1] = w / H[j + 1, j]

        # Left subspace (acting on M†)
        z = M.conj().T @ W[:, j]
        for i in range(j + 1):
            G[i, j] = np.vdot(W[:, i], z)
            z -= G[i, j] * W[:, i]
        G[j + 1, j] = la.norm(z)
        if G[j + 1, j] < 1e-12:
            break
        W[:, j + 1] = z / G[j + 1, j]

    return V[:, :m], W[:, :m], H[:m, :m], G[:m, :m]


# --- Rayleigh-Ritz Projection ---


def bi_orthogonal_projection(omega, V, W):
    """Compute the reduced bi-orthogonal projection of M(omega)."""
    M = M_matrix(omega)

    Mproj = (
        W.conj().swapaxes(1, 0) @ move_last_two_axes_to_beginning(M) @ V
    )  # Bi-orthogonal projection

    return move_last_two_axes_to_beginning(Mproj)


# --- Solve for Eigenvalues ---
def determinant(omega, V):
    """Compute determinant of the reduced M(omega)."""
    return np.linalg.det(reduced_M_matrix(omega, V))


omega0 = 0.2 - 2.32j
omega1 = 12.92 + 0.0001j

nc = 101

omegasr = np.linspace(omega0.real, omega1.real, nc)
omegasi = np.linspace(omega0.imag, omega1.imag, nc)

re, im = np.meshgrid(omegasr, omegasi)
omegas = re + 1j * im

# matrix = M_matrix(omegas)

# plt.figure()
# detM = np.linalg.det(matrix.T).T
# plt.pcolormesh(omegasr, omegasi, np.log10(np.abs(detM)), cmap="inferno")
# plt.colorbar()
# plt.title(r"det $M(\omega)$")
# plt.xlim(omegasr[0], omegasr[-1])
# plt.ylim(omegasi[0], omegasi[-1])
# plt.xlabel(r"Re $\omega$")
# plt.ylabel(r"Im $\omega$")
# plt.pause(0.1)


# omega_arnoldi = (omega0 + omega1) / 2
# # Krylov Basis Selection
# M_sample = M_matrix(np.array([omega_arnoldi]))[
#     :, :, 0
# ]  # Compute M at an arbitrary frequency
# b = np.random.randn(N) + 1j * np.random.randn(N)  # Random initial vector
# c = np.random.randn(N) + 1j * np.random.randn(N)  # Random initial vector
# m_subspace = 4  # Dimension of the reduced space
# V, W, _, _ = two_sided_arnoldi(M_sample, b, c, m_subspace)  # Compute Krylov basis


# def M_matrix_red(omega):
#     return bi_orthogonal_projection(omega, V, W)


# fun = M_matrix

# import time

# t = -time.time()

# plt.figure()
# evs, modes = nonlinear_eigensolver(
#     M_matrix,
#     omega0,
#     omega1,
#     plot_solver=True,
# )
# t += time.time()


# tred = -time.time()
# evs_red, modes_red = nonlinear_eigensolver(
#     M_matrix_red,
#     omega0,
#     omega1,
#     plot_solver=True,
# )
# tred += time.time()


# tapp = -time.time()
# evs_app, modes_app = nonlinear_eigensolver(
#     M_matrix,
#     omega0,
#     omega1,
#     guesses=evs_red,
#     plot_solver=True,
# )
# tapp += time.time()

# plt.clf()
# plt.plot(evs.real, evs.imag, ".g")
# plt.plot(evs_red.real, evs_red.imag, "*r")
# plt.plot(evs_app.real, evs_app.imag, "+k")


# print(evs)
# print(evs_red)
# print(evs_app)
# print(t)
# print(tred)
# print(tapp)
# print(t / tred)
# print(t / (tapp + tred))


import numpy as np
from scipy.special import hankel2, gamma, factorial


def greens_function_series(r, k, n_terms=4):
    """
    Calculate the Green's function G(r, omega) for a thin elastic plate using series representation.

    Parameters:
    r (float): Radial distance from the source point.
    k (float): Wavenumber

    Returns:
    G (complex): Value of the Green's function at (r, omega).
    """

    # Series representation
    G = 0
    for n in range(n_terms):
        G += (
            (-1) ** n
            * (k * r / 2) ** (2 * n)
            * (1 - (1j) ** (2 * n))
            / (factorial(n) * gamma(n + 1))
        )
    G *= 1j / (8 * k**2)
    return G


import numpy as np
from scipy.special import digamma, factorial


def greens_function_series(r, k, n_terms=20):
    """Compute the Green's function G_0(r) for a thin elastic plate using its series expansion."""
    sum_series = 0

    for m in range(1, n_terms):  # Start from m=1 (first term in sum)
        term = (-1) ** m / (factorial(m) ** 2) * (k * r / 2) ** (2 * m)
        correction = 1 - (2j / np.pi) * (digamma(m + 1) + digamma(m))
        sum_series += term * correction

    G0 = (1j / (8 * k**2)) * sum_series
    return G0


r = 1e-31
k = 1
g0 = greens_function(r, k)
g1 = greens_function_series(r, k)
print(g0)
print(g1)

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1, digamma, factorial


# Function to compute the series expansion of H0(z) - H0(i*z)
def series_expansion_H0(z, n_terms=10):
    result = 0
    for m in range(1, n_terms + 1):
        # Calculate the coefficient a_m
        a_m = -2 / ((2 * m + 1) ** 2)
        # Accumulate the series
        result += a_m * (z / 2) ** (4 * m + 2)
    return result


def green_function_series(r, k, n_terms=20):
    """Compute the Green's function G_0(r) for a thin elastic plate using its series expansion."""
    return (1j / (8 * k**2)) * series_expansion_H0(k * r, n_terms)


def exact_green_function(r, k):
    """Calculate the exact Green's function G_0(r) using the Hankel function."""
    H0_real = hankel1(0, k * r)
    H0_imag = hankel1(0, 1j * k * r)
    return (1j / (8 * k**2)) * (H0_real - H0_imag)


# Parameters
k = 1.0  # Wavenumber
r_values = np.linspace(0.01, 1.0, 50)  # Avoid r=0 (singularity)
n_terms = 30  # Number of terms in the series

# Compute values for comparison
G0_approx_values = np.array([green_function_series(r, k, n_terms) for r in r_values])
G0_exact_values = np.array([exact_green_function(r, k) for r in r_values])

# Compute relative errors
relative_errors = np.abs(G0_approx_values - G0_exact_values) / np.abs(G0_exact_values)

# Plot relative error
plt.figure(figsize=(7, 5))
plt.semilogy(r_values, relative_errors, "r-", label="Relative Error")
plt.xlabel(r"$r$")
plt.ylabel("Relative Error")
plt.title("Error in Series Expansion of $G_0(r)$")
plt.legend()
plt.grid()
plt.show()

# Print example results for small r
for r, G_approx, G_exact, err in zip(
    r_values[:5], G0_approx_values[:5], G0_exact_values[:5], relative_errors[:5]
):
    print(
        f"r = {r:.3f}, Approx = {G_approx:.6g}, Exact = {G_exact:.6g}, Rel. Error = {err:.2e}"
    )
