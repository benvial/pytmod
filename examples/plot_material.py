# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


"""
Time-Modulated Material Eigenmodes
==================================

This example demonstrates how to work with time-modulated materials using
``pytmod``. We will explore the Floquet eigenmodes of a material whose
permittivity varies sinusoidally in time, and study their properties including
temporal profiles, convergence characteristics, and dispersion relations.

This example reproduces results from :cite:t:`zurita-sanchez2009`.
"""

# %%
# Imports and setup
# -----------------
# We begin by importing the necessary packages and configuring matplotlib for
# interactive plotting.

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import pytmod as pm

plt.ion()
plt.close("all")


# %%
# Background
# ----------
# When a material's permittivity varies periodically in time as
# :math:`\epsilon(t) = \epsilon_0 + \Delta\epsilon \sin(\Omega t)`,
# the wave equation inside the material becomes:
#
# .. math::
#     \nabla^2 E = \epsilon(t) \frac{\partial^2 E}{\partial t^2}
#
# Due to the periodicity, Floquet theory applies: solutions can be written as
# a superposition of harmonics at frequencies :math:`\omega + n\Omega`, where
# :math:`n` is an integer. This leads to an eigenvalue problem where the
# eigenvalues correspond to effective wavenumbers :math:`k_n` and the
# eigenvectors describe the temporal structure of each mode.


# %%
# Defining a time-modulated material
# ----------------------------------
# We define a material with sinusoidal permittivity modulation:
#
# .. math::
#     \epsilon(t) = \epsilon_0 + \Delta\epsilon \sin(\Omega t)
#
# where :math:`\epsilon_0 = 5.25` is the background permittivity,
# :math:`\Delta\epsilon = 2` is the modulation amplitude, and
# :math:`\Omega = 1` is the modulation frequency.
#
# The Fourier coefficients for this modulation are:
# :math:`\epsilon_{-1} = -i\Delta\epsilon/2`,
# :math:`\epsilon_0 = \epsilon_0`,
# :math:`\epsilon_{+1} = +i\Delta\epsilon/2`.

Omega = 1.0  # Modulation frequency
eps0 = 5.25  # Background permittivity
deps = 2.0  # Modulation amplitude

# Fourier coefficients for sin(Ωt) modulation
# The array is ordered as [ε₋₁, ε₀, ε₊₁]
eps_fourier = [
    -deps / (2 * 1j),  # ε₋₁ = -iΔε/2
    eps0,  # ε₀
    deps / (2 * 1j),  # ε₊₁ = +iΔε/2
]

# Create the Material object
mat = pm.Material(eps_fourier, Omega)

print(f"Material created with modulation period T = {mat.modulation_period:.4f}")
print(f"Number of harmonics (before padding): {mat.nh}")
print(f"Central harmonic index Nh: {mat.Nh}")


# %%
# Visualizing the permittivity in time domain
# -------------------------------------------
# Let's reconstruct and plot the time-domain permittivity to verify our
# Fourier representation matches the intended sinusoidal modulation.

T = mat.modulation_period
t = np.linspace(0, 3 * T, 3000)  # Three periods for visualization
eps_time = mat.get_eps_time(t)

plt.figure(figsize=(8, 4))
plt.plot(t / T, eps_time.real, color="#c24c4c", linewidth=2)
plt.xlabel(r"Normalized time $t/T$", fontsize=12)
plt.ylabel(r"Re $\epsilon(t)$", fontsize=12)
plt.title("Time-modulated permittivity", fontsize=14)
plt.xlim(0, 3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# Computing Floquet eigenmodes
# ----------------------------
# At a given frequency ω, the eigenvalue problem yields:
#
# - **Eigenvalues** :math:`k_n^2`: These correspond to the effective
#   wavenumbers of the Floquet modes
# - **Eigenvectors** :math:`e_n`: These describe the temporal structure
#   of each mode across the Floquet harmonics
#
# Let's compute the eigenmodes at frequency ω = 0.8.

omega = 0.8
eigenvalues, eigenvectors = mat.eigensolve(omega)

print(f"\nEigenvalues (k²) at ω = {omega}:")
print(eigenvalues)
print("\nCorresponding wavenumbers k = ±√(k²):")
print(np.sqrt(eigenvalues))


# %%
# Visualizing eigenmodes in the time domain
# -----------------------------------------
# Each eigenvector represents a mode's amplitude across Floquet harmonics.
# We convert these frequency-domain representations to the time domain using
# the :meth:`~pytmod.Material.freq2time` method to see the actual temporal
# evolution of each mode.

plt.figure(figsize=(10, 6))
for i in range(min(4, len(eigenvalues))):  # Plot first 4 modes
    k_n = np.sqrt(eigenvalues[i])
    mode_freq = eigenvectors[:, i]
    mode_time = mat.freq2time(mode_freq, t)

    plt.plot(
        t / T, mode_time.real, label=f"Mode {i}: k ≈ {k_n.real:.3f}", linewidth=1.5
    )

plt.xlabel(r"Normalized time $t/T$", fontsize=12)
plt.ylabel(r"Re $e_n(t)$", fontsize=12)
plt.title("Time-domain Floquet eigenmodes", fontsize=14)
plt.xlim(0, 3)
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# Convergence study: Truncation of the Floquet expansion
# ------------------------------------------------------
# The Floquet expansion must be truncated for numerical computation.
# Here we study how the eigenvalues and eigenvectors converge as we
# increase the number of harmonics (controlled by the ``Npad`` parameter).
#
# We compute the first three eigenvalues for increasing truncation orders
# and compare them to a highly converged reference calculation.

Nmax = 15  # Maximum truncation order to test
Npads = range(Nmax)
eigenvalues_convergence = []
modes_convergence = []

print("\nPerforming convergence study...")

for Npad in Npads:
    # Create material with increased padding
    mat_test = pm.Material(eps_fourier, Omega, Npad=Npad)

    # Solve eigenvalue problem (without sign selection, with sorting)
    ksq, modes = mat_test.eigensolve(omega, sign=False, sort=True)

    # Sort by eigenvalue magnitude
    idx = np.argsort(ksq)
    ksq = ksq[idx]
    modes = modes[:, idx]

    # Store first 3 eigenvalues
    eigenvalues_convergence.append(ksq[:3].real)

    # Store first 3 modes in time domain
    modes_time = []
    for i in range(3):
        mode = modes[:, i]
        mode_time = mat_test.freq2time(mode, t)
        modes_time.append(mode_time)
    modes_convergence.append(modes_time)

    if Npad % 5 == 0:
        print(f"  Npad = {Npad:2d}: N = {mat_test.nh:2d} harmonics")

eigenvalues_convergence = np.array(eigenvalues_convergence)
modes_convergence = np.array(modes_convergence)


# %%
# Plotting eigenvalue convergence
# -------------------------------
# We plot how the first three eigenvalues converge as the number of
# Floquet harmonics increases.

Ns = 3 + 2 * np.array(Npads)  # Total number of harmonics

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: Eigenvalues vs truncation order
ax = axes[0]
for i in range(3):
    ax.plot(
        Ns,
        eigenvalues_convergence[:, i],
        marker="o",
        markersize=4,
        label=f"Mode {i}",
        linewidth=1.5,
    )
ax.set_xlabel("Number of harmonics $N$", fontsize=12)
ax.set_ylabel("Eigenvalue Re($k^2$)", fontsize=12)
ax.set_title("Eigenvalue convergence", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Compute reference values with high truncation
mat_ref = pm.Material(eps_fourier, Omega, Npad=100)
ksq_ref, _ = mat_ref.eigensolve(omega, sign=False, sort=True)
idx = np.argsort(ksq_ref)
ksq_ref = ksq_ref[idx]
reference_values = ksq_ref[:3].real

# Right panel: Relative error
ax = axes[1]
for i in range(3):
    rel_error = np.abs(1 - eigenvalues_convergence[:, i] / reference_values[i])
    ax.semilogy(
        Ns, rel_error, marker="o", markersize=4, label=f"Mode {i}", linewidth=1.5
    )
ax.set_xlabel("Number of harmonics $N$", fontsize=12)
ax.set_ylabel("Relative error", fontsize=12)
ax.set_title("Convergence error", fontsize=14)
ax.set_xlim(3, 2 * Nmax + 1)
ax.legend()
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.show()


# %%
# Visualizing eigenstate convergence
# ----------------------------------
# We visualize how the temporal profiles of the eigenstates converge
# as the truncation order increases. Darker colors indicate higher
# truncation orders (more harmonics included).

cmap = mpl.colormaps["Blues"]
colors = cmap(np.linspace(0.3, 1.0, 10))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, ax in enumerate(axes):
    ax.set_title(f"Mode {i}", fontsize=14)

    # Plot modes for different truncation orders
    for j in range(10):
        ax.plot(
            t / T,
            modes_convergence[j, i].real,
            color=colors[j],
            linewidth=1.2,
            alpha=0.8,
        )

    ax.set_xlim(0, 3)
    ax.set_xlabel(r"$t/T$", fontsize=11)
    ax.set_ylabel("Mode amplitude" if i == 0 else "", fontsize=11)
    ax.grid(True, alpha=0.3)

# Add a colorbar to indicate truncation order
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=3, vmax=21))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
cbar.set_label("Number of harmonics $N$", fontsize=11)

plt.suptitle("Eigenstate convergence with increasing truncation", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


# %%
# Computing the dispersion relation
# ---------------------------------
# The dispersion relation :math:`\omega(k)` or :math:`k(\omega)` describes
# how waves propagate in the time-modulated medium. We can compute it in
# two ways:
#
# 1. Fix ω and solve for k (using :meth:`~pytmod.Material.eigensolve`)
# 2. Fix k and solve for ω (using :meth:`~pytmod.Material.eigensolve_omega`)
#
# These two approaches should give consistent results where they overlap.

# Create a material with moderate truncation for dispersion calculation
mat_disp = pm.Material(eps_fourier, Omega, Npad=1)

# Method 1: Fix ω, solve for k
omegas = np.linspace(0.01, 1.0, 100)
k_vs_omega, _ = mat_disp.eigensolve(omegas, sign=False, sort=True)

# Method 2: Fix k, solve for ω
ks = np.linspace(0.0, 3.0, 500)
omega_vs_k, _ = mat_disp.eigensolve_omega(ks, sort=True)

# Filter to valid frequency range (0 ≤ Re(ω) ≤ Ω)
omega_vs_k = np.where(np.real(omega_vs_k) <= Omega, omega_vs_k, np.nan)
omega_vs_k = np.where(np.real(omega_vs_k) >= 0, omega_vs_k, np.nan)


# %%
# Plotting the dispersion relation
# --------------------------------
# We plot both the real and imaginary parts of the dispersion relation,
# comparing the two computational approaches.

fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# Real part
ax = axes[0]
line1 = ax.plot(
    ks / Omega,
    omega_vs_k.T.real / Omega,
    "-",
    color="#5066d4",
    linewidth=2,
    label=r"$\omega$ vs. $k$ (fixed $k$)",
)
line2 = ax.plot(
    k_vs_omega.T.real / Omega,
    omegas / Omega,
    "o",
    color="#d4507a",
    markerfacecolor="none",
    markersize=4,
    markeredgewidth=1,
    label=r"$k$ vs. $\omega$ (fixed $\omega$)",
)
ax.set_xlabel(r"Normalized wavenumber $k/K$", fontsize=12)
ax.set_ylabel(r"Re $\omega/\Omega$", fontsize=12)
ax.set_title("Dispersion relation: Real part", fontsize=14)
ax.set_xlim(0, 3)
ax.set_ylim(0, 1)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# Imaginary part
ax = axes[1]
ax.plot(
    ks / Omega,
    omega_vs_k.T.imag / Omega,
    ".",
    color="#5066d4",
    markersize=3,
    label=r"$\omega$ vs. $k$",
)
ax.set_xlabel(r"Normalized wavenumber $k/K$", fontsize=12)
ax.set_ylabel(r"Im $\omega/\Omega$", fontsize=12)
ax.set_title("Dispersion relation: Imaginary part", fontsize=14)
ax.set_xlim(0, 3)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# Summary
# -------
# This example demonstrated:
#
# 1. **Creating a time-modulated material**: Using Fourier coefficients to
#    represent sinusoidal permittivity modulation.
#
# 2. **Computing Floquet eigenmodes**: Solving the eigenvalue problem to
#    find the wavenumbers and temporal profiles of modes.
#
# 3. **Convergence analysis**: How eigenvalues and eigenstates converge as
#    the Floquet truncation order increases. The exponential convergence
#    is typical for smooth modulations.
#
# 4. **Dispersion relations**: Computing and visualizing the frequency-wavenumber
#    relationship using two complementary approaches.
#
# The eigenmodes of time-modulated materials form the basis for understanding
# more complex phenomena like scattering from time-modulated scatterers and
# parametric amplification effects.
