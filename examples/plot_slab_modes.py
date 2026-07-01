# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


"""
Quasi-Normal Modes of a Time-Modulated Slab
===========================================

This example demonstrates the computation of quasi-normal modes (QNMs) for a
slab of time-modulated material. QNMs are the natural resonances of the
open system, characterized by complex frequencies where the field decays
exponentially in time and grows exponentially in space.
"""

# %%
# Imports and setup
# -----------------

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pytmod as pm

# %%
# Background
# ----------
# Quasi-normal modes (QNMs) are the eigenmodes of an open optical system
# with outgoing boundary conditions. Unlike normal modes in closed cavities,
# QNMs have complex frequencies:
#
# .. math::
#     \omega_n = \omega_n' - i\gamma_n
#
# where :math:`\omega_n'` is the resonant frequency and :math:`\gamma_n` is
# the decay rate (linewidth). The imaginary part reflects the fact that energy
# leaks out of the open system.
#
# For a time-modulated slab, the QNM condition is given by:
#
# .. math::
#     \det M(\omega) = 0
#
# where :math:`M(\omega)` is the boundary condition matrix. This nonlinear
# eigenvalue problem is solved using a contour integration method that finds
# all poles within a specified region of the complex frequency plane.


# %%
# Problem parameters
# ------------------
# We define the material and geometry parameters:
#
# - Background permittivity :math:`\epsilon_0 = 5.25`
# - Modulation amplitude :math:`\Delta\epsilon = 1`
# - Modulation frequency :math:`\Omega = 1`
# - Slab thickness :math:`L = 2`
# - Floquet truncation with ``Npad = 6`` (13 harmonics total)

Omega = 1.0  # Modulation frequency
Npad = 6  # Floquet truncation padding
eps0 = 5.25  # Background permittivity
deps = 1.0  # Modulation amplitude
L = 2.0  # Slab thickness

# Fourier coefficients for sinusoidal modulation
eps_fourier = [
    -deps / (2 * 1j),  # ε₋₁ = -iΔε/2
    eps0,  # ε₀ (background)
    deps / (2 * 1j),  # ε₊₁ = +iΔε/2
]

# Create material and slab
mat = pm.Material(eps_fourier, Omega, Npad)
slab = pm.Slab(mat, L)

print(f"Material: eps0 = {eps0}, deps = {deps}, Omega = {Omega}")
print(f"Slab thickness: L = {L}")
print(f"Number of Floquet harmonics: {mat.nh}")


# %%
# Solving the nonlinear eigenvalue problem
# ----------------------------------------
# We search for QNMs in a rectangular region of the complex frequency plane.
# The search region is defined by two corner points:
#
# - Bottom-left: :math:`\omega_0 = 0.65 - 0.32i`
# - Top-right: :math:`\omega_1 = 0.92 - 0.019i`
#
# The contour integration method finds all poles of :math:`M(\omega)^{-1}`
# within this region.

# Define search region corners
omega0 = 0.65 - 0.32j
omega1 = 0.92 - 0.019j

print("\nSearching for QNMs in region:")
print(f"  Bottom-left:  ω = {omega0}")
print(f"  Top-right:    ω = {omega1}")

# Solve nonlinear eigenvalue problem
evs, modes, modes_left = slab.eigensolve(
    omega0,
    omega1,
    peak_ref=6,
    recursive=True,
    tol=1e-6,
    return_left=True,
)

evs = np.array(evs)
Nevs = len(evs)
print(f"\nNumber of QNMs found: {Nevs}")
if Nevs > 0:
    print("\nQNM frequencies (ω/Ω):")
    for i, ev in enumerate(evs):
        print(f"  Mode {i}: ω = {ev.real:.4f} - {abs(ev.imag):.4f}i")


# %%
# Visualizing the complex frequency plane
# ---------------------------------------
# We create a colormap of :math:`\log_{10}|\det M(\omega)|` over the search
# region. The QNMs appear as dark spots (poles) where the determinant
# approaches zero. We also plot:
#
# - Black crosses: Static slab resonances (unmodulated reference)
# - Red crosses: Found QNMs of the time-modulated slab

# Create frequency grid for visualization
nc = 101
omegasr = np.linspace(omega0.real, omega1.real, nc)
omegasi = np.linspace(omega0.imag, omega1.imag, nc)
re, im = np.meshgrid(omegasr, omegasi)
omegas = re + 1j * im

# Compute determinant on the grid
kns, ens = mat.eigensolve(omegas)
matrix_slab_c = slab.build_matrix(omegas, kns, ens)
matrix_slab_c = np.transpose(matrix_slab_c, (2, 3, 0, 1))
D = np.linalg.det(matrix_slab_c)

# Plot determinant map
plt.figure(figsize=(10, 8))
plt.pcolormesh(
    omegasr / Omega, omegasi / Omega, np.log10(np.abs(D)), cmap="BuPu", shading="auto"
)
plt.colorbar(label=r"$\log_{10}|\det M(\omega)|$")
plt.title(r"Characteristic function: $\det M(\omega)$", fontsize=14)

# Plot static slab resonances (black crosses)
for i in range(10):
    eigenvalue_static = slab.eigenvalue_static(i)
    plt.plot(
        eigenvalue_static.real / Omega,
        eigenvalue_static.imag / Omega,
        "k+",
        markersize=8,
        label="Static" if i == 0 else "",
    )

# Plot additional static resonances with Floquet shifts
Nh = mat.Nh
for i in range(-50, 50):
    eigenvalue_static = slab.eigenvalue_static(i)
    for n in range(-Nh, Nh + 1):
        plt.plot(
            eigenvalue_static.real / Omega - n,
            eigenvalue_static.imag / Omega,
            "sk",
            mfc="none",
            markersize=8,
            label="Static shifted" if (i == 0 and n == 0) else "",
        )

# Plot found QNMs (red crosses)
if Nevs != 0:
    plt.plot(
        evs.real / Omega, evs.imag / Omega, "xr", markersize=10, mew=2, label="QNM"
    )

plt.xlim(omegasr[0] / Omega, omegasr[-1] / Omega)
plt.ylim(omegasi[0] / Omega, omegasi[-1] / Omega)
plt.xlabel(r"Re $\omega/\Omega$", fontsize=12)
plt.ylabel(r"Im $\omega/\Omega$", fontsize=12)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()


# %%
# Compute QNM field distributions
# -------------------------------
# For each found QNM, we compute the spatial field distribution. The QNM
# fields are computed at t = 0 and normalized for visualization.

# Time and space grids
T = mat.modulation_period
t = np.linspace(0, 3 * T, 300)
Lhom = 6 * L
x = np.linspace(-Lhom, Lhom + L, 1000)

qnms = []
qnms_left = []

print("\nComputing QNM field distributions...")

for imode in range(Nevs):
    omega = evs[imode]
    kns, ens = mat.eigensolve(omega)
    Eis = slab.init_incident_field(omega)
    psi = slab.extract_coefficients(modes[:, imode], Eis, kns, ens)
    E = slab.get_scattered_field(x, t, omega, psi, kns, ens)
    qnms.append(E)

    # Compute left eigenmode (using conjugate for biorthogonality)
    psi_left = slab.extract_coefficients(
        modes_left[:, imode], Eis, kns.conj(), ens.conj()
    )
    Eleft = slab.get_scattered_field(x, t, omega, psi_left, kns, ens)
    qnms_left.append(Eleft)

print(f"Computed {Nevs} right QNMs and {Nevs} left QNMs")


# %%
# Plot right QNMs
# ---------------
# We visualize the spatial profiles of the right QNMs (outgoing solutions).
# Each mode is offset vertically for clarity.

plt.figure(figsize=(10, 6))
for imode in range(Nevs):
    mode = qnms[imode][:, 0].real
    mode /= np.max(np.abs(mode)) * 2  # Normalize for visualization
    plt.plot(
        x / L - 0.5,
        1 * imode + mode.real,
        linewidth=1.5,
        label=f"Mode {imode}: ω = {evs[imode].real:.3f} - {abs(evs[imode].imag):.3f}i",
    )

# Mark slab boundaries
plt.axvline(-0.5, color="#949494", linewidth=1, linestyle="-")
plt.axvline(0.5, color="#949494", linewidth=1, linestyle="-")
plt.xlabel(r"Normalized position $x/L$", fontsize=12)
plt.ylabel("Normalized field amplitude", fontsize=12)
plt.title("Right Quasi-Normal Modes", fontsize=14)
plt.xlim(x[0] / L - 0.5, x[-1] / L - 0.5)
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()


# %%
# Plot left QNMs
# --------------
# We also visualize the left QNMs, which satisfy the adjoint eigenvalue
# problem and are important for modal expansions and normalization.

plt.figure(figsize=(10, 6))
for imode in range(Nevs):
    mode = qnms_left[imode][:, 0].real
    mode /= np.max(np.abs(mode)) * 2  # Normalize for visualization
    plt.plot(
        x / L - 0.5,
        1 * imode + mode.real,
        linewidth=1.5,
        label=f"Mode {imode}: ω = {evs[imode].real:.3f} - {abs(evs[imode].imag):.3f}i",
    )

# Mark slab boundaries
plt.axvline(-0.5, color="#949494", linewidth=1, linestyle="-")
plt.axvline(0.5, color="#949494", linewidth=1, linestyle="-")
plt.xlabel(r"Normalized position $x/L$", fontsize=12)
plt.ylabel("Normalized field amplitude", fontsize=12)
plt.title("Left Quasi-Normal Modes", fontsize=14)
plt.xlim(x[0] / L - 0.5, x[-1] / L - 0.5)
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()


# %%
# Animate QNM time evolution
# --------------------------
# We create animations showing the time evolution of both right and left
# QNMs. The animations demonstrate the exponential growth in space but decay in time characteristic of
# quasi-normal modes.

# Animate right QNM (mode 4)
if Nevs > 4:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"Right QNM 4: ω = {evs[4].real:.3f} - {abs(evs[4].imag):.3f}i")
    anim = slab.animate_field(x, t, qnms[4], (fig, ax))
    plt.tight_layout()
    plt.show()

    # Animate left QNM (mode 4)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"Left QNM 4: ω = {evs[4].real:.3f} - {abs(evs[4].imag):.3f}i")
    anim_left = slab.animate_field(x, t, qnms_left[4], (fig, ax))
    plt.tight_layout()
    plt.show()


# %%
# Summary
# -------
# This example demonstrated:
#
# 1. **Computing QNMs** using the nonlinear eigenvalue solver that finds
#    poles of the scattering matrix in the complex frequency plane.
#
# 2. **Complex frequency plane analysis** showing how QNMs appear as poles
#    of the characteristic function det M(ω).
#
# 3. **Static vs. modulated comparison** - the black crosses show where
#    unmodulated slab resonances would be, while red crosses show how
#    time-modification shifts these resonances.
#
# 4. **Right and left QNMs** - the biorthogonal eigenmodes required for
#    complete modal expansions in non-Hermitian systems.
#
# 5. **Field visualization** - spatial profiles and time evolution of the
#    quasi-normal modes showing their localized nature within the slab
#    and outgoing radiation in the surrounding medium.
#
# The Floquet-shifted static resonances (additional black squares) illustrate
# how time modulation creates replicas of the original resonances at frequencies
# shifted by multiples of the modulation frequency.
