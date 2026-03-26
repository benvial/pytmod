# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


"""
Quasi-Normal Modes of a Time-Modulated Cylinder
===============================================

This example demonstrates the computation of quasi-normal modes (QNMs) for a
cylindrical scatterer with time-modulated permittivity. QNMs are the natural
resonances of the open system, characterized by complex frequencies.
"""

# %%
# Imports and setup
# -----------------

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pytmod as pm

plt.ion()
plt.close("all")


# %%
# Background
# ----------
# Quasi-normal modes (QNMs) are the eigenmodes of an open optical system with
# outgoing boundary conditions. For a time-modulated cylinder, the QNM condition
# is given by:
#
# .. math::
#     \det M_n(\omega) = 0
#
# where :math:`M_n(\omega)` is the boundary condition matrix for azimuthal
# mode number :math:`n`. Due to the cylindrical symmetry, modes with different
# :math:`n` decouple, allowing us to solve for each azimuthal order separately.
#
# The QNMs have complex frequencies :math:`\omega_n = \omega_n' - i\gamma_n`
# where the imaginary part represents the decay rate due to radiation.


# %%
# Problem parameters
# ------------------
# We define the material and geometry parameters:
#
# - Background permittivity :math:`\epsilon_0 = 3`
# - Modulation amplitude :math:`\Delta\epsilon = 0.3`
# - Modulation frequency :math:`\Omega = 0.6`
# - Cylinder radius :math:`R = \lambda`
# - Floquet truncation with ``Npad = 0`` (3 harmonics total)
# - Polarization: TM

# Material parameters
eps0 = 3.0  # Background permittivity
deps = 0.3  # Modulation amplitude
Omega = 0.6  # Modulation frequency
Npad = 0  # Floquet truncation padding

# Geometry
Lbda = 2 * np.pi / Omega  # Modulation wavelength
R = Lbda  # Cylinder radius

# Polarization
polarization = "TM"
plot_type = "polar"

print(f"Material: eps0 = {eps0}, deps = {deps}, Omega = {Omega}")
print(f"Cylinder radius: R = {R:.4f} (R/λ = {R / Lbda:.2f})")
print(f"Polarization: {polarization}")


def get_eps_fourier(deps):
    """
    Generate Fourier coefficients for sinusoidal modulation.

    Parameters
    ----------
    deps : float
        Modulation amplitude Δε

    Returns
    -------
    list
        Fourier coefficients [ε₋₁, ε₀, ε₊₁]
    """
    return [
        -deps / (2 * 1j),  # ε₋₁ = -iΔε/2
        eps0,  # ε₀ (background)
        deps / (2 * 1j),  # ε₊₁ = +iΔε/2
    ]


# %%
# Create the time-modulated cylinder
# ----------------------------------
# We create the material and cylinder objects.

eps_fourier = get_eps_fourier(deps)
mat = pm.Material(eps_fourier, Omega, Npad)
cyl = pm.Cylinder(mat, R)

print(f"\nMaterial has {mat.nh} Floquet harmonics")


# %%
# Define search regions
# ---------------------
# We define the search regions for QNMs in the complex frequency plane.
# Different regions may contain different numbers of modes.

# Modulated cylinder search region
omega0 = 0.01 - 0.15j
omega1 = 0.101 - 0.00001j

# Static cylinder reference region
omega0_stat = -0.191 - 0.15j
omega1_stat = 2.19 - 0.0001j

print("\nSearch region for modulated cylinder:")
print(f"  Bottom-left:  ω = {omega0}")
print(f"  Top-right:    ω = {omega1}")


# %%
# Compute static cylinder resonances
# ----------------------------------
# As a reference, we compute the QNMs of the unmodulated (static) cylinder
# using the analytical formula.

evs0 = [
    cyl.eigenvalue_static(omega0_stat, omega1_stat, n, polarization) for n in range(3)
]

eigenvalue_static = np.sort(np.hstack(evs0))

print("\nStatic cylinder resonance frequencies (ω/Ω):")
for i, ev in enumerate(eigenvalue_static):
    print(f"  Mode {i}: ω = {ev.real:.4f} - {abs(ev.imag):.4f}i")


# %%
# Solve for QNMs of the modulated cylinder
# ----------------------------------------
# We solve the nonlinear eigenvalue problem for each azimuthal mode number
# :math:`n`. The contour integration method finds all poles within the
# specified search region.


def eigensolve(n):
    """Solve the nonlinear eigenvalue problem for azimuthal mode n."""
    evs, modes, modes_left = cyl.eigensolve(
        omega0,
        omega1,
        peak_ref=6,
        recursive=True,
        tol=1e-6,
        plot_solver=False,
        n=n,
        polarization=polarization,
        return_left=True,
    )
    evs = np.array(evs)
    modes = np.array(modes)
    modes_left = np.array(modes_left)
    return evs, modes, modes_left


# Solve for azimuthal modes n = 0, 1, 2
n_range = range(3)
eigenpairs = [eigensolve(n) for n in n_range]
evs = np.sort(np.hstack([e[0] for e in eigenpairs]))

nmodes = len(evs)
print(f"\nNumber of QNMs found for modulated cylinder: {nmodes}")
print("\nModulated cylinder QNM frequencies (ω/Ω):")
for i, ev in enumerate(evs):
    print(f"  Mode {i}: ω = {ev.real:.4f} - {abs(ev.imag):.4f}i")


# %%
# Verify matrix derivative
# ------------------------
# We verify the analytical derivative of the boundary condition matrix by
# comparing with finite differences.

print("\nVerifying matrix derivative (dM/dω):")
print("=" * 50)

for omega in evs[:3]:  # Check first 3 modes
    print(f"\nMode at ω = {omega:.4f}")

    # Finite difference step
    dw = 1e-7

    # Compute eigenmodes at ω ± dω
    ev_p, mo_p = mat.eigensolve(omega + dw)
    ev_m, mo_m = mat.eigensolve(omega - dw)

    # Build matrices and compute finite difference
    n = 0
    M_plus = cyl.build_matrix(omega + dw, ev_p, mo_p, n=n, polarization=polarization)
    M_minus = cyl.build_matrix(omega - dw, ev_m, mo_m, n=n, polarization=polarization)
    dM_fd = (M_plus - M_minus) / (2 * dw)

    # Compute analytical derivative
    ev, mo, mo_l = mat.eigensolve(omega, left=True, normalize=True)
    dM_analytic = cyl.build_dmatrix_domega(
        omega, ev, mo, mo_l, n=n, polarization=polarization
    )

    # Compute relative error
    err = dM_analytic - dM_fd
    rel_err = 100 * (np.abs(err / np.linalg.norm(dM_fd))).max()
    print(f"  max |dM_analytic - dM_fd|/||dM_fd||: {rel_err:.12f}%")


# %%
# Visualize the complex frequency plane
# -------------------------------------
# We create a colormap of :math:`\log_{10}|\det M(\omega)|` over the search
# region. QNMs appear as dark spots (poles) where the determinant approaches
# zero.

# Create frequency grid for visualization
nc = 101
omegasr = np.linspace(omega0.real, omega1.real, nc)
omegasi = np.linspace(omega0.imag, omega1.imag, nc)
re, im = np.meshgrid(omegasr, omegasi)
omegas = re + 1j * im

# Compute determinant on the grid
kns, ens = mat.eigensolve(omegas)
D = 1
for n in n_range:
    matrix_c = cyl.build_matrix(
        omegas, kns, ens, n=n, polarization=polarization, alt=True
    )
    matrix_c = np.transpose(matrix_c, (2, 3, 0, 1))
    D *= np.linalg.det(matrix_c)

# Plot determinant map
plt.figure(figsize=(10, 8))
plt.pcolormesh(
    omegasr / Omega, omegasi / Omega, np.log10(np.abs(D)), cmap="BuPu", shading="auto"
)
plt.colorbar(label=r"$\log_{10}|\det M(\omega)|$")
plt.title(r"Characteristic function: $\det M(\omega)$", fontsize=14)

# Plot static resonances with Floquet shifts (blue circles)
for q in range(-mat.Nh, mat.Nh + 1):
    label = "Static shifted" if q == 0 else None
    evs_shift = eigenvalue_static + q * Omega
    plt.plot(
        evs_shift.real / Omega,
        evs_shift.imag / Omega,
        "ob",
        markerfacecolor="none",
        markersize=6,
        label=label,
    )

# Plot modulated QNMs (red crosses)
plt.plot(
    evs.real / Omega,
    evs.imag / Omega,
    "xr",
    markersize=10,
    mew=2,
    label="Modulated QNMs",
)

plt.xlim(omegasr[0] / Omega, omegasr[-1] / Omega)
plt.ylim(omegasi[0] / Omega, omegasi[-1] / Omega)
plt.xlabel(r"Re $\omega/\Omega$", fontsize=12)
plt.ylabel(r"Im $\omega/\Omega$", fontsize=12)
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# Visualize QNM field distributions
# ---------------------------------
# We compute and visualize the spatial field distributions for each QNM.
# Both right (outgoing) and left (adjoint) eigenmodes are shown.

# Spatial grid
Nx = 101
Ny = 101
M = 1.5
x = np.linspace(-M, M, Nx) * Lbda
y = np.linspace(-M, M, Ny) * Lbda

# Azimuthal range for field reconstruction
n_az_max = 7
n_range_plt = range(-n_az_max, n_az_max + 1)

# Time grid (just 2 time points for visualization)
T = 2 * np.pi / Omega
t = np.linspace(0, 10, 2) * T

print("\nComputing QNM field distributions...")

for n in n_range:
    omega_ns, phi_ns, phi_ns_left = eigenpairs[n]
    if len(omega_ns) == 0:
        continue

    print(f"\nAzimuthal mode n = {n}: {len(omega_ns)} QNMs found")

    for imode in range(len(omega_ns)):
        omega_n = omega_ns[imode]
        phi_n = phi_ns[:, imode]
        phi_n_left = phi_ns_left[:, imode]

        # Reshape for single mode
        omega_n = np.array([omega_n])
        phi_n = np.array([phi_n]).T
        phi_n_left = np.array([phi_n_left]).T

        # Get material eigenmodes
        eigenvalues, modes, modes_left = mat.eigensolve(
            omega_n, left=True, normalize=True
        )

        # Normalize QNM
        matrix_derivative = cyl.build_dmatrix_domega(
            omega_n, eigenvalues, modes, modes_left, n=n, polarization=polarization
        )
        phi_n, phi_n_left = cyl.normalize(phi_n, phi_n_left, matrix_derivative)

        # Compute right mode field
        fields = cyl.get_mode(
            x, y, omega_n, phi_n, eigenvalues, modes, n, polarization, t, n_range_plt
        )

        # Compute left mode field
        eigenvalues, modes = mat.eigensolve(omega_n.conj())
        fields_left = cyl.get_mode(
            x,
            y,
            omega_n.conj(),
            phi_n_left,
            eigenvalues,
            modes,
            n,
            polarization,
            t,
            n_range_plt,
        )

        # Plot fields
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        ax = ax.T
        it = 0

        cyl.plot_mode(
            x,
            y,
            fields_left,
            ax=ax[0],
            time_index=it,
            plot_type=plot_type,
            polarization=polarization,
        )

        cyl.plot_mode(
            x,
            y,
            fields,
            ax=ax[1],
            time_index=it,
            plot_type=plot_type,
            polarization=polarization,
        )

        field_name = "E_z" if polarization == "TM" else "H_z"
        plt.suptitle(
            f"${field_name}$ {polarization} mode {imode}, $n={n}$\n"
            f"$\omega/\Omega = {omega_n[0] / Omega:.2f}$, $t = {t[it] / T:.2f}T$",
            fontsize=12,
        )
        fig.text(0.5, 0.02, "Left QNM    Right QNM", ha="center", fontsize=10)
        plt.tight_layout()
        plt.pause(0.1)


# %%
# Summary
# -------
# This example demonstrated:
#
# 1. **Computing QNMs for a time-modulated cylinder** using the nonlinear
#    eigenvalue solver for each azimuthal mode number :math:`n`.
#
# 2. **Static vs. modulated comparison** - blue circles show unmodulated
#    cylinder resonances (with Floquet shifts), while red crosses show how
#    time modulation modifies these resonances.
#
# 3. **Matrix derivative verification** - comparing analytical and finite
#    difference derivatives to validate the implementation.
#
# 4. **Right and left QNMs** - visualizing both the outgoing (right) and
#    adjoint (left) eigenmodes, which form a biorthogonal set.
#
# 5. **Field patterns** - spatial distributions showing the localized nature
#    of QNMs within the cylinder and their outgoing radiation pattern.
#
# The azimuthal symmetry allows each mode number :math:`n` to be solved
# independently, significantly simplifying the problem compared to arbitrary
# geometries.
