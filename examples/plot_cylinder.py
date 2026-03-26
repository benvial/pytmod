# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


"""
Scattering from a Time-Modulated Cylinder
=========================================

This example demonstrates the scattering of electromagnetic waves from a
cylinder with time-modulated permittivity. We compute the total and
scattered fields in 2D, showing the field distribution around the cylinder.
"""

# %%
# Imports and setup
# -----------------

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pytmod as pm
from pytmod.check_boundary import run_checks

plt.close("all")
plt.ion()


# %%
# Background
# ----------
# This example solves the 2D scattering problem from a cylindrical scatterer
# with time-modulated material properties. The approach uses:
#
# 1. Floquet theory to handle the time modulation
# 2. Cylindrical wave expansion (Mie-type theory) for the spatial dependence
# 3. Boundary condition matching at the cylinder surface
#
# The cylinder has radius R and its permittivity varies sinusoidally in time.
# An incident plane wave excites the structure, and we compute the resulting
# field distribution.


# %%
# Problem parameters
# ------------------
# We define the material and geometry parameters:
#
# - Background permittivity :math:`\epsilon_0 = 3`
# - Modulation amplitude :math:`\Delta\epsilon = 0.3`
# - Modulation frequency :math:`\Omega = 0.6`
# - Cylinder radius :math:`R = \lambda` (equal to modulation wavelength)
# - Floquet truncation with ``Npad = 0`` (3 harmonics total)

# Material parameters
eps0 = 3.0  # Background permittivity
deps = 2  # Modulation amplitude
Omega = 0.6  # Modulation frequency
Npad = 0  # Floquet truncation padding (3 harmonics: -1, 0, +1)

# Geometry
Lbda = 2 * np.pi / Omega  # Modulation wavelength
R = Lbda  # Cylinder radius

# Frequency range (single frequency for this example)
Nomega = 1
omegas = np.linspace(0.8, 2.1, Nomega) * Omega

# Polarization: "TE" or "TM"
polarization = "TM"

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
# We create the material and cylinder objects, then compute the material
# eigenmodes at the incident frequencies.

eps_fourier = get_eps_fourier(deps)
mat = pm.Material(eps_fourier, Omega, Npad)
eigenvalues, modes = mat.eigensolve(omegas)

cyl = pm.Cylinder(mat, R)

print(f"\nMaterial has {mat.nh} Floquet harmonics")
print(f"Eigenvalues (k²): {eigenvalues}")


# %%
# Solve the scattering problem
# ----------------------------
# We solve the scattering problem by iterating over azimuthal mode numbers
# :math:`n_{az}`. For each mode, we:
#
# 1. Initialize the incident plane wave
# 2. Build the boundary condition matrix
# 3. Solve for the scattering coefficients
# 4. Extract the internal field coefficients

n_az_max = 9
n_range = range(-n_az_max, n_az_max + 1)

solution_all = []
rhs_all = []
inner_coeffs_all = []

print(f"\nSolving for azimuthal modes: {list(n_range)}")

for n_az in n_range:
    # Initialize incident field (plane wave)
    incident_field, incident_angles = cyl.init_incident_field(omegas)
    incident_field[mat.Nh] = 1
    incident_angles[mat.Nh] = 0

    # Build and solve the scattering matrix
    matrix = cyl.build_matrix(omegas, eigenvalues, modes, n_az, polarization)
    rhs = cyl.build_rhs(omegas, incident_field, incident_angles, n_az)
    solution = cyl.solve(matrix, rhs)

    # Get internal field coefficients
    inner_coeffs = cyl.get_inner_coefficients(
        omegas, eigenvalues, modes, n_az, polarization, rhs, solution, check=True
    )

    inner_coeffs_all.append(inner_coeffs)
    solution_all.append(solution)
    rhs_all.append(rhs)

# Convert lists to arrays
solution_all = np.array(solution_all)
inner_coeffs_all = np.array(inner_coeffs_all)
rhs_all = np.array(rhs_all)

print(f"Scattering problem solved for {len(n_range)} azimuthal modes")


# %%
# Compute field distributions
# ---------------------------
# We compute the field on a 2D spatial grid and for multiple time snapshots.
# The field includes contributions from all Floquet harmonics and azimuthal modes.

# Spatial grid
Nx = 101
Ny = 101
M = 3.5
x = np.linspace(-M, M, Nx) * Lbda
y = np.linspace(-M, M, Ny) * Lbda

# Time grid
T = 2 * np.pi / Omega
t = np.linspace(0, 10, 300) * T

# Compute fields
fields = cyl.get_fields(
    x,
    y,
    omegas,
    solution_all,
    inner_coeffs_all,
    eigenvalues,
    modes,
    incident_field,
    incident_angles,
    n_range,
    t,
    polarization,
)

field_name = "E_z" if polarization == "TM" else "H_z"

print(f"\nField computed on grid: {Nx} x {Ny} spatial points x {len(t)} time points")


# %%
# Plot field distributions
# ------------------------
# We visualize the field at a specific time snapshot, showing both the
# real part (phase information) and magnitude.

field_type = "total"  # Options: "total", "scattered", "incident"
it = 0  # Time index
iomega = 0  # Frequency index

# Plot real part
plt.figure(figsize=(10, 8))
plt.pcolormesh(
    x / R,
    y / R,
    np.real(fields[field_type])[:, :, iomega, it],
    cmap="RdBu_r",
    shading="auto",
)
plt.xlabel(r"$x/R$", fontsize=12)
plt.ylabel(r"$y/R$", fontsize=12)
plt.colorbar(label=f"Re ${field_name}$")
cyl.plot(normalize=True)
plt.title(
    rf"{polarization}, {field_type} field Re$\,{field_name}$ at $t={t[it] / T:.2f}T$"
)
plt.axis("scaled")
plt.tight_layout()
plt.show()

# Plot magnitude
plt.figure(figsize=(10, 8))
plt.pcolormesh(
    x / R,
    y / R,
    np.abs(fields[field_type])[:, :, iomega, it],
    cmap="magma",
    shading="auto",
)
plt.xlabel(r"$x/R$", fontsize=12)
plt.ylabel(r"$y/R$", fontsize=12)
plt.colorbar(label=f"$|{field_name}|$")
cyl.plot(normalize=True, color="w")
plt.title(
    rf"{polarization}, {field_type} field $|{field_name}|$ at $t={t[it] / T:.2f}T$"
)
plt.axis("scaled")
plt.tight_layout()
plt.show()


# %%
# Field cross-sections
# --------------------
# We extract field cuts along the x and y axes to visualize the field
# variation through the cylinder center.

# x-cut (horizontal line through center)
Nxc = 1501
xc = np.linspace(-M, M, Nxc) * Lbda
y0 = 0

fields_cut_x = cyl.get_fields(
    xc,
    y0,
    omegas,
    solution_all,
    inner_coeffs_all,
    eigenvalues,
    modes,
    incident_field,
    incident_angles,
    n_range,
    t,
    polarization,
)

# y-cut (vertical line through center)
Nyc = 1501
x0 = 0
yc = np.linspace(-M, M, Nyc) * Lbda

fields_cut_y = cyl.get_fields(
    x0,
    yc,
    omegas,
    solution_all,
    inner_coeffs_all,
    eigenvalues,
    modes,
    incident_field,
    incident_angles,
    n_range,
    t,
    polarization,
)

# Plot cross-sections
plt.figure(figsize=(10, 6))
plt.plot(
    xc / R,
    np.real(fields_cut_x[field_type])[0, :, iomega, it],
    label="Re x-cut",
    color="#d55151",
    linestyle="-",
)
plt.plot(
    yc / R,
    np.real(fields_cut_y[field_type])[:, 0, iomega, it],
    label="Re y-cut",
    color="#3cabdf",
    linestyle="-",
)
plt.plot(
    xc / R,
    np.imag(fields_cut_x[field_type])[0, :, iomega, it],
    label="Im x-cut",
    color="#d55151",
    linestyle="--",
)
plt.plot(
    yc / R,
    np.imag(fields_cut_y[field_type])[:, 0, iomega, it],
    label="Im y-cut",
    color="#3cabdf",
    linestyle="--",
)
# Mark cylinder boundaries
plt.axvline(1, color="#949494", linewidth=1, linestyle=":")
plt.axvline(-1, color="#949494", linewidth=1, linestyle=":")
plt.xlabel(r"$\nu/R$", fontsize=12)
plt.ylabel(f"${field_name}$", fontsize=12)
plt.legend()
plt.title(rf"{polarization}, {field_type} field cross-sections at $t={t[it] / T:.2f}T$")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# Animate field evolution
# -----------------------
# We create an animation showing the time evolution of the field. The
# animation demonstrates how the field pattern changes as the material
# permittivity varies in time.

anim = cyl.animate_field(
    x,
    y,
    t,
    fields[field_type],
    field_type=field_type,
    iomega=iomega,
    normalize=True,
    cmap="RdBu_r",
    interval=1,
)

plt.show()


# %%
# Boundary condition verification
# -------------------------------
# We verify that the computed fields satisfy the boundary conditions at the
# cylinder surface by running diagnostic checks.


print("\nRunning boundary condition checks...")
run_checks(
    cyl,
    omegas,
    eigenvalues,
    modes,
    rhs_all,
    solution_all,
    inner_coeffs_all,
    n_range=n_range,
    polarization=polarization,
)


# %%
# Summary
# -------
# This example demonstrated:
#
# 1. **2D scattering from a time-modulated cylinder** using Floquet-Mie theory.
#
# 2. **Azimuthal mode expansion** - the solution is built by summing over
#    angular momentum modes (n_az from -9 to +9).
#
# 3. **Field visualization** - both 2D spatial distributions and 1D cross-sections
#    showing the field behavior inside and outside the cylinder.
#
# 4. **Time evolution** - animation showing how the field pattern evolves as
#    the material permittivity varies periodically.
#
# 5. **Boundary condition verification** - ensuring the solution satisfies
#    electromagnetic continuity at the cylinder surface.
#
# The example shows how time modulation affects the scattering characteristics
# compared to a static cylinder, with sideband generation and modified field
# patterns.
