# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


"""
Electric Field in a Time-Modulated Slab
=======================================

This example visualizes the electric field distribution in and around a
time-modulated slab when illuminated by a plane wave. We compute the
spatiotemporal field pattern and analyze the energy balance between
reflected and transmitted sidebands.
"""

# %%
# Imports and setup
# -----------------

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pytmod as pm

# %%
# Background
# ----------
# When a plane wave interacts with a time-modulated slab, the scattered field
# contains not only the incident frequency but also sidebands at frequencies
# shifted by integer multiples of the modulation frequency (Floquet harmonics).
#
# In this example, we:
#
# 1. Define a time-modulated slab with sinusoidal permittivity variation
# 2. Solve for the Floquet eigenmodes of the material
# 3. Apply boundary conditions to find reflected and transmitted fields
# 4. Compute the total field in space and time
# 5. Visualize the field as both an animation and a space-time map
# 6. Calculate the energy balance across all sidebands


# %%
# Problem parameters
# ------------------
# We define the material and geometry parameters:
#
# - Background permittivity :math:`\epsilon_0 = 5.25`
# - Modulation amplitude :math:`\Delta\epsilon = 2`
# - Modulation frequency :math:`\Omega = 1`
# - Slab thickness :math:`L = 5`
# - Incident frequency :math:`\omega = 1.2\Omega`
# - Floquet truncation with ``Npad = 7`` (15 harmonics total)

# Material parameters
eps0 = 5.25  # Background permittivity
deps = 2.0  # Modulation amplitude
Omega = 1.0  # Modulation frequency
Npad = 7  # Floquet truncation padding

# Geometry
L = 5.0  # Slab thickness
Ei0 = 1.0  # Incident field amplitude

# Incident wave frequency
omega = 1.2 * Omega

# Fourier coefficients for sinusoidal modulation
eps_fourier = [
    -deps / (2 * 1j),  # ε₋₁ = -iΔε/2
    eps0,  # ε₀ (background)
    deps / (2 * 1j),  # ε₊₁ = +iΔε/2
]

print(f"Incident frequency: ω = {omega:.2f} (ω/Ω = {omega / Omega:.2f})")
print(f"Slab thickness: L = {L}")
print(f"Modulation amplitude: Δε = {deps}")


# %%
# Solve material eigenproblem
# ---------------------------
# First, we create the time-modulated material and compute its Floquet
# eigenmodes at the incident frequency. These eigenmodes describe the
# propagating waves inside the slab.

mat = pm.Material(eps_fourier, Omega, Npad)
kns, ens = mat.eigensolve(omega)
slab = pm.Slab(mat, L)

print(f"\nMaterial has {mat.nh} Floquet harmonics (Nh = {mat.Nh})")
print(f"Eigenvalues (wavenumbers): {kns}")


# %%
# Build and solve the scattering problem
# --------------------------------------
# We construct the boundary condition matrix for the slab and solve for
# the field coefficients. The system couples:
#
# - Forward and backward propagating waves inside the slab
# - Reflected waves in the incident medium (x < 0)
# - Transmitted waves in the output medium (x > L)

# Build the boundary condition matrix
matrix_slab = slab.build_matrix(omega, kns, ens)

# Initialize incident field (only central harmonic has nonzero amplitude)
Eis = slab.init_incident_field(omega)
Ninc = mat.Nh
Eis[Ninc] = Ei0

# Build right-hand side (boundary conditions)
rhs_slab = slab.build_rhs(omega, Eis)

# Solve the linear system
solution = slab.solve(matrix_slab, rhs_slab)

# Extract field coefficients
Eslab_plus, Eslab_minus, Er, Et = slab.extract_coefficients(solution, Eis, kns, ens)

print("\nScattering problem solved successfully!")


# %%
# Reflection and transmission coefficients
# ----------------------------------------
# We compute the reflection and transmission coefficients for each Floquet
# sideband. The coefficients are normalized by the incident field amplitude.

# Normalized coefficients
rn = Er / Ei0
tn = Et / Ei0

# Display coefficients in a table
pd.set_option("display.float_format", lambda x: f"{x:.4e}")
pd.set_option("display.max_rows", None)

# Power coefficients (|r|² and |t|²)
Rn = np.abs(rn) ** 2
Tn = np.abs(tn) ** 2

coeffs_n = pd.DataFrame(
    data={"r_n": rn, "t_n": tn, "R_n": Rn, "T_n": Tn}, index=range(-mat.Nh, mat.Nh + 1)
)
coeffs_n.index.name = "n"

print("\nReflection and transmission coefficients by sideband:")
print("=" * 60)
coeffs_n  # noqa: B018


# %%
# Energy balance
# --------------
# We verify energy conservation by summing the power in all reflected and
# transmitted sidebands. For a lossless system, R + T = 1.

R = np.sum(Rn)  # Total reflected power
T = np.sum(Tn)  # Total transmitted power

coeffs_sum = pd.DataFrame(
    data={"R": R, "T": T, "Balance (R + T)": R + T}, index=["Total"]
)

print("\nEnergy balance:")
print("=" * 60)
coeffs_sum  # noqa: B018


# %%
# Compute the field in space and time
# -----------------------------------
# We now reconstruct the total electric field E(x, t) on a spatial grid
# covering the incident region, the slab, and the transmitted region.
# The time domain covers several modulation periods.

# Time grid (3 modulation periods)
T_period = mat.modulation_period
t = np.linspace(0, 3 * T_period, 300)

# Space grid (3L on each side of the slab)
Lhom = 3 * L
x = np.linspace(-Lhom, Lhom + L, 1000)

# Field coefficients tuple
psi = (Eslab_plus, Eslab_minus, Er, Et)

# Compute scattered field
Es = slab.get_scattered_field(x, t, omega, psi, kns, ens)

# Compute incident field
Einc = slab.get_incident_field(x, t, omega, Eis)

# Total field
E = Einc + Es

print(f"\nField computed on grid: {len(x)} spatial points x {len(t)} time points")


# %%
# Animate the field
# -----------------
# We create an animation showing the time evolution of the electric field.
# The animation shows:
#
# - The incident wave approaching from the left
# - The reflected wave traveling backward
# - The field inside the time-modulated slab (shaded region)
# - The transmitted wave exiting to the right
# - The time-varying permittivity of the slab (color intensity)

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title(
    rf"Electric field evolution: $\omega = {omega / Omega:.1f}\,\Omega$", fontsize=14
)
anim = slab.animate_field(x, t, E, (fig, ax))

# Uncomment to save animation as GIF:
# writer = animation.PillowWriter(fps=15, metadata=dict(artist='pytmod'), bitrate=1800)
# anim.save('field.gif', writer=writer)

plt.tight_layout()
plt.show()


# %%
# Space-time map
# --------------
# As an alternative visualization, we create a space-time map (x-t diagram)
# showing the field amplitude across all positions and times. This reveals
# the interference patterns and sideband generation more clearly.

plt.figure(figsize=(10, 6))
plt.pcolormesh(
    x / L - 0.5,  # Normalized position (slab centered at 0)
    t / T_period,  # Normalized time
    np.real(E.T),  # Real part of field
    cmap="RdBu_r",  # Diverging colormap
    shading="auto",
)

# Mark slab boundaries
plt.axvline(-0.5, color="#949494", linewidth=2, linestyle="-", label="Slab boundaries")
plt.axvline(0.5, color="#949494", linewidth=2, linestyle="-")

plt.ylim(0, t[-1] / T_period)
plt.xlabel(r"Normalized position $x/L$", fontsize=12)
plt.ylabel(r"Normalized time $t/T$", fontsize=12)
plt.title(
    rf"Space-time field map: $\omega = {omega / Omega:.1f}\,\Omega$, $\Delta\epsilon = {deps}$",
    fontsize=14,
)

cb = plt.colorbar()
cb.ax.set_title("Re $E$")

plt.legend(loc="upper right")
plt.tight_layout()
plt.show()


# %%
# Summary
# -------
# This example demonstrated:
#
# 1. **Computing the total field** in a time-modulated slab including
#    incident, reflected, and transmitted contributions.
#
# 2. **Sideband analysis** showing the distribution of power across
#    Floquet harmonics (n = 0, ±1, ±2, ...).
#
# 3. **Energy conservation** verification by summing power in all sidebands.
#
# 4. **Visualization techniques** including time animation and space-time
#    maps that reveal the complex spatiotemporal field structure.
#
# The field visualization clearly shows how the time modulation creates
# additional frequency components that propagate at different phase velocities,
# leading to the characteristic beating patterns observed in the space-time map.
