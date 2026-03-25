# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


"""
Quasi-Normal Modes of a Time-Modulated Slab (FEM)
=================================================

This example demonstrates the computation of quasi-normal modes (QNMs) for a
one-dimensional slab with time-periodic permittivity using the finite element
method (FEM). The solver uses FEniCSx/dolfinx for spatial discretization and
SLEPc for solving the resulting quadratic eigenvalue problem.

The example compares FEM-computed eigenvalues with:
1. Analytical solutions for the static (unmodulated) slab
2. Results from the semi-analytical NLEVP solver (if available)
"""

# %%
# Imports and setup
# -----------------

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Import the FEM slab solver
try:
    from pytmod.fem.slab_solver import FEMSlabSolver

    HAS_FEM = True
except ImportError:
    HAS_FEM = False
    print(
        "Warning: FEM dependencies not available. "
        "Install dolfinx, gmsh, slepc4py to run this example."
    )

import pytmod as pm

plt.ion()
plt.close("all")


# %%
# Background
# ----------
# For a time-modulated slab with permittivity:
#
# .. math::
#     \\epsilon(t) = \\epsilon_0 + \\Delta\\epsilon \\sin(\\Omega t)
#
# The Floquet formalism leads to a quadratic eigenvalue problem in the
# complex frequency :math:`\\omega`:
#
# .. math::
#     (A_0 + \\omega A_1 + \\omega^2 A_2) \\mathbf{u} = 0
#
# where the matrices :math:`A_0, A_1, A_2` come from the discretized weak
# form of Maxwell's equations. The eigenvalues :math:`\\omega_n` are the
# quasi-normal mode frequencies, characterized by:
#
# - Real part: resonant frequency
# - Imaginary part: decay rate (negative for passive systems)
#
# The FEM approach discretizes the spatial domain and uses Perfectly Matched
# Layers (PMLs) to simulate outgoing boundary conditions.


# %%
# Problem parameters
# ------------------
# We define the material and geometry parameters:
#
# - Background permittivity :math:`\\epsilon_0 = 50`
# - Modulation amplitude :math:`\\Delta\\epsilon = 20`
# - Modulation frequency :math:`\\Omega = 1`
# - Slab thickness :math:`L = 1.5`
# - Floquet harmonics: [-1, 0, 1] (3 harmonics total)

# Material parameters
eps0 = 50.0  # Background permittivity
deps = 20.0  # Modulation amplitude
Omega = 1.0  # Modulation frequency

# Geometry
Lslab = 1.5  # Slab thickness

# Floquet harmonics to include
harmonics = [-1, 0, 1]

print(f"Material: eps0 = {eps0}, deps = {deps}, Omega = {Omega}")
print(f"Slab thickness: L = {Lslab}")
print(f"Floquet harmonics: {harmonics}")


# %%
# Create the FEM solver
# ---------------------
# We instantiate the FEMSlabSolver with the problem parameters.
# The solver will:
#
# 1. Generate a 1D mesh with gmsh (including PMLs)
# 2. Set up the variational forms for the quadratic EVP
# 3. Assemble the PETSc matrices

if HAS_FEM:
    solver = FEMSlabSolver(
        eps0=eps0,
        deps=deps,
        Omega=Omega,
        Lslab=Lslab,
        harmonics=harmonics,
        Lpml=2 * Lslab,  # PML thickness
        lc_min=5e-3,  # Mesh resolution
        order=2,  # FEM order
    )

    print("\nFEM solver created:")
    print(f"  - PML stretching: s = {solver.spml}")
    print(f"  - Epsilon coefficients: {solver.epsilon_coeffs}")

    # Run setup (mesh generation, material assignment, form compilation)
    print("\nSetting up problem...")
    solver.setup()
    print("Setup complete!")


# %%
# Solve the eigenvalue problem
# ----------------------------
# We solve the quadratic eigenvalue problem using SLEPc.
# The target parameter helps the solver find eigenvalues near a specific
# value (here, 0.1).

if HAS_FEM:
    print("\nSolving eigenvalue problem...")
    neig = 50
    target = 0.1
    eigenvalues, eigenvectors = solver.solve(
        neig=neig,
        target=target,
        tol=1e-6,
        quad=True,  # Quadratic eigenvalue problem
    )

    print(f"Number of converged eigenvalues: {len(eigenvalues)}")

    # Sort eigenvalues by real part
    srt = np.argsort(eigenvalues.real)
    evs = eigenvalues[srt]
    eigenvectors = eigenvectors[:, srt]

    # Filter to physical modes (positive real part, small imaginary part)
    mask = (evs.real > 0.1) & (evs.real < 0.8) & (evs.imag > -0.1)
    evs_filtered = evs[mask]

    print("Filtered eigenvalues in range [0.1, 0.8]:")
    for i, ev in enumerate(evs_filtered[:10]):
        print(f"  Mode {i}: ω = {ev.real:.6f} {ev.imag:+.6f}i")


# %%
# Analytical reference: static slab
# ---------------------------------
# For validation, we compute the analytical eigenvalues of the static
# (unmodulated) slab and add Floquet shifts.

if HAS_FEM:
    evs_ana = solver.get_analytical_static_eigenvalues(n_modes=10)

    # Filter to relevant range
    mask_ana = (evs_ana.real > 0.1) & (evs_ana.real < 0.8)
    evs_ana_filtered = evs_ana[mask_ana]

    print("\nAnalytical static eigenvalues (with Floquet shifts):")
    for i, ev in enumerate(evs_ana_filtered[:10]):
        print(f"  Mode {i}: ω = {ev.real:.6f} {ev.imag:+.6f}i")


# %%
# Comparison with semi-analytical NLEVP solver
# --------------------------------------------
# We also compute the eigenvalues using the semi-analytical NLEVP solver
# from pytmod for comparison.

eps_fourier = [
    -deps / (2 * 1j),  # ε₋₁ = -iΔε/2
    eps0,  # ε₀ (background)
    deps / (2 * 1j),  # ε₊₁ = +iΔε/2
]

mat = pm.Material(eps_fourier, Omega, Npad=1)
slab = pm.Slab(mat, Lslab)

# Define search region for NLEVP solver
omega0 = 0.65 - 0.32j
omega1 = 0.92 - 0.019j

print("\nSearching for QNMs with NLEVP solver in region:")
print(f"  Bottom-left:  ω = {omega0}")
print(f"  Top-right:    ω = {omega1}")

# Solve with NLEVP
evs_nlevp, modes_nlevp, modes_left_nlevp = slab.eigensolve(
    omega0,
    omega1,
    peak_ref=6,
    recursive=True,
    tol=1e-6,
    return_left=True,
)

evs_nlevp = np.array(evs_nlevp)
print(f"Number of QNMs found (NLEVP): {len(evs_nlevp)}")

for i, ev in enumerate(evs_nlevp):
    print(f"  Mode {i}: ω = {ev.real:.6f} {ev.imag:+.6f}i")


# %%
# Visualize eigenvalues in complex plane
# --------------------------------------
# We plot the eigenvalues found by both methods in the complex frequency
# plane, along with the analytical static reference.

if HAS_FEM and len(eigenvalues) > 0:
    plt.figure(figsize=(10, 8))

    # Plot FEM eigenvalues
    plt.plot(-evs.real, evs.imag, "sk", mfc="none", markersize=6, label="FEM (all)")

    # Plot analytical static eigenvalues
    plt.plot(
        evs_ana_filtered.real,
        evs_ana_filtered.imag,
        "+b",
        markersize=8,
        label="Analytical (static)",
    )

    # Plot NLEVP eigenvalues
    plt.plot(evs_nlevp.real, evs_nlevp.imag, "xr", markersize=10, mew=2, label="NLEVP")

    # Plot PML branch lines
    theta = np.angle(solver.spml)
    s = np.linspace(-1, 2, 100)
    for i in harmonics:
        plt.plot(
            s + i * Omega,
            -s * np.cos(theta),
            "k",
            alpha=0.2,
            linewidth=0.5,
        )

    plt.xlabel(r"Re $\omega$", fontsize=12)
    plt.ylabel(r"Im $\omega$", fontsize=12)
    plt.title("Quasi-Normal Modes in Complex Frequency Plane", fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(-0.05, 0.05)
    plt.tight_layout()
    plt.show()


# %%
# Compare FEM and NLEVP results
# -----------------------------
# We compute the relative error between FEM and NLEVP eigenvalues
# for modes that are close to each other.

if HAS_FEM and len(eigenvalues) > 0 and len(evs_nlevp) > 0:
    print("\n" + "=" * 70)
    print("Comparison: FEM vs NLEVP")
    print("=" * 70)
    print(f"{'Mode':<6} {'NLEVP (Re/Im)':<25} {'FEM (Re/Im)':<25} {'Error %'}")
    print("-" * 70)

    # Match eigenvalues
    for j, ev_nlevp in enumerate(evs_nlevp):
        # Find closest FEM eigenvalue
        distances = np.abs(evs - ev_nlevp)
        idx = np.argmin(distances)
        ev_fem = evs[idx]

        err = np.abs(ev_nlevp - ev_fem) / np.abs(ev_fem) * 100

        print(
            f"{j + 1:<6} "
            f"{ev_nlevp.real:10.6f} {ev_nlevp.imag:10.6f}  "
            f"{ev_fem.real:10.6f} {ev_fem.imag:10.6f}  "
            f"{err:.2e}"
        )


# %%
# Visualize mode profiles (FEM)
# -----------------------------
# We extract and plot the spatial profiles of selected QNMs computed
# by the FEM solver.

if HAS_FEM and len(eigenvalues) > 0:
    # Get DOF coordinates
    dof_coords = solver.get_dof_coordinates()
    x = dof_coords[:, 1] - solver.L_total / 2  # Center at origin

    # Extract mode amplitudes for each harmonic
    modes = solver.extract_mode_amplitudes(eigenvectors)

    # Select modes to plot (in the range of interest)
    mode_indices = np.where(mask)[0][:6]  # First 6 valid modes

    if len(mode_indices) > 0:
        fig, axes = plt.subplots(
            len(mode_indices),
            len(harmonics),
            figsize=(12, 2 * len(mode_indices)),
            squeeze=False,
        )

        for i, mode_idx in enumerate(mode_indices):
            for j, harm_idx in enumerate(range(len(harmonics))):
                ax = axes[i, j]
                mode_profile = modes[harm_idx][:, mode_idx]

                # Plot real and imaginary parts
                ax.plot(
                    x / Lslab,
                    np.real(mode_profile),
                    "-",
                    color="#3cabdf",
                    label="Re",
                    linewidth=1.5,
                )
                ax.plot(
                    x / Lslab,
                    np.imag(mode_profile),
                    "--",
                    color="#d55151",
                    label="Im",
                    linewidth=1.5,
                )

                # Mark slab boundaries
                ax.axvline(-0.5, color="#949494", linewidth=1, linestyle=":")
                ax.axvline(0.5, color="#949494", linewidth=1, linestyle=":")

                if i == 0:
                    ax.set_title(f"Harmonic q={harmonics[j]}", fontsize=10)
                if j == 0:
                    ax.set_ylabel(
                        f"Mode {mode_idx}\n"
                        f"$\\omega$={evs[mode_idx].real:.3f}{evs[mode_idx].imag:+.3f}i",
                        fontsize=9,
                    )
                if i == len(mode_indices) - 1:
                    ax.set_xlabel(r"$x/L$", fontsize=10)

                ax.set_xlim(x[0] / Lslab, x[-1] / Lslab)

        axes[0, -1].legend(loc="upper right", fontsize=8)
        plt.suptitle("FEM QNM Spatial Profiles", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()


# %%
# Convergence study: mesh refinement
# ----------------------------------
# We study how the eigenvalues converge as the mesh is refined.
# This is important for validating FEM results.

if HAS_FEM:
    print("\nConvergence study: mesh refinement")
    print("=" * 50)

    lc_values = [1e-2, 7.5e-3, 5e-3, 2.5e-3]
    eigenvalues_convergence = []

    for lc in lc_values:
        print(f"  Mesh size lc = {lc:.4f}...")

        solver_conv = FEMSlabSolver(
            eps0=eps0,
            deps=0,  # Static case for cleaner convergence
            Omega=Omega,
            Lslab=Lslab,
            harmonics=[0],  # Single harmonic for speed
            lc_min=lc,
            order=2,
        )
        solver_conv.setup()

        evs_conv, _ = solver_conv.solve(
            neig=10,
            target=0.3,
            tol=1e-8,
            quad=True,
        )

        # Sort and take first mode
        if len(evs_conv) > 0:
            ev_sorted = np.sort(evs_conv[evs_conv.real > 0])
            eigenvalues_convergence.append(ev_sorted[0] if len(ev_sorted) > 0 else None)
        else:
            eigenvalues_convergence.append(None)

    print("\nFirst eigenvalue vs mesh size:")
    for lc, ev in zip(lc_values, eigenvalues_convergence, strict=False):
        if ev is not None:
            print(f"  lc = {lc:.4f}: ω = {ev.real:.6f} {ev.imag:+.6f}i")


# %%
# Summary
# -------
# This example demonstrated:
#
# 1. **FEM setup for time-modulated slabs**: Creating a mesh with PMLs,
#    setting up variational forms for the quadratic eigenvalue problem.
#
# 2. **Solving the QNM problem**: Using SLEPc to find quasi-normal modes
#    in the complex frequency plane.
#
# 3. **Validation**: Comparing FEM results with:
#    - Analytical static slab eigenvalues (with Floquet shifts)
#    - Semi-analytical NLEVP solver results
#
# 4. **Mode visualization**: Plotting spatial profiles of QNMs for each
#    Floquet harmonic component.
#
# 5. **Convergence study**: Checking mesh convergence for validation.
#
# The FEM approach provides a flexible framework that can be extended to:
# - More complex geometries (2D/3D)
# - Spatially varying permittivity profiles
# - Nonlinear materials
#
# Key observations:
# - The quadratic eigenvalue problem captures the full time-modulation effects
# - PMLs effectively implement outgoing boundary conditions
# - Results agree well with semi-analytical methods for simple geometries
