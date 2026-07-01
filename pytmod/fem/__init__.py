# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod

"""
Finite Element Method (FEM) module for pytmod.

This module provides FEM-based solvers for time-modulated electromagnetic
problems using FEniCSx/dolfinx.

Available Classes
-----------------
FEMSlabSolver
    Solver for 1D time-modulated slab eigenvalue problems using FEM.

Examples
--------
>>> from pytmod.fem import FEMSlabSolver
>>> solver = FEMSlabSolver(eps0=50, deps=20, Omega=1, Lslab=1.5)
>>> solver.setup()
>>> eigenvalues, eigenvectors = solver.solve(neig=50)
"""

from __future__ import annotations

# Optional import - FEM dependencies may not be installed
try:
    from .slab_solver import FEMSlabSolver

    __all__ = ["FEMSlabSolver"]
except ImportError:
    # Dependencies not available (dolfinx, gmsh, slepc4py)
    __all__ = []
