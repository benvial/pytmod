# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod

"""
FEM Solver for Time-Modulated Slabs.

This module provides a finite element method (FEM) solver for computing
quasi-normal modes (QNMs) of one-dimensional slabs with time-periodic
permittivity modulation. The solver uses FEniCSx/dolfinx for the finite
element discretization and SLEPc for the eigenvalue problem.
"""

from __future__ import annotations

import gmsh
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc


class FEMSlabSolver:
    """
    FEM solver for time-modulated slab eigenvalue problems.

    This class sets up and solves the quadratic eigenvalue problem for
    a 1D slab with time-periodic permittivity using the Floquet formalism.

    Parameters
    ----------
    eps0 : float
        Background permittivity (constant component).
    deps : float
        Modulation amplitude (sinusoidal component).
    Omega : float
        Modulation frequency.
    Lslab : float
        Slab thickness.
    harmonics : list[int], optional
        List of Floquet harmonic indices to include. Default is [-1, 0, 1].
    Lsupsub : float, optional
        Superstrate and substrate thickness. Default is 1.01 * Lslab.
    Lpml : float, optional
        Perfectly matched layer (PML) thickness. Default is 2 * Lslab.
    lc_min : float, optional
        Minimum element size for mesh generation. Default is 5e-3.
    apml : float, optional
        PML real coefficient. Default is 1.
    bpml : float, optional
        PML imaginary coefficient. Default is 1.
    element_type : str, optional
        Finite element type. Default is "Lagrange".
    order : int, optional
        Finite element order. Default is 2.

    Attributes
    ----------
    domain : Mesh
        The dolfinx mesh object.
    ct : MeshTags
        Cell tags for material regions.
    V : FunctionSpace
        Vector function space for the electric field.
    epsilon_coeffs : dict
        Fourier coefficients of the permittivity.

    Examples
    --------
    >>> solver = FEMSlabSolver(
    ...     eps0=50, deps=20, Omega=1, Lslab=1.5, harmonics=[-1, 0, 1]
    ... )
    >>> solver.generate_mesh()
    >>> solver.setup_variational_forms()
    >>> eigenvalues, eigenvectors = solver.solve(neig=50, target=0.1)

    Notes
    -----
    The time-modulated permittivity is modeled as:

    .. math::
        \\epsilon(t) = \\epsilon_0 + \\Delta\\epsilon \\sin(\\Omega t)

    The Fourier coefficients are:
    :math:`\\epsilon_{-1} = -i\\Delta\\epsilon/2`,
    :math:`\\epsilon_0 = \\epsilon_0`,
    :math:`\\epsilon_{+1} = +i\\Delta\\epsilon/2`.

    The eigenvalue problem is quadratic in :math:`\\omega`:

    .. math::
        (A_0 + \\omega A_1 + \\omega^2 A_2) \\mathbf{u} = 0

    where the matrices come from the discretized variational forms.

    References
    ----------
    .. [1] Zurita-Sánchez et al., "Reflection and transmission of light
           in time-periodic and space-periodic media", J. Opt. Soc. Am. A
           (2009).
    """

    def __init__(
        self,
        eps0: float,
        deps: float,
        Omega: float,
        Lslab: float,
        harmonics: list[int] | None = None,
        Lsupsub: float | None = None,
        Lpml: float | None = None,
        lc_min: float = 5e-3,
        apml: float = 1.0,
        bpml: float = 1.0,
        element_type: str = "Lagrange",
        order: int = 2,
    ):
        self.eps0 = eps0
        self.deps = deps
        self.Omega = Omega
        self.Lslab = Lslab
        self.harmonics = harmonics if harmonics is not None else [-1, 0, 1]
        self.Lsupsub = Lsupsub if Lsupsub is not None else 1.01 * Lslab
        self.Lpml = Lpml if Lpml is not None else 2 * Lslab
        self.lc_min = lc_min
        self.apml = apml
        self.bpml = bpml
        self.element_type = element_type
        self.order = order

        # PML complex stretching parameter
        self.spml = apml - 1j * bpml

        # Compute epsilon Fourier coefficients
        self.epsilon_coeffs = self._compute_epsilon_coeffs()

        # Initialize attributes that will be set later
        self.domain = None
        self.ct = None
        self.ft = None
        self.V = None
        self.A0 = None
        self.A1 = None
        self.A2 = None
        self.bcs = []

    def _compute_epsilon_coeffs(self) -> dict[int, complex]:
        """
        Compute Fourier coefficients of the permittivity.

        Returns
        -------
        dict[int, complex]
            Dictionary mapping harmonic index to Fourier coefficient.
        """
        coeffs = {0: self.eps0}
        if -1 in self.harmonics:
            coeffs[-1] = -self.deps / (2j)
        if 1 in self.harmonics:
            coeffs[1] = self.deps / (2j)
        return coeffs

    def generate_mesh(self) -> None:
        """
        Generate the 1D mesh with gmsh.

        Creates a layered mesh with:
        - Bottom PML
        - Substrate
        - Slab (time-modulated region)
        - Superstrate
        - Top PML

        The mesh is stored in ``self.domain`` and cell tags in ``self.ct``.
        """
        gmsh.initialize()
        gmsh.model.add("1D_slab")

        # Layer definitions
        layer_thicknesses = {
            "pml_bottom": self.Lpml,
            "substrate": self.Lsupsub,
            "slab": self.Lslab,
            "superstrate": self.Lsupsub,
            "pml_top": self.Lpml,
        }

        L = sum(layer_thicknesses.values())

        # Create points along the line
        y = 0
        points = []
        labels = []

        for name, thickness in layer_thicknesses.items():
            p = gmsh.model.geo.addPoint(0, y, 0, self.lc_min)
            points.append(p)
            y += thickness
            labels.append(name)

        # Add topmost point
        points.append(gmsh.model.geo.addPoint(0, y, 0, self.lc_min))

        # Create line segments
        lines = []
        line_tags_by_layer = {}
        for i, name in enumerate(labels):
            lab = gmsh.model.geo.addLine(points[i], points[i + 1])
            lines.append(lab)
            line_tags_by_layer[name] = lab

        # Add physical groups
        for name, tag in line_tags_by_layer.items():
            gmsh.model.addPhysicalGroup(1, [tag], tag)
            gmsh.model.setPhysicalName(1, tag, name)

        # Generate mesh
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(1)

        # Convert to dolfinx mesh
        self.domain, self.ct, self.ft = model_to_mesh(
            gmsh.model, MPI.COMM_WORLD, 0, gdim=2
        )
        gmsh.finalize()

        # Store total length for reference
        self.L_total = L

    def assign_material_properties(self) -> None:
        """
        Assign material properties (permittivity and permeability) to mesh regions.

        Sets up:
        - PML regions: complex stretched permittivity/permeability (spml)
        - Other regions: will be handled in variational forms

        The properties are stored in ``self.epsilon`` and ``self.mu`` functions.
        """
        # Scalar function space for material properties
        Q = fem.functionspace(self.domain, ("DG", 0))
        material_tags = np.unique(self.ct.values)

        self.epsilon = fem.Function(Q)
        self.mu = fem.Function(Q)
        self.epsilon.x.array[:] = 1.0 + 0j
        self.mu.x.array[:] = 1.0 + 0j

        # Tag mapping: pml_bottom=1, substrate=2, slab=3, superstrate=4, pml_top=5
        for tag in material_tags:
            cells = self.ct.find(tag)
            if tag in [1, 5]:  # PML regions
                eps_ = self.spml
                mu_ = self.spml
                self.epsilon.x.array[cells] = np.full_like(
                    cells, eps_, dtype=PETSc.ScalarType
                )
                self.mu.x.array[cells] = np.full_like(
                    cells, mu_, dtype=PETSc.ScalarType
                )

    def setup_variational_forms(self) -> None:
        """
        Set up the variational forms for the quadratic eigenvalue problem.

        The eigenvalue problem is:

        .. math::
            (A_0 + \\omega A_1 + \\omega^2 A_2) \\mathbf{u} = 0

        where :math:`\\omega` is the eigenvalue (complex frequency).

        This method creates the matrices A0, A1, A2 and stores them as
        PETSc matrices in ``self.A0``, ``self.A1``, ``self.A2``.
        """
        N_harm = len(self.harmonics)

        # Vector function space for electric field (one component per harmonic)
        self.V = fem.functionspace(
            self.domain, (self.element_type, self.order, (N_harm,))
        )

        # Measure with subdomain data
        dx = ufl.Measure("dx", domain=self.domain, subdomain_data=self.ct)

        # Trial and test functions
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        # Domain tags: slab is 3, others are 1,2,4,5
        dom_tmod = 3
        dom_cst = (1, 2, 4, 5)

        # Initialize forms
        a0, a1, a2 = 0, 0, 0

        # Build coupling terms from time-modulated permittivity
        for jm, m in enumerate(self.harmonics):
            for jn, n in enumerate(self.harmonics):
                um = u[jm]
                vn = v[jn]
                delta = n - m

                if delta not in self.epsilon_coeffs:
                    continue

                eps_mn = self.epsilon_coeffs[delta]
                Omega_n = n * self.Omega
                coeffs_ = (
                    np.array([Omega_n**2, 2 * Omega_n, 1], dtype=PETSc.ScalarType)
                    * eps_mn
                )
                coeffs = [
                    fem.Constant(self.domain, PETSc.ScalarType(coeff))
                    for coeff in coeffs_
                ]

                a0 -= coeffs[0] * ufl.inner(um, vn) * dx(dom_tmod)
                a1 += coeffs[1] * ufl.inner(um, vn) * dx(dom_tmod)
                a2 -= coeffs[2] * ufl.inner(um, vn) * dx(dom_tmod)

            # Add terms for constant permittivity regions
            vm = v[jm]
            um = u[jm]
            Omega_m = m * self.Omega

            # Curl-curl term (1/mu * grad(u) * grad(v))
            a0 += 1 / self.mu * ufl.inner(ufl.grad(um), ufl.grad(vm)) * dx

            # Constant permittivity terms
            a0 -= (
                self.epsilon
                * fem.Constant(self.domain, PETSc.ScalarType(Omega_m**2))
                * ufl.inner(um, vm)
                * dx(dom_cst)
            )
            a1 += (
                self.epsilon
                * fem.Constant(self.domain, PETSc.ScalarType(2 * Omega_m))
                * ufl.inner(um, vm)
                * dx(dom_cst)
            )
            a2 -= self.epsilon * ufl.inner(um, vm) * dx(dom_cst)

        # Assemble PETSc matrices
        self.A0 = petsc.assemble_matrix(fem.form(a0), bcs=self.bcs)
        self.A0.assemble()
        self.A1 = petsc.assemble_matrix(fem.form(a1), bcs=self.bcs)
        self.A1.assemble()
        self.A2 = petsc.assemble_matrix(fem.form(a2), bcs=self.bcs)
        self.A2.assemble()

    def solve(
        self,
        neig: int = 200,
        tol: float = 1e-6,
        target: complex = 0.1,
        quad: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the quadratic eigenvalue problem.

        Parameters
        ----------
        neig : int, optional
            Number of eigenvalues to compute. Default is 200.
        tol : float, optional
            Tolerance for the eigenvalue solver. Default is 1e-6.
        target : float or complex, optional
            Target value for eigenvalue search. Default is 0.1.
        quad : bool, optional
            If True, solve the quadratic eigenvalue problem.
            If False, solve the generalized eigenvalue problem with
            :math:`\\omega^2` as eigenvalue. Default is True.

        Returns
        -------
        eigenvalues : np.ndarray
            Array of computed eigenvalues (complex frequencies).
        eigenvectors : np.ndarray
            Array of eigenvectors, shape (ndof, nconv).
        """
        if quad:
            eps_solver = SLEPc.PEP().create(MPI.COMM_WORLD)
            eps_solver.setOperators([self.A0, self.A1, self.A2])
            eps_solver.setProblemType(SLEPc.PEP.ProblemType.GENERAL)
        else:
            eps_solver = SLEPc.EPS().create(MPI.COMM_WORLD)
            eps_solver.setOperators(self.A0, self.A2)
            eps_solver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

        eps_solver.setWhichEigenpairs(eps_solver.Which.TARGET_MAGNITUDE)
        st = eps_solver.getST()
        st.setType("sinvert")
        st.setTransform(False)

        target_ = target if quad else target**2
        eps_solver.setTolerances(tol=tol)
        eps_solver.setTarget(target_)
        eps_solver.setDimensions(neig)
        eps_solver.setFromOptions()
        eps_solver.solve()

        # Extract results
        eigenvalues = []
        eigenvectors = []
        nconv = eps_solver.getConverged()

        if MPI.COMM_WORLD.rank == 0:
            for i in range(min(nconv, neig)):
                vr, vi = self.A0.getVecs()
                val = eps_solver.getEigenpair(i, vr, vi)
                eigenvalues.append(val)

                # Extract eigenvector for each harmonic
                vec = vr.array
                eigenvectors.append(vec)

        eigenvalues = np.array(eigenvalues)
        eigenvectors = np.array(eigenvectors).T if eigenvectors else np.array([])

        # Convert to omega if not quadratic
        if not quad:
            eigenvalues = np.sqrt(eigenvalues)

        # Store solver info
        self.solver_info = {
            "nconv": nconv,
            "neig_requested": neig,
            "tol": tol,
            "target": target,
            "quad": quad,
        }

        return eigenvalues, eigenvectors

    def get_dof_coordinates(self) -> np.ndarray:
        """
        Get the coordinates of the degrees of freedom.

        Returns
        -------
        np.ndarray
            Coordinates of DOFs, shape (ndof, 2).
        """
        if self.V is None:
            msg = (
                "Function space not initialized. Call setup_variational_forms() first."
            )
            raise RuntimeError(msg)
        return self.V.tabulate_dof_coordinates()

    def extract_mode_amplitudes(self, eigenvectors: np.ndarray) -> list[np.ndarray]:
        """
        Extract mode amplitudes for each Floquet harmonic.

        Parameters
        ----------
        eigenvectors : np.ndarray
            Eigenvectors from solve(), shape (ndof, nconv).

        Returns
        -------
        list[np.ndarray]
            List of mode amplitudes for each harmonic.
            Each element has shape (ndof_per_harm, nconv).
        """
        N_harm = len(self.harmonics)
        modes = []
        for jm in range(N_harm):
            mode_jm = eigenvectors[jm::N_harm, :]
            modes.append(mode_jm)
        return modes

    def get_analytical_static_eigenvalues(self, n_modes: int = 10) -> np.ndarray:
        """
        Compute analytical eigenvalues for the static (unmodulated) slab.

        This provides a reference for validation.

        Parameters
        ----------
        n_modes : int, optional
            Number of modes to compute. Default is 10.

        Returns
        -------
        np.ndarray
            Analytical eigenvalues for the static slab.
        """
        n0 = np.sqrt(self.eps0)
        r21 = (n0 - 1) / (n0 + 1)

        evs_ana = []
        for i in range(n_modes):
            omega_ana = np.pi * (i + 1) / (self.Lslab * n0)
            omega_ana += 1j * np.log(r21) / (self.Lslab * n0)
            for j in self.harmonics:
                omega_ana_shift = omega_ana + j * self.Omega
                evs_ana.append(omega_ana_shift)

        return np.sort(evs_ana)

    def setup(self) -> None:
        """
        Convenience method to run all setup steps.

        This method calls in sequence:
        1. generate_mesh()
        2. assign_material_properties()
        3. setup_variational_forms()
        """
        self.generate_mesh()
        self.assign_material_properties()
        self.setup_variational_forms()


if __name__ == "__main__":
    # Quick self-test
    print("FEMSlabSolver self-test")
    print("=" * 50)

    # Create solver with default parameters
    solver = FEMSlabSolver(
        eps0=50,
        deps=20,
        Omega=1,
        Lslab=1.5,
        harmonics=[-1, 0, 1],
    )

    print(f"Background permittivity: eps0 = {solver.eps0}")
    print(f"Modulation amplitude: deps = {solver.deps}")
    print(f"Modulation frequency: Omega = {solver.Omega}")
    print(f"Slab thickness: Lslab = {solver.Lslab}")
    print(f"Floquet harmonics: {solver.harmonics}")
    print(f"Epsilon coefficients: {solver.epsilon_coeffs}")

    print("\nSelf-test completed successfully!")
