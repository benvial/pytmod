# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .eig import nonlinear_eigensolver
from .helpers import (
    dimhandler,
    dot,
    fresnel,
    matvecprod,
    normalize_modes,
    vecmatprod,
)


class Slab:
    """
    Slab object

    Parameters
    ----------
    material : Material
        The material of the slab
    thickness : float
        The thickness of the slab
    eps_plus : float, optional
        The permittivity of the medium above the slab
    eps_minus : float, optional
        The permittivity of the medium below the slab
    """

    def __init__(self, material, thickness, eps_plus=1, eps_minus=1, alt=True):
        self.material = material
        self.thickness = thickness
        self.eps_plus = eps_plus
        self.eps_minus = eps_minus
        self.dim = self.material.nh * 2
        self.alt = alt

    def __repr__(self):
        return f"Slab(thickness={self.thickness}, material={self.material}), eps_plus={self.eps_plus}, eps_minus={self.eps_minus}"

    def __str__(self):
        return self.__repr__()

    @dimhandler
    def build_matrix(self, omegas, eigenvalues, modes):
        """
        Build the matrix of the linear system to be solved.

        Parameters
        ----------
        omegas : array_like
            The frequencies at which to solve the system.
        eigenvalues : array_like
            The eigenvalues of the material.
        modes : array_like
            The eigenvectors of the material.

        Returns
        -------
        matrix_slab : array_like
            The matrix of the linear system.
        """
        omegas = np.array(omegas)
        Nh = self.material.Nh
        eigenvalues = eigenvalues.T
        modes = modes.T

        harm_index = np.arange(-Nh, Nh + 1)
        # harm_index = np.hstack([harm_index, harm_index])
        harm_index = np.broadcast_to(harm_index, eigenvalues.shape)
        harm_index = np.transpose(harm_index)

        omegas_shift = omegas - harm_index * self.material.modulation_frequency
        omegas_shift = np.transpose(omegas_shift)

        L = self.thickness
        phi_plus = np.exp(1j * eigenvalues * L)
        phi_minus = np.exp(-1j * eigenvalues * L)

        ks = np.broadcast_to(eigenvalues[:, :, np.newaxis], modes.shape)
        phi_plus = np.broadcast_to(phi_plus[:, :, np.newaxis], modes.shape)
        phi_minus = np.broadcast_to(phi_minus[:, :, np.newaxis], modes.shape)
        omegas_shift = np.broadcast_to(omegas_shift[:, :, np.newaxis], modes.shape)

        ks = np.transpose(ks, (0, 2, 1))
        modes = np.transpose(modes, (0, 2, 1))
        phi_plus = np.transpose(phi_plus, (0, 2, 1))
        phi_minus = np.transpose(phi_minus, (0, 2, 1))
        # omegas_shift = np.transpose(omegas_shift, (0, 2, 1))

        n_plus = self.eps_plus**0.5
        n_minus = self.eps_minus**0.5
        if self.alt:
            m11 = (n_plus + ks / omegas_shift) * modes
            m12 = (n_plus - ks / omegas_shift) * modes
            m21 = (n_minus - ks / omegas_shift) * phi_plus * modes
            m22 = (n_minus + ks / omegas_shift) * phi_minus * modes

        else:
            m11 = (omegas_shift * n_plus + ks) * modes
            m12 = (omegas_shift * n_plus - ks) * modes
            m21 = (omegas_shift * n_minus - ks) * phi_plus * modes
            m22 = (omegas_shift * n_minus + ks) * phi_minus * modes

        matrix_slab = np.block([[m11, m12], [m21, m22]])
        return np.transpose(matrix_slab, (1, 2, 0))

    @dimhandler
    def build_dmatrix_domega(self, omegas, eigenvalues, modes, modes_left):
        """
        Build the of the linear system to be solved.

        Parameters
        ----------
        omegas : array_like
            The frequencies at which to solve the system.
        eigenvalues : array_like
            The eigenvalues of the material.
        modes : array_like
            The eigenvectors of the material.
        modes_left : array_like
            The left eigenvectors of the material.

        Returns
        -------
        matrix_slab : array_like
            The matrix derivative wrt omega of the linear system.
        """
        omegas = np.array(omegas)
        Nh = self.material.Nh

        dmatrix = self.material.build_dmatrix_domega(omegas)

        dmodes = self.material.get_deigenmodes_right_domega(
            omegas,
            eigenvalues,
            modes,
            modes_left,
            dmatrix,
        )

        deigenvalues = self.material.get_deigenvalues_domega(
            omegas,
            eigenvalues,
            modes,
            modes_left,
            dmatrix,
        )

        eigenvalues = eigenvalues.T
        modes = modes.T
        dmodes = dmodes.T
        deigenvalues = deigenvalues.T

        harm_index = np.arange(-Nh, Nh + 1)
        harm_index = np.broadcast_to(harm_index, eigenvalues.shape)
        harm_index = np.transpose(harm_index)

        omegas_shift = omegas - harm_index * self.material.modulation_frequency
        omegas_shift = np.transpose(omegas_shift)

        L = self.thickness
        phi_plus = np.exp(1j * eigenvalues * L)
        phi_minus = np.exp(-1j * eigenvalues * L)

        ks = np.broadcast_to(eigenvalues[:, :, np.newaxis], modes.shape)
        phi_plus = np.broadcast_to(phi_plus[:, :, np.newaxis], modes.shape)
        phi_minus = np.broadcast_to(phi_minus[:, :, np.newaxis], modes.shape)
        omegas_shift = np.broadcast_to(omegas_shift[:, :, np.newaxis], modes.shape)
        dks = np.broadcast_to(deigenvalues[:, :, np.newaxis], modes.shape)

        dphi_plus = 1j * L * dks * phi_plus
        dphi_minus = -1j * L * dks * phi_minus

        ks = np.transpose(ks, (0, 2, 1))
        modes = np.transpose(modes, (0, 2, 1))
        phi_plus = np.transpose(phi_plus, (0, 2, 1))
        phi_minus = np.transpose(phi_minus, (0, 2, 1))
        dks = np.transpose(dks, (0, 2, 1))
        dmodes = np.transpose(dmodes, (0, 2, 1))
        dphi_plus = np.transpose(dphi_plus, (0, 2, 1))
        dphi_minus = np.transpose(dphi_minus, (0, 2, 1))

        n_plus = self.eps_plus**0.5
        n_minus = self.eps_minus**0.5

        if self.alt:
            dm11 = (n_plus + ks / omegas_shift) * dmodes + (
                dks - ks / omegas_shift
            ) / omegas_shift * modes
            dm12 = (n_plus - ks / omegas_shift) * dmodes - (
                dks - ks / omegas_shift
            ) / omegas_shift * modes
            dm21 = (n_minus - ks / omegas_shift) * (
                modes * dphi_plus + dmodes * phi_plus
            ) - (dks - ks / omegas_shift) / omegas_shift * phi_plus * modes
            dm22 = (n_minus + ks / omegas_shift) * (
                modes * dphi_minus + dmodes * phi_minus
            ) + (dks - ks / omegas_shift) / omegas_shift * phi_minus * modes
        else:
            dm11 = (omegas_shift * n_plus + ks) * dmodes + (n_plus + dks) * modes
            dm12 = (omegas_shift * n_plus - ks) * dmodes + (n_plus - dks) * modes
            dm21 = (omegas_shift * n_minus - ks) * (
                modes * dphi_plus + dmodes * phi_plus
            ) + (n_minus - dks) * phi_plus * modes
            dm22 = (omegas_shift * n_minus + ks) * (
                modes * dphi_minus + dmodes * phi_minus
            ) + (n_minus + dks) * phi_minus * modes

        dmatrix_slab = np.block([[dm11, dm12], [dm21, dm22]])
        return np.transpose(dmatrix_slab, (1, 2, 0))

    def build_rhs(self, omegas, Eis):
        """
        Build the right-hand side (RHS) of the linear system for the slab.

        Parameters
        ----------
        omegas : array_like
            The frequencies at which to solve the system.
        Eis : array_like
            The incident electric fields.

        Returns
        -------
        rhs_slab : array_like
            The RHS matrix of the linear system.
        """

        omegas = np.array(omegas)
        Eis = np.array(Eis)
        rhs_slab = np.zeros((2 * self.material.nh, *omegas.shape), dtype=np.complex128)
        for n in range(self.material.nh):
            rhs_slab[n] = self.eps_plus**0.5 * 2 * Eis[n]
            rhs_slab[n + self.material.nh] = (
                self.eps_minus**0.5 * 2 * Eis[n + self.material.nh]
            )
            if not self.alt:
                nshift = self.material.index_shift(n)
                omegas_shift = omegas - nshift * self.material.modulation_frequency
                rhs_slab[n] *= omegas_shift
                rhs_slab[n + self.material.nh] *= omegas_shift
        return rhs_slab

    def solve(self, matrix_slab, rhs_slab):
        """
        Solve the linear system defined by the matrix and RHS of the slab.

        Parameters
        ----------
        matrix_slab : array_like
            The matrix of the linear system.
        rhs_slab : array_like
            The right-hand side of the linear system.

        Returns
        -------
        solution : array_like
            The solution of the linear system.
        """
        if matrix_slab.ndim == 2:
            return np.linalg.solve(matrix_slab, rhs_slab)
        sol = np.empty_like(rhs_slab)
        if matrix_slab.ndim == 3:
            for i in range(matrix_slab.shape[-1]):
                sol[:, i] = np.linalg.solve(matrix_slab[:, :, i], rhs_slab[:, i])
            return sol
        if matrix_slab.ndim == 4:
            for i in range(matrix_slab.shape[-2]):
                for j in range(matrix_slab.shape[-1]):
                    sol[:, i, j] = np.linalg.solve(
                        matrix_slab[:, :, i, j], rhs_slab[:, i, j]
                    )
            return sol
        msg = f"Unsupported number of dimensions: {matrix_slab.ndim}"
        raise ValueError(msg)

    def _extract_coefficients(self, solution, Eis, kns, ens):
        phi_plus = np.exp(1j * kns * self.thickness)
        phi_minus = np.exp(-1j * kns * self.thickness)
        nh = self.material.nh
        Eslab_plus = solution[:nh]
        Eslab_minus = solution[nh : 2 * nh]
        Er = ens @ (Eslab_plus + Eslab_minus) - Eis[:nh]
        Et = (
            ens * phi_plus @ Eslab_plus
            + ens * phi_minus @ Eslab_minus
            - Eis[nh : 2 * nh]
        )
        return Eslab_plus, Eslab_minus, Er, Et

    def extract_coefficients(self, solution, Eis, kns, ens):
        """
        Extracts the coefficients of the waves from the solution of the linear system.

        Parameters
        ----------
        solution : array_like
            The solution of the linear system.
        Eis : array_like
            The incident electric fields.
        kns : array_like
            The eigenvalues of the slab time-modulated medium.
        ens : array_like
            The eigenvectors of the slab time-modulated medium.

        Returns
        -------
        Eslab_plus : array_like
            The coefficients of the forward propagating waves inside the slab.
        Eslab_minus : array_like
            The coefficients of the backward propagating waves inside the slab.
        Er : array_like
            The coefficients of the reflected waves.
        Et : array_like
            The coefficients of the transmitted waves.
        """
        if solution.ndim == 1:
            return self._extract_coefficients(solution, Eis, kns, ens)

        Eslab_plus = np.empty_like(kns)
        Eslab_minus = np.empty_like(kns)
        Er = np.empty_like(kns)
        Et = np.empty_like(kns)
        if solution.ndim == 2:
            for i in range(solution.shape[-1]):
                Eslab_plus[:, i], Eslab_minus[:, i], Er[:, i], Et[:, i] = (
                    self._extract_coefficients(
                        solution[:, i], Eis[:, i], kns[:, i], ens[:, :, i]
                    )
                )
            return Eslab_plus, Eslab_minus, Er, Et
        if solution.ndim == 3:
            for i in range(solution.shape[-2]):
                for j in range(solution.shape[-1]):
                    (
                        Eslab_plus[:, i, j],
                        Eslab_minus[:, i, j],
                        Er[:, i, j],
                        Et[:, i, j],
                    ) = self._extract_coefficients(
                        solution[:, i, j],
                        Eis[:, i, j],
                        kns[:, i, j],
                        ens[:, :, i, j],
                    )
            return Eslab_plus, Eslab_minus, Er, Et
        msg = f"Unsupported number of dimensions: {solution.ndim}"
        raise ValueError(msg)

    def fresnel_static(self, omegas):
        """
        Compute the Fresnel coefficients for a static slab with the same thickness
        and dielectric properties as the current slab.

        Parameters
        ----------
        omegas : array_like
            The frequencies at which to compute the Fresnel coefficients

        Returns
        -------
        rf : array_like
            The reflection Fresnel coefficient
        tf : array_like
            The transmission Fresnel coefficient
        """
        eps_slab = self.material.eps_fourier[self.material.Nh]
        return fresnel(omegas, eps_slab, self.eps_plus, self.eps_minus, self.thickness)

    def eigenvalue_static(self, n):
        """
        Calculate the static eigenvalue for a given mode number.

        Parameters
        ----------
        n : int
            The mode number for which the static eigenvalue is calculated.

        Returns
        -------
        complex
            The static eigenvalue corresponding to the specified mode number,
            based on the slab's thickness and dielectric properties.
        """

        eps_plus = self.eps_plus
        eps_minus = self.eps_minus
        eps_slab = self.material.eps_fourier[self.material.Nh]
        r21 = (eps_slab**0.5 - eps_plus**0.5) / (eps_plus**0.5 + eps_slab**0.5)
        r23 = (eps_slab**0.5 - eps_minus**0.5) / (eps_slab**0.5 + eps_minus**0.5)
        alpha = r21 * r23
        return (
            1 / (self.thickness * eps_slab**0.5) * (n * np.pi + 1j / 2 * np.log(alpha))
        )

    def eigensolve(self, *args, jac=False, **kwargs):
        """
        Solve the eigenvalue problem of the time-modulated slab.

        Parameters
        ----------
        *args : array_like
            Arguments to be passed to `nonlinear_eigensolver`.
        **kwargs : dict
            Keyword arguments to be passed to `nonlinear_eigensolver`.

        Returns
        -------
        eigenvalues : array_like
            The eigenvalues of the system.
        modes : array_like
            The eigenvectors of the system.
        """
        if "dim" not in kwargs:
            kwargs["dim"] = self.material.nh * 2

        if jac:

            def _build_matrix(omegas):
                eigenvalues, modes, modes_left = self.material.eigensolve(
                    omegas, left=True, normalize=True
                )

                return self.build_matrix(
                    omegas, eigenvalues, modes
                ), self.build_dmatrix_domega(omegas, eigenvalues, modes, modes_left)

            return nonlinear_eigensolver(_build_matrix, *args, dfunc=True, **kwargs)

        def _build_matrix(omegas):
            eigenvalues, modes, modes_left = self.material.eigensolve(
                omegas, left=True, normalize=True
            )
            return self.build_matrix(omegas, eigenvalues, modes)

        return nonlinear_eigensolver(_build_matrix, *args, **kwargs)

    def init_incident_field(self, omegas):
        """
        Initialize the incident field.

        Parameters
        ----------
        omegas : array_like
            The frequencies at which to initialize the incident field.

        Returns
        -------
        incident_field : array_like
            The initialized incident field.
        """
        omegas = np.array(omegas)
        return np.zeros((2 * self.material.nh, *omegas.shape), dtype=np.complex128)

    def get_incident_field(self, x, t, omega, Eis):
        """
        Compute the incident field at the given points in space and time.

        Parameters
        ----------
        x : array_like
            The points in space at which to compute the incident field.
        t : array_like
            The points in time at which to compute the incident field.
        omega : float
            The frequency at which the incident field is computed.
        Eis : array_like
            The Fourier coefficients of the incident field.

        Returns
        -------
        Einc : array_like
            The incident field at the specified points in space and time.
        """
        Nt = len(t)
        Nx = len(x)
        nh = self.material.nh
        Nh = self.material.Nh
        eps_plus = self.eps_plus
        eps_minus = self.eps_minus
        Omega = self.material.modulation_frequency
        L = self.thickness

        Einc = np.zeros((Nx, Nt), dtype=np.complex128)
        for ix, x_ in enumerate(x):
            if x_ < 0:
                _E = 0
                for n in range(-Nh, Nh + 1):
                    kn = eps_plus**0.5 * (omega - n * Omega)
                    _E += Eis[n + Nh] * np.exp(
                        1j * (kn * (x_) - (omega - n * Omega) * t)
                    )
                Einc[ix] = _E
            elif x_ > L:
                _E = 0
                for n in range(-Nh, Nh + 1):
                    kn = eps_minus**0.5 * (omega - n * Omega)
                    _E += Eis[n + nh + Nh] * np.exp(
                        -1j * (kn * (x_ - L) + (omega - n * Omega) * t)
                    )
                Einc[ix] = _E
            else:
                pass
        return Einc

    def get_scattered_field(
        self,
        x,
        t,
        omega,
        psi,
        ks,
        modes,
    ):
        """
        Compute the scattered electric field at positions x and times t.

        Parameters
        ----------
        x : array_like
            The positions at which to compute the scattered field.
        t : array_like
            The times at which to compute the scattered field.
        omega : float
            The frequency of the incident wave.
        psi : tuple
            The coefficients of the waves inside the slab, as returned by
            extract_coefficients.
        ks : array_like
            The eigenvalues of the slab time-modulated medium.
        modes : array_like
            The eigenvectors of the slab time-modulated medium.

        Returns
        -------
        E : array_like
            The scattered electric field at positions x and times t.
        """
        Nt = len(t)
        Nx = len(x)
        Eslab_plus, Eslab_minus, Enr, Ent = psi
        Omega = self.material.modulation_frequency
        eps_plus = self.eps_plus
        eps_minus = self.eps_minus
        nh = self.material.nh
        Nh = self.material.Nh
        L = self.thickness

        E = np.zeros((Nx, Nt), dtype=np.complex128)
        for ix, x_ in enumerate(x):
            if x_ < 0:
                _E = 0
                for n in range(-Nh, Nh + 1):
                    kn = eps_plus**0.5 * (omega - n * Omega)
                    _E += Enr[n + Nh] * np.exp(
                        -1j * (kn * (x_) + (omega - n * Omega) * t)
                    )
                E[ix] = _E
            elif x_ > L:
                _E = 0
                for n in range(-Nh, Nh + 1):
                    kn = eps_minus**0.5 * (omega - n * Omega)
                    _E += Ent[n + Nh] * np.exp(
                        1j * (kn * (x_ - L) - (omega - n * Omega) * t)
                    )
                E[ix] = _E
            else:
                _E = 0
                for p in range(nh):
                    _En = 0
                    for n in range(-Nh, Nh + 1):
                        _En += (
                            (
                                Eslab_plus[p] * np.exp(1j * ks[p] * x_)
                                + Eslab_minus[p] * np.exp(-1j * ks[p] * x_)
                            )
                            * modes[n + Nh, p]
                            * np.exp(-1j * (omega - n * Omega) * t)
                        )
                    _E += _En
                E[ix] = _E
        return E

    def animate_field(self, x, t, E, fig_ax=None):
        """
        Create an animation of the electric field over time within the slab.

        Parameters
        ----------
        x : array_like
            The spatial positions at which the electric field is evaluated.
        t : array_like
            The temporal points at which the electric field is evaluated.
        E : array_like
            The electric field values at the specified positions and times.
        fig_ax : tuple, optional
            A tuple containing a matplotlib figure and axes. If None, a new figure
            and axes are created.

        Returns
        -------
        ani : matplotlib.animation.FuncAnimation
            The animation object displaying the evolution of the electric field.
        """

        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax

        L = self.thickness
        T = self.material.modulation_period
        Emaxi = np.max(E.real)
        Emini = np.min(E.real)
        xs = x / L - 0.5
        ymax = np.max(np.real(E))
        ymin = np.min(np.real(E))
        ymaxmin = max(np.abs(ymax), np.abs(ymin))
        ymax += ymaxmin * 0.2
        ymin -= ymaxmin * 0.2

        eps_time = self.material.get_eps_time(t)
        eps_slab = eps_time[0].real
        eps_slab_map = eps_slab * np.ones((1, 1))

        xslab = [-0.5, 0.5]
        yslab = [ymin, ymax]
        eps_min = np.min(eps_time.real)
        eps_max = np.max(eps_time.real)

        plt.axvline(-0.5, color="#949494", lw=1)
        plt.axvline(0.5, color="#949494", lw=1)
        plt.xlabel("$x/L$")
        plt.ylabel("Re $E$")
        plt.ylim(Emini, Emaxi)
        cax = plt.pcolormesh(
            xslab,
            yslab,
            eps_slab_map,
            cmap="Reds",
            vmin=eps_min,
            vmax=eps_max,
            shading="flat",
            alpha=0.5,
        )
        (line,) = ax.plot(xs, np.real(E[:, 0]), c="#4c4c4c")
        cbar = plt.colorbar()
        cbar.ax.set_title(r"$\epsilon(t)$")

        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ymin, ymax)
        # title = ax.set_title(rf"$t = {t[0]/T:.2f}T$")
        title = ax.text(
            0.13, 0.93, rf"$t = {t[0] / T:.2f}\,T$", transform=ax.transAxes, ha="center"
        )

        def animate(it):  # pragma: no cover
            eps_slab = eps_time[it].real
            cax.set_array(eps_slab * np.ones((1, 1)))
            line.set_ydata(np.real(E[:, it]))
            title.set_text(f"$t = {t[it] / T:.2f}T$")
            return (
                line,
                cax,
            )

        return animation.FuncAnimation(
            fig, animate, blit=False, repeat=True, frames=len(t) - 1, interval=10
        )

    def get_modes_normalization(self, modes_right, modes_left, matrix_derivative):
        dim = modes_right.shape[1]
        normas = np.zeros((dim,) + modes_right.shape[2:], dtype=complex)
        for i in range(dim):
            normas[i] = (
                dot(
                    modes_left[:, i],
                    matvecprod(matrix_derivative[:, :, i], modes_right[:, i]),
                )
                ** 0.5
            )

        return normas

    def normalize(self, modes_right, modes_left, matrix_derivative, max_index=0):
        normas = self.get_modes_normalization(
            modes_right, modes_left, matrix_derivative
        )
        return normalize_modes(normas, modes_right, modes_left, max_index=max_index)

    # def normalize(self, modes_right, modes_left, matrix_derivative):
    #     max_indices = np.argmax(np.abs(modes_right), axis=0)
    #     # max_indices = np.zeros_like(max_indices)
    #     modes_right_max_values = np.take_along_axis(
    #         modes_right, np.expand_dims(max_indices, axis=0), axis=0
    #     )
    #     modes_right = modes_right / modes_right_max_values
    #     # modes_right = normalize_batch(modes_right, modes_right[self.material.Nh])
    #     normas = self.get_modes_normalization(
    #         modes_right, modes_left, matrix_derivative
    #     )
    #     modes_left = normalize_batch(modes_left, normas**2)
    #     return modes_right, modes_left

    def scalar_product(
        self,
        modes_right,
        modes_left,
        eigenvalue_right,
        eigenvalue_left,
        matrix_right,
        matrix_left,
        matrix_derivative,
        diag=True,
    ):
        if diag:
            return dot(modes_left, matvecprod(matrix_derivative, modes_right))
        R = dot(modes_left, matvecprod(matrix_right, modes_right))
        L = dot(vecmatprod(modes_left, matrix_left), modes_right)
        return (L - R) / (eigenvalue_right - eigenvalue_left)

    def get_multiplicities(self, eigenvalues, M=None, tol=1e-6):
        """
        Compute the multiplicity of each eigenvalue.

        Parameters
        ----------
        eigenvalues : array
            The eigenvalues
        M : array, optional
            The matrix at each eigenvalue. If None, it is computed.
        tol : float
            The tolerance for the rank computation

        Returns
        -------
        multiplicities : array
            The multiplicity of each eigenvalue
        """
        if M is None:
            evs_mat, modes_mat, _ = self.material.eigensolve(
                eigenvalues, left=True, normalize=True
            )
            M = self.build_matrix(eigenvalues, evs_mat, modes_mat).swapaxes(-1, 0)
        return self.dim - np.linalg.matrix_rank(M, tol=tol)
