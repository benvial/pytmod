# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


"""Cylinder scattering module for static and time-modulated cylinders.

This module provides functions for computing Mie scattering coefficients,
scattering efficiencies, and quasi-normal modes (QNMs) of infinite
dielectric cylinders. It supports both TM (E along z) and TE (H along z)
polarizations.

The static (unmodulated) cylinder functions serve as the foundation and
validation baseline for the Floquet-Mie extension to time-periodic
permittivity.

Notes
-----
All quantities use dimensionless units internally:
- Lengths normalized by cylinder radius R
- Frequencies normalized by c/R
- Size parameter X = omega * R / c = k_0 * R

Sign convention: e^{-i omega t} time dependence (physics convention).
Outgoing waves: H_n^{(1)}(kr) for r -> infinity.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy import special

from .eig import nonlinear_eigensolver
from .helpers import (
    dimhandler,
    dot,
    matmatprod,
    matvecprod,
    move_first_axes_to_end,
    move_first_two_axes_to_end,
    move_last_axes_to_beginning,
    move_last_two_axes_to_beginning,
    normalize_modes,
    vecmatprod,
)

# ============================================================
# Bessel function utilities
# ============================================================


def jn(n, z):
    """Bessel function of the first kind J_n(z) for complex argument.

    Parameters
    ----------
    n : int
        Order of the Bessel function.
    z : complex or array_like
        Argument (may be complex).

    Returns
    -------
    complex or ndarray
        Value of J_n(z).

    Notes
    -----
    Uses `scipy.special.jv` which handles complex arguments natively.
    """
    return special.jv(n, z)


def yn(n, z):
    """Bessel function of the second kind Y_n(z) for complex argument.

    Parameters
    ----------
    n : int
        Order of the Bessel function.
    z : complex or array_like
        Argument (may be complex).

    Returns
    -------
    complex or ndarray
        Value of Y_n(z).

    Notes
    -----
    Uses `scipy.special.yv` which handles complex arguments natively.
    """
    return special.yv(n, z)


def h1n(n, z):
    """Hankel function of the first kind H_n^{(1)}(z).

    Parameters
    ----------
    n : int
        Order of the Hankel function.
    z : complex or array_like
        Argument (may be complex).

    Returns
    -------
    complex or ndarray
        Value of H_n^{(1)}(z) = J_n(z) + i Y_n(z).

    Notes
    -----
    Uses `scipy.special.hankel1` for complex arguments.
    """
    return special.hankel1(n, z)


def jn_prime(n, z):
    """Derivative of the Bessel function J_n'(z).

    Parameters
    ----------
    n : int
        Order of the Bessel function.
    z : complex or array_like
        Argument (may be complex).

    Returns
    -------
    complex or ndarray
        Value of dJ_n/dz evaluated at z.

    Notes
    -----
    Uses `scipy.special.jvp` which handles complex arguments natively.
    """
    return special.jvp(n, z)


def h1n_prime(n, z):
    """Derivative of the Hankel function H_n^{(1)'}(z).

    Parameters
    ----------
    n : int
        Order of the Hankel function.
    z : complex or array_like
        Argument (may be complex).

    Returns
    -------
    complex or ndarray
        Value of dH_n^{(1)}/dz evaluated at z.

    Notes
    -----
    Uses `scipy.special.h1vp` which handles complex arguments natively.
    """
    return special.h1vp(n, z)


def jn_double_prime(n, z):
    return special.jvp(n, z, n=2)


def h1n_double_prime(n, z):
    return special.h1vp(n, z, n=2)


# ============================================================
# Static Mie scattering coefficients
# ============================================================


def mie_coefficient_TM(n, size_param, m):
    """TM Mie scattering coefficient b_n for a 2D cylinder.

    Computes the scattering coefficient for angular order n for TM
    polarization (E field along the cylinder axis z).

    Parameters
    ----------
    n : int
        Angular order (azimuthal mode number). Can be negative.
    size_param : complex or array_like
        Size parameter X = k_0 R = omega R / c (may be complex for QNM
        searches).
    m : complex
        Relative refractive index m = sqrt(epsilon_r) of the cylinder.

    Returns
    -------
    complex or ndarray
        TM scattering coefficient b_n(X).

    Notes
    -----
    The TM Mie coefficient for an infinite 2D cylinder is [1]_:

    .. math::

        b_n = -\\frac{m J_n(mX) J_n'(X) - J_n(X) J_n'(mX)}
                     {m J_n(mX) H_n^{(1)'}(X) - H_n^{(1)}(X) J_n'(mX)}

    QNMs correspond to the poles of b_n, i.e., zeros of the denominator.

    References
    ----------
    .. [1] Bohren & Huffman, "Absorption and Scattering of Light by Small
       Particles" (1983), adapted for 2D cylinder geometry.
    """
    X = np.asarray(size_param, dtype=complex)
    mX = m * X

    Jn_mX = jn(n, mX)
    Jn_X = jn(n, X)
    Jnp_mX = jn_prime(n, mX)
    Jnp_X = jn_prime(n, X)
    H1n_X = h1n(n, X)
    H1np_X = h1n_prime(n, X)

    numerator = m * Jn_mX * Jnp_X - Jn_X * Jnp_mX
    denominator = m * Jn_mX * H1np_X - H1n_X * Jnp_mX

    return -numerator / denominator


def mie_coefficient_TE(n, size_param, m):
    """TE Mie scattering coefficient a_n for a 2D cylinder.

    Computes the scattering coefficient for angular order n for TE
    polarization (H field along the cylinder axis z).

    Parameters
    ----------
    n : int
        Angular order (azimuthal mode number). Can be negative.
    size_param : complex or array_like
        Size parameter X = k_0 R = omega R / c (may be complex for QNM
        searches).
    m : complex
        Relative refractive index m = sqrt(epsilon_r) of the cylinder.

    Returns
    -------
    complex or ndarray
        TE scattering coefficient a_n(X).

    Notes
    -----
    The TE Mie coefficient for an infinite 2D cylinder is [1]_:

    .. math::

        a_n = -\\frac{J_n(mX) J_n'(X) - m J_n(X) J_n'(mX)}
                     {J_n'(mX) H_n^{(1)}(X) - m J_n(mX) H_n^{(1)'}(X)}

    Note the m <-> 1/m duality with the TM case.

    References
    ----------
    .. [1] Bohren & Huffman, "Absorption and Scattering of Light by Small
       Particles" (1983), adapted for 2D cylinder geometry.
    """
    X = np.asarray(size_param, dtype=complex)
    mX = m * X

    Jn_mX = jn(n, mX)
    Jn_X = jn(n, X)
    Jnp_mX = jn_prime(n, mX)
    Jnp_X = jn_prime(n, X)
    H1n_X = h1n(n, X)
    H1np_X = h1n_prime(n, X)

    # TE: swap m <-> 1/m relative to TM in the numerator/denominator
    numerator = Jn_mX * Jnp_X - m * Jn_X * Jnp_mX
    denominator = Jnp_mX * H1n_X - m * Jn_mX * H1np_X

    return -numerator / denominator


def mie_coefficients_TM(n_max, size_param, m):
    """TM Mie scattering coefficients b_n for orders n = 0, ..., n_max.

    Parameters
    ----------
    n_max : int
        Maximum angular order.
    size_param : complex or array_like
        Size parameter X = k_0 R.
    m : complex
        Relative refractive index.

    Returns
    -------
    ndarray
        Array of TM coefficients b_0, b_1, ..., b_{n_max}.
        Shape: (n_max + 1,) + size_param.shape.
    """
    return np.array([mie_coefficient_TM(n, size_param, m) for n in range(n_max + 1)])


def mie_coefficients_TE(n_max, size_param, m):
    """TE Mie scattering coefficients a_n for orders n = 0, ..., n_max.

    Parameters
    ----------
    n_max : int
        Maximum angular order.
    size_param : complex or array_like
        Size parameter X = k_0 R.
    m : complex
        Relative refractive index.

    Returns
    -------
    ndarray
        Array of TE coefficients a_0, a_1, ..., a_{n_max}.
        Shape: (n_max + 1,) + size_param.shape.
    """
    return np.array([mie_coefficient_TE(n, size_param, m) for n in range(n_max + 1)])


# ============================================================
# Scattering efficiency
# ============================================================


def scattering_efficiency_TM(size_param, m, n_max=20):
    """Total TM scattering efficiency Q_sca for a static 2D cylinder.

    Parameters
    ----------
    size_param : float or array_like
        Size parameter X = k_0 R (real, positive).
    m : complex
        Relative refractive index.
    n_max : int, optional
        Maximum angular order for convergence. Default is 20.

    Returns
    -------
    float or ndarray
        Scattering efficiency Q_sca = sigma_sca / (2R).

    Notes
    -----
    For a 2D cylinder the scattering cross section per unit length is:

    .. math::

        \\sigma_{\\rm sca} = \\frac{2}{k_0} \\left(|b_0|^2 +
        2\\sum_{n=1}^{n_{\\rm max}} |b_n|^2\\right)

    so Q_sca = sigma_sca / (2R) = (1/X)(|b_0|^2 + 2 sum |b_n|^2).
    """
    X = np.asarray(size_param, dtype=float)
    bn = mie_coefficients_TM(n_max, X, m)

    # b_0 counts once, b_n (n>=1) count twice due to +n and -n symmetry
    sigma = np.abs(bn[0]) ** 2 + 2 * np.sum(np.abs(bn[1:]) ** 2, axis=0)
    return (2.0 / X) * sigma


def scattering_efficiency_TE(size_param, m, n_max=20):
    """Total TE scattering efficiency Q_sca for a static 2D cylinder.

    Parameters
    ----------
    size_param : float or array_like
        Size parameter X = k_0 R (real, positive).
    m : complex
        Relative refractive index.
    n_max : int, optional
        Maximum angular order for convergence. Default is 20.

    Returns
    -------
    float or ndarray
        Scattering efficiency Q_sca = sigma_sca / (2R).

    Notes
    -----
    Analogous to TM but using TE coefficients a_n.
    """
    X = np.asarray(size_param, dtype=float)
    an = mie_coefficients_TE(n_max, X, m)

    sigma = np.abs(an[0]) ** 2 + 2 * np.sum(np.abs(an[1:]) ** 2, axis=0)
    return (2.0 / X) * sigma


def scattering_efficiency(size_param, m, n_max=20, polarization="TM"):
    """Total scattering efficiency Q_sca for a static 2D cylinder.

    Parameters
    ----------
    size_param : float or array_like
        Size parameter X = k_0 R (real, positive).
    m : complex
        Relative refractive index.
    n_max : int, optional
        Maximum angular order for convergence. Default is 20.
    polarization : str, optional
        Polarization: 'TM' or 'TE'. Default is 'TM'.

    Returns
    -------
    float or ndarray
        Scattering efficiency Q_sca.

    Raises
    ------
    ValueError
        If polarization is not 'TM' or 'TE'.
    """
    if polarization.upper() == "TM":
        return scattering_efficiency_TM(size_param, m, n_max)
    if polarization.upper() == "TE":
        return scattering_efficiency_TE(size_param, m, n_max)
    msg = f"Unknown polarization '{polarization}'. Must be 'TM' or 'TE'."
    raise ValueError(msg)


# ============================================================
# QNM denominators (for root-finding)
# ============================================================


def mie_denominator_TM(n, size_param, m):
    """Denominator of the TM Mie coefficient b_n.

    Parameters
    ----------
    n : int
        Angular order.
    size_param : complex or array_like
        Size parameter X = k_0 R (may be complex).
    m : complex
        Relative refractive index.

    Returns
    -------
    complex or ndarray
        Value of m J_n(mX) H_n^{(1)'}(X) - H_n^{(1)}(X) J_n'(mX).

    Notes
    -----
    QNMs are the complex frequencies omega where this function vanishes.
    In dimensionless units (R = c = 1), X = omega.
    """
    X = np.asarray(size_param, dtype=complex)
    mX = m * X
    return m * jn(n, mX) * h1n_prime(n, X) - h1n(n, X) * jn_prime(n, mX)


def mie_denominator_TE(n, size_param, m):
    """Denominator of the TE Mie coefficient a_n.

    Parameters
    ----------
    n : int
        Angular order.
    size_param : complex or array_like
        Size parameter X = k_0 R (may be complex).
    m : complex
        Relative refractive index.

    Returns
    -------
    complex or ndarray
        Value of J_n'(mX) H_n^{(1)}(X) - m J_n(mX) H_n^{(1)'}(X).

    Notes
    -----
    QNMs are the complex frequencies omega where this function vanishes.
    """
    X = np.asarray(size_param, dtype=complex)
    mX = m * X
    return jn_prime(n, mX) * h1n(n, X) - m * jn(n, mX) * h1n_prime(n, X)


# ============================================================
# Cylinder class
# ============================================================


class Cylinder:
    """Cylinder scatterer with (optionally) time-modulated permittivity.

    Parameters
    ----------
    material : Material
        The material of the cylinder (may be time-modulated).
    radius : float
        The radius of the cylinder.
    eps_ext : float, optional
        The permittivity of the surrounding medium. Default is 1 (vacuum).

    Attributes
    ----------
    material : Material
        The cylinder material.
    radius : float
        The cylinder radius R.
    eps_ext : float
        The external permittivity.
    dim : int
        The dimension of the linear system (material.nh).

    Notes
    -----
    In the current implementation, the external medium is assumed to be
    static (time-invariant) and homogeneous.

    For time-modulated cylinders, the Floquet-Mie matrix is built using
    the material eigenvalue problem. The dimension of the system is
    material.nh (number of Floquet harmonics).
    """

    def __init__(self, material, radius, eps_ext=1.0):
        self.material = material
        self.radius = radius
        self.eps_ext = eps_ext
        self.dim = self.material.nh

    def __repr__(self):
        return (
            f"Cylinder(radius={self.radius}, material={self.material}, "
            f"eps_ext={self.eps_ext})"
        )

    def __str__(self):
        return self.__repr__()

    def static(self, Npad=0):
        """Return a static version of the cylinder.

        Parameters
        ----------
        Npad : int, optional
            Number of padding zeros for Fourier coefficients.

        Returns
        -------
        Cylinder
            A cylinder with static material.
        """
        material_static = self.material.static(Npad)
        return Cylinder(
            material_static,
            self.radius,
            self.eps_ext,
        )

    @property
    def eps_static(self):
        """Static (DC) permittivity of the cylinder material.

        Returns
        -------
        complex
            The zeroth Fourier coefficient of the permittivity.
        """
        return self.material.eps_fourier[self.material.Nh]

    @property
    def m_static(self):
        """Static refractive index of the cylinder.

        Returns
        -------
        complex
            sqrt(eps_static / eps_ext).
        """
        return (self.eps_static / self.eps_ext) ** 0.5

    def mie_coefficient(self, omega, n=0, polarization="TM"):
        """Static Mie scattering coefficient for a given frequency.

        Parameters
        ----------
        omega : complex or array_like
            Frequency (in units of c/R).
        n : int, optional
            Angular order. Default is 0.
        polarization : str, optional
            'TM' or 'TE'. Default is 'TM'.

        Returns
        -------
        complex or ndarray
            Scattering coefficient b_n (TM) or a_n (TE).
        """

        n_ext = self.eps_ext**0.5
        X = n_ext * np.asarray(omega) * self.radius
        m = self.m_static

        if polarization == "TM":
            return mie_coefficient_TM(n, X, m)
        if polarization == "TE":
            return mie_coefficient_TE(n, X, m)
        msg = f"Unknown polarization '{polarization}'"
        raise ValueError(msg)

    def scattering_efficiency(self, omega, n_max=20, polarization="TM"):
        """Scattering efficiency for a static cylinder.

        Parameters
        ----------
        omega : float or array_like
            Frequency (in units of c/R).
        n_max : int, optional
            Maximum angular order. Default is 20.
        polarization : str, optional
            'TM' or 'TE'. Default is 'TM'.

        Returns
        -------
        float or ndarray
            Q_sca = sigma_sca / (2R).
        """
        n_ext = self.eps_ext**0.5
        X = n_ext * np.asarray(omega) * self.radius
        m = self.m_static
        return scattering_efficiency(X, m, n_max=n_max, polarization=polarization)

    @dimhandler
    def build_sub_matrices(self, omegas, eigenvalues, modes, n=0, polarization="TM"):
        """Build the Floquet-Mie matrix for the cylinder.

        Parameters
        ----------
        omegas : array_like
            The frequencies at which to solve the system.
        eigenvalues : array_like
            The eigenvalues of the material (wavenumbers k_j).
        modes : array_like
            The eigenvectors of the material.
        n : int, optional
            Angular order. Default is 0.
        polarization : str, optional
            Polarization: 'TM' or 'TE'. Default is 'TM'.

        Returns
        -------
        matrix : array_like
            The Floquet-Mie matrix of shape (nh, nh, *omegas.shape).

        """
        R = self.radius
        n_ext = self.eps_ext**0.5
        omegas = np.array(omegas)
        Nh = self.material.Nh
        # eigenvalues = move_first_axes_to_end(eigenvalues)
        # modes = move_first_two_axes_to_end(modes)
        eigenvalues = eigenvalues.T
        modes = modes.T

        harm_index = np.arange(-Nh, Nh + 1)
        harm_index = np.broadcast_to(harm_index, eigenvalues.shape)
        harm_index = np.transpose(harm_index)

        omegas_shift = omegas - harm_index * self.material.modulation_frequency
        omegas_shift = np.transpose(omegas_shift)

        k_int = eigenvalues
        k_ext = n_ext * omegas_shift

        J_ext = jn(n, k_ext * R)
        H_ext = h1n(n, k_ext * R)
        J_prime_ext = k_ext * jn_prime(n, k_ext * R)
        H_prime_ext = k_ext * h1n_prime(n, k_ext * R)

        J_prime_int = jn_prime(n, k_int * R)

        if polarization == "TM":
            f = modes  # f[w,j,q] = e_{jq}
            G = (k_int * J_prime_int)[
                ..., :, None
            ] * modes  # G_before[w,j,q] = k_j J_n'(k_j R) e_{jq}
        else:
            # f[w, j, q] = e_{jq} / omega_q  — omega_q is on the q axis
            f = modes / omegas_shift[:, np.newaxis, :]  # (nw, j, q) / (nw, 1, q)

            # G[w, j, q] = k_j^3 J_n'(k_j R) / omega_q^3 * e_{jq}
            G = (
                (k_int**3 * J_prime_int)[..., :, None]  # (nw, j, 1)
                * modes  # (nw, j, q)
                / omegas_shift[:, np.newaxis, :] ** 3  # (nw, 1, q)
            )
        F = jn(n, k_int * R)[..., :, None] * f

        P = J_ext
        Q = H_ext
        if polarization == "TM":
            P_prime = J_prime_ext
            Q_prime = H_prime_ext
        else:
            P_prime = J_prime_ext / self.eps_ext
            Q_prime = H_prime_ext / self.eps_ext

        F = np.transpose(F, (0, 2, 1))
        G = np.transpose(G, (0, 2, 1))
        return P, Q, P_prime, Q_prime, F, G

    @dimhandler
    def build_matrix(
        self, omegas, eigenvalues, modes, n=0, polarization="TM", alt=False
    ):
        """Build the Floquet-Mie matrix for the cylinder.

        Parameters
        ----------
        omegas : array_like
            The frequencies at which to solve the system.
        eigenvalues : array_like
            The eigenvalues of the material (wavenumbers k_j).
        modes : array_like
            The eigenvectors of the material.
        n : int, optional
            Angular order. Default is 0.
        polarization : str, optional
            Polarization: 'TM' or 'TE'. Default is 'TM'.

        Returns
        -------
        matrix : array_like
            The Floquet-Mie matrix of shape (nh, nh, *omegas.shape).

        """
        P, Q, P_prime, Q_prime, F, G = self.build_sub_matrices(
            omegas, eigenvalues, modes, n=n, polarization=polarization
        )

        X = np.linalg.solve(F, np.eye(self.dim))

        Xp = (
            X * P[..., None, :]
        )  # (nw, nh, nh) * (nw, 1, nh) — scales columns: F^{-1} diag(P)
        Xq = (
            X * Q[..., None, :]
        )  # (nw, nh, nh) * (nw, 1, nh) — scales columns: F^{-1} diag(Q)

        idx = np.arange(self.dim)
        G_ = move_last_two_axes_to_beginning(G)

        L = move_first_two_axes_to_end(
            matmatprod(G_, move_last_two_axes_to_beginning(Xq))
        )
        L[:, idx, idx] -= Q_prime  # L = GF^{-1}Q - Q'

        R = -move_first_two_axes_to_end(
            matmatprod(G_, move_last_two_axes_to_beginning(Xp))
        )
        R[:, idx, idx] += P_prime  # R = P' - GF^{-1}P
        if alt:
            return np.transpose(L, (1, 2, 0))
        S = np.linalg.solve(R, L)
        return np.transpose(S, (1, 2, 0))

    def build_rhs(self, omegas, incident_field, incident_angles, n=0):
        """Build the right-hand side (RHS) of the linear system.

        Parameters
        ----------
        omegas : array_like
            The frequencies at which to solve the system.
        incident_field : array_like
            The incident field coefficients.
        incident_angles : array_like
            The incident field angles.
        n : int, optional
            The azimuthal mode number. Default is 0.

        Returns
        -------
        rhs : array_like
            The RHS vector of shape (nh, *omegas.shape).
        """
        omegas = np.array(omegas)
        incident_field = np.array(incident_field)
        incident_angles = np.array(incident_angles)
        rhs = np.zeros((self.material.nh, *omegas.shape), dtype=np.complex128)

        for q in range(self.material.nh):
            rhs[q] = incident_field[q] * 1j**n * np.exp(-1j * n * incident_angles[q])

        return rhs

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
            The initialized incident amplitudes.
        incident_angles : array_like
            The initialized incident angles.
        """
        omegas = np.array(omegas)
        Eis, thetais = (
            np.zeros((self.material.nh, *omegas.shape), dtype=np.complex128),
            np.zeros((self.material.nh, *omegas.shape), dtype=np.complex128),
        )
        return Eis, thetais

    def solve(self, matrix, rhs):
        """Solve the linear system defined by the matrix and RHS.

        Parameters
        ----------
        matrix : array_like
            The matrix of the linear system.
        rhs : array_like
            The right-hand side of the linear system.

        Returns
        -------
        solution : array_like
            The solution of the linear system.
        """
        if matrix.ndim == 2:
            return np.linalg.solve(matrix, rhs)
        sol = np.empty_like(rhs)
        if matrix.ndim == 3:
            for i in range(matrix.shape[-1]):
                sol[:, i] = np.linalg.solve(matrix[:, :, i], rhs[:, i])
            return sol
        if matrix.ndim == 4:
            for i in range(matrix.shape[-2]):
                for j in range(matrix.shape[-1]):
                    sol[:, i, j] = np.linalg.solve(matrix[:, :, i, j], rhs[:, i, j])
            return sol
        msg = f"Unsupported number of dimensions: {matrix.ndim}"
        raise ValueError(msg)

    def eigensolve(self, *args, jac=False, **kwargs):
        """Solve the eigenvalue problem of the time-modulated cylinder.

        Parameters
        ----------
        *args : array_like
            Arguments to be passed to `nonlinear_eigensolver`.
        jac : bool, optional
            Whether to compute the Jacobian. Default is False.
        **kwargs : dict
            Keyword arguments to be passed to `nonlinear_eigensolver`.

        Returns
        -------
        eigenvalues : array_like
            The eigenvalues of the system (complex frequencies).
        modes : array_like
            The eigenvectors of the system.
        """
        if "dim" not in kwargs:
            kwargs["dim"] = self.material.nh
        if "polarization" in kwargs:
            polarization = kwargs.pop("polarization")
        else:
            polarization = "TM"
        if "n" in kwargs:
            n = kwargs.pop("n")
        else:
            n = 0
        if "alt" in kwargs:
            alt = kwargs.pop("alt")
        else:
            alt = True

        if jac:

            def _build_matrix(omegas):
                eigenvalues, modes, modes_left = self.material.eigensolve(
                    omegas, left=True, normalize=True
                )
                return self.build_matrix(
                    omegas,
                    eigenvalues,
                    modes,
                    n=n,
                    polarization=polarization,
                    alt=alt,
                ), self.build_dmatrix_domega(omegas, eigenvalues, modes, modes_left)

            return nonlinear_eigensolver(_build_matrix, *args, dfunc=True, **kwargs)

        def _build_matrix(omegas):
            eigenvalues, modes, modes_left = self.material.eigensolve(
                omegas, left=True, normalize=True
            )
            return self.build_matrix(
                omegas,
                eigenvalues,
                modes,
                n=n,
                polarization=polarization,
                alt=alt,
            )

        return nonlinear_eigensolver(_build_matrix, *args, **kwargs)

    @dimhandler
    def build_dmatrix_domega(
        self, omegas, eigenvalues, modes, modes_left, n=0, polarization="TM"
    ):
        """Build the derivative of the scattering matrix M wrt omega.

        M is defined by M b = a (b = scattered amplitudes, a = input amplitudes).
        M = R^{-1} L  where
            L = G F^{-1} Q - Q'
            R = P' - G F^{-1} P
        so
            dM/domega = R^{-1} (dL/domega - dR/domega M)

        with
            dL/domega = (dG - G X dF) X Q + G X dQ - dQ'
            dR/domega = dP' - (dG - G X dF) X P - G X dP

        where X = F^{-1} and dots denote d/domega.

        Parameters
        ----------
        omegas : array_like
            The frequencies at which to compute the derivative.
        eigenvalues : array_like
            The eigenvalues of the material (k_j), shape (nh, *omegas.shape).
        modes : array_like
            The right eigenvectors of the material, shape (nh, nh, *omegas.shape).
        modes_left : array_like
            The left eigenvectors of the material, shape (nh, nh, *omegas.shape).
        n : int, optional
            Azimuthal order. Default is 0.
        polarization : str, optional
            'TM' or 'TE'. Default is 'TM'.

        Returns
        -------
        dmatrix : array_like
            dM/domega, shape (nh, nh, *omegas.shape).
        """
        R_cyl = self.radius
        n_ext = self.eps_ext**0.5
        omegas = np.array(omegas)
        Nh = self.material.Nh
        nh = self.material.nh

        # ------------------------------------------------------------------
        # Reuse build_sub_matrices for P, Q, P', Q', F, G
        # F, G: (nw, nh, nh);  P, Q, P', Q': (nw, nh)
        # ------------------------------------------------------------------
        P, Q, P_prime, Q_prime, F, G = self.build_sub_matrices(
            omegas, eigenvalues, modes, n=n, polarization=polarization
        )

        # X = F^{-1}, shape (nw, nh, nh)
        X = np.linalg.solve(F, np.eye(nh))

        # Rebuild L, R, M
        Xp = X * P[..., None, :]
        Xq = X * Q[..., None, :]
        idx = np.arange(nh)
        G_ = move_last_two_axes_to_beginning(G)

        L_mat = move_first_two_axes_to_end(
            matmatprod(G_, move_last_two_axes_to_beginning(Xq))
        )
        L_mat[:, idx, idx] -= Q_prime

        R_mat = -move_first_two_axes_to_end(
            matmatprod(G_, move_last_two_axes_to_beginning(Xp))
        )
        R_mat[:, idx, idx] += P_prime

        M_mat = np.linalg.solve(R_mat, L_mat)  # (nw, nh, nh)

        # ------------------------------------------------------------------
        # Derivatives of interior eigenvalues and eigenvectors wrt omega
        # ------------------------------------------------------------------
        dmat_mat = self.material.build_dmatrix_domega(omegas)

        dk = self.material.get_deigenvalues_domega(
            omegas, eigenvalues, modes, modes_left, dmatrix=dmat_mat
        )  # (nh, *omegas.shape)

        de = self.material.get_deigenmodes_right_domega(
            omegas, eigenvalues, modes, modes_left, dmatrix=dmat_mat
        )  # (nh, nh, *omegas.shape)

        # Reshape to match build_sub_matrices conventions: (nw, j) and (nw, j, q)
        k_int = eigenvalues.T  # (nw, j)
        dk_int = dk.T  # (nw, j)
        E_wjq = modes.T  # (nw, j, q)
        de_wjq = de.T  # (nw, j, q)

        # Floquet-shifted exterior frequencies omega_q, shape (nw, q)
        harm_index = np.arange(-Nh, Nh + 1)
        harm_index = np.broadcast_to(harm_index, k_int.shape)
        harm_index = np.transpose(harm_index)
        omegas_shift = omegas - harm_index * self.material.modulation_frequency
        omegas_shift = np.transpose(omegas_shift)  # (nw, q)

        k_ext = n_ext * omegas_shift  # (nw, q)

        # ------------------------------------------------------------------
        # Interior Bessel values at r = R, indexed by j
        # J_n''(x) = (n^2 - x^2)/x^2 J_n(x) - (1/x) J_n'(x)
        # ------------------------------------------------------------------
        kR_int = k_int * R_cyl
        Jn_int = jn(n, kR_int)
        Jnp_int = jn_prime(n, kR_int)
        Jnpp_int = (n**2 - kR_int**2) / kR_int**2 * Jn_int - Jnp_int / kR_int

        # ------------------------------------------------------------------
        # dF/domega and dG/domega — polarization dependent
        # ------------------------------------------------------------------
        if polarization == "TM":
            # F[w,j,q] = J_n(k_j R) e_{jq}
            # dF[w,j,q] = R J_n'(k_j R) dk_j e_{jq} + J_n(k_j R) de_{jq}
            alpha = R_cyl * Jnp_int * dk_int  # (nw, j)
            dF_wjq = alpha[..., :, None] * E_wjq + Jn_int[..., :, None] * de_wjq

            # G[w,j,q] = k_j J_n'(k_j R) e_{jq}
            # dG[w,j,q] = [dk_j J_n'(k_j R) + k_j R J_n''(k_j R) dk_j] e_{jq}
            #            + k_j J_n'(k_j R) de_{jq}
            beta = dk_int * Jnp_int + k_int * R_cyl * Jnpp_int * dk_int
            dG_wjq = (
                beta[..., :, None] * E_wjq + (k_int * Jnp_int)[..., :, None] * de_wjq
            )
        else:
            # TE: F[w,j,q] = J_n(k_j R) / omega_q * e_{jq}
            # dF[w,j,q] = R J_n'(k_j R) dk_j / omega_q * e_{jq}
            #            - J_n(k_j R) / omega_q^2 * e_{jq}
            #            + J_n(k_j R) / omega_q * de_{jq}
            oq = omegas_shift[:, np.newaxis, :]  # (nw, 1, q)
            alpha_TE = R_cyl * Jnp_int * dk_int  # (nw, j)
            dF_wjq = (
                alpha_TE[..., :, None] / oq * E_wjq
                - Jn_int[..., :, None] / oq**2 * E_wjq
                + Jn_int[..., :, None] / oq * de_wjq
            )

            # TE: G[w,j,q] = k_j^3 J_n'(k_j R) / omega_q^3 * e_{jq}
            # dG[w,j,q] = [3 k_j^2 dk_j J_n'(k_j R) + k_j^3 R J_n''(k_j R) dk_j] / omega_q^3 * e_{jq}
            #            - 3 k_j^3 J_n'(k_j R) / omega_q^4 * e_{jq}
            #            + k_j^3 J_n'(k_j R) / omega_q^3 * de_{jq}
            k3Jnp = k_int**3 * Jnp_int  # (nw, j)
            d_k3Jnp = (
                3 * k_int**2 * dk_int * Jnp_int + k_int**3 * R_cyl * Jnpp_int * dk_int
            )
            oq3 = oq**3  # (nw, 1, q)
            oq4 = oq**4
            dG_wjq = (
                d_k3Jnp[..., :, None] / oq3 * E_wjq
                - 3 * k3Jnp[..., :, None] / oq4 * E_wjq
                + k3Jnp[..., :, None] / oq3 * de_wjq
            )

        dF = np.transpose(dF_wjq, (0, 2, 1))  # (nw, q, j)
        dG = np.transpose(dG_wjq, (0, 2, 1))  # (nw, q, j)

        # ------------------------------------------------------------------
        # Derivatives of exterior diagonal matrices
        # d(k_q^ext)/domega = n_ext  for all q
        # dP_q  = n_ext R J_n'(k_q R)
        # dQ_q  = n_ext R H_n^(1)'(k_q R)
        # dP'_q = n_ext J_n'(k_q R) + n_ext k_q R J_n''(k_q R)
        # dQ'_q = n_ext H_n^(1)'(k_q R) + n_ext k_q R H_n^(1)''(k_q R)
        # ------------------------------------------------------------------
        kR_ext = k_ext * R_cyl
        Jn_ext = jn(n, kR_ext)
        Jnp_ext = jn_prime(n, kR_ext)
        H1n_ext = h1n(n, kR_ext)
        H1np_ext = h1n_prime(n, kR_ext)
        Jnpp_ext = (n**2 - kR_ext**2) / kR_ext**2 * Jn_ext - Jnp_ext / kR_ext
        H1npp_ext = (n**2 - kR_ext**2) / kR_ext**2 * H1n_ext - H1np_ext / kR_ext

        dP = n_ext * R_cyl * Jnp_ext
        dQ = n_ext * R_cyl * H1np_ext
        dP_prime = n_ext * Jnp_ext + n_ext * k_ext * R_cyl * Jnpp_ext
        dQ_prime = n_ext * H1np_ext + n_ext * k_ext * R_cyl * H1npp_ext

        if polarization == "TE":
            dP_prime /= self.eps_ext
            dQ_prime /= self.eps_ext

        # ------------------------------------------------------------------
        # Assemble dL and dR
        # GX = G F^{-1}, shape (nw, q, j)
        # A  = dG - GX dF
        # dL = A X Q + GX dQ - diag(dQ')
        # dR = diag(dP') - A X P - GX dP
        # ------------------------------------------------------------------
        GX = np.einsum("...ij,...jk->...ik", G, X)  # (nw, q, j)
        GX_dF = np.einsum("...ij,...jk->...ik", GX, dF)  # (nw, q, j)
        A = dG - GX_dF  # (nw, q, j)

        AX = np.einsum("...ij,...jk->...ik", A, X)  # (nw, q, j)
        AXQ = AX * Q[..., None, :]  # A F^{-1} diag(Q)
        GXdQ = GX * dQ[..., None, :]  # G F^{-1} diag(dQ)

        dL = AXQ + GXdQ
        dL[:, idx, idx] -= dQ_prime

        AXP = AX * P[..., None, :]
        GXdP = GX * dP[..., None, :]

        dR = -AXP - GXdP
        dR[:, idx, idx] += dP_prime

        # ------------------------------------------------------------------
        # dM/domega = R^{-1} (dL - dR M)
        # ------------------------------------------------------------------
        dR_M = np.einsum("...ij,...jk->...ik", dR, M_mat)
        rhs = dL - dR_M
        dM = np.linalg.solve(R_mat, rhs)  # (nw, q, j)

        return np.transpose(dM, (1, 2, 0))  # (nh, nh, *omegas.shape)

    def get_modes_normalization(self, modes_right, modes_left, matrix_derivative):
        """Compute the normalization constants for the modes.

        Parameters
        ----------
        modes_right : array_like
            The right eigenvectors.
        modes_left : array_like
            The left eigenvectors.
        matrix_derivative : array_like
            The derivative of the matrix wrt omega.

        Returns
        -------
        normas : array_like
            The normalization constants.
        """
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
        """Normalize the eigenmodes.

        Parameters
        ----------
        modes_right : array_like
            The right eigenvectors.
        modes_left : array_like
            The left eigenvectors.
        matrix_derivative : array_like
            The derivative of the matrix wrt omega.
        max_index : int, optional
            The index of the maximum value for normalization.

        Returns
        -------
        modes_right : array_like
            The normalized right eigenvectors.
        modes_left : array_like
            The normalized left eigenvectors.
        """
        normas = self.get_modes_normalization(
            modes_right, modes_left, matrix_derivative
        )
        return normalize_modes(normas, modes_right, modes_left, max_index=max_index)

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
        """Compute the scalar product between left and right modes.

        Parameters
        ----------
        modes_right : array_like
            The right eigenvectors.
        modes_left : array_like
            The left eigenvectors.
        eigenvalue_right : array_like
            The right eigenvalues.
        eigenvalue_left : array_like
            The left eigenvalues.
        matrix_right : array_like
            The matrix at the right eigenvalue.
        matrix_left : array_like
            The matrix at the left eigenvalue.
        matrix_derivative : array_like
            The derivative of the matrix wrt omega.
        diag : bool, optional
            Whether to compute only diagonal elements.

        Returns
        -------
        product : array_like
            The scalar product.
        """
        if diag:
            return dot(modes_left, matvecprod(matrix_derivative, modes_right))
        R = dot(modes_left, matvecprod(matrix_right, modes_right))
        L = dot(vecmatprod(modes_left, matrix_left), modes_right)
        return (L - R) / (eigenvalue_right - eigenvalue_left)

    def get_multiplicities(self, eigenvalues, M=None, tol=1e-6):
        """Compute the multiplicity of each eigenvalue.

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

    def get_inner_coefficients(
        self, omegas, eigenvalues, modes, n, polarization, rhs, solution, check=False
    ):
        P, Q, P_prime, Q_prime, F, G = self.build_sub_matrices(
            omegas, eigenvalues, modes, n=n, polarization=polarization
        )
        a = rhs
        b = solution
        a_ = move_last_axes_to_beginning(a)
        b_ = move_last_axes_to_beginning(b)
        V_c = P * a_ + Q * b_
        c = np.linalg.solve(F, V_c[..., None])[..., 0]
        c = move_first_axes_to_end(c)
        if check:
            c_ = move_last_axes_to_beginning(c)
            np.einsum("...ij,...j->...i", G, c_)
            V_c_prime = P_prime * a_ + Q_prime * b_
            c_prime = np.linalg.solve(G, V_c_prime[..., None])[..., 0]
            c_prime = move_first_axes_to_end(c_prime)
            assert np.allclose(c, c_prime)
        return c

    def get_fields(
        self,
        x,
        y,
        omegas,
        solution,
        inner_coeffs,
        eigenvalues,
        modes,
        incident_field,
        incident_angles,
        n_range,
        t,
        polarization="TM",
    ):
        """Compute the total, scattered, and incident fields on a 2D grid.

        Parameters
        ----------
        x : array_like, shape (nx,)
            x-coordinates of the evaluation grid.
        y : array_like, shape (ny,) or scalar
            y-coordinates of the evaluation grid.  A scalar ``y`` produces a
            1-row grid (x-cut).
        omegas : array_like, shape (nw,)
            Frequencies at which the scattering problem was solved.
        solution : array_like, shape (n_az, nh, nw)
            Scattered amplitudes b, one row per azimuthal order.
        inner_coeffs : array_like, shape (n_az, nh, nw)
            Interior coefficients c, one row per azimuthal order.
        eigenvalues : array_like, shape (nh, nw)
            Material wavenumbers k_j.
        modes : array_like, shape (nh, nh, nw)
            Material eigenvectors.
        incident_field : array_like, shape (nh, nw)
            Incident field amplitudes A_q.
        incident_angles : array_like, shape (nh, nw)
            Incident angles theta_q (radians).
        n_range : sequence of int
            Azimuthal orders used.
        t : float or array_like, shape (nt,)
            Time(s) at which to evaluate the field.  When ``t`` is a scalar
            or a 1-element array the output has shape ``(ny, nx, nw)``; when
            ``t`` is an array of length ``nt > 1`` the output has shape
            ``(ny, nx, nw, nt)``.
        polarization : {'TM', 'TE'}, optional
            Polarization.  Default is ``'TM'``.

        Returns
        -------
        dict with keys ``'total'``, ``'scattered'``, ``'incident'``, each
        an ndarray of shape ``(ny, nx, nw)`` (scalar t) or
        ``(ny, nx, nw, nt)`` (array t with nt > 1).
        """
        R = self.radius
        Omega = self.material.modulation_frequency
        mat = self.material
        Nh = mat.Nh
        nh = mat.nh

        omegas_arr = np.asarray(omegas)
        omega_shape = omegas_arr.shape
        nw = omegas_arr.size
        omegas_f = omegas_arr.reshape(nw)
        eigenvalues = eigenvalues.T
        modes = modes.T

        # Parse time: support scalar and 1-D array.
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        nt = t_arr.size
        scalar_t = (np.ndim(t) == 0) or (t_arr.size == 1)

        # Flatten all frequency-dependent arrays to a single frequency axis,
        # then reshape back at the end.
        incident_field_f = np.asarray(incident_field).reshape(nh, nw)
        incident_angles_f = np.asarray(incident_angles).reshape(nh, nw)
        solution_f = np.asarray(solution).reshape(len(n_range), nh, nw)
        inner_coeffs_f = np.asarray(inner_coeffs).reshape(len(n_range), nh, nw)
        eigenvalues_f = np.asarray(eigenvalues).reshape(nh, nw)
        modes_f = np.asarray(modes).reshape(nh, nh, nw)

        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)

        q_vals = np.arange(-Nh, Nh + 1)
        n_vals = np.asarray(n_range)

        # Reconstruct rhs_f (input amplitudes a_q^{(n)}) from incident field data
        # Shape: (n_az, nh, nw)
        rhs_f = np.zeros((len(n_vals), nh, nw), dtype=complex)
        for ind_n, n in enumerate(n_vals):
            for iq in range(nh):
                rhs_f[ind_n, iq, :] = (
                    incident_field_f[iq, :]
                    * (1j**n)
                    * np.exp(-1j * n * incident_angles_f[iq, :])
                )

        # omega_q[q, w], k_ext_q[q, w]
        omega_q = omegas_f[None, :] - q_vals[:, None] * Omega  # (nh, nw)
        k_ext_q = omega_q * self.eps_ext**0.5  # (nh, nw)

        # time_phase_q:
        #   scalar t  → shape (nh, nw)
        #   array  t  → shape (nh, nw, nt)
        if scalar_t:
            time_phase_q = np.exp(-1j * omega_q * t_arr[0])  # (nh, nw)
        else:
            # omega_q: (nh, nw, 1), t_arr: (nt,) → broadcast to (nh, nw, nt)
            time_phase_q = np.exp(-1j * omega_q[:, :, None] * t_arr[None, None, :])

        # ----------------------------------------------------------------
        #  Incident plane-wave field  (exterior, exact)
        # ----------------------------------------------------------------
        phi = (
            X[:, :, None, None] * np.cos(incident_angles_f)[None, None, :, :]
            + Y[:, :, None, None] * np.sin(incident_angles_f)[None, None, :, :]
        )  # (ny, nx, nh, nw)

        # exp(i k_q r·s_q): (ny, nx, nh, nw)
        plane_phase = np.exp(1j * k_ext_q[None, None, :, :] * phi)

        if scalar_t:
            # time_phase_q: (nh, nw), broadcast as (ny, nx, nh, nw)
            field_inc = np.sum(
                incident_field_f[None, None, :, :]
                * plane_phase
                * time_phase_q[None, None, :, :],
                axis=2,
            )  # (ny, nx, nw)
        else:
            # time_phase_q: (nh, nw, nt) → (1, 1, nh, nw, nt)
            field_inc = np.einsum(
                "xypw,xypw,pwt->xywt",
                incident_field_f[None, None, :, :]
                * np.ones((r.shape[0], r.shape[1], 1, 1)),
                plane_phase,
                time_phase_q,
            )  # (ny, nx, nw, nt)

        # ----------------------------------------------------------------
        #  Incident field — cylindrical expansion (for interior region)
        # ----------------------------------------------------------------
        ang_n = np.exp(1j * theta[:, :, None] * n_vals[None, None, :])  # (ny, nx, n_az)
        kR_ext = r[:, :, None, None] * k_ext_q[None, None, :, :]  # (ny, nx, nh, nw)

        if scalar_t:
            a_time = rhs_f * time_phase_q[None, :, :]  # (n_az, nh, nw)
        else:
            a_time = (
                rhs_f[:, :, :, None] * time_phase_q[None, :, :, :]
            )  # (n_az, nh, nw, nt)

        field_inc_cyl = 0
        for ind_n, n in enumerate(n_vals):
            J_inc_n = jn(n, kR_ext)  # (ny, nx, nh, nw)
            if scalar_t:
                field_inc_cyl += ang_n[:, :, ind_n, None] * np.einsum(
                    "xyqw,qw->xyw", J_inc_n, a_time[ind_n]
                )
            else:
                field_inc_cyl += ang_n[:, :, ind_n, None, None] * np.einsum(
                    "xyqw,qwt->xywt", J_inc_n, a_time[ind_n]
                )

        # Use plane wave outside (exact), cylindrical expansion inside (consistent with BCs)
        if scalar_t:
            field_inc = np.where(r[:, :, None] > R, field_inc, field_inc_cyl)
        else:
            field_inc = np.where(r[:, :, None, None] > R, field_inc, field_inc_cyl)

        # ----------------------------------------------------------------
        #  Scattered field outside
        # ----------------------------------------------------------------
        if scalar_t:
            b_time = solution_f * time_phase_q[None, :, :]  # (n_az, nh, nw)
        else:
            b_time = (
                solution_f[:, :, :, None] * time_phase_q[None, :, :, :]
            )  # (n_az, nh, nw, nt)

        field_scatt_out = np.zeros_like(field_inc)
        for ind_n, n in enumerate(n_vals):
            H_ext_n = h1n(n, kR_ext)  # (ny, nx, nh, nw)
            if scalar_t:
                field_scatt_out += ang_n[:, :, ind_n, None] * np.einsum(
                    "xyqw,qw->xyw", H_ext_n, b_time[ind_n]
                )
            else:
                field_scatt_out += ang_n[:, :, ind_n, None, None] * np.einsum(
                    "xyqw,qwt->xywt", H_ext_n, b_time[ind_n]
                )

        # ----------------------------------------------------------------
        #  Scattered field inside
        # ----------------------------------------------------------------
        if polarization == "TM":
            f = modes_f  # (nh, nw) alias (j, q, w) after .T
        else:
            f = modes_f / omega_q[None, :, :]  # (j, q, w) → h_{jq}

        if scalar_t:
            f_time = np.einsum("jqw,qw->jw", f, time_phase_q)  # (nh, nw)
        else:
            f_time = np.einsum("jqw,qwt->jwt", f, time_phase_q)  # (nh, nw, nt)

        if scalar_t:
            coeff_njw = inner_coeffs_f * f_time[None, :, :]  # (n_az, nh, nw)
        else:
            coeff_njw = (
                inner_coeffs_f[:, :, :, None] * f_time[None, :, :, :]
            )  # (n_az, nh, nw, nt)

        kR_int = (
            r[:, :, None, None] * eigenvalues_f[None, None, :, :]
        )  # (ny, nx, nh, nw)
        field_scatt_in = np.zeros_like(field_inc)
        for ind_n, n in enumerate(n_vals):
            J_int_n = jn(n, kR_int)  # (ny, nx, nh, nw)
            if scalar_t:
                field_scatt_in += ang_n[:, :, ind_n, None] * np.einsum(
                    "xyjw,jw->xyw", J_int_n, coeff_njw[ind_n]
                )
            else:
                field_scatt_in += ang_n[:, :, ind_n, None, None] * np.einsum(
                    "xyjw,jwt->xywt", J_int_n, coeff_njw[ind_n]
                )

        # ----------------------------------------------------------------
        #  Assemble and reshape
        # ----------------------------------------------------------------
        if scalar_t:
            mask = r[:, :, None] > R
        else:
            mask = r[:, :, None, None] > R

        field_scatt = np.where(
            mask,
            field_scatt_out,
            field_scatt_in - field_inc_cyl,
        )

        if scalar_t:
            field_inc = field_inc.reshape(*r.shape, *omega_shape)
            field_scatt = field_scatt.reshape(*r.shape, *omega_shape)
        else:
            # shape is (ny, nx, nw, nt) already — reshape nw part back
            field_inc = field_inc.reshape(*r.shape, *omega_shape, nt)
            field_scatt = field_scatt.reshape(*r.shape, *omega_shape, nt)

        field_tot = field_inc + field_scatt
        return {"incident": field_inc, "scattered": field_scatt, "total": field_tot}

    def plot(self, color="k", ax=None, normalize=False):
        ax = ax or plt.gca()
        R = 1 if normalize else self.radius
        circle = plt.Circle((0, 0), R, ec=color, fill=False)
        ax.add_patch(circle)
        return circle

    def animate_field(
        self,
        x,
        y,
        t,
        E,
        polarization="TM",
        field_type="total",
        iomega=0,
        normalize=True,
        cmap="RdBu_r",
        fig_ax=None,
        interval=50,
        blit=False,
    ):
        """Create an animation of the 2D field map around the cylinder over time.

        The input field array ``E`` of shape ``(ny, nx, nt)`` is the pre-computed
        field at each time step.  Since :meth:`get_fields` accepts an array ``t``,
        the simplest way to produce ``E`` is::

            fi = self.get_fields(x, y, omegas, ..., t_anim, polarization)
            E  = fi["total"][:, :, 0, :]   # pick iomega=0 → shape (ny, nx, nt)

        Parameters
        ----------
        x : array_like, shape (nx,)
            The x-coordinates of the field grid (same units as :attr:`radius`).
        y : array_like, shape (ny,)
            The y-coordinates of the field grid.
        t : array_like, shape (nt,)
            The time values at which the field was evaluated.
        E : array_like, shape (ny, nx, nt)
            The (real or complex) field snapshots.  The real part
            ``Re[E[:, :, it]]`` is displayed at frame ``it``.
        field_type : str, optional
            Label for the colorbar title, e.g. ``'total'``, ``'scattered'``,
            ``'incident'``.  Default is ``'total'``.
        iomega : int, optional
            Index of the frequency slice to display (relevant when ``E`` has
            been evaluated at multiple frequencies).  Default is ``0``.
        normalize : bool, optional
            If ``True``, axes and cylinder radius are normalised by
            :attr:`radius`.  Default is ``True``.
        cmap : str, optional
            Matplotlib colormap for the field.  Default is ``'RdBu_r'``.
        fig_ax : tuple or None, optional
            ``(fig, ax)`` pair.  If ``None``, a new figure is created.
        interval : int, optional
            Delay between frames in milliseconds.  Default is 50.
        blit : bool, optional
            Whether to use blitting for faster animation. Default is False.

        Returns
        -------
        ani : matplotlib.animation.FuncAnimation
            The animation object.

        Notes
        -----
        Typical usage::

            t_anim = np.linspace(0, 2 * np.pi / Omega, 60)
            frames = []
            for ti in t_anim:
                fi = self.get_fields(x, y, omegas, solution_all, inner_coeffs_all,
                                    eigenvalues, modes, incident_field, incident_angles,
                                    n_range, ti, polarization)
                frames.append(np.real(fi["total"][:, :, 0]))
            E_anim = np.stack(frames, axis=-1)           # (ny, nx, nt)
            ani = self.animate_field(x, y, t_anim, E_anim)
            plt.show()
        """
        x = np.asarray(x)
        y = np.asarray(y)
        t = np.asarray(t)
        E = np.asarray(E)

        R_plot = 1.0 if normalize else self.radius
        scale = 1.0 / self.radius if normalize else 1.0
        xs = x * scale
        ys = y * scale

        T = self.material.modulation_period
        eps_time = self.material.get_eps_time(t)  # (nt,)
        eps_min = np.min(eps_time.real)
        eps_max = np.max(eps_time.real)

        # Create spatial mask for cylinder interior
        X, Y = np.meshgrid(xs, ys)
        r = np.sqrt(X**2 + Y**2)
        mask_inside = r <= R_plot

        # Colour limits: symmetric about zero, with 20 % margin
        emax = np.max(np.abs(np.real(E)))
        vmax = emax * 1.05
        vmin = -vmax

        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax

        # Static elements
        ax.set_xlabel(r"$x/R$" if normalize else r"$x$")
        ax.set_ylabel(r"$y/R$" if normalize else r"$y$")
        ax.set_aspect("equal")
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys[0], ys[-1])

        # Initial permittivity map (frame 0)
        eps_map = np.where(mask_inside, eps_time[0].real, self.eps_ext)

        # Initial field pcolormesh frame (frame 0)
        pcm = ax.pcolormesh(
            xs,
            ys,
            np.real(E[:, :, iomega, 0]),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        eps_cax = ax.pcolormesh(
            xs,
            ys,
            eps_map,
            cmap="Grays",
            vmin=eps_min,
            vmax=eps_max,
            shading="auto",
            alpha=0.2,
        )

        circle = self.plot(normalize=normalize, ax=ax)
        ax.add_patch(circle)

        # Field colorbar
        cbar_field = fig.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
        field_name = "E_z" if polarization == "TM" else "H_z"
        cbar_field.ax.set_title(rf"Re ${field_name}$")

        # Permittivity colorbar
        cbar_eps = fig.colorbar(eps_cax, ax=ax, shrink=0.8, pad=0.08)
        cbar_eps.ax.set_title(r"$\varepsilon(t)$")

        ax.set_title(rf"{polarization}, {field_type} field")

        # Time annotation
        title = ax.text(
            0.02,
            0.96,
            rf"$t = {t[0] / T:.2f}\,T$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "alpha": 0.6,  # transparency (0 = fully transparent, 1 = opaque)
                "edgecolor": "none",  # no border
                "boxstyle": "round,pad=0.2",
            },
        )

        def animate(it):  # pragma: no cover
            pcm.set_array(np.real(E[:, :, iomega, it]).ravel())
            eps_map = np.where(mask_inside, eps_time[it].real, self.eps_ext)
            eps_cax.set_array(eps_map.ravel())
            title.set_text(rf"$t = {t[it] / T:.2f}\,T$")
            return (pcm, eps_cax, title)

        return animation.FuncAnimation(
            fig, animate, blit=blit, repeat=True, frames=len(t), interval=interval
        )

    def plot_mode(
        self,
        x,
        y,
        fields,
        ax=None,
        time_index=0,
        freq_index=0,
        field_type="scattered",
        plot_type="cartesian",
        polarization="TM",
        normalize=True,
        show_cylinder=True,
        cmap=None,
        figsize=(10, 5),
    ):
        """Plot a QNM or scattered field mode on a 2D grid.

        Creates a two-panel plot showing either (Re, Im) components for
        cartesian representation or (Amplitude, Phase) for polar representation.

        Parameters
        ----------
        x : array_like, shape (nx,)
            x-coordinates of the evaluation grid.
        y : array_like, shape (ny,)
            y-coordinates of the evaluation grid.
        fields : dict or ndarray
            Field data to plot. If dict, should have keys like 'scattered',
            'total', 'incident'. If ndarray, should have shape
            (ny, nx, nw, nt) or (ny, nx).
        ax : array_like of Axes, optional
            Two matplotlib axes to plot on. If None, creates new figure.
        time_index : int, optional
            Time index to plot (for time-dependent fields). Default is 0.
        freq_index : int, optional
            Frequency index to plot (for multi-frequency fields). Default is 0.
        field_type : str, optional
            Which field component to plot if fields is a dict.
            Default is 'scattered'.
        plot_type : {'cartesian', 'polar'}, optional
            Type of plot: 'cartesian' shows Re/Im, 'polar' shows Amplitude/Phase.
            Default is 'cartesian'.
        polarization : {'TM', 'TE'}, optional
            Polarization type for axis labels. Default is 'TM'.
        normalize : bool, optional
            If True, normalize coordinates by cylinder radius. Default is True.
        show_cylinder : bool, optional
            If True, overlay cylinder boundary. Default is True.
        cmap : list of str, optional
            Colormaps for the two panels. Default depends on plot_type.
        figsize : tuple, optional
            Figure size if ax is None. Default is (10, 5).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object (if created).
        ax : array of Axes
            The two axes objects.

        Examples
        --------
        >>> fields = cyl.get_mode(x, y, omega_n, phi_n, eigenvalues, modes,
        ...                       n, "TM", t, n_range)
        >>> fig, ax = cyl.plot_mode(x, y, fields, plot_type="polar")
        """
        import matplotlib.pyplot as plt

        # Validate plot_type
        if plot_type not in ["cartesian", "polar"]:
            msg = "plot_type must be either 'cartesian' or 'polar'"
            raise ValueError(msg)

        # Extract field data
        if isinstance(fields, dict):
            field_data = fields[field_type]
        else:
            field_data = fields

        # Handle different field shapes
        field_data = np.asarray(field_data)
        ndim = field_data.ndim

        # Extract the correct slice based on dimensions
        if ndim == 4:  # (ny, nx, nw, nt)
            field = field_data[:, :, freq_index, time_index]
        elif ndim == 3:  # (ny, nx, nw) or (ny, nx, nt)
            if field_data.shape[2] > 10:  # Assume time dimension
                field = field_data[:, :, time_index]
            else:  # Assume frequency dimension
                field = field_data[:, :, freq_index]
        elif ndim == 2:  # (ny, nx)
            field = field_data
        else:
            msg = f"Unsupported field dimensions: {ndim}"
            raise ValueError(msg)

        # Set up axes
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=figsize)
        else:
            ax = np.atleast_1d(ax)
            if len(ax) < 2:
                msg = "ax must contain at least 2 axes"
                raise ValueError(msg)
            fig = ax[0].figure

        # Set component labels and default colormaps
        if plot_type == "cartesian":
            comp = ["Re", "Im"]
            default_cmap = ["RdBu_r", "RdBu_r"]
        else:  # polar
            comp = ["Amplitude", "Phase"]
            default_cmap = ["BuPu", "twilight"]

        if cmap is None:
            cmap = default_cmap

        # Scaling for coordinates
        scale = 1.0 / self.radius if normalize else 1.0
        xs = np.asarray(x) * scale
        ys = np.asarray(y) * scale

        # Field name for labels
        field_name = "E_z" if polarization == "TM" else "H_z"

        # Plot both panels
        for i in range(2):
            # Extract data component
            if plot_type == "polar":
                z = np.abs(field) if i == 0 else np.angle(field)
                label = rf"$|{field_name}|$" if i == 0 else rf"arg$({field_name})$"
            else:
                z = np.real(field) if i == 0 else np.imag(field)
                label = rf"Re ${field_name}$" if i == 0 else rf"Im ${field_name}$"

            # Create pcolormesh
            pcm = ax[i].pcolormesh(
                xs,
                ys,
                z,
                cmap=cmap[i],
                shading="auto",
            )

            # Add colorbar
            cbar = plt.colorbar(pcm, ax=ax[i], shrink=0.8)
            cbar.set_label(label)

            # Add cylinder boundary
            if show_cylinder:
                self.plot(normalize=normalize, ax=ax[i], color="k")

            # Set labels and title
            ax[i].set_xlabel(r"$x/R$" if normalize else r"$x$")
            ax[i].set_ylabel(r"$y/R$" if normalize else r"$y$")
            ax[i].set_title(f"{comp[i]}")
            ax[i].set_aspect("equal")

        plt.tight_layout()
        return fig, ax

    def eigenvalue_static(self, omega0, omega1, n, polarization):
        """Return the static eigenvalues of the cylinder."""
        evs, modes = self.static().eigensolve(
            omega0,
            omega1,
            peak_ref=6,
            recursive=True,
            tol=1e-6,
            plot_solver=False,
            n=n,
            polarization=polarization,
        )
        return evs

    def get_mode(
        self, x, y, omega_n, phi_n, eigenvalues, modes, n, polarization, t, n_range_plt
    ):
        solution_all = []
        rhs_all = []
        inner_coeffs_all = []
        for n_az in n_range_plt:
            incident_field, incident_angles = self.init_incident_field(omega_n)
            rhs = self.build_rhs(omega_n, incident_field, incident_angles, n_az)
            if n_az == n:
                solution = phi_n
                inner_coeffs = self.get_inner_coefficients(
                    omega_n,
                    eigenvalues,
                    modes,
                    n_az,
                    polarization,
                    rhs,
                    solution,
                    check=False,
                )
            elif n_az == -n and n != 0:
                # solution = phi_n.conj().T
                solution = ((-1) ** n) * phi_n
                inner_coeffs = self.get_inner_coefficients(
                    omega_n,
                    eigenvalues,
                    modes,
                    n_az,
                    polarization,
                    rhs,
                    solution,
                    check=False,
                )
            else:
                solution = np.zeros_like(rhs)
                inner_coeffs = np.zeros_like(rhs)
            inner_coeffs_all.append(inner_coeffs)
            solution_all.append(solution)
            rhs_all.append(rhs)

        solution_all = np.array(solution_all)
        inner_coeffs_all = np.array(inner_coeffs_all)
        rhs_all = np.array(rhs_all)

        return self.get_fields(
            x,
            y,
            omega_n,
            solution_all,
            inner_coeffs_all,
            eigenvalues,
            modes,
            incident_field,
            incident_angles,
            n_range_plt,
            t,
            polarization,
        )
