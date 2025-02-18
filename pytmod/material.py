# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod
from __future__ import annotations

import numpy as np

from .helpers import (
    build_field_time,
    dot,
    index_shift,
    matvecprod,
    move_first_two_axes_to_end,
    move_last_axes_to_beginning,
    move_last_two_axes_to_beginning,
    normalize_modes,
    pad,
)


class Material:
    """
    Material object

    Parameters
    ----------
    eps_fourier : array_like
        The Fourier coefficients of the dielectric function
    modulation_frequency : float
        The modulation frequency of the dielectric function
    Npad : int, optional
        The number of components to pad the dielectric function with

    Raises
    ------
    ValueError
        If the length of `eps_fourier` is even
    """

    def __init__(self, eps_fourier, modulation_frequency, Npad=0):
        self._eps_fourier = np.array(eps_fourier)

        #: Syntax also works for class variables.
        self.modulation_frequency = modulation_frequency
        self._Npad = Npad
        if self.nh % 2 == 0:
            msg = "The length of eps_fourier must be odd"
            raise ValueError(msg)

    def __repr__(self):
        return (
            f"Material(modulation_frequency={self.modulation_frequency}, "
            f"eps_fourier={self.eps_fourier})"
        )

    def __str__(self):
        return self.__repr__()

    def pad(self, x):
        """
        Pad an array with zeros if `Npad` is positive

        Parameters
        ----------
        x : array_like
            The array to pad

        Returns
        -------
        y : array_like
            The padded array
        """
        if self.Npad > 0:
            return pad(x, self._Npad)
        return x

    @property
    def eps_fourier(self):
        """
        The Fourier coefficients of the dielectric function

        Returns
        -------
        eps_fourier : array_like
            The Fourier coefficients of the dielectric function
        """
        return self.pad(self._eps_fourier)

    @eps_fourier.setter
    def eps_fourier(self, coeff):
        """
        Set the Fourier coefficients of the dielectric function

        Parameters
        ----------
        coeff : array_like
            The Fourier coefficients of the dielectric function

        Returns
        -------
        None
        """
        self._eps_fourier = coeff

    @property
    def Npad(self):
        """
        The number of zeros to pad the Fourier coefficients with

        Returns
        -------
        Npad : int
            The number of zeros to pad the Fourier coefficients with
        """
        if not isinstance(self._Npad, int) or self._Npad < 0:
            msg = "Npad must be a positive integer"
            raise ValueError(msg)
        return self._Npad

    @Npad.setter
    def Npad(self, N):
        """
        Set the number of zeros to pad the Fourier coefficients with

        Parameters
        ----------
        N : int
            The number of zeros to pad the Fourier coefficients with

        Returns
        -------
        None
        """
        self._Npad = N

    @property
    def modulation_period(self):
        """
        The modulation period of the dielectric function

        Returns
        -------
        modulation_period : float
            The modulation period of the dielectric function
        """
        return 2 * np.pi / self.modulation_frequency

    @property
    def nh(self):
        """
        The length of the Fourier coefficients array

        Returns
        -------
        nh : int
            The length of the Fourier coefficients array
        """
        return len(self.eps_fourier)

    @property
    def Nh(self):
        """
        The integer corresponding to order 0 in the Fourier coefficients array

        Returns
        -------
        Nh : int
            The integer corresponding to order 0 in the Fourier coefficients array
        """
        return int((self.nh - 1) / 2)

    def index_shift(self, i):
        """
        Shift an index to the index of the Fourier coefficient of the same order
        in the padded array.

        Parameters
        ----------
        i : int
            The index in the unpadded array

        Returns
        -------
        int
            The corresponding index in the padded array
        """
        return index_shift(i, self.Nh)

    def _gamma(self, m, omega):
        return (omega - self.modulation_frequency * m) ** 2

    def _dgamma_domega(self, m, omega):
        return 2 * (omega - self.modulation_frequency * m)

    def build_matrix(self, omegas):
        """
        Build the matrix of the linear system to be solved.

        Parameters
        ----------
        omegas : array_like
            The frequencies at which to solve the system.

        Returns
        -------
        matrix : array_like
            The matrix of the linear system.
        """
        omegas = np.array(omegas)
        # integ = bk.where(bk.int32(omegas.real / Omega) == omegas.real / Omega)
        # omegas[integ] += 1e-12
        nh = self.nh
        matrix = np.zeros((nh, nh, *omegas.shape), dtype=np.complex128)
        for m in range(nh):
            mshift = self.index_shift(m)
            for n in range(nh):
                dmn = m - n
                coeff = (
                    0
                    if abs(dmn) > self.Nh
                    else self._gamma(mshift, omegas) * self.eps_fourier[dmn + self.Nh]
                )
                matrix[m, n] = coeff
        return matrix

    def build_dmatrix_domega(self, omegas):
        """
        Build the matrix derivative wrt omega of the linear system to be solved.

        Parameters
        ----------
        omegas : array_like
            The frequencies at which to solve the system.

        Returns
        -------
        dmatrix : array_like
            The matrix matrix derivative wrt omega.
        """
        omegas = np.array(omegas)
        # integ = bk.where(bk.int32(omegas.real / Omega) == omegas.real / Omega)
        # omegas[integ] += 1e-12
        nh = self.nh
        dmatrix = np.zeros((nh, nh, *omegas.shape), dtype=np.complex128)
        for m in range(nh):
            mshift = self.index_shift(m)
            for n in range(nh):
                dmn = m - n
                coeff = (
                    0
                    if abs(dmn) > self.Nh
                    else self._dgamma_domega(mshift, omegas)
                    * self.eps_fourier[dmn + self.Nh]
                )
                dmatrix[m, n] = coeff
        return dmatrix

    def eigensolve(self, omegas, matrix=None, left=False, normalize=True, sort=True):
        """
        Solve the eigenvalue problem for the material.

        Parameters
        ----------
        omegas : array_like
            The frequencies at which to solve the system.
        matrix : array_like, optional
            The matrix of the linear system. If None, it will be built.
        left : bool, optional
            Whether to compute the left eigenvectors. Defaults to False.
        normalize : bool, optional
            Whether to normalize the left and right eigenvectors. Defaults to False.
        sort : bool, optional
            Whether to sort the eigenvalues. Defaults to True.

        Returns
        -------
        eigenvalues : array_like
            The eigenvalues of the material.
        modes_right : array_like
            The right eigenvectors of the material.
        modes_left : array_like
            The left eigenvectors of the material, if left is True.
        """
        left0 = left
        if normalize:
            left = True
        omegas = np.array(omegas)
        if matrix is None:
            matrix = self.build_matrix(omegas)
        mat = move_first_two_axes_to_end(matrix)
        if left:
            matH = np.swapaxes(mat, -1, -2).conj()
            k2h, modes_left = np.linalg.eig(matH)
            modes_left = modes_left.conj()
        k2, modes_right = np.linalg.eig(mat)
        eigenvalues = (k2 + 0j) ** 0.5

        if sort:
            sort_indices = np.argsort(eigenvalues.real, axis=-1)

            # Sort eigenvalues
            eigenvalues = np.take_along_axis(eigenvalues, sort_indices, axis=-1)

            # Sort eigenvectors accordingly
            sorted_indices_expanded = np.expand_dims(
                sort_indices, axis=-2
            )  # Shape: (..., 1, N)
            modes_right = np.take_along_axis(
                modes_right, sorted_indices_expanded, axis=-1
            )

        eigenvalues = move_last_axes_to_beginning(eigenvalues)
        modes_right = move_last_two_axes_to_beginning(modes_right)

        if left:
            if sort:
                modes_left = np.take_along_axis(
                    modes_left, sorted_indices_expanded, axis=-1
                )
            modes_left = move_last_two_axes_to_beginning(modes_left)
            if normalize:
                modes_right, modes_left = self.normalize(modes_right, modes_left)
            if left0:
                return eigenvalues, modes_right, modes_left

            return eigenvalues, modes_right

        return eigenvalues, modes_right

    def get_modes_normalization(self, modes_right, modes_left):
        """
        Compute the normalization constants for the modes.

        Parameters
        ----------
        modes_right : array_like
            The right eigenvectors of the material.
        modes_left : array_like
            The left eigenvectors of the material.

        Returns
        -------
        normas : array_like
            The normalization constants.
        """
        return (
            np.array([dot(modes_left[:, i], modes_right[:, i]) for i in range(self.nh)])
            ** 0.5
        )

    def normalize(self, modes_right, modes_left):
        """
        Normalize the eigenmodes of the material.

        Parameters
        ----------
        modes_right : array_like
            The right eigenvectors of the material.
        modes_left : array_like
            The left eigenvectors of the material.

        Returns
        -------
        modes_right : array_like
            The normalized right eigenvectors of the material.
        modes_left : array_like
            The normalized left eigenvectors of the material.

        Notes
        -----
        First, the eigenmodes are normalized so that the left and right
        eigenmodes are biorthogonal. Then, the right eigenmodes are
        normalized so that the maximum value of each eigenmode is 1.
        """
        normas = self.get_modes_normalization(modes_right, modes_left)
        return normalize_modes(normas, modes_right, modes_left)

    def get_deigenvalues_domega(
        self,
        omegas,
        eigenvalues,
        normalized_modes_right,
        normalized_modes_left,
        dmatrix=None,
    ):
        """
        Compute the derivative of the eigenvalues wrt omega.

        Parameters
        ----------
        omegas : array_like
            The frequencies at which to compute the derivative.
        eigenvalues : array_like
            The eigenvalues of the material.
        normalized_modes_right : array_like
            The normalized right eigenvectors of the material.
        normalized_modes_left : array_like
            The normalized left eigenvectors of the material.
        dmatrix : array_like, optional
            The matrix derivative wrt omega. Defaults to None.

        Returns
        -------
        deigenvalues : array_like
            The derivative of the eigenvalues wrt omega.
        """
        if dmatrix is None:
            dmatrix = self.build_dmatrix_domega(omegas)

        return np.array(
            [
                dot(
                    normalized_modes_left[:, i],
                    matvecprod(dmatrix, normalized_modes_right[:, i]),
                )
                / (2 * eigenvalues[i])
                / dot(normalized_modes_left[:, i], normalized_modes_right[:, i])
                for i in range(self.nh)
            ]
        )

    def get_deigenmodes_right_domega(
        self,
        omegas,
        eigenvalues,
        normalized_modes_right,
        normalized_modes_left,
        dmatrix=None,
    ):
        """
        Compute the derivative of the right eigenmodes wrt omega.

        Parameters
        ----------
        omegas : array_like
            The frequencies at which to compute the derivative.
        eigenvalues : array_like
            The eigenvalues of the material.
        normalized_modes_right : array_like
            The normalized right eigenvectors of the material.
        normalized_modes_left : array_like
            The normalized left eigenvectors of the material.
        dmatrix : array_like, optional
            The matrix derivative wrt omega. Defaults to None.

        Returns
        -------
        deigenmodes_right : array_like
            The derivative of the right eigenmodes wrt omega.
        """
        if dmatrix is None:
            dmatrix = self.build_dmatrix_domega(omegas)

        max_indices = np.argmax(np.abs(normalized_modes_right), axis=0)
        modes_right_max_values = np.take_along_axis(
            normalized_modes_right, np.expand_dims(max_indices, axis=1), axis=0
        )
        deigvecs = []
        for i in range(self.nh):
            s = 0
            cjs = np.zeros_like(eigenvalues)
            for j in range(self.nh):
                if i != j:
                    cj = (
                        dot(
                            normalized_modes_left[:, j],
                            matvecprod(dmatrix, normalized_modes_right[:, i]),
                        )
                        / (eigenvalues[i] ** 2 - eigenvalues[j] ** 2)
                        / dot(normalized_modes_left[:, j], normalized_modes_right[:, j])
                    )
                    cjs[j] = cj
                    s += cj * modes_right_max_values[i, j]
            cjs[i] = -s
            dv = 0
            for j in range(self.nh):
                dv += cjs[j] * normalized_modes_right[:, j]
            deigvecs.append(dv)
        return np.swapaxes(np.array(deigvecs), 1, 0)

    def freq2time(self, coeff, t):
        """
        Compute the time-domain representation of a coefficient array.

        Parameters
        ----------
        coeff : array_like
            The coefficient array in the frequency domain
        t : array_like
            The time array at which to compute the time-domain representation

        Returns
        -------
        array_like
            The time-domain representation of the coefficient array
        """
        return build_field_time(coeff, self.modulation_frequency, self.Nh, t)

    def get_eps_time(self, t):
        """
        Compute the time-domain representation of the dielectric function.

        Parameters
        ----------
        t : array_like
            The time array at which to compute the time-domain representation

        Returns
        -------
        array_like
            The time-domain representation of the dielectric function
        """
        return self.freq2time(self.eps_fourier, t)


# def _get_maxis(ur):
#     return bk.argmax(bk.abs(ur), axis=(0))
