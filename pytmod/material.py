#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


from .helpers import *
import numpy as bk


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
        self._eps_fourier = bk.array(eps_fourier)
        self.modulation_frequency = modulation_frequency
        self._Npad = Npad

        if self.nh % 2 == 0:
            raise ValueError("The length of eps_fourier must be odd")

    @property
    def eps_fourier(self):
        if self._Npad > 0:
            return pad(self._eps_fourier, self._Npad)
        return self._eps_fourier

    @eps_fourier.setter
    def eps_fourier(self, coeff):
        self._eps_fourier = coeff

    @property
    def Npad(self):
        return self._Npad

    @Npad.setter
    def Npad(self, N):
        self._Npad = N
        if self._Npad < 0:
            raise ValueError("Npad must be >0")

    @property
    def modulation_period(self):
        return 2 * bk.pi / self.modulation_frequency

    @property
    def nh(self):
        return len(self.eps_fourier)

    @property
    def Nh(self):
        return int((self.nh - 1) / 2)

    def index_shift(self, i):
        return i - self.Nh

    def gamma(self, m, omega):
        return (omega - self.modulation_frequency * m) ** 2

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
        omegas = bk.array(omegas)
        # integ = bk.where(bk.int32(omegas.real / Omega) == omegas.real / Omega)
        # omegas[integ] += 1e-12
        nh = self.nh
        matrix = bk.zeros((nh, nh) + omegas.shape, dtype=bk.complex128)
        for m in range(nh):
            mshift = self.index_shift(m)
            for n in range(nh):
                dmn = m - n
                coeff = (
                    0
                    if abs(dmn) > self.Nh
                    else self.gamma(mshift, omegas) * self.eps_fourier[dmn + self.Nh]
                )
                matrix[m, n] = coeff
        return matrix

    def eigensolve(self, omegas, matrix=None):
        """
        Solve the eigenvalue problem given by the matrix.

        Parameters
        ----------
        omegas : array_like
            The frequencies at which to solve the system.
        matrix : array_like, optional
            The matrix of the linear system. If None, it will be built.

        Returns
        -------
        eigenvalues : array_like
            The eigenvalues of the system.
        modes : array_like
            The eigenvectors of the system.
        """
        omegas = bk.array(omegas)
        if matrix is None:
            matrix = self.build_matrix(omegas)
        mat = move_first_two_axes_to_end(matrix)
        k2, modes = bk.linalg.eig(mat)
        eigenvalues = (k2 + 0j) ** 0.5
        eigenvalues = move_last_axes_to_beginning(eigenvalues)
        modes = move_last_two_axes_to_beginning(modes)
        return eigenvalues, modes
