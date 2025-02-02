#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


from .helpers import *
from .eig import nonlinear_eigensolver
import numpy as bk


class Slab:
    def __init__(self, material, thickness, eps_plus=1, eps_minus=1):
        self.material = material
        self.thickness = thickness
        self.eps_plus = eps_plus
        self.eps_minus = eps_minus

    @dimhandler
    def build_matrix(self, omegas, eigenvalues, modes):
        omegas = bk.array(omegas)
        Nh = self.material.Nh
        eigenvalues = eigenvalues.T
        modes = modes.T
        # modes = bk.transpose(modes, (2, 0, 1))

        harm_index = bk.arange(-Nh, Nh + 1)
        harm_index = bk.broadcast_to(harm_index, eigenvalues.shape)

        harm_index = bk.transpose(harm_index)
        omegas_shift = omegas - harm_index * self.material.modulation_frequency
        omegas_shift = bk.transpose(omegas_shift)
        L = self.thickness
        phi_plus = bk.exp(1j * eigenvalues * L)
        phi_minus = bk.exp(-1j * eigenvalues * L)
        ks = bk.broadcast_to(eigenvalues[:, :, bk.newaxis], modes.shape)
        phi_plus = bk.broadcast_to(phi_plus[:, :, bk.newaxis], modes.shape)
        phi_minus = bk.broadcast_to(phi_minus[:, :, bk.newaxis], modes.shape)
        omegas_shift = bk.broadcast_to(omegas_shift[:, :, bk.newaxis], modes.shape)
        ks = bk.transpose(ks, (0, 2, 1))
        phi_plus = bk.transpose(phi_plus, (0, 2, 1))
        phi_minus = bk.transpose(phi_minus, (0, 2, 1))
        modes = bk.transpose(modes, (0, 2, 1))
        matrix_slab = bk.block(
            [
                [
                    (omegas_shift * self.eps_plus**0.5 + ks) * modes,
                    (omegas_shift * self.eps_plus**0.5 - ks) * modes,
                ],
                [
                    (omegas_shift * self.eps_minus**0.5 - ks) * phi_plus * modes,
                    (omegas_shift * self.eps_minus**0.5 + ks) * phi_minus * modes,
                ],
            ]
        )
        matrix_slab = bk.transpose(matrix_slab, (1, 2, 0))
        return matrix_slab

    def build_rhs(self, omegas, Eis):
        omegas = bk.array(omegas)
        Eis = bk.array(Eis)
        rhs_slab = bk.zeros((2 * self.material.nh,) + omegas.shape, dtype=bk.complex128)
        for n in range(self.material.nh):
            nshift = self.material.index_shift(n)
            omegas_shift = omegas - nshift * self.material.modulation_frequency
            rhs_slab[n] = self.eps_plus**0.5 * 2 * Eis[n] * omegas_shift
        return rhs_slab

    def solve(self, matrix_slab, rhs_slab):
        if matrix_slab.ndim == 2:
            return bk.linalg.solve(matrix_slab, rhs_slab)
        sol = bk.empty_like(rhs_slab)
        if matrix_slab.ndim == 3:
            for i in range(matrix_slab.shape[-1]):
                sol[:, i] = bk.linalg.solve(matrix_slab[:, :, i], rhs_slab[:, i])
            return sol
        if matrix_slab.ndim == 4:
            for i in range(matrix_slab.shape[-2]):
                for j in range(matrix_slab.shape[-1]):
                    sol[:, i, j] = bk.linalg.solve(
                        matrix_slab[:, :, i, j], rhs_slab[:, i, j]
                    )
            return sol
        raise ValueError(f"Unsupported number of dimensions: {matrix_slab.ndim}")

    def _extract_coefficients(self, solution, Eis, kns, ens):
        phi_plus = bk.exp(1j * kns * self.thickness)
        phi_minus = bk.exp(-1j * kns * self.thickness)
        nh = self.material.nh
        C = solution[:nh]
        D = solution[nh : 2 * nh]
        Er = ens @ (C + D) - Eis
        Et = ens * phi_plus @ C + ens * phi_minus @ D
        return C, D, Er, Et

    def extract_coefficients(self, solution, Eis, kns, ens):
        if solution.ndim == 1:
            return self._extract_coefficients(solution, Eis, kns, ens)
        C = bk.empty_like(Eis)
        D = bk.empty_like(Eis)
        Er = bk.empty_like(Eis)
        Et = bk.empty_like(Eis)
        if solution.ndim == 2:
            for i in range(solution.shape[-1]):
                C[:, i], D[:, i], Er[:, i], Et[:, i] = self._extract_coefficients(
                    solution[:, i], Eis[:, i], kns[:, i], ens[:, :, i]
                )
            return C, D, Er, Et
        if solution.ndim == 3:
            for i in range(solution.shape[-2]):
                for j in range(solution.shape[-1]):

                    C[:, i, j], D[:, i, j], Er[:, i, j], Et[:, i, j] = (
                        self._extract_coefficients(
                            solution[:, i, j],
                            Eis[:, i, j],
                            kns[:, i, j],
                            ens[:, :, i, j],
                        )
                    )
            return C, D, Er, Et
        raise ValueError(f"Unsupported number of dimensions: {solution.ndim}")

    def fresnel_static(self, omegas):
        eps_slab = self.material.eps_fourier[self.material.Nh]
        return fresnel(omegas, eps_slab, self.eps_plus, self.eps_minus, self.thickness)

    def eigenvalue_static(self, n):
        eps_plus = self.eps_plus
        eps_minus = self.eps_minus
        eps_slab = self.material.eps_fourier[self.material.Nh]
        r12 = (eps_plus**0.5 - eps_slab**0.5) / (eps_plus**0.5 + eps_slab**0.5)
        r23 = (eps_slab**0.5 - eps_minus**0.5) / (eps_slab**0.5 + eps_minus**0.5)
        alpha = -r12 * r23
        return (
            1 / (self.thickness * eps_slab**0.5) * (n * bk.pi + 1j / 2 * bk.log(alpha))
        )

    def eigensolve(self, *args, **kwargs):
        def _build_matrix(omegas):
            eigenvalues, modes = self.material.eigensolve(omegas)
            out = self.build_matrix(omegas, eigenvalues, modes)
            return out

        return nonlinear_eigensolver(_build_matrix, *args, **kwargs)
