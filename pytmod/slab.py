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
            rhs_slab[n] = (self.eps_plus**0.5 + 1) * Eis[n] * omegas_shift
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
        return fresnel(omegas, eps_slab, self.eps_plus, self.eps_minus)

    def eigenvalue_static(self, n):
        eps_slab = self.material.eps_fourier[self.material.Nh]

        alpha = (eps_slab**0.5 + self.eps_plus) / (eps0**0.5 - self.eps_minus)

        return 1 / (self.thickness * eps_slab**0.5) * (n * bk.pi - 1j * bk.log(alpha))


if __name__ == "__main__":
    import sys

    eps0 = 5.25
    deps = 0.85

    eps_fourier = [
        -deps / (2 * 1j),
        eps0,
        deps / (2 * 1j),
    ]
    # eps_fourier = [eps0]
    Omega = 1  # 5707963267948966

    from material import Material

    Npad = int(sys.argv[1])
    mat = Material(eps_fourier, Omega, Npad=Npad)
    Omega = mat.modulation_frequency
    Ln = 2
    L = Ln / eps0**0.5 / Omega
    L = Ln / eps0**0.5  # / Omega
    slab = Slab(mat, L)
    omegas0D = 0.15
    omegas1D = bk.array([0.15, 0.15])
    omegas2D = bk.array([[0.15, 0.15], [0.15, 0.15]])
    for om in [omegas0D, omegas1D, omegas2D]:
        om = bk.array(om)

        kns, ens = mat.eigensolve(om)

        matrix_slab = slab.build_matrix(om, kns, ens)

        Eis = bk.ones((slab.material.nh,) + om.shape, dtype=bk.complex128)

        rhs_slab = slab.build_rhs(om, Eis)
        solution = slab.solve(matrix_slab, rhs_slab)
        C, D, Er, Et = slab.extract_coefficients(solution, Eis, kns, ens)

    # sys.exit(0)
    import matplotlib.pyplot as plt

    plt.ion()

    Nomega = 1000
    omegas = bk.linspace(0, 10 * Omega, Nomega)
    EPS = 1e-4
    omegas = bk.linspace(0 + EPS, 10 * Omega - EPS, Nomega)
    # omegas = bk.array([0.5])
    # omegas = omegas2D

    kns, ens = mat.eigensolve(omegas)

    Nharmo_plot = 0

    # # plt.plot(kns.T.real,omegas.real,".",c="#d42323", ms=5, mew=0)

    # omegas = bk.array([8.5])

    # rconv = []
    # tconv = []
    # for Npad in range(9):
    #     mat = Material(eps_fourier, Omega, Npad=Npad)
    #     slab = Slab(mat, L)
    #     kns, ens = mat.eigensolve(omegas)
    #     matrix_slab = slab.build_matrix(omegas)
    #     Eis = bk.zeros((slab.material.nh,) + omegas.shape, dtype=bk.complex128)
    #     Ei0 = 1
    #     Eis[mat.Nh] = Ei0
    #     rhs_slab = slab.build_rhs(omegas, Eis)
    #     solution = slab.solve(matrix_slab, rhs_slab)
    #     C, D, Er, Et = slab.extract_coefficients(solution, Eis, kns, ens)
    #     rn = Er / Ei0
    #     tn = Et / Ei0
    #     _r = rn[mat.Nh, 0]
    #     _t = tn[mat.Nh, 0]
    #     print(f"{Npad}  |  {_r:.9f}   |   {_t:.9f}")
    #     rconv.append(_r)
    #     tconv.append(_t)
    # rconv = bk.array(rconv)
    # tconv = bk.array(tconv)
    # plt.plot(tconv.imag)

    mat = Material(eps_fourier, Omega, Npad=Npad)
    slab = Slab(mat, L)
    kns, ens = mat.eigensolve(omegas)
    matrix_slab = slab.build_matrix(omegas, kns, ens)
    Eis = bk.zeros((slab.material.nh,) + omegas.shape, dtype=bk.complex128)
    Ei0 = 1
    Eis[mat.Nh] = Ei0
    rhs_slab = slab.build_rhs(omegas, Eis)
    solution = slab.solve(matrix_slab, rhs_slab)
    C, D, Er, Et = slab.extract_coefficients(solution, Eis, kns, ens)
    rn = Er / Ei0
    tn = Et / Ei0

    plt.figure()
    imode = mat.Nh + Nharmo_plot

    log = False
    r_ = bk.abs(rn[imode])
    t_ = bk.abs(tn[imode])
    if log:
        r_ = bk.log10(r_)
        t_ = bk.log10(t_)

    plt.plot(omegas, r_, "-", c="#e49649", label=rf"$r$")
    plt.plot(omegas, t_, "-", c="#5000ca", label=rf"$t$")
    plt.title(rf"harmonic ${{{Nharmo_plot}}}$")
    plt.legend()
    plt.pause(0.1)

    omega0 = 0.12 - 2.1 * 1j
    omega1 = 8.1 - 1e-12 * 1j

    tol = 1e-12
    peak_ref = 4
    max_iter = 15

    nc = 201

    omegasr = bk.linspace(omega0.real, omega1.real, nc)
    omegasi = bk.linspace(omega0.imag, omega1.imag, nc)

    re, im = bk.meshgrid(omegasr, omegasi)
    omegas = re + 1j * im

    kns, ens = mat.eigensolve(omegas)
    matrix_slab_c = slab.build_matrix(omegas, kns, ens)

    matrix_slab_c = bk.transpose(matrix_slab_c, (2, 3, 0, 1))

    # D = bk.min(bk.linalg.eigvals(matrix_slab_c), axis=-1)
    D = bk.linalg.det(matrix_slab_c)
    plt.figure()
    plt.pcolormesh(omegasr, omegasi, bk.log10(bk.abs(D)), cmap="inferno")
    plt.colorbar()
    plt.title("Det slab")
    for i in range(0, 10):
        eigenvalue_static = slab.eigenvalue_static(i)
        plt.plot(eigenvalue_static.real, eigenvalue_static.imag, "xg")

    plt.xlim(omegasr[0], omegasr[-1])
    plt.ylim(omegasi[0], omegasi[-1])
    plt.pause(0.1)
