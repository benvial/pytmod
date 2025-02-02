#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


# if __name__ == "__main__":
#     import sys

#     eps0 = 5.25
#     deps = 0.85

#     eps_fourier = [
#         -deps / (2 * 1j),
#         eps0,
#         deps / (2 * 1j),
#     ]
#     # eps_fourier = [eps0]
#     Omega = 1  # 5707963267948966

#     from material import Material

#     Npad = int(sys.argv[1])
#     mat = Material(eps_fourier, Omega, Npad=Npad)
#     Omega = mat.modulation_frequency
#     Ln = 2
#     L = Ln / eps0**0.5 / Omega
#     L = Ln / eps0**0.5  # / Omega
#     slab = Slab(mat, L)
#     omegas0D = 0.15
#     omegas1D = bk.array([0.15, 0.15])
#     omegas2D = bk.array([[0.15, 0.15], [0.15, 0.15]])
#     for om in [omegas0D, omegas1D, omegas2D]:
#         om = bk.array(om)

#         kns, ens = mat.eigensolve(om)

#         matrix_slab = slab.build_matrix(om, kns, ens)

#         Eis = bk.ones((slab.material.nh,) + om.shape, dtype=bk.complex128)

#         rhs_slab = slab.build_rhs(om, Eis)
#         solution = slab.solve(matrix_slab, rhs_slab)
#         C, D, Er, Et = slab.extract_coefficients(solution, Eis, kns, ens)

#     # sys.exit(0)
#     import matplotlib.pyplot as plt

#     plt.ion()

#     Nomega = 1000
#     omegas = bk.linspace(0, 10 * Omega, Nomega)
#     EPS = 1e-4
#     omegas = bk.linspace(0 + EPS, 10 * Omega - EPS, Nomega)
#     # omegas = bk.array([0.5])
#     # omegas = omegas2D

#     kns, ens = mat.eigensolve(omegas)

#     Nharmo_plot = 0

#     # # plt.plot(kns.T.real,omegas.real,".",c="#d42323", ms=5, mew=0)

#     # omegas = bk.array([8.5])

#     # rconv = []
#     # tconv = []
#     # for Npad in range(9):
#     #     mat = Material(eps_fourier, Omega, Npad=Npad)
#     #     slab = Slab(mat, L)
#     #     kns, ens = mat.eigensolve(omegas)
#     #     matrix_slab = slab.build_matrix(omegas)
#     #     Eis = bk.zeros((slab.material.nh,) + omegas.shape, dtype=bk.complex128)
#     #     Ei0 = 1
#     #     Eis[mat.Nh] = Ei0
#     #     rhs_slab = slab.build_rhs(omegas, Eis)
#     #     solution = slab.solve(matrix_slab, rhs_slab)
#     #     C, D, Er, Et = slab.extract_coefficients(solution, Eis, kns, ens)
#     #     rn = Er / Ei0
#     #     tn = Et / Ei0
#     #     _r = rn[mat.Nh, 0]
#     #     _t = tn[mat.Nh, 0]
#     #     print(f"{Npad}  |  {_r:.9f}   |   {_t:.9f}")
#     #     rconv.append(_r)
#     #     tconv.append(_t)
#     # rconv = bk.array(rconv)
#     # tconv = bk.array(tconv)
#     # plt.plot(tconv.imag)

#     mat = Material(eps_fourier, Omega, Npad=Npad)
#     slab = Slab(mat, L)
#     kns, ens = mat.eigensolve(omegas)
#     matrix_slab = slab.build_matrix(omegas, kns, ens)
#     Eis = bk.zeros((slab.material.nh,) + omegas.shape, dtype=bk.complex128)
#     Ei0 = 1
#     Eis[mat.Nh] = Ei0
#     rhs_slab = slab.build_rhs(omegas, Eis)
#     solution = slab.solve(matrix_slab, rhs_slab)
#     C, D, Er, Et = slab.extract_coefficients(solution, Eis, kns, ens)
#     rn = Er / Ei0
#     tn = Et / Ei0

#     plt.figure()
#     imode = mat.Nh + Nharmo_plot

#     log = False
#     r_ = bk.abs(rn[imode])
#     t_ = bk.abs(tn[imode])
#     if log:
#         r_ = bk.log10(r_)
#         t_ = bk.log10(t_)

#     plt.plot(omegas, r_, "-", c="#e49649", label=rf"$r$")
#     plt.plot(omegas, t_, "-", c="#5000ca", label=rf"$t$")
#     plt.title(rf"harmonic ${{{Nharmo_plot}}}$")
#     plt.legend()
#     plt.pause(0.1)

#     omega0 = 0.12 - 2.1 * 1j
#     omega1 = 8.1 - 1e-12 * 1j

#     tol = 1e-12
#     peak_ref = 4
#     max_iter = 15

#     nc = 201

#     omegasr = bk.linspace(omega0.real, omega1.real, nc)
#     omegasi = bk.linspace(omega0.imag, omega1.imag, nc)

#     re, im = bk.meshgrid(omegasr, omegasi)
#     omegas = re + 1j * im

#     kns, ens = mat.eigensolve(omegas)
#     matrix_slab_c = slab.build_matrix(omegas, kns, ens)

#     matrix_slab_c = bk.transpose(matrix_slab_c, (2, 3, 0, 1))

#     # D = bk.min(bk.linalg.eigvals(matrix_slab_c), axis=-1)
#     D = bk.linalg.det(matrix_slab_c)
#     plt.figure()
#     plt.pcolormesh(omegasr, omegasi, bk.log10(bk.abs(D)), cmap="inferno")
#     plt.colorbar()
#     plt.title("Det slab")
#     for i in range(0, 10):
#         eigenvalue_static = slab.eigenvalue_static(i)
#         plt.plot(eigenvalue_static.real, eigenvalue_static.imag, "xg")

#     plt.xlim(omegasr[0], omegasr[-1])
#     plt.ylim(omegasi[0], omegasi[-1])
#     plt.pause(0.1)
