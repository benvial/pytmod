#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


"""
Slab
=======================

A simple example.

"""


####################################################################################
# Check results from :cite:t:`zurita-sanchez2009`


import matplotlib.pyplot as plt
import pytmod as pm
import numpy as bk

plt.ion()
plt.close("all")


eps0 = 5.25
Omega = 1
Npad = 5

Nomega = 500
omegas = bk.linspace(0, 1 * Omega, Nomega)


def get_eps_fourier(deps):
    eps_fourier = [
        -deps / (2 * 1j),
        eps0,
        deps / (2 * 1j),
    ]
    return eps_fourier


####################################################################################
# Figure 3

fig3, ax3 = plt.subplots(3, 1, figsize=(4, 9))
for i, deps in enumerate([0.085, 0.85, 3.4]):
    eps_fourier = get_eps_fourier(deps)
    mat = pm.Material(eps_fourier, Omega, Npad)
    kns, ens = mat.eigensolve(omegas)
    ax3[i].plot(
        kns.T.real / eps0**0.5, omegas.real / Omega, ".", c="#5000ca", ms=3, mew=0
    )
    ax3[i].set_xlim(0, 3)
    ax3[i].set_ylim(0, 1)
    ax3[i].set_title(rf"$\Delta \epsilon = {deps}$")

fig3.supxlabel(r"normalized wave vector $kc/(\Omega\sqrt{\epsilon_0)}$")
fig3.supylabel(r"normalized frequency $\omega/\Omega$")
fig3.tight_layout()

####################################################################################
# Figures 4 and 5

Nomega = 1500
omegas = bk.linspace(0.0001, 10 + 0.00001 * Omega, Nomega)

Nharmo_plot = 0

fig4, ax4 = plt.subplots(2, 2)
fig5, ax5 = plt.subplots(2, 2)

for i, Ln in enumerate([0.5, 2]):
    for j, deps in enumerate([0.085, 0.85]):
        L = Ln / eps0**0.5 / Omega
        eps_fourier = get_eps_fourier(deps)
        mat = pm.Material(eps_fourier, Omega, Npad)
        kns, ens = mat.eigensolve(omegas)
        slab = pm.Slab(mat, L)
        matrix_slab = slab.build_matrix(omegas, kns, ens)
        Eis = bk.zeros((slab.material.nh,) + omegas.shape, dtype=bk.complex128)
        Ei0 = 1
        Eis[mat.Nh] = Ei0
        rhs_slab = slab.build_rhs(omegas, Eis)
        solution = slab.solve(matrix_slab, rhs_slab)
        C, D, Er, Et = slab.extract_coefficients(solution, Eis, kns, ens)
        rn = Er / Ei0
        tn = Et / Ei0

        imode = mat.Nh + Nharmo_plot
        r_ = bk.abs(rn[imode])
        t_ = bk.abs(tn[imode])
        ax4[i, j].plot(omegas, t_, "-", c="#5000ca", label=rf"$t$")
        ax4[i, j].plot(omegas, r_, "-", c="#e49649", label=rf"$r$")
        ax4[i, j].set_title(rf"$L_N = {Ln}, $n=${Nharmo_plot}$")
        ax4[i, j].set_ylim(0, 1)
        ax4[i, j].set_xlim(0, 10)

        r_ = bk.angle(rn[imode]) / bk.pi
        t_ = bk.angle(tn[imode]) / bk.pi
        ax5[i, j].plot(omegas, t_, "-", c="#5000ca", label=rf"$t$")
        ax5[i, j].plot(omegas, r_, "-", c="#e49649", label=rf"$r$")
        ax5[i, j].set_title(rf"$L_N = {Ln}, $n=${Nharmo_plot}$")
        ax5[i, j].set_ylim(-1, 1)
        ax5[i, j].set_xlim(0, 10)

ax4[0, 1].legend()
ax5[0, 1].legend()
fig4.supxlabel(r"normalized frequency $\omega_0/\Omega$")
fig4.supylabel(rf"magnitude")
fig5.supxlabel(r"normalized frequency $\omega_0/\Omega$")
fig5.supylabel(rf"phase")
fig4.tight_layout()
fig5.tight_layout()


####################################################################################
# Figures 6 and 7


fig6, ax6 = plt.subplots(2, 2)
fig7, ax7 = plt.subplots(2, 2)


deps = 0.085
for i, Ln in enumerate([0.5, 8]):
    L = Ln / eps0**0.5 / Omega
    eps_fourier = get_eps_fourier(deps)
    mat = pm.Material(eps_fourier, Omega, Npad)
    kns, ens = mat.eigensolve(omegas)
    slab = pm.Slab(mat, L)
    matrix_slab = slab.build_matrix(omegas, kns, ens)
    Eis = bk.zeros((slab.material.nh,) + omegas.shape, dtype=bk.complex128)
    Ei0 = 1
    Eis[mat.Nh] = Ei0
    rhs_slab = slab.build_rhs(omegas, Eis)
    solution = slab.solve(matrix_slab, rhs_slab)
    C, D, Er, Et = slab.extract_coefficients(solution, Eis, kns, ens)
    rn = Er / Ei0
    tn = Et / Ei0
    for j, Nharmo_plot in enumerate([1, -1]):
        imode = mat.Nh + Nharmo_plot
        r_ = bk.abs(rn[imode])
        t_ = bk.abs(tn[imode])
        ax6[i, j].plot(omegas, t_, "-", c="#5000ca", label=rf"$t$")
        ax6[i, j].plot(omegas, r_, "-", c="#e49649", label=rf"$r$")
        ax6[i, j].set_title(rf"$L_N = {Ln}, $n=${Nharmo_plot}$")
        ax6[i, j].set_ylim(0)
        ax6[i, j].set_xlim(0, 10)

        r_ = bk.angle(rn[imode]) / bk.pi
        t_ = bk.angle(tn[imode]) / bk.pi
        ax7[i, j].plot(omegas, t_, "-", c="#5000ca", label=rf"$t$")
        ax7[i, j].plot(omegas, r_, "-", c="#e49649", label=rf"$r$")
        ax7[i, j].set_title(rf"$L_N = {Ln}, $n=${Nharmo_plot}$")
        ax7[i, j].set_ylim(-1, 1)
        xmax = 2 if Ln == 8 else 10
        ax7[i, j].set_xlim(0, xmax)
        if i != 1:
            ax6[i, j].set_xticklabels([])
            ax7[i, j].set_xticklabels([])

ax6[0, 1].legend()
ax7[0, 1].legend()
fig6.supxlabel(r"normalized frequency $\omega_0/\Omega$")
fig6.supylabel(rf"magnitude")
fig7.supxlabel(r"normalized frequency $\omega_0/\Omega$")
fig7.supylabel(rf"phase")
fig6.tight_layout()
fig7.tight_layout()

####################################################################################
# Figures 8 and 9

fig8, ax8 = plt.subplots(4, 2, figsize=(6, 9))
fig9, ax9 = plt.subplots(4, 2, figsize=(6, 9))


deps = 0.85
for j, Ln in enumerate([0.5, 8]):
    L = Ln / eps0**0.5 / Omega
    eps_fourier = get_eps_fourier(deps)
    mat = pm.Material(eps_fourier, Omega, Npad)
    kns, ens = mat.eigensolve(omegas)
    slab = pm.Slab(mat, L)
    matrix_slab = slab.build_matrix(omegas, kns, ens)
    Eis = bk.zeros((slab.material.nh,) + omegas.shape, dtype=bk.complex128)
    Ei0 = 1
    Eis[mat.Nh] = Ei0
    rhs_slab = slab.build_rhs(omegas, Eis)
    solution = slab.solve(matrix_slab, rhs_slab)
    C, D, Er, Et = slab.extract_coefficients(solution, Eis, kns, ens)
    rn = Er / Ei0
    tn = Et / Ei0
    for i, Nharmo_plot in enumerate([1, -1, 2, -2]):
        imode = mat.Nh + Nharmo_plot
        r_ = bk.abs(rn[imode])
        t_ = bk.abs(tn[imode])
        ax8[i, j].plot(omegas, t_, "-", c="#5000ca", label=rf"$t$")
        ax8[i, j].plot(omegas, r_, "-", c="#e49649", label=rf"$r$")
        ax8[i, j].set_title(rf"$L_N = {Ln}, n = {Nharmo_plot}$")
        ax8[i, j].set_ylim(0)
        ax8[i, j].set_xlim(0, 10)

        r_ = bk.angle(rn[imode]) / bk.pi
        t_ = bk.angle(tn[imode]) / bk.pi
        ax9[i, j].plot(omegas, t_, "-", c="#5000ca", label=rf"$t$")
        ax9[i, j].plot(omegas, r_, "-", c="#e49649", label=rf"$r$")
        ax9[i, j].set_title(rf"$L_N = {Ln}, n = {Nharmo_plot}$")
        ax9[i, j].set_ylim(-1, 1)
        xmax = 2 if Ln == 8 else 10
        ax9[i, j].set_xlim(0, xmax)
        if (i, j) != (3, 0) and (i, j) != (3, 1):
            ax8[i, j].set_xticklabels([])
            ax9[i, j].set_xticklabels([])

ax8[1, 0].legend()
ax9[1, 0].legend()
fig8.supxlabel(r"normalized frequency $\omega_0/\Omega$")
fig8.supylabel(rf"magnitude")
fig9.supxlabel(r"normalized frequency $\omega_0/\Omega$")
fig9.supylabel(rf"phase")

fig8.tight_layout()
fig9.tight_layout()
