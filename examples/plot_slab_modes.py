#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


"""
Slab quasi normal modes
===========================

Solve the slab nonlinear eigenproblem and plot the quasi normal modes

"""


####################################################################################
# Imports and parameters


import matplotlib.pyplot as plt
import pytmod as pm
import numpy as bk
import sys

plt.ion()
plt.close("all")


Omega = 1.3711034416945151
Omega = 1
Npad = 11

eps0 = 5.25
deps = 1

eps_fourier = [
    -deps / (2 * 1j),
    eps0,
    deps / (2 * 1j),
]
# eps_fourier = [eps0]

L = 2
mat = pm.Material(eps_fourier, Omega, Npad)
slab = pm.Slab(mat, L)

omega0 = 0.65 - 0.32j
omega1 = 0.92 - 0.019j

nc = 101

omegasr = bk.linspace(omega0.real, omega1.real, nc)
omegasi = bk.linspace(omega0.imag, omega1.imag, nc)

re, im = bk.meshgrid(omegasr, omegasi)
omegas = re + 1j * im

# for Omega in bk.linspace(0.1, 1.7, 51):
#     Nh = mat.Nh
#     Nh = 50
#     M = 50
#     ms = bk.arange(-M, M+1)
#     eigenvalue_static = slab.eigenvalue_static(ms)
#     static_evs = []
#     for n in range(-Nh, Nh + 1):
#         static_evs.append(eigenvalue_static-n*Omega)
#     static_evs = bk.array(static_evs)
#     plt.plot(static_evs.real, static_evs.imag, ".g")
#     plt.plot(eigenvalue_static.real, eigenvalue_static.imag, "or")
#     plt.axvline(Omega)

#     plt.xlim(omegasr[0], omegasr[-1])
#     plt.ylim(omegasi[0], omegasi[-1])
#     plt.xlabel(r"Re $\omega/\Omega$")
#     plt.ylabel(r"Im $\omega/\Omega$")
#     plt.pause(0.1)
#     plt.clf()


# sys.exit(0)


evs, modes = slab.eigensolve(
    omega0,
    omega1,
    peak_ref=6,
    recursive=True,
    tol=1e-6,
    plot_solver=True,
    # peaks_estimate="det",
)

print("eigenvalues:")
print(evs)
evs = bk.array(evs)
Nevs = len(evs)

plt.figure()


kns, ens = mat.eigensolve(omegas)
matrix_slab_c = slab.build_matrix(omegas, kns, ens)
matrix_slab_c = bk.transpose(matrix_slab_c, (2, 3, 0, 1))

D = bk.linalg.det(matrix_slab_c)
# D = bk.min(bk.abs(bk.linalg.eigvals(matrix_slab_c)), axis=-1)

plt.pcolormesh(omegasr / Omega, omegasi / Omega, bk.log10(bk.abs(D)), cmap="inferno")
plt.colorbar()
plt.title(r"det $M(\omega)$")
for i in range(0, 10):
    eigenvalue_static = slab.eigenvalue_static(i)
    plt.plot(eigenvalue_static.real / Omega, eigenvalue_static.imag / Omega, "xg")


if Nevs != 0:
    plt.plot(evs.real / Omega, evs.imag / Omega, "+w")
plt.xlim(omegasr[0] / Omega, omegasr[-1] / Omega)
plt.ylim(omegasi[0] / Omega, omegasi[-1] / Omega)
plt.xlabel(r"Re $\omega/\Omega$")
plt.ylabel(r"Im $\omega/\Omega$")
plt.pause(0.1)

Nh = mat.Nh
for i in range(-50, 50):
    eigenvalue_static = slab.eigenvalue_static(i)
    for n in range(-Nh, Nh + 1):
        plt.plot(
            eigenvalue_static.real / Omega - n, eigenvalue_static.imag / Omega, "xg"
        )


if Nevs != 0:
    kns_eig, ens_eig = mat.eigensolve(evs)
    matrix_slab_eig = slab.build_matrix(evs, kns_eig, ens_eig)
    matrix_slab_eig = bk.transpose(matrix_slab_eig, (2, 0, 1))
    Deig = bk.linalg.det(matrix_slab_eig)

    print("det(eigenvalues):")
    print(bk.abs(Deig))


####################################################################################
# Get the field


T = mat.modulation_period
t = bk.linspace(0, 3 * T, 300)
Lhom = 6 * L
x = bk.linspace(-Lhom, Lhom + L, 1000)

qnms = []
for imode in range(Nevs):
    omega = evs[imode]
    solution = modes[:, imode]
    kns, ens = mat.eigensolve(omega)
    Eis = slab.init_incident_field(omega)
    psi = slab.extract_coefficients(solution, Eis, kns, ens)
    E = slab.get_scattered_field(x, t, omega, psi, kns, ens)
    qnms.append(E)

####################################################################################
# Plot QNMs

plt.figure()
for imode in range(Nevs):
    mode = qnms[imode][:, 0].real
    mode /= bk.max(bk.abs(mode)) * 2
    plt.plot(x / L - 0.5, 1 * imode + mode.real)
plt.axvline(-0.5, color="#949494", lw=1)
plt.axvline(0.5, color="#949494", lw=1)
plt.xlabel("$x/L$")
plt.ylabel("$E(t=0)$")
plt.tight_layout()
plt.show()


####################################################################################
# Animate the field


anim = slab.animate_field(x, t, qnms[0])


####################################################################################
# Space time map

plt.figure()
plt.pcolormesh(x / L - 0.5, t / T, bk.real(qnms[0].T), cmap="RdBu_r")
plt.axvline(-0.5, color="#949494", lw=1)
plt.axvline(0.5, color="#949494", lw=1)
plt.ylim(0, t[-1] / T)
plt.xlabel("$x/L$")
plt.ylabel("$t/T$")
cb = plt.colorbar()
cb.ax.set_title("Re $E$")
plt.tight_layout()
plt.show()
