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
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pytmod as pm

plt.ion()
plt.close("all")


Omega = 1
Npad = 6

eps0 = 5.25
deps = 1

eps_fourier = [
    -deps / (2 * 1j),
    eps0,
    deps / (2 * 1j),
]

L = 2
mat = pm.Material(eps_fourier, Omega, Npad)
slab = pm.Slab(mat, L)

omega0 = 0.65 - 0.32j
omega1 = 0.92 - 0.019j

nc = 101

omegasr = np.linspace(omega0.real, omega1.real, nc)
omegasi = np.linspace(omega0.imag, omega1.imag, nc)

re, im = np.meshgrid(omegasr, omegasi)
omegas = re + 1j * im


evs, modes = slab.eigensolve(
    omega0,
    omega1,
    peak_ref=6,
    recursive=True,
    tol=1e-6,
    plot_solver=True,
    # peaks_estimate="det",
)

evs = np.array(evs)
Nevs = len(evs)

plt.figure()


kns, ens = mat.eigensolve(omegas)
matrix_slab_c = slab.build_matrix(omegas, kns, ens)
matrix_slab_c = np.transpose(matrix_slab_c, (2, 3, 0, 1))

D = np.linalg.det(matrix_slab_c)
# D = bk.min(bk.abs(bk.linalg.eigvals(matrix_slab_c)), axis=-1)

plt.pcolormesh(omegasr / Omega, omegasi / Omega, np.log10(np.abs(D)), cmap="inferno")
plt.colorbar()
plt.title(r"det $M(\omega)$")
for i in range(10):
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
    matrix_slab_eig = np.transpose(matrix_slab_eig, (2, 0, 1))
    Deig = np.linalg.det(matrix_slab_eig)


####################################################################################
# Get the field


T = mat.modulation_period
t = np.linspace(0, 3 * T, 300)
Lhom = 6 * L
x = np.linspace(-Lhom, Lhom + L, 1000)

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
    mode /= np.max(np.abs(mode)) * 2
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
plt.pcolormesh(x / L - 0.5, t / T, np.real(qnms[0].T), cmap="RdBu_r")
plt.axvline(-0.5, color="#949494", lw=1)
plt.axvline(0.5, color="#949494", lw=1)
plt.ylim(0, t[-1] / T)
plt.xlabel("$x/L$")
plt.ylabel("$t/T$")
cb = plt.colorbar()
cb.ax.set_title("Re $E$")
plt.tight_layout()
plt.show()
