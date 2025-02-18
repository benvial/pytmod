# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


"""
Electric field in a time-modulated slab
===============================================

Calculate the field in response to an incident plane wave.

"""

####################################################################################
# First import the packages
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pytmod as pm

plt.ion()
plt.close("all")


####################################################################################
# Define the parameters

eps0 = 5.25
deps = 2
Omega = 1
Npad = 7
Ei0 = 1
L = 5

omega = 1.2 * Omega

eps_fourier = [
    -deps / (2 * 1j),
    eps0,
    deps / (2 * 1j),
]

# eps_fourier = [eps0]


####################################################################################
# Solve material eigenproblem

mat = pm.Material(eps_fourier, Omega, Npad)
kns, ens = mat.eigensolve(omega)
slab = pm.Slab(mat, L)


####################################################################################
# Build the matrix for the slab

matrix_slab = slab.build_matrix(omega, kns, ens)

####################################################################################
# Build the rhs for the slab

Eis = slab.init_incident_field(omega)
Ninc = mat.Nh
Eis[Ninc] = Ei0
rhs_slab = slab.build_rhs(omega, Eis)

####################################################################################
# Solve slab problem

solution = slab.solve(matrix_slab, rhs_slab)

####################################################################################
# Extract field coefficients
Eslab_plus, Eslab_minus, Er, Et = slab.extract_coefficients(solution, Eis, kns, ens)


####################################################################################
# Extract reflection and transmission coefficients

rn = Er / Ei0
tn = Et / Ei0


pd.set_option("display.float_format", lambda x: f"{x:.4e}")

Rn = np.abs(rn) ** 2
Tn = np.abs(tn) ** 2

coeffs_n = pd.DataFrame(
    data={"r_n": rn, "t_n": tn, "R_n": Rn, "T_n": Tn}, index=range(-mat.Nh, mat.Nh + 1)
)
# coeffs_n.index.name = "n"
coeffs_n  # noqa: B018

####################################################################################
# Total reflection and transmission coefficients

R = np.sum(Rn)
T = np.sum(Tn)

coeffs_sum = pd.DataFrame(data={"R": R, "T": T, "Balance": R + T}, index=[""])
coeffs_sum  # noqa: B018


####################################################################################
# Get the field

T0 = 2 * np.pi / omega
T = mat.modulation_period
t = np.linspace(0, 3 * T, 300)
Lhom = 3 * L
x = np.linspace(-Lhom, Lhom + L, 1000)
psi = Eslab_plus, Eslab_minus, Er, Et
Es = slab.get_scattered_field(x, t, omega, psi, kns, ens)
Einc = slab.get_incident_field(x, t, omega, Eis)
E = Einc + Es

####################################################################################
# Animate the field

fig, ax = plt.subplots()
ax.set_title(rf"$\omega = {omega / Omega}\,\Omega$")
anim = slab.animate_field(x, t, E, (fig, ax))

# writer = animation.PillowWriter(fps=15,
#                                 metadata=dict(artist='pytmod'),
#                                 bitrate=1800)
# anim.save('field.gif', writer=writer)

####################################################################################
# Space time map

plt.figure()
plt.pcolormesh(x / L - 0.5, t / T, np.real(E.T), cmap="RdBu_r")
plt.axvline(-0.5, color="#949494", lw=1)
plt.axvline(0.5, color="#949494", lw=1)
# for i in range(5):
#     plt.axhline(i, color="#949494", lw=1, ls=":")
#     plt.axhline(i * T0 / T, color="#949494", lw=1, ls="--")

plt.ylim(0, t[-1] / T)
plt.xlabel("$x/L$")
plt.ylabel("$t/T$")
cb = plt.colorbar()
cb.ax.set_title("Re $E$")
plt.tight_layout()
plt.show()
