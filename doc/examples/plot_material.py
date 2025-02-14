#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


"""
Time modulated material
=======================

We will learn how to build a time modulated material and how to calculate the
eigenvalues and eigenvectors of this material.

"""


####################################################################################
# Check results from :cite:t:`zurita-sanchez2009`


import matplotlib as mpl
import matplotlib.pyplot as plt
import pytmod as pm
import numpy as bk

plt.ion()
plt.close("all")


##############################################################################################
# We define the modulation as :math:`\epsilon = \epsilon_0 + \Delta\epsilon \sin(\Omega t)`

Omega = 1
eps0 = 5.25
deps = 4


eps_fourier = [
    -deps / (2 * 1j),
    eps0,
    deps / (2 * 1j),
]

mat = pm.Material(eps_fourier, Omega)

T = mat.modulation_period

t = bk.linspace(0, 3 * T, 3000)
eps_time = mat.get_eps_time(t)

plt.figure()
plt.plot(t / T, eps_time.real, c="#c24c4c")
plt.xlabel(r"$t/T$")
plt.ylabel(r"Re $\epsilon(t)$")
plt.xlim(0, 3)
plt.show()


####################################################################################
# Compute the eigenvalues and eigenvectors of the material

omega = 0.8
kns, ens = mat.eigensolve(omega)

####################################################################################
# The eigenvalues are

print(kns.real)

####################################################################################
# We plot the eigenmodes in the time domain

plt.figure()
for i in range(len(kns)):
    kn = kns[i].real
    mode = ens[i]
    mode_time = mat.freq2time(mode, t)
    plt.plot(t / T, mode_time.real, label=i)
    plt.xlabel(r"$t/T$")
    plt.ylabel(r"Re $e_n(t)$")
    plt.xlim(0, 3)
    plt.legend()
    plt.show()


####################################################################################
# Lets's study the convergence with respect to the truncation of the Fourier series

Nmax = 15
Npads = range(Nmax)
ev_cv = []
en_cv = []
for Npad in Npads:
    mat = pm.Material(eps_fourier, Omega, Npad=Npad)
    kns, ens = mat.eigensolve(omega)
    idx = bk.argsort(kns)
    kns = kns[idx]
    ens = ens[:, idx]

    ev_cv.append(kns[:3].real)
    modes_time = []
    for i in range(3):
        mode = ens[:, i]
        mode_time = mat.freq2time(mode, t)
        modes_time.append(mode_time)
    en_cv.append(modes_time)
ev_cv = bk.array(ev_cv)
en_cv = bk.array(en_cv)

####################################################################################
# Eigenvalues convergence


Ns = 3 + 2 * bk.array(Npads)


plt.figure()
for i in range(3):
    plt.plot(Ns, ev_cv[:, i], label=i)
plt.xlabel("$N$")
plt.ylabel("eigenvalue")
plt.legend()
plt.show()


####################################################################################
# Relative error

mat = pm.Material(eps_fourier, Omega, Npad=100)
kns, ens = mat.eigensolve(omega)
idx = bk.argsort(kns)
kns = kns[idx]
ens = ens[:, idx]
evs_check = kns[:3].real
ens_check = ens[::3].real


plt.figure()
for i in range(3):
    plt.plot(Ns, bk.abs(1 - ev_cv[:, i] / evs_check[i]), label=i)
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


# _x = bk.linspace(3, 2*Nmax+1, 100)
# plt.plot(_x, bk.exp(-_x), "--k")
plt.xlabel("$N$")
plt.ylabel("relative error")
plt.xlim(3, 21)
xticks = bk.arange(3, 2 * Nmax + 2, 2)
plt.xticks(xticks, xticks)
plt.legend()
plt.tight_layout()
plt.show()


####################################################################################
# Eigenstates convergence

cmap = mpl.colormaps["Blues"]
colors = cmap(bk.linspace(0, 1, 11)[1:])

for i in range(3):
    plt.figure()
    plt.title(i)
    for j in range(10):
        plt.plot(t / T, en_cv[j, i].T.real, c=colors[j])
        plt.pause(0.01)
    plt.xlim(0, 3)
    plt.xlabel("$t/T$")
    plt.ylabel("mode")
    plt.tight_layout()
    plt.show()
