#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod

import sys
import numpy as bk
import matplotlib.pyplot as plt


plt.ion()
plt.close("all")

pi = bk.pi

Npad = int(sys.argv[1])
sort = bool(int(sys.argv[2]))

bk.random.seed(1)

Omega = 1  # time modulation period
T = 2 * pi / Omega


Nomega = 101
EPS = 1e-4
# omegas = bk.linspace(0 + EPS, 8.1 * Omega - EPS, Nomega, dtype=bk.complex128)
omegas = bk.linspace(0 + EPS, 10 * Omega - EPS, Nomega, dtype=bk.complex128)

# omegas -= 0.47j


eps0 = 5.25
deps = 3.22

eps_fourier = [
    -deps / (2 * 1j),
    eps0,
    deps / (2 * 1j),
]  # must be odd (running from -n to n)


# nh = 5
# eps_fourier = 2 * (0.5 - bk.random.rand(nh)) * 0.6

# eps_fourier = [eps0]

nh = len(eps_fourier)
Nh = int((nh - 1) / 2)

t = bk.linspace(0, T, 1100)


def pad(coeffs, Npad):
    return bk.array([0] * Npad + list(coeffs) + [0] * Npad)


eps_fourier = pad(eps_fourier, Npad)


nh = len(eps_fourier)
Nh = int((nh - 1) / 2)


def index_shift(i):
    return i - Nh


def build_field(coeffs, t):
    field = 0
    for i, coeff in enumerate(coeffs):
        n = index_shift(i)
        print(n)
        field += coeff * bk.exp(n * 1j * Omega * t)
    return field


def gamma(m, omega):
    return (omega - Omega * m) ** 2


def build_matrix(omegas):
    if bk.array(omegas).shape == ():
        omegas = bk.array([omegas])
    matrix = bk.zeros(omegas.shape + (nh, nh), dtype=bk.complex128)
    for m in range(nh):
        mshift = index_shift(m)
        for n in range(nh):
            dmn = m - n
            coeff = (
                0 if abs(dmn) > Nh else gamma(mshift, omegas) * eps_fourier[dmn + Nh]
            )
            matrix[:, m, n] = coeff
    return matrix


Ln = 2
L = Ln / eps0**0.5 / Omega
eps_plus = 1
eps_minus = 1
Ei = 1


alpha = (eps0**0.5 + 1) / (eps0**0.5 - 1)

omega_slab_ana = bk.array(
    [1 / (L * eps0**0.5) * (n * bk.pi - 1j * bk.log(alpha)) for n in range(10)]
)


def compute(omegas):
    rn = bk.zeros(omegas.shape + (nh,), dtype=bk.complex128)
    tn = bk.zeros(omegas.shape + (nh,), dtype=bk.complex128)
    matrix_freq = bk.zeros(omegas.shape + (nh, nh), dtype=bk.complex128)
    matrix_slab_freq = bk.zeros(omegas.shape + (2 * nh, 2 * nh), dtype=bk.complex128)
    ks = bk.zeros(omegas.shape + (nh,), dtype=bk.complex128)
    modes_omega = bk.zeros(omegas.shape + (nh, nh), dtype=bk.complex128)

    for iomega, omega in enumerate(omegas):
        matrix = build_matrix(omega)[0]
        matrix_freq[iomega] = matrix
        k2, modes = bk.linalg.eig(matrix)
        k = (k2 + 0j) ** 0.5

        if sort:
            isort = bk.argsort(k.real, axis=0)
            k = k[isort]
            modes = modes[:, isort]

        harm_index = bk.arange(-Nh, Nh + 1)
        # harm_index = harm_index[isort]

        denom = omega - harm_index * Omega
        phi_plus = bk.exp(1j * k * L)
        phi_minus = bk.exp(-1j * k * L)
        Q = denom / (k + EPS)

        matrix_slab = bk.block(
            [
                [(1 + Q * eps_plus**0.5) * modes, (1 - Q * eps_plus**0.5) * modes],
                [
                    (1 - Q * eps_minus**0.5) * phi_plus * modes,
                    (1 + Q * eps_minus**0.5) * phi_minus * modes,
                ],
            ]
        )
        rhs_slab = bk.zeros_like(matrix_slab[:, 0])
        rhs_slab[Nh] = 2 * Ei
        # if sort:
        #     rhs_slab[:nh] = rhs_slab[:nh][isort]

        sol = bk.linalg.solve(matrix_slab, rhs_slab)
        C = sol[:nh]
        D = sol[nh:]

        # if sort:
        #     C = C[isort]
        #     D = D[isort]
        dn0 = bk.zeros_like(C)
        dn0[Nh] = 1

        # if sort:
        #     dn0 = dn0[isort]
        rn[iomega] = eps_plus**0.5 / Ei * Q * modes @ (C - D) - dn0
        zi = eps_minus**0.5 / Ei * Q * modes
        tn[iomega] = zi * phi_plus @ C - zi * phi_minus @ D

        # tn[iomega] = C[2]

        matrix_slab_freq[iomega] = matrix_slab
        ks[iomega] = k
        modes_omega[iomega] = modes

        # print(matrix_slab)
        # print(rhs_slab)
        # print(sol)

        # M = matrix_slab
        # print(M)
        # plt.close("all")
        # plt.imshow(M.real)
        # plt.colorbar()
        # plt.title("slab matrix")
        # plt.savefig("matrix_slab.png")

    return matrix_freq, ks, modes_omega, rn, tn, matrix_slab_freq


matrix_freq, kns, modes_omega, rn, tn, matrix_slab_freq = compute(omegas)

# M = matrix_slab_freq[-1]
# print(M)

# plt.close("all")
# plt.imshow(M.real)
# plt.colorbar()
# plt.title("slab matrix")
# plt.savefig("matrix_slab.png")


isort = bk.zeros_like(kns, dtype=bk.int32)
for iomega in range(len(omegas)):
    isort[iomega] = bk.argsort(kns[iomega].real, axis=0)
    kns[iomega] = kns[iomega, isort[iomega]]
    modes_omega[iomega] = modes_omega[iomega, :, isort[iomega]]


#####################################################

plt.figure()
imode = Nh
# for imode in range(nh):
plt.plot(omegas, bk.abs(rn[:, imode]), "-", c="#e49649")
plt.plot(omegas, bk.abs(tn[:, imode]), "-", c="#4d5bde")


# fresnel coefficients


def fresnel(omegas):

    r13 = (eps_plus**0.5 - eps0**0.5) / (eps_plus**0.5 + eps0**0.5)
    r32 = (eps0**0.5 - eps_minus**0.5) / (eps0**0.5 + eps_minus**0.5)
    t13 = (2 * eps_plus**0.5) / (eps_plus**0.5 + eps0**0.5)
    t32 = (2 * eps0**0.5) / (eps0**0.5 + eps_minus**0.5)

    rf = (r13 + r32 * bk.exp(1j * 2 * omegas * eps0**0.5 * L)) / (
        1 + r13 * r32 * bk.exp(1j * 2 * omegas * eps0**0.5 * L)
    )
    tf = (t13 * t32 * bk.exp(1j * 2 * eps0**0.5 * omegas * L)) / (
        1 + r13 * r32 * bk.exp(1j * 2 * omegas * eps0**0.5 * L)
    )
    return rf, tf


rf, tf = fresnel(omegas)
plt.plot(omegas, bk.abs(rf), "--", c="#e49649")
plt.plot(omegas, bk.abs(tf), "--", c="#4d5bde")

sys.exit(0)

# plt.ylim(0, 1)


fig, ax = plt.subplots(1, 2)

for i in range(nh):
    ax[0].plot(kns[:, i].real, omegas.real, "or", ms=5, mew=0)
    ax[1].plot(kns[:, i].imag, omegas.real, "or", ms=5, mew=0)

ax[0].plot(omegas.real * eps0**0.5, omegas.real, "-b", ms=5, mew=0)
ax[1].plot(omegas.imag * eps0**0.5, omegas.real, "-b", ms=5, mew=0)


####### omega vs k


def build_matrix_alt(ks):
    if bk.array(ks).shape == ():
        ks = bk.array([ks])
    A = bk.zeros(ks.shape + (nh, nh), dtype=bk.complex128)
    B = bk.zeros(ks.shape + (nh, nh), dtype=bk.complex128)
    C = bk.zeros(ks.shape + (nh, nh), dtype=bk.complex128)

    ones = bk.eye(nh, dtype=bk.complex128)
    ones = bk.broadcast_to(ones, A.shape)
    zeros = bk.zeros(ks.shape + (nh, nh), dtype=bk.complex128)
    for m in range(nh):
        mshift = index_shift(m)
        for n in range(nh):
            dmn = m - n
            if abs(dmn) <= Nh:
                eps_mn = eps_fourier[dmn + Nh]
                A[:, m, n] = eps_mn
                B[:, m, n] = -2 * m * Omega * eps_mn
                C[:, m, n] = -(m**2) * Omega**2 * eps_mn
            if dmn == 0:
                C[:, m, n] += ks**2
    matrix_lhs = bk.block([[C, zeros], [zeros, ones]])
    matrix_rhs = bk.block([[B, A], [ones, zeros]])
    # matrix_rhs_inv = bk.linalg.inv(matrix_rhs)
    # return matrix_rhs_inv @ matrix_lhs
    return bk.linalg.solve(matrix_rhs, matrix_lhs)


Nks = Nomega
ks = bk.linspace(-1, 5 * Omega, Nks)


# omegans = bk.zeros(ks.shape + (1 * nh,), dtype=bk.complex128)
# modes_omega = bk.zeros(ks.shape + (1 * nh, 1 * nh), dtype=bk.complex128)
omegans = bk.zeros(ks.shape + (2 * nh,), dtype=bk.complex128)
modes_omega = bk.zeros(ks.shape + (2 * nh, 2 * nh), dtype=bk.complex128)
for ik, k in enumerate(ks):
    matrix_alt = build_matrix_alt(k)[0]
    e, v = bk.linalg.eig(matrix_alt)
    isort = bk.argsort(e)
    e = e[isort]
    v = v[:, isort]
    # omegans[ik], modes_omega[ik] = e[:nh], v[:nh, :nh]
    omegans[ik], modes_omega[ik] = e, v


for i in range(nh):
    ax[0].plot(ks, omegans[:, i].real, "-k", ms=5, mew=0)
    ax[1].plot(ks, omegans[:, i].imag, ".k", ms=5, mew=0)

ax[0].set_xlabel(r"$k$")
ax[0].set_ylabel(r"$\omega$")
ax[1].set_xlabel(r"$k$")
ax[1].set_ylabel(r"$\omega$")
plt.tight_layout()


# sys.exit(0)


########### complex plane

nc = 201
Nc = int((nc - 1) / 2)

omegasr = bk.linspace(0.01, 2 * Omega, nc)
omegasi = bk.linspace(-1, -0.2 * Omega, nc)

re, im = bk.meshgrid(omegasr, omegasi)
omegas = re + 1j * im


rf, tf = fresnel(omegas)

rn_c = bk.zeros(omegas.shape + (nh,), dtype=bk.complex128)
tn_c = bk.zeros(omegas.shape + (nh,), dtype=bk.complex128)
kn_c = bk.zeros(omegas.shape + (nh,), dtype=bk.complex128)

matrix_c = bk.zeros(omegas.shape + (nh, nh), dtype=bk.complex128)
matrix_slab_c = bk.zeros(omegas.shape + (2 * nh, 2 * nh), dtype=bk.complex128)

for iomega, omega in enumerate(omegas):
    (
        matrix_c[iomega],
        kn_c[iomega],
        _,
        rn_c[iomega],
        tn_c[iomega],
        matrix_slab_c[iomega],
    ) = compute(omega)


# plt.plot(omegasr, bk.abs(rn_c[Nc, :, Nh]), ".", c="#e49649")
# plt.plot(omegasr, bk.abs(tn_c[Nc, :, Nh]), ".", c="#4d5bde")

plt.figure()
plt.pcolormesh(omegasr, omegasi, bk.log10(bk.abs(rf)), cmap="inferno")
plt.colorbar()
plt.title("r fresnel")
plt.plot(omega_slab_ana.real, omega_slab_ana.imag, "xg")
plt.xlim(omegasr[0], omegasr[-1])
plt.show()

i = Nh
# for i in range(nh):
plt.figure()
plt.pcolormesh(omegasr, omegasi, bk.log10(bk.abs(rn_c[:, :, i])), cmap="inferno")
plt.colorbar()
plt.title("r")
plt.plot(omega_slab_ana.real, omega_slab_ana.imag, "xg")
plt.xlim(omegasr[0], omegasr[-1])
plt.show()


plt.figure()
plt.pcolormesh(omegasr, omegasi, bk.log10(bk.abs(tf)), cmap="inferno")
plt.colorbar()
plt.title("t fresnel")
plt.plot(omega_slab_ana.real, omega_slab_ana.imag, "xg")
plt.xlim(omegasr[0], omegasr[-1])
plt.show()

i = Nh
# for i in range(nh):
plt.figure()
plt.pcolormesh(omegasr, omegasi, bk.log10(bk.abs(tn_c[:, :, i])), cmap="inferno")
plt.colorbar()
plt.title("t")
plt.plot(omega_slab_ana.real, omega_slab_ana.imag, "xg")
plt.xlim(omegasr[0], omegasr[-1])
plt.show()


# # D = bk.min(bk.linalg.eigvals(matrix_c), axis=-1)
# D = bk.linalg.det(matrix_c)
# plt.figure()
# plt.pcolormesh(omegasr, omegasi, bk.log10(bk.abs(D)), cmap="inferno")
# plt.colorbar()
# plt.title("Det mat")
# plt.show()


# D = bk.min(bk.linalg.eigvals(matrix_slab_c), axis=-1)
D = bk.linalg.det(matrix_slab_c)
plt.figure()
plt.pcolormesh(omegasr, omegasi, bk.log10(bk.abs(D)), cmap="inferno")
plt.colorbar()
plt.title("Det slab")
plt.plot(omega_slab_ana.real, omega_slab_ana.imag, "xg")
plt.xlim(omegasr[0], omegasr[-1])
plt.show()


sys.exit(0)
####


kr = bk.linspace(0.01, 4 * Omega, nc)
ki = bk.linspace(-1, 1.2 * Omega, nc)

re, im = bk.meshgrid(kr, ki)
ks = re + 1j * im


ks = kr

if True:
    # def compute_alt(ks):
    rn = bk.zeros(ks.shape + (nh,), dtype=bk.complex128)
    tn = bk.zeros(ks.shape + (nh,), dtype=bk.complex128)
    matrix_freq = bk.zeros(ks.shape + (2 * nh, 2 * nh), dtype=bk.complex128)
    matrix_slab_freq = bk.zeros(ks.shape + (2 * nh, 2 * nh), dtype=bk.complex128)
    omegans = bk.zeros(ks.shape + (nh,), dtype=bk.complex128)
    modes_omega = bk.zeros(ks.shape + (nh, nh), dtype=bk.complex128)

    for ik, k in enumerate(ks):
        matrix = build_matrix_alt(k)[0]
        omegas, modes = bk.linalg.eig(matrix)
        # isort = omegas.real >= 0
        # omegas = omegas[isort]
        # modes = modes[:, isort]
        # isort = bk.argsort(omegas.real)
        # omegas = omegas[isort]
        # modes = modes[:, isort]
        omegas, modes = omegas[::2], modes[::2, ::2]
        harm_index = bk.arange(-Nh, Nh + 1)
        denom = omegas - harm_index * Omega
        phi_plus = bk.exp(1j * k * L)
        phi_minus = bk.exp(-1j * k * L)
        Q = denom / (k + EPS)
        matrix_slab = bk.block(
            [
                [(1 + Q * eps_plus**0.5) * modes, (1 - Q * eps_plus**0.5) * modes],
                [
                    (1 - Q * eps_minus**0.5) * phi_plus * modes,
                    (1 + Q * eps_minus**0.5) * phi_minus * modes,
                ],
            ]
        ).T
        # plt.close("all")
        # plt.imshow(matrix_slab.real)
        rhs_slab = bk.zeros_like(matrix_slab[:, 0])
        rhs_slab[Nh] = 2 * Ei
        sol = bk.linalg.solve(matrix_slab, rhs_slab)
        C = sol[:nh]
        D = sol[nh:]
        dn0 = bk.zeros_like(C)
        dn0[Nh] = 1
        rn[ik] = eps_plus**0.5 / Ei * Q * modes @ (C - D) - dn0
        zi = eps_minus**0.5 / Ei * Q * modes
        tn[ik] = zi * phi_plus @ C - zi * phi_minus @ D
        matrix_slab_freq[ik] = matrix_slab
        omegans[ik] = omegas
        modes_omega[ik] = modes
        matrix_freq[ik] = matrix
    # return matrix_freq, omegans, modes_omega, rn, tn, matrix_slab_freq


# matrix_freq, omegans, modes_omega, rn, tn, matrix_slab_freq = compute_alt(ks)

plt.figure()
plt.plot(kr, bk.abs(rn[:, Nh]), ".", c="#e49649")
plt.plot(kr, bk.abs(tn[:, Nh]), ".", c="#4d5bde")
