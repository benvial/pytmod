#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod

import sys
import numpy as bk
import matplotlib.pyplot as plt
import importlib

import eig

importlib.reload(eig)
from eig import nonlinear_eigensolver


plt.ion()
# plt.close("all")

pi = bk.pi

Omega = 1  # time modulation period

sort = bool(int(sys.argv[2]))

Nomega = 1000
EPS = 1e-4
# omegas = bk.linspace(-Omega / 2, Omega / 2, 5501)
omegas = bk.linspace(0 + EPS, 10 * Omega - EPS, Nomega)
# omegas = bk.linspace(1,1, 2) * Omega * 0.74

omegas = bk.array([8.5])


T = 2 * pi / Omega

eps0 = 5.25
deps = 0.85


bk.random.seed(1)

eps_fourier = [
    -deps / (2 * 1j),
    eps0,
    deps / (2 * 1j),
]  # must be odd (running from -n to n)


# eps_fourier = [eps0]

# nh = 11
# eps_fourier = 2 * (0.5 - bk.random.rand(nh)) * 0.6

nh = len(eps_fourier)
Nh = int((nh - 1) / 2)


t = bk.linspace(0, 2 * T, 151)

Npad = int(sys.argv[1])


Ln = 2

L = Ln / eps0**0.5 / Omega
eps_plus = 1
eps_minus = 1
Ei = 1


Nharmo_plot = 0


omega0 = 5.12 - 2.1 * 1j
omega1 = 8.1 - 1e-12 * 1j

tol = 1e-12
peak_ref = 4
max_iter = 15

nc = 201

plt_fresnel = True


alpha = (eps0**0.5 + 1) / (eps0**0.5 - 1)

omega_slab_ana = bk.array(
    [1 / (L * eps0**0.5) * (n * bk.pi - 1j * bk.log(alpha)) for n in range(10)]
)


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
        field += coeff * bk.exp(n * 1j * Omega * t)
    return field


# eps_time = build_field(eps_fourier, t)
# plt.plot(t, eps_time.real)
# plt.plot(t, eps_time.imag, "--")

# eps_time1 = eps0 + deps * bk.sin(Omega * t)
# plt.plot(t, eps_time1.real)
# plt.plot(t, eps_time1.imag, "--")


def gamma(m, omega):
    return (omega - Omega * m) ** 2


def build_matrix(omegas):
    # integ = bk.where(bk.int32(omegas.real / Omega) == omegas.real / Omega)
    # omegas[integ] += 1e-12
    matrix = bk.zeros((nh, nh) + omegas.shape, dtype=bk.complex128)
    for m in range(nh):
        mshift = index_shift(m)
        for n in range(nh):
            dmn = m - n
            coeff = (
                0 if abs(dmn) > Nh else gamma(mshift, omegas) * eps_fourier[dmn + Nh]
            )
            matrix[m, n] = coeff
    return matrix


def eigensolve_material(omegas):
    matrix = build_matrix(omegas)
    k2, modes = bk.linalg.eig(matrix.T)
    eigenvalues = (k2 + 0j) ** 0.5
    if sort:

        isort = bk.zeros_like(eigenvalues, dtype=bk.int32)
        for iomega in range(len(omegas)):
            isort[iomega] = bk.argsort(eigenvalues[iomega].real, axis=0)
            eigenvalues[iomega] = eigenvalues[iomega, isort[iomega]]
            modes[iomega] = modes[iomega, :, isort[iomega]]
    return eigenvalues, modes, matrix


# def build_matrix_slab(omegas):
if True:
    omegas = bk.array([0.5])
    eigenvalues, modes, matrix = eigensolve_material(omegas)

    if sort:
        harm_index = bk.array([bk.arange(-Nh, Nh + 1)[i] for i in isort])
        harm_index = bk.broadcast_to(harm_index[:, :, bk.newaxis], omegas.shape)
        harm_index = bk.transpose(harm_index, (2, 1, 0))
    else:
        harm_index = bk.arange(-Nh, Nh + 1)
        harm_index = bk.broadcast_to(harm_index, eigenvalues.shape)

    harm_index = bk.transpose(harm_index)
    omegas_shift = omegas - harm_index * Omega
    omegas_shift = bk.transpose(omegas_shift)
    phi_plus = bk.exp(1j * eigenvalues * L)
    phi_minus = bk.exp(-1j * eigenvalues * L)
    ks = bk.broadcast_to(eigenvalues[:, :, bk.newaxis], modes.shape)
    phi_plus = bk.broadcast_to(phi_plus[:, :, bk.newaxis], modes.shape)
    phi_minus = bk.broadcast_to(phi_minus[:, :, bk.newaxis], modes.shape)
    omegas_shift = bk.broadcast_to(omegas_shift[:, :, bk.newaxis], modes.shape)
    ks = bk.transpose(ks, (0, 2, 1))
    phi_plus = bk.transpose(phi_plus, (0, 2, 1))
    phi_minus = bk.transpose(phi_minus, (0, 2, 1))
    modesT = modes
    matrix_slab = bk.block(
        [
            [
                (omegas_shift * eps_plus**0.5 + ks) * modes,
                (omegas_shift * eps_plus**0.5 - ks) * modes,
            ],
            [
                (omegas_shift * eps_minus**0.5 - ks) * phi_plus * modes,
                (omegas_shift * eps_minus**0.5 + ks) * phi_minus * modes,
            ],
        ]
    )

    A = (omegas_shift * eps_plus**0.5 + ks) * modes
    print(modes)
    print(modes.shape)
    print(A)
    print(A.shape)
    plt.figure()
    plt.imshow(matrix_slab[0].real)
    plt.colorbar()
    plt.title("dev")
    sys.exit(0) 

    # return matrix_slab, eigenvalues, modes, matrix, phi_plus, phi_minus, ks


def compute(omegas):
    matrix_slab, eigenvalues, modes, matrix, phi_plus, phi_minus, ks = (
        build_matrix_slab(omegas)
    )
    rhs_slab = bk.zeros_like(matrix_slab[:, :, 0])
    rhs_slab[:, Nh] = (eps_plus**0.5 + 1) * Ei * omegas
    if sort:
        for i in range(matrix_slab.shape[0]):
            rhs_slab[i, :nh] = rhs_slab[i, :nh][isort[i]]

    sol = bk.empty_like(rhs_slab)
    # Loop through each batch
    for i in range(matrix_slab.shape[0]):
        sol[i] = bk.linalg.solve(matrix_slab[i], rhs_slab[i])

    C = sol[:, :nh]
    D = sol[:, nh : 2 * nh]

    if sort:
        for i in range(matrix_slab.shape[0]):
            C[i] = C[i][isort[i]]
            D[i] = D[i][isort[i]]

    rn = bk.zeros_like(C)
    tn = bk.zeros_like(C)
    dn0 = bk.zeros_like(C)
    dn0[:, Nh] = 1
    for i in range(matrix_slab.shape[0]):
        rn[i] = modes[i] @ (C[i] + D[i]) / Ei - dn0[i]
        tn[i] = modes[i] / Ei * phi_plus[i] @ C[i] + modes[i] / Ei * phi_minus[i] @ D[i]

    return matrix, eigenvalues, modes, rn, tn, C, D, matrix_slab



omegas = bk.array([0.5])

matrix, eigenvalues, modes, rn, tn, Cs, Ds, matrix_slab = compute(omegas)
plt.figure()
plt.imshow(matrix_slab[0].real)
plt.colorbar()
plt.title("dev")


print(rn[0, Nh])
print(tn[0, Nh])
sys.exit(0)

# plt.figure()
# imode = Nh
# for imode in range(nh):
#     plt.plot(omegas, bk.abs(rn[:, imode]), "-", c="#e49649")
#     plt.plot(omegas, bk.abs(tn[:, imode]), "-", c="#5000ca")

plt.figure()
imode = Nh + Nharmo_plot

log = False
r_ = bk.abs(rn[:, imode])
t_ = bk.abs(tn[:, imode])
if log:
    r_ = bk.log10(r_)
    t_ = bk.log10(t_)

plt.plot(omegas, r_, "-", c="#e49649", label=rf"$r$")
plt.plot(omegas, t_, "-", c="#5000ca", label=rf"$t$")
plt.title(rf"harmonic ${{{Nharmo_plot}}}$")
plt.legend()
plt.pause(0.1)

# plt.figure()
# plt.imshow(bk.log10(bk.abs(matrix_slab[0])))
# plt.axis("scaled")
# plt.colorbar()


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
if plt_fresnel:
    plt.plot(omegas, bk.abs(rf), "--", c="#e49649")
    plt.plot(omegas, bk.abs(tf), "--", c="#5000ca")


# plt.ylim(0, 1)


# plt.figure()
# Rf = bk.abs(rf) ** 2
# Tf = bk.abs(tf) ** 2

# R = bk.abs(rn[:, Nh]) ** 2
# T = bk.abs(tn[:, Nh]) ** 2

# plt.plot(omegas, Rf, c="#e49649")
# plt.plot(omegas, Tf, c="#5000ca")
# plt.plot(omegas, Rf + Tf, c="#3b3c46")

# plt.plot(omegas, R, "--", c="#e49649")
# plt.plot(omegas, T, "--", c="#5000ca")
# plt.plot(omegas, R + T, "--", c="#3b3c46")


########### complex plane


Nc = int((nc - 1) / 2)

omegasr = bk.linspace(omega0.real, omega1.real, nc)
omegasi = bk.linspace(omega0.imag, omega1.imag, nc)

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
        _,
        kn_c[iomega],
        _,
        rn_c[iomega],
        tn_c[iomega],
        _,
        _,
        matrix_slab_c[iomega],
    ) = compute(omega)


# plt.plot(omegasr, bk.abs(rn_c[Nc, :, Nh]), ".", c="#e49649")
# plt.plot(omegasr, bk.abs(tn_c[Nc, :, Nh]), ".", c="#5000ca")

plt.figure()
plt.pcolormesh(omegasr, omegasi, bk.log10(bk.abs(rf)), cmap="inferno")
plt.colorbar()
plt.title("r fresnel")
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


# #################


# # plt.figure()
# # for i in range(nh):
# #     plt.plot(eigenvalues[:, i].real / eps0**0.5, omegas)

# # plt.xlim(0, 3)
# # plt.ylim(0, 1)


# # plt.figure()
# # for i in range(nh):
# #     plt.plot(eigenvalues[:, i].real, omegas, ".r", ms=5, mew=0)

# # for i in range(nh):
# #     plt.plot((omegas + i * Omega) * eps_fourier[Nh].real ** 0.5, omegas, "--k")
# #     plt.plot(-(omegas + (-nh + i) * Omega) * eps_fourier[Nh].real ** 0.5, omegas, "--k")

# # # plt.figure()
# # # for i in range(nh):
# # #     plt.plot(eigenvalues[:, i].imag, omegas, ".b")


# iomega = int(len(omegas) / 2)


# # sys.exit(0)


# # plt.figure()

# # Nmax = 0
# # modesplot = range(-Nmax, Nmax + 1)
# # modesplot = [0, 1, 2]

# # print("nh = ", nh)

# # for imode in modesplot:
# #     # imode += Nh
# #     n = index_shift(imode + Nh)
# #     print("n = ", n)
# #     # print("harm = ", index_shift(imode))
# #     print("imode = ", imode)
# #     mode = modes[iomega, imode]
# #     mode_time = build_field(mode, t)
# #     plt.plot(t, bk.abs(mode_time), label=imode)
# #     plt.title(eigenvalues[iomega, imode])
# # plt.legend()
# # print(eigenvalues[iomega, imode])
# # # plt.plot(t, mode_time.imag, "--")


# # sys.exit(0)

# # ortho = bk.zeros((nh, nh), dtype=bk.complex128)
# # for p in range(nh):
# #     for q in range(nh):
# #         ps = 0
# #         for m in range(nh):
# #             for n in range(nh):
# #                 dmn = n - m
# #                 coeff = 0 if abs(dmn) > Nh else eps_fourier[dmn + Nh]
# #                 ps += coeff * modes[iomega, p, m] * modes[iomega, q, n].conj()
# #         ortho[p, q] = ps


# # ortho /= bk.diag(ortho)

# # print(ortho)

# # plt.figure()
# # plt.imshow(ortho.real)


# normas = bk.zeros((nh), dtype=bk.complex128)
# for p in range(nh):
#     ps = 0
#     for m in range(nh):
#         for n in range(nh):
#             dmn = n - m
#             coeff = 0 if abs(dmn) > Nh else eps_fourier[dmn + Nh]
#             ps += coeff * modes[iomega, p, m] * modes[iomega, p, n].conj()
#     normas[p] = ps

# # modes /= normas**0.5


def build_matrix_slab_eig(omegas):
    # print(omegas.shape)
    omegas = bk.array(omegas)
    isscalar = omegas.shape == ()
    if isscalar:
        omegas = bk.array([[omegas]])
    #     # return build_matrix_slab(omegas)
    # elif len(omegas.shape) == 1:
    #     omegas = bk.array([omegas])
    matrix_slab_c = bk.zeros((2 * nh, 2 * nh) + omegas.shape, dtype=bk.complex128)
    for iomega, omega in enumerate(omegas):
        M = build_matrix_slab(omega)[0]
        # M = bk.linalg.inv(M)
        M = bk.transpose(M, (2, 1, 0))
        matrix_slab_c[:, :, iomega] = M
    # print(matrix_slab_c.shape)
    if isscalar:
        return matrix_slab_c[:, :, 0, 0]
    # print(matrix_slab_c)
    # matrix_slab_c = bk.linalg.inv(matrix_slab_c)
    return matrix_slab_c
    # # matrix_slab_c = matrix_slab_c.T
    # # print(matrix_slab_c.shape)
    # print(omegas.shape)
    # return build_matrix_slab(omegas)[0]


matrix_slab_c = build_matrix_slab_eig(omegas)
matrix_slab_c = bk.transpose(matrix_slab_c, (2, 3, 0, 1))
# D = bk.min(bk.linalg.eigvals(matrix_slab_c), axis=-1)
D = bk.linalg.det(matrix_slab_c)
plt.figure()
plt.pcolormesh(omegasr, omegasi, bk.log10(bk.abs(D)), cmap="inferno")
plt.colorbar()
plt.title("Det slab")
for i in range(-Nh, Nh + 1):
    plt.plot(omega_slab_ana.real + i * Omega, omega_slab_ana.imag, "og")

plt.xlim(omegasr[0], omegasr[-1])
plt.ylim(omegasi[0], omegasi[-1])
plt.pause(0.1)


# print(Mslab.shape)
# Mslab = build_matrix_slab_eig(1)
# print(Mslab.shape)
# Mslab = build_matrix_slab_eig(bk.array([12]))
# print(Mslab.shape)

# plt.close("all")

plt.figure()
evs, vs = nonlinear_eigensolver(
    build_matrix_slab_eig,
    omega0,
    omega1,
    # guesses=bk.array([0.0]),
    recursive=True,
    lambda_tol=tol,
    return_left=False,
    peak_ref=peak_ref,
    max_iter=max_iter,
    plot_solver=True,
    peaks_estimate="det",
    weight="max element",
)

if len(evs) > 0:

    print("eigenvalues = ", evs)

    matrix = build_matrix_slab_eig(evs[:, bk.newaxis])
    det = bk.linalg.det(matrix.T)[0]
    print("|det| = ", bk.abs(det))

    plt.plot(evs.real, evs.imag, "xw")
    plt.pause(0.1)
else:
    print("no eigenvalues found")


i = Nh + Nharmo_plot
# for i in range(nh):
plt.figure()
plt.pcolormesh(omegasr, omegasi, bk.log10(bk.abs(rn_c[:, :, i])), cmap="inferno")
plt.colorbar()
plt.title(rf"$r_{{{Nharmo_plot}}}$")
plt.plot(omega_slab_ana.real, omega_slab_ana.imag, "xg")
if len(evs) > 0:
    plt.plot(evs.real, evs.imag, "xw")
plt.xlim(omegasr[0], omegasr[-1])
plt.ylim(omegasi[0], omegasi[-1])


# for i in range(nh):
plt.figure()
plt.pcolormesh(omegasr, omegasi, bk.log10(bk.abs(tn_c[:, :, i])), cmap="inferno")
plt.colorbar()
plt.title(rf"$t_{{{Nharmo_plot}}}$")
plt.plot(omega_slab_ana.real, omega_slab_ana.imag, "xg")
if len(evs) > 0:
    plt.plot(evs.real, evs.imag, "xw")
plt.xlim(omegasr[0], omegasr[-1])
plt.ylim(omegasi[0], omegasi[-1])


#### field


mode_field = False

if mode_field:

    imode = 1
    sol = vs[:, imode]
    C = sol[:nh]
    D = sol[nh : 2 * nh]
    omega = evs[imode]

    ks, modes, matrix = eigensolve_material(omega)
    phi_plus = bk.exp(1j * ks * L)
    phi_minus = bk.exp(-1j * ks * L)
    Enr = modes @ (C + D)
    Ent = modes * phi_plus @ C + modes * phi_minus @ D

else:
    omegas = bk.array([3.3])
    matrix, eigenvalues, modes, rn, tn, Cs, Ds, matrix_slab = compute(omegas)

    omega = omegas[0]

    C = Cs[0]
    D = Ds[0]

    modes = modes[0]
    ks = eigenvalues[0]

    Enr = rn[0]
    Ent = tn[0]

### ---------


Lhom = 3 * L

t = bk.linspace(0, 2 * T, 151)

Nx = 1500
Nt = len(t)
x = bk.linspace(-Lhom, Lhom + L, Nx)
E = bk.zeros((Nx, Nt), dtype=bk.complex128)
for ix, x_ in enumerate(x):
    if x_ < 0:
        _E = 0
        for n in range(-Nh, Nh + 1):
            kn = eps_plus**0.5 * (omega - n * Omega)
            _E += Enr[n + Nh] * bk.exp(-1j * (kn * (x_) + (omega - n * Omega) * t))
        E[ix] = _E
    elif x_ > L:
        _E = 0
        for n in range(-Nh, Nh + 1):
            kn = eps_minus**0.5 * (omega - n * Omega)
            _E += Ent[n + Nh] * bk.exp(1j * (kn * (x_ - L) - (omega - n * Omega) * t))
        E[ix] = _E
    else:
        _E = 0
        for p in range(0, nh):
            _En = 0
            for n in range(-Nh, Nh + 1):
                _En += (
                    (C[p] * bk.exp(1j * ks[p] * x_) + D[p] * bk.exp(-1j * ks[p] * x_))
                    * modes[n + Nh, p]
                    * bk.exp(-1j * (omega - n * Omega) * t)
                )
            _E += _En
        E[ix] = _E


if not mode_field:
    Einc = bk.zeros((Nx, Nt), dtype=bk.complex128)
    kinc = eps_plus**0.5 * omega
    for ix, x_ in enumerate(x):
        if x_ < 0:
            Einc[ix] = Ei * bk.exp(1j * (kinc * x_ - omega * t))

    E += Einc

plt.figure()
plt.axvline(-0.5, color="#c0c0c0")
plt.axvline(0.5, color="#c0c0c0")
line = plt.plot(x / L - 0.5, bk.real(E[:, 0]), c="#cc4646")
plt.xlabel("$x/L$")

# plt.figure()
# plt.axvline(-0.5, color="#c0c0c0")
# plt.axvline(0.5, color="#c0c0c0")
# for it, t_ in enumerate(t):
#     line = plt.plot(x / L - 0.5, bk.real(E[:, it]), c="#cc4646")
#     plt.xlabel("$x/L$")
#     plt.pause(0.01)
#     [l.remove() for l in line]


plt.figure()
plt.pcolormesh(x / L - 0.5, t / T, bk.abs(E.T))
plt.xlabel("$x/L$")
plt.ylabel("$t/T$")
