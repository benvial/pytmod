#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod

import pytest
import numpy as bk

bk.random.seed(1234)
from pytmod import Material, Slab
from pytmod.eig import get_residual
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


eps0 = 5.25
deps = 0.85

eps_fourier = [1, 6, 1]
mat = Material(eps_fourier, 1, Npad=1)

slab = Slab(mat, 3)


def test_multidim():
    test = []
    omegas0D = 0.15
    omegas1D = bk.array([0.15, 0.15])
    omegas2D = bk.array([[0.15, 0.15], [0.15, 0.15]])
    for om in [omegas0D, omegas1D, omegas2D]:
        om = bk.array(om)
        kns, ens = mat.eigensolve(om)
        matrix_slab = slab.build_matrix(om, kns, ens)
        Eis = slab.init_incident_field(om)
        rhs_slab = slab.build_rhs(om, Eis)
        solution = slab.solve(matrix_slab, rhs_slab)
        C, D, Er, Et = slab.extract_coefficients(solution, Eis, kns, ens)
        test.append(matrix_slab)
        r, t = slab.fresnel_static(om)
        evstatic = slab.eigenvalue_static(om)
    assert bk.allclose(test[0], test[1][:, :, 0])
    assert bk.allclose(test[0], test[2][:, :, 0, 0])


def test_fresnel():
    eps_fourier = [6]
    mat = Material(eps_fourier, 1)
    slab = Slab(mat, 3, eps_plus=2, eps_minus=4)
    print(slab)

    omegas = bk.linspace(0.1, 2.3, 5)
    kns, ens = mat.eigensolve(omegas)
    matrix_slab = slab.build_matrix(omegas, kns, ens)
    Eis = slab.init_incident_field(omegas)
    Ei0 = 1
    Eis[mat.Nh] = Ei0
    rhs_slab = slab.build_rhs(omegas, Eis)
    solution = slab.solve(matrix_slab, rhs_slab)
    C, D, Er, Et = slab.extract_coefficients(solution, Eis, kns, ens)
    rs, ts = slab.fresnel_static(omegas)
    r = Er / Ei0
    t = Et / Ei0
    assert bk.allclose(rs, r)
    assert bk.allclose(ts, t)
    assert bk.allclose(
        bk.abs(rs) ** 2 + slab.eps_minus**0.5 / slab.eps_plus**0.5 * bk.abs(ts) ** 2, 1
    )


def test_eigensolve():
    eps_fourier = [6]
    mat = Material(eps_fourier, 1)
    slab = Slab(mat, 3, eps_plus=2, eps_minus=4)
    evs, modes = slab.eigensolve(
        0.01 - 1j,
        4 - 0.001,
        peak_ref=4,
        recursive=True,
        tol=1e-7,
    )

    evstatic = bk.array([slab.eigenvalue_static(n) for n in range(1, 10)])

    assert bk.allclose(evs, evstatic)

    eps_fourier = [1, 6, -1]
    mat = Material(eps_fourier, 1)
    slab = Slab(mat, 3)
    evs, modes = slab.eigensolve(
        0.01 - 0.4j,
        0.9 - 0.001j,
        peak_ref=3,
        recursive=True,
        plot_solver=True,
        tol=1e-7,
    )

    evs, modes = slab.eigensolve(
        0.01 - 0.4j,
        0.25 - 0.001j,
        peak_ref=2,
        recursive=False,
        plot_solver=True,
        peaks_estimate="eig",
    )
    omega = evs[0]
    kns, smodes = mat.eigensolve(omega)
    M = slab.build_matrix(omega, kns, smodes)

    res = get_residual(M, modes[:, 0])
    assert res < 1e-6
    print(res)

    for return_left in [True, False]:
        slab.eigensolve(0.01 - 0.4j, 0.02 - 0.001j, return_left=return_left)

    for recursive in [True, False]:
        slab.eigensolve(
            0.01 - 0.4j,
            2.25 - 0.001j,
            guesses=[1 - 1j],
            return_left=True,
            recursive=recursive,
            tol=1e-6,
        )
    slab.eigensolve(
        0.1 - 0.4j,
        0.2 - 0.1j,
        strategy="random",
        N_grid=(2, 2),
        peak_ref=1,
        verbose=True,
        init_vect="random",
        dim=slab.material.nh * 2,
        refine=True,
    )
    slab.eigensolve(
        1.1 - 0.4j,
        1.2 - 0.2j,
        strategy="grid",
        N_grid=(2, 2),
        peak_ref=1,
    )

    with pytest.raises(ValueError):
        slab.eigensolve(
            0.01 - 0.4j,
            0.25 - 0.001j,
            init_vect="random",
            dim=None,
        )
    with pytest.raises(ValueError):
        slab.eigensolve(0.01 - 0.4j, 0.25 - 0.001j, strategy="unkown")
    with pytest.raises(ValueError):
        slab.eigensolve(0.01 - 0.4j, 0.25 - 0.001j, init_vect="unkown")

    for return_left in [True, False]:
        slab.eigensolve(
            0.25 - 0.4j, 0.251 - 0.001j, return_left=return_left, recursive=True
        )
    for weight in [
        "rayleigh",
        "rayleigh symmetric",
        "rayleigh asymmetric",
        "max element",
    ]:

        slab.eigensolve(
            0.01 - 0.4j,
            0.25 - 0.001j,
            weight=weight,
        )

    with pytest.raises(ValueError):
        slab.eigensolve(
            0.01 - 0.4j,
            0.25 - 0.001j,
            weight="unknown",
        )


def test_matrix_derivative():

    eps_fourier = [0.5, 2.0, 0.5]
    eps_fourier = [2.0]
    modulation_frequency = 1.2
    material = Material(eps_fourier, modulation_frequency)
    omegas = bk.linspace(1, 10, 5)

    eigenvalues, modes_right, modes_left = material.eigensolve(
        omegas, left=True, normalize=True
    )

    slab = Slab(material, 3)

    dM = slab.build_dmatrix_domega(omegas, eigenvalues, modes_right, modes_left)
    M = slab.build_matrix(omegas, eigenvalues, modes_right)
    delta_omega = 1e-9

    deigenvalues, dmodes_right, dmodes_left = material.eigensolve(
        omegas + delta_omega, left=True, normalize=True
    )
    delta_M = slab.build_matrix(omegas + delta_omega, deigenvalues, dmodes_right)
    dM_fd = (delta_M - M) / delta_omega
    assert bk.allclose(dM_fd, dM, atol=1e-6)


def test_incident():
    omega = 0.7
    Ei0 = 1

    eps_fourier = [1, 6, -4]
    mat = Material(eps_fourier, 3)
    kns, ens = mat.eigensolve(omega)
    slab = Slab(mat, 3)
    Nh = mat.Nh

    for i in range(-Nh, Nh + 1):
        Ninc = mat.Nh + i
        Eis = slab.init_incident_field(omega)
        Eis[Ninc] = Ei0
        rhs_slab = slab.build_rhs(omega, Eis)
        matrix_slab = slab.build_matrix(omega, kns, ens)
        solution = slab.solve(matrix_slab, rhs_slab)
        Eslab_plus, Eslab_minus, Er, Et = slab.extract_coefficients(
            solution, Eis, kns, ens
        )
        rn1 = Er / Ei0
        tn1 = Et / Ei0

        Ninc = mat.Nh + i + mat.nh
        Eis = slab.init_incident_field(omega)
        Eis[Ninc] = Ei0
        rhs_slab = slab.build_rhs(omega, Eis)
        matrix_slab = slab.build_matrix(omega, kns, ens)
        solution = slab.solve(matrix_slab, rhs_slab)
        Eslab_plus, Eslab_minus, Er, Et = slab.extract_coefficients(
            solution, Eis, kns, ens
        )
        rn2 = Er / Ei0
        tn2 = Et / Ei0

        assert bk.allclose(rn1, tn2)
        assert bk.allclose(rn2, tn1)


def test_solve_raises_error():
    slab = Slab(None, None)  # Assuming a default constructor for simplicity
    matrix_slab = bk.zeros((0, 0, 0, 0, 0), dtype=bk.complex128)  # 5D array
    rhs_slab = bk.zeros((0, 0, 0), dtype=bk.complex128)  # 3D array
    with pytest.raises(ValueError, match="Unsupported number of dimensions"):
        slab.solve(matrix_slab, rhs_slab)


def test_extract_coefficients_raises_error():
    slab = Slab(None, None)  # Assuming a default constructor for simplicity
    Eis, kns, ens = None, None, None
    solution = bk.zeros((0, 0, 0, 0, 0), dtype=bk.complex128)  # 3D array
    with pytest.raises(ValueError, match="Unsupported number of dimensions"):
        slab.extract_coefficients(solution, Eis, kns, ens)


def test_field():
    eps0 = 5.25
    deps = 2
    Omega = 1
    Npad = 7
    Ei0 = 1
    L = 5
    omega = 1.0 * Omega + 1e-12
    eps_fourier = [
        -deps / (2 * 1j),
        eps0,
        deps / (2 * 1j),
    ]
    mat = Material(eps_fourier, Omega, Npad)
    kns, ens = mat.eigensolve(omega)
    slab = Slab(mat, L)
    matrix_slab = slab.build_matrix(omega, kns, ens)
    Eis = slab.init_incident_field(omega)
    Ninc = mat.Nh
    Eis[Ninc] = Ei0
    rhs_slab = slab.build_rhs(omega, Eis)
    solution = slab.solve(matrix_slab, rhs_slab)
    Eslab_plus, Eslab_minus, Er, Et = slab.extract_coefficients(solution, Eis, kns, ens)

    T0 = 2 * bk.pi / omega
    T = mat.modulation_period
    t = bk.linspace(0, 3 * T, 6)
    Lhom = 3 * L
    x = bk.linspace(-Lhom, Lhom + L, 10)
    psi = Eslab_plus, Eslab_minus, Er, Et
    Es = slab.get_scattered_field(x, t, omega, psi, kns, ens)
    Einc = slab.get_incident_field(x, t, omega, Eis)
    E = Einc + Es

    fig, ax = plt.subplots()
    anim1 = slab.animate_field(x, t, E, (fig, ax))
    anim2 = slab.animate_field(x, t, E)
