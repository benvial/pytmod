#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod

import pytest
import numpy as bk
from pytmod import Material, Slab

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
        Eis = bk.ones((slab.material.nh,) + om.shape, dtype=bk.complex128)
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

    omegas = bk.linspace(0.1, 2.3, 5)
    kns, ens = mat.eigensolve(omegas)
    matrix_slab = slab.build_matrix(omegas, kns, ens)
    matrix_slab = slab.build_matrix(omegas, kns, ens)
    Eis = bk.zeros((slab.material.nh,) + omegas.shape, dtype=bk.complex128)
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
        0.01 - 1j,
        4 - 0.001,
        peak_ref=4,
        recursive=True,
        tol=1e-7,
    )

    evs, modes = slab.eigensolve(
        0.01 - 1j,
        2 - 0.001,
        peak_ref=1,
        recursive=False,
        tol=1e-2,
    )


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
