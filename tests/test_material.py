# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


from __future__ import annotations

import numpy as np
import pytest

from pytmod import Material
from pytmod.helpers import dot


def test_material_initialization():
    eps_fourier = [0.5, 1.0, 0.5]
    modulation_frequency = 2.0
    material = Material(eps_fourier, modulation_frequency)
    assert np.array_equal(material.eps_fourier, eps_fourier)
    assert material.modulation_frequency == modulation_frequency
    assert material.modulation_period == np.pi

    material = Material(eps_fourier, modulation_frequency, 2)
    assert material.Npad == 2
    assert material.nh == 7
    material.Npad += 1
    assert material.nh == 9
    assert material.Npad == 3

    material = Material(eps_fourier, modulation_frequency, 3)
    material.eps_fourier = [1]
    assert material.Npad == 3
    assert len(material.eps_fourier) == 7
    assert material.nh == 7


def test_material_initialization_even_length():
    eps_fourier = [0.5, 1.0]
    modulation_frequency = 2.0
    with pytest.raises(ValueError, match="The length of eps_fourier must be odd"):
        Material(eps_fourier, modulation_frequency)


def test_build_matrix():
    eps_fourier = [0.5, 1.0, 0.5]
    modulation_frequency = 2.0
    material = Material(eps_fourier, modulation_frequency)
    omegas = np.array([1.0, 2.0, 3.0])
    matrix = material.build_matrix(omegas)
    assert matrix.shape == (material.nh, material.nh, len(omegas))


def test_eigensolve():
    eps_fourier = [0.5, 1.0, 0.5]
    modulation_frequency = 1.2
    material = Material(eps_fourier, modulation_frequency)
    omegas = np.linspace(1, 10, 10)
    eigenvalues, modes = material.eigensolve(omegas)
    assert eigenvalues.shape == (material.nh, len(omegas))
    assert modes.shape == (material.nh, material.nh, len(omegas))
    eigenvalues, modes_right, modes_left = material.eigensolve(omegas, left=True)
    modes_right, modes_left = material.normalize(modes_right, modes_left)

    for i in range(material.nh):
        for j in range(material.nh):
            test = dot(modes_left[:, i], modes_right[:, j])
            val = 1 if i == j else 0
            assert np.allclose(test, val)


def test_matrix_derivative():
    eps_fourier = [0.5, 1.0, 0.5]
    modulation_frequency = 1.2
    material = Material(eps_fourier, modulation_frequency)
    omegas = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    matrix = material.build_matrix(omegas)
    delta_omega = 1e-6
    delta_matrix = material.build_matrix(omegas + delta_omega)

    dmatrix_fd = (delta_matrix - matrix) / delta_omega

    dmatrix = material.build_dmatrix_domega(omegas)

    assert np.allclose(dmatrix_fd, dmatrix)


def test_deigenpairs_domega():
    material = Material([0.5, 1.0, 0.5], 1)

    omega0 = 0.65 - 0.32j
    omega1 = 0.92 - 0.019j
    nc = 11
    omegasr = np.linspace(omega0.real, omega1.real, nc)
    omegasi = np.linspace(omega0.imag, omega1.imag, nc)
    re, im = np.meshgrid(omegasr, omegasi)
    omegas = re + 1j * im

    dmatrix = material.build_dmatrix_domega(omegas)
    eigenvalues, modes_right, modes_left = material.eigensolve(
        omegas, left=True, normalize=True
    )

    eigenvalues_1, _ = material.eigensolve(omegas, normalize=False)
    assert np.allclose(eigenvalues_1, eigenvalues)
    for i in range(material.nh):
        for j in range(material.nh):
            test = dot(modes_right[:, i], modes_left[:, j])
            val = 1 if i == j else 0
            assert np.allclose(test, val)

    deigenvalues_nomatrix = material.get_deigenvalues_domega(
        omegas,
        eigenvalues,
        modes_right,
        modes_left,
    )
    deigenvalues = material.get_deigenvalues_domega(
        omegas, eigenvalues, modes_right, modes_left, dmatrix
    )
    assert np.allclose(deigenvalues_nomatrix, deigenvalues)
    delta_omega = 1e-6
    material.build_matrix(omegas + delta_omega)
    delta_eigenvalues, delta_vr, delta_vl = material.eigensolve(
        omegas + delta_omega, left=True, normalize=True
    )

    deigenvalues_fd = (delta_eigenvalues - eigenvalues) / delta_omega
    dvr_fd = (delta_vr - modes_right) / delta_omega
    dvr = material.get_deigenmodes_right_domega(
        omegas, eigenvalues, modes_right, modes_left, dmatrix
    )
    dvr_nomatrix = material.get_deigenmodes_right_domega(
        omegas, eigenvalues, modes_right, modes_left
    )
    assert np.allclose(dvr_nomatrix, dvr)
    assert np.allclose(deigenvalues_fd, deigenvalues)
    assert np.allclose(dvr_fd, dvr, atol=1e-6)


def test_gamma():
    eps_fourier = [0.5, 1.0, 0.5]
    modulation_frequency = 2.0
    material = Material(eps_fourier, modulation_frequency)
    m = 1
    omega = 3.0
    gamma_value = material._gamma(m, omega)
    expected_value = (omega - material.modulation_frequency * m) ** 2
    assert gamma_value == expected_value


def test_index_shift():
    eps_fourier = [0.5, 1.0, 0.5]
    modulation_frequency = 2.0
    material = Material(eps_fourier, modulation_frequency)
    for i in range(material.nh):
        assert material.index_shift(i) == i - material.Nh


def test_Npad_negative():
    eps_fourier = [0.5, 1.0, 0.5]
    modulation_frequency = 2.0
    with pytest.raises(ValueError, match="Npad must be a positive integer"):
        Material(eps_fourier, modulation_frequency, Npad=-1)
    with pytest.raises(ValueError, match="Npad must be a positive integer"):
        Material(eps_fourier, modulation_frequency, Npad=1.5)
