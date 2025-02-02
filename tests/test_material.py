#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3


import pytest
import numpy as bk
from pytmod import Material


def test_material_initialization():
    eps_fourier = [0.5, 1.0, 0.5]
    modulation_frequency = 2.0
    material = Material(eps_fourier, modulation_frequency)
    assert bk.array_equal(material.eps_fourier, eps_fourier)
    assert material.modulation_frequency == modulation_frequency
    assert material.modulation_period == bk.pi

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
    with pytest.raises(ValueError):
        Material(eps_fourier, modulation_frequency)


def test_build_matrix():
    eps_fourier = [0.5, 1.0, 0.5]
    modulation_frequency = 2.0
    material = Material(eps_fourier, modulation_frequency)
    omegas = bk.array([1.0, 2.0, 3.0])
    matrix = material.build_matrix(omegas)
    assert matrix.shape == (material.nh, material.nh, len(omegas))


def test_eigensolve():
    eps_fourier = [0.5, 1.0, 0.5]
    modulation_frequency = 2.0
    material = Material(eps_fourier, modulation_frequency)
    omegas = bk.array([1.0, 2.0, 3.0])
    eigenvalues, modes = material.eigensolve(omegas)
    assert eigenvalues.shape == (material.nh, len(omegas))
    assert modes.shape == (material.nh, material.nh, len(omegas))


def test_gamma():
    eps_fourier = [0.5, 1.0, 0.5]
    modulation_frequency = 2.0
    material = Material(eps_fourier, modulation_frequency)
    m = 1
    omega = 3.0
    gamma_value = material.gamma(m, omega)
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
    with pytest.raises(ValueError):
        Material(eps_fourier, modulation_frequency, Npad=-1)
    with pytest.raises(ValueError):
        Material(eps_fourier, modulation_frequency, Npad=1.5)
