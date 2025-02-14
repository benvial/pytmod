#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


import pytest
import numpy as np
from pytmod.eig import polyeig, gram_schmidt


def test_polyeig_single_matrix():
    N = 4
    m = 6
    A = np.random.rand(m, N, N)
    e, X = polyeig(A)


def test_polyeig_exception_for_non_square_matrix():
    N = 4
    m = 6
    A = np.random.rand(m, N, N + 1)
    with pytest.raises(Exception) as excinfo:
        polyeig(A)
    assert str(excinfo.value) == "Matrices must be square"


def test_polyeig_exception_for_different_shapes():
    N = 4
    A1 = np.random.rand(N, N)
    A2 = np.random.rand(N + 1, N + 1)
    A = [A1, A2]
    with pytest.raises(Exception) as excinfo:
        polyeig(A)
    assert str(excinfo.value) == "All matrices must have the same shapes"


def test_polyeig_exception_for_empty_list():
    A = []
    with pytest.raises(Exception) as excinfo:
        polyeig(A)
    assert str(excinfo.value) == "Provide at least one matrix"


def test_gram_schmidt_two_vectors():
    N = 4
    A = np.random.rand(N, N)
    dM = np.random.rand(N, N)
    result = gram_schmidt(A, dM)
