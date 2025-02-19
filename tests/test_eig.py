# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


from __future__ import annotations

import numpy as np
import pytest

from pytmod.eig import gram_schmidt, polyeig

rng = np.random.default_rng(12345)


def test_polyeig_single_matrix():
    N = 4
    m = 6
    A = rng.random((m, N, N))
    e, X = polyeig(A)


def test_polyeig_error_for_non_square_matrix():
    N = 4
    m = 6
    A = rng.random((m, N, N + 1))
    with pytest.raises(ValueError, match="Matrices must be square"):
        polyeig(A)


def test_polyeig_error_for_different_shapes():
    N = 4
    A1 = rng.random((N, N))
    A2 = rng.random((N + 1, N + 1))
    A = [A1, A2]
    with pytest.raises(ValueError, match="All matrices must have the same shapes"):
        polyeig(A)


def test_polyeig_error_for_empty_list():
    A = []
    with pytest.raises(ValueError, match="Provide at least one matrix"):
        polyeig(A)


def test_gram_schmidt_two_vectors():
    N = 5
    A = rng.random((N, N)) + 1j * rng.random((N, N))
    dM = rng.random((N, N)) + 1j * rng.random((N, N))
    gram_schmidt(A, dM)
