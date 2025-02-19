# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


from __future__ import annotations

import numpy as np
import pytest

from pytmod.helpers import (
    dimhandler,
    dot,
    matmatprod,
    matvecprod,
    move_first_axes_to_end,
)

rng = np.random.default_rng(12345)


# Define a dummy function to test the decorator
def dummy_function(self, omegas, eigenvalues, modes):
    _ = eigenvalues, omegas, self
    return modes


# Apply the decorator to the dummy function
dummy_function = dimhandler(dummy_function)
omegas = 1.0
eigenvalues = np.array([1, 2, 3])
modes = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


class TestDimHandler:
    def test_scalar_input(self):
        omegas = 1.0
        eigenvalues = np.array([1, 2, 3])
        modes = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = dummy_function(None, omegas, eigenvalues, modes)
        assert np.allclose(result, modes)

    def test_1d_array_input(self):
        N = 2
        mx = 5
        omegas = rng.random(mx)
        eigenvalues = rng.random((N, mx))
        modes = rng.random((N, N, mx))
        result = dummy_function(None, omegas, eigenvalues, modes)
        assert np.allclose(result, modes)

    def test_2d_array_input(self):
        N = 2
        mx, my = 3, 5
        omegas = rng.random((mx, my))
        eigenvalues = rng.random((N, mx, my))
        modes = rng.random((N, N, mx, my))
        result = dummy_function(None, omegas, eigenvalues, modes)
        assert np.allclose(result, modes)

    def test_invalid_input(self):
        N = 2
        mx, my, mz = 7, 3, 5
        omegas = rng.random((mx, my, mz))
        eigenvalues = rng.random((N, mx, my, mz))
        modes = rng.random((N, N, mx, my, mz))
        with pytest.raises(ValueError, match="Unsupported number of dimensions"):
            dummy_function(None, omegas, eigenvalues, modes)


def test_dot_product_1d_arrays():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    expected_result = np.array([32])
    assert np.allclose(dot(a, b), expected_result)


def test_dot_product_2d_arrays():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    expected_result = np.array([26, 44])
    print(dot(a, b))
    assert np.allclose(dot(a, b), expected_result)


def test_dot_product_arrays_different_shapes():
    a = np.array([1, 2, 3])
    b = np.array([[4, 5], [6, 7]])
    with pytest.raises(
        ValueError,
        match="operands could not be broadcast together with remapped shapes",
    ):
        dot(a, b)


def test_dot_product_non_numeric_values():
    a = np.array([1, 2, 3])
    b = np.array(["a", "b", "c"])
    with pytest.raises(TypeError):
        dot(a, b)


def test_dot_product_empty_arrays():
    a = np.array([])
    b = np.array([])
    expected_result = np.array([])
    assert np.allclose(dot(a, b), expected_result)


def test_matvecprod_2d_matrix_1d_vector():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    result = matvecprod(a, b)
    expected = np.array([17, 39])
    assert np.allclose(result, expected)


def test_matvecprod_3d_matrix_2d_vector():
    N = 2
    mx, my = 7, 3
    a = rng.random((N, N, mx, my))
    b = rng.random((N, mx, my))
    matvecprod(a, b)


def test_matmatprod_2d_arrays():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    expected_result = np.array([[19, 22], [43, 50]])
    print(matmatprod(a, b))
    assert np.allclose(matmatprod(a, b), expected_result)


def test_matmatprod_3d_arrays():
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    expected_result = np.array([[[48, 76], [56, 88]], [[136, 172], [160, 200]]])

    assert np.allclose(matmatprod(a, b), expected_result)


def test_matmatprod_non_numeric_arrays():
    a = np.array([[1, 2], [3, 4]], dtype=object)
    b = np.array([["5", 6], [7, 8]], dtype=object)
    with pytest.raises(TypeError):
        matmatprod(a, b)


def test_move_first_axes_to_end_1d():
    arr = np.array([1, 2, 3])
    result = move_first_axes_to_end(arr)
    assert np.array_equal(result, arr)


def test_move_first_axes_to_end_2d():
    arr = np.array([[1, 2], [3, 4]])
    expected = np.array([[1, 3], [2, 4]])
    result = move_first_axes_to_end(arr)
    assert np.array_equal(result, expected)


def test_move_first_axes_to_end_3d():
    arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    expected = np.array([[[1, 5], [2, 6]], [[3, 7], [4, 8]]])
    result = move_first_axes_to_end(arr)
    assert np.array_equal(result, expected)


def test_move_first_axes_to_end_4d():
    arr = np.array(
        [
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        ]
    )
    expected = np.array(
        [
            [[[1, 9], [2, 10]], [[3, 11], [4, 12]]],
            [[[5, 13], [6, 14]], [[7, 15], [8, 16]]],
        ]
    )
    result = move_first_axes_to_end(arr)
    assert np.array_equal(result, expected)
