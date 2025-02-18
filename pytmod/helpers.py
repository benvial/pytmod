# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod

from __future__ import annotations

import numpy as np


def normalize_batch(modes, normas):
    dim = modes.shape[1]
    return np.array([modes[:, i] / normas[i] for i in range(dim)]).swapaxes(1, 0)


def normalize_modes(normas, modes_right, modes_left):
    """
    Normalize the eigenmodes.

    Parameters
    ----------
    modes_right : array_like
        The right eigenvectors
    modes_left : array_like
        The left eigenvectors

    Returns
    -------
    modes_right : array_like
        The normalized right eigenvectors.
    modes_left : array_like
        The normalized left eigenvectors.

    Notes
    -----
    First, the eigenmodes are normalized so that the left and right
    eigenmodes are biorthogonal. Then, the right eigenmodes are
    normalized so that the maximum value of each eigenmode is 1.
    """
    # normalize so that left and right eigenmodes are biorthogonal
    modes_right = normalize_batch(modes_right, normas)
    modes_left = normalize_batch(modes_left, normas)

    # normalize so max value for the right eigenmodes is 1
    max_indices = np.argmax(np.abs(modes_right), axis=0)
    modes_right_max_values = np.take_along_axis(
        modes_right, np.expand_dims(max_indices, axis=0), axis=0
    )
    modes_right = modes_right / modes_right_max_values
    modes_left = modes_left * modes_right_max_values

    return modes_right, modes_left


def dot(a, b):
    """
    Compute the dot product of two arrays using Einstein summation convention.

    Parameters
    ----------
    a : array_like
        The first input array.
    b : array_like
        The second input array.

    Returns
    -------
    array
        The dot product of the input arrays.
    """

    return np.einsum("i...,i...", a, b)


def matvecprod(a, b):
    """
    Compute the matrix-vector product of two arrays using Einstein summation convention.

    Parameters
    ----------
    a : array_like
        The matrix input array.
    b : array_like
        The vector input array.

    Returns
    -------
    array
        The matrix-vector product of the input arrays.
    """
    return np.einsum("ij...,j...->i...", a, b)


def vecmatprod(a, b):
    """
    Compute the vector-matrix product of two arrays using Einstein summation convention.

    Parameters
    ----------
    a : array_like
        The matrix input array.
    b : array_like
        The vector input array.

    Returns
    -------
    array
        The matrix-vector product of the input arrays.
    """
    return np.einsum("i...,ij...->j...", a, b)


def matmatprod(a, b):
    """
    Compute the matrix-matrix product of two arrays using Einstein summation convention.

    Parameters
    ----------
    a : array_like
        The first matrix input array.
    b : array_like
        The second matrix input array.

    Returns
    -------
    array
        The matrix-matrix product of the input arrays.
    """
    return np.einsum("ij...,jk...->ik...", a, b)


def pad(coefficients, padding_size):
    """
    Pad an array with zeros if `padding_size` is positive

    Parameters
    ----------
    coefficients : array_like
        The array to pad
    padding_size : int
        The number of components to pad the array with

    Returns
    -------
    y : array_like
        The padded array
    """
    return np.array([0] * padding_size + list(coefficients) + [0] * padding_size)


def index_shift(i, Nh):
    """
    Shift an index to the index of the Fourier coefficient of the same order
    in the padded array.

    Parameters
    ----------
    i : int
        The index in the unpadded array
    Nh : int
        The integer corresponding to order 0 in the padded array

    Returns
    -------
    int
        The corresponding index in the padded array
    """
    return i - Nh


def build_field_time(coeffs, Omega, Nh, t):
    """
    Build the time dependent field from the coefficients of the Fourier series.

    Parameters
    ----------
    coeffs : array_like
        The coefficients of the Fourier series.
    Omega : float
        The frequency of the modulation.
    Nh : int
        The integer corresponding to order 0 in the padded array.
    t : array_like
        The time points at which to compute the field.

    Returns
    -------
    array_like
        The field at the specified time points.
    """
    field = 0
    for i, coeff in enumerate(coeffs):
        n = index_shift(i, Nh)
        field += coeff * np.exp(n * 1j * Omega * t)
    return field


def move_first_two_axes_to_end(arr):
    """
    Move the first two axes of an array to the end.

    Parameters
    ----------
    arr : array_like
        The input array.

    Returns
    -------
    array_like
        The array with the first two axes moved to the end.
    """
    if arr.ndim > 2:
        return np.moveaxis(arr, [0, 1], [-2, -1])
    return arr  # Return unchanged if ndim <= 2


def move_first_axes_to_end(arr):
    """
    Move the first axis of an array to the end.

    Parameters
    ----------
    arr : array_like
        The input array.

    Returns
    -------
    array_like
        The array with the first axis moved to the end.
    """

    if arr.ndim > 1:
        return np.moveaxis(arr, [0], [-1])
    return arr


def move_last_two_axes_to_beginning(arr):
    """
    Move the last two axes of an array to the beginning.

    Parameters
    ----------
    arr : array_like
        The input array.

    Returns
    -------
    array_like
        The array with the last two axes moved to the beginning,
        or unchanged if the number of dimensions is 2 or less.
    """

    if arr.ndim > 2:
        return np.moveaxis(arr, [-2, -1], [0, 1])
    return arr  # Return unchanged if ndim <= 2


def move_last_axes_to_beginning(arr):
    """
    Move the last axis of an array to the beginning.

    Parameters
    ----------
    arr : array_like
        The input array.

    Returns
    -------
    array_like
        The array with the last axis moved to the beginning,
        or unchanged if the number of dimensions is 1.
    """
    if arr.ndim > 1:
        return np.moveaxis(arr, [-1], [0])
    return arr


def fresnel(omegas, eps_slab, eps_plus, eps_minus, L):
    """
    Compute the Fresnel coefficients for a static slab with the given thickness
    and dielectric properties.

    Parameters
    ----------
    omegas : array_like
        The frequencies at which to compute the Fresnel coefficients
    eps_slab : complex
        The permittivity of the slab
    eps_plus : complex
        The permittivity of the material above the slab
    eps_minus : complex
        The permittivity of the material below the slab
    L : float
        The thickness of the slab

    Returns
    -------
    rf, tf : array_like
        The reflection and transmission Fresnel coefficients, respectively
    """
    r12 = (eps_plus**0.5 - eps_slab**0.5) / (eps_plus**0.5 + eps_slab**0.5)
    r23 = (eps_slab**0.5 - eps_minus**0.5) / (eps_slab**0.5 + eps_minus**0.5)
    t12 = (2 * eps_plus**0.5) / (eps_plus**0.5 + eps_slab**0.5)
    t23 = (2 * eps_slab**0.5) / (eps_minus**0.5 + eps_slab**0.5)

    rf = (r12 + r23 * np.exp(1j * 2 * omegas * eps_slab**0.5 * L)) / (
        1 + r12 * r23 * np.exp(1j * 2 * omegas * eps_slab**0.5 * L)
    )
    tf = (t12 * t23 * np.exp(1j * eps_slab**0.5 * omegas * L)) / (
        1 + r12 * r23 * np.exp(1j * 2 * omegas * eps_slab**0.5 * L)
    )
    return rf, tf


def dimhandler(f):
    """
    A decorator to handle different dimensionalities of input arrays for a function.

    This decorator adjusts the input arrays' dimensions to ensure compatibility with the
    wrapped function, which expects certain input shapes. It supports scalar, 1D, and 2D
    array inputs for `omegas`, and modifies `eigenvalues`, `modes`, and any additional
    arguments accordingly.

    Parameters
    ----------
    f : function
        The function to be decorated. It should accept the parameters `self`, `omegas`,
        `eigenvalues`, `modes`, and additional arguments.

    Returns
    -------
    function
        A wrapper function that processes the dimensionality of the input arrays and
        calls the original function with appropriately shaped inputs.

    Raises
    ------
    ValueError
        If the number of dimensions in `omegas` is unsupported.
    """

    def wrapper(self, omegas, eigenvalues, modes, *args):
        omegas = np.array(omegas)
        dim = omegas.ndim
        if dim == 0:
            return f(
                self,
                np.array([omegas]),
                eigenvalues[:, np.newaxis],
                modes[:, :, np.newaxis],
                *(arg[:, :, np.newaxis] for arg in args),
            )[:, :, 0]
        if dim == 1:
            return f(self, omegas, eigenvalues, modes, *args)
        if dim == 2:
            out = []
            for j, _omegas in enumerate(omegas):
                _eigenvalues = eigenvalues[:, j]
                _modes = modes[:, :, j]
                _args = [arg[:, :, j] for arg in args]
                out.append(f(self, _omegas, _eigenvalues, _modes, *_args))
            return np.transpose(np.stack(out, axis=-1), (0, 1, 3, 2))
        msg = f"Unsupported number of dimensions: {omegas.ndim}"
        raise ValueError(msg)

    return wrapper
