#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


import numpy as bk


def pad(coefficients, padding_size):
    return bk.array([0] * padding_size + list(coefficients) + [0] * padding_size)


def build_field_time(coeffs, Omega, t):
    field = 0
    for i, coeff in enumerate(coeffs):
        n = index_shift(i)
        field += coeff * bk.exp(n * 1j * self.Omega * t)
    return field


def move_first_two_axes_to_end(arr):
    if arr.ndim > 2:
        return bk.moveaxis(arr, [0, 1], [-2, -1])
    return arr  # Return unchanged if ndim <= 2


def move_first_axes_to_end(arr):
    if arr.ndim > 1:
        return bk.moveaxis(arr, [0], [-1])
    return arr


def move_last_two_axes_to_beginning(arr):
    if arr.ndim > 2:
        return bk.moveaxis(arr, [-2, -1], [0, 1])
    return arr  # Return unchanged if ndim <= 2


def move_last_axes_to_beginning(arr):
    if arr.ndim > 1:
        return bk.moveaxis(arr, [-1], [0])
    return arr


def block(blocks):
    # First, concatenate horizontally along axis 1 (columns)
    row_stacked = [bk.concatenate(row, axis=1) for row in blocks]

    # Then, concatenate vertically along axis 0 (rows)
    return bk.concatenate(row_stacked, axis=0)


def fresnel(omegas, eps_slab, eps_plus, eps_minus, L):

    r12 = (eps_plus**0.5 - eps_slab**0.5) / (eps_plus**0.5 + eps_slab**0.5)
    r23 = (eps_slab**0.5 - eps_minus**0.5) / (eps_slab**0.5 + eps_minus**0.5)
    t12 = (2 * eps_plus**0.5) / (eps_plus**0.5 + eps_slab**0.5)
    t23 = (2 * eps_slab**0.5) / (eps_minus**0.5 + eps_slab**0.5)

    rf = (r12 + r23 * bk.exp(1j * 2 * omegas * eps_slab**0.5 * L)) / (
        1 + r12 * r23 * bk.exp(1j * 2 * omegas * eps_slab**0.5 * L)
    )
    tf = (t12 * t23 * bk.exp(1j * eps_slab**0.5 * omegas * L)) / (
        1 + r12 * r23 * bk.exp(1j * 2 * omegas * eps_slab**0.5 * L)
    )
    return rf, tf


def dimhandler(f):
    def wrapper(self, omegas, eigenvalues, modes):
        omegas = bk.array(omegas)
        dim = omegas.ndim
        if dim == 0:
            return f(
                self,
                bk.array([omegas]),
                eigenvalues[:, bk.newaxis],
                modes[:, :, bk.newaxis],
            )[:, :, 0]
        if dim == 1:
            return f(self, omegas, eigenvalues, modes)
        if dim == 2:
            out = []
            for j, _omegas in enumerate(omegas):
                _eigenvalues = eigenvalues[:, j]
                _modes = modes[:, :, j]
                out.append(f(self, _omegas, _eigenvalues, _modes))
            return bk.transpose(bk.stack(out, axis=-1), (0, 1, 3, 2))
        raise ValueError(f"Unsupported number of dimensions: {omegas.ndim}")

    return wrapper
