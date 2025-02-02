#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3


from pytmod import Material
import numpy as bk

def test_material():
    eps0 = 5.25
    deps = 0.85

    eps_fourier = [
        -deps / (2 * 1j),
        eps0,
        deps / (2 * 1j),
    ]
    Omega = 2.54

    # mat = Material(eps_fourier[:-1], Omega)

    mat = Material(eps_fourier, Omega, Npad=0)
    # print(mat.modulation_frequency)
    # mat.modulation_frequency = 10

    # print(mat.eps_fourier)
    # print(mat.nh)

    # mat.Npad = 1
    # print(mat.eps_fourier)
    # print(mat.nh)

    # mat.Npad = 1
    # print(mat.eps_fourier)
    # print(mat.nh)

    # mat.eps_fourier = [1]
    # print(mat.eps_fourier)
    # print(mat.nh)

    omegas = bk.array([0.15, 0.48])
    for i, om in enumerate([omegas, 0.15]):
        M = mat.build_matrix(om)
        kns, ens = mat.eigensolve(om)

        print(M.shape)
        print(kns.shape)
        print(ens.shape)
        if i == 0:
            kns = kns[:, i]
            ens = ens[:, :, i]
        print(kns)
        print(ens)
        print("----------")
