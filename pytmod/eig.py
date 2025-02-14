#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod

__all__ = ["nonlinear_eigensolver"]

import logging
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import scipy.linalg as la
from skimage.feature import peak_local_max


import numpy as bk


def get_backend():
    return "numpy"


def cut_in_four(z0, z1, ratio_re=0.5, ratio_im=0.5):
    x0, x1, y0, y1 = z0.real, z1.real, z0.imag, z1.imag
    xm, ym = x0 + ratio_re * (x1 - x0), y0 + ratio_im * (y1 - y0)
    zm = xm + 1j * ym
    zb = xm + 1j * y0
    zt = xm + 1j * y1
    zl = x0 + 1j * ym
    zr = x1 + 1j * ym
    return (z0, zm), (zb, zr), (zm, z1), (zl, zt)


def get_residual(M, phi):
    """
    Compute the residual of the eigenvalue problem M @ phi = 0.

    Parameters
    ----------
    M : array
        The matrix of the eigenvalue problem
    phi : array
        The eigenvector

    Returns
    -------
    float
        The norm of the residual
    """

    return bk.abs(phi @ M @ phi)


def eig(C, D):
    """
    Solve the eigenvalue problem C @ phi = D @ phi * omega.

    Parameters
    ----------
    C : array
        The matrix of the eigenvalue problem
    D : array
        The matrix of the eigenvalue problem

    Returns
    -------
    omega : array
        The eigenvalues
    phi : array
        The eigenvectors
    """
    if get_backend() == "torch":
        invD = bk.linalg.inv(D)
        return bk.linalg.eig(invD @ C)
    return la.eig(C, D)


class ConvergenceError(Exception):
    pass


class EigenvalueError(Exception):
    pass


def null(M, mult=1):
    """
    Compute the nullspace of a matrix M.

    Parameters
    ----------
    M : array
        The matrix
    mult : int, optional
        The multiplicity of the nullspace

    Returns
    -------
    array
        The nullspace of M
    """
    e, v = bk.linalg.eig(M)
    srt = bk.argsort(bk.abs(e))
    return v[:, srt[0:mult]]


def block(matrices):
    """
    Stack a list of matrices into a single matrix.

    Parameters
    ----------
    matrices : list of arrays
        The matrices to stack

    Returns
    -------
    array
        The stacked matrix
    """
    if get_backend() == "torch":
        return bk.cat([bk.cat(m, dim=1) for m in matrices], dim=0)
    return bk.block(matrices)


def dot(a, b):
    """
    Compute the dot product of two arrays, either with torch.matmul or numpy.dot.

    Parameters
    ----------
    a : array
        The first array
    b : array
        The second array

    Returns
    -------
    array
        The dot product of a and b
    """
    if get_backend() == "torch":
        return bk.matmul(a, b)
    else:
        return bk.dot(a, b)


# adapted from https://github.com/DavidPowell/OpenModes/blob/161bd0b30036c98caf4ab0cd463032a4ba22a382/openmodes/eig.py#L193


def eig_newton(
    func,
    lambda_0,
    x_0=None,
    tol=1e-6,
    max_iter=20,
    func_gives_der=False,
    G=None,
    args=[],
    weight="rayleigh symmetric",
    y_0=None,
):
    """Solve a nonlinear eigenvalue problem by Newton iteration

    Parameters
    ----------
    func : function
        The function with input `lambda` which returns the matrix
    lambda_0 : complex
        The starting guess for the eigenvalue
    x_0 : ndarray
        The starting guess for the eigenvector
    tol : float
        The relative tolerance in the eigenvalue for convergence
    max_iter : int
        The maximum number of iterations to perform
    func_gives_der : boolean, optional
        If `True`, then the function also returns the derivative as the second
        returned value. If `False` finite differences will be used instead,
        which will have reduced accuracy
    args : list, optional
        Any additional arguments to be supplied to `func`
    weight : string, optional
        How to perform the weighting of the eigenvector

        'max element' : The element with largest magnitude will be preserved

        'rayleigh' : Rayleigh iteration for Hermitian matrices will be used

        'rayleigh symmetric' : Rayleigh iteration for complex symmetric
        (i.e. non-Hermitian) matrices will be used

        'rayleigh asymmetric' : Rayleigh iteration for general matrices

    y_0 : ndarray, optional
        For 'rayleigh asymmetric weighting', this is required as the initial
        guess for the left eigenvector

    Returns
    -------
    res : dictionary
        A dictionary containing the following members:

        `eigval` : the eigenvalue

        'eigvec' : the eigenvector

        'iter_count' : the number of iterations performed

        'delta_lambda' : the change in the eigenvalue on the final iteration


    See:
    1.  P. Lancaster, Lambda Matrices and Vibrating Systems.
        Oxford: Pergamon, 1966.

    2.  A. Ruhe, “Algorithms for the Nonlinear Eigenvalue Problem,”
        SIAM J. Numer. Anal., vol. 10, no. 4, pp. 674–689, Sep. 1973.

    """

    # import time
    # import numpy as np
    # t = -time.time()

    has_y_0 = y_0 is not None
    has_x_0 = x_0 is not None
    x_s = x_0
    lambda_s = lambda_0

    logging.debug("Searching for zeros with eig_newton")
    logging.debug("Starting guess %+.4e %+.4ej" % (lambda_0.real, lambda_0.imag))

    converged = False

    if not func_gives_der:
        # evaluate at an arbitrary nearby starting point to allow finite
        # differences to be taken
        step = tol * 10
        step = max(1e-3, step)
        step = (1 + 1j) * step
        # lambda_sm = lambda_0 * (1 + (1 + 1j) * step)
        lambda_sm = lambda_0 + step
        T_sm = func(lambda_sm, *args)

    # lambda_history = [lambda_s]
    # plt.plot(lambda_s.real,lambda_s.imag,"xk")
    # plt.xlim(0.2,1.3)
    # plt.ylim(-0.3,-0.0)

    for iter_count in range(max_iter):
        if func_gives_der:
            T_s, T_ds = func(lambda_s, *args)
        else:
            T_s = func(lambda_s, *args)
            T_ds = (T_s - T_sm) / (lambda_s - lambda_sm)

        if not has_x_0 and iter_count == 0:
            x_s = null(T_s)[:, 0]
            x_s /= (x_s @ T_ds @ x_s) ** 0.5

        if not has_y_0 and iter_count == 0 and weight.lower() == "rayleigh asymmetric":
            y_s = null(T_s.conj().T)[:, 0]
            y_s /= (y_s @ T_ds.conj().T @ y_s) ** 0.5

        w = dot(T_ds, x_s)
        if get_backend() == "torch":
            # import torch
            # A = torch.ones(3,3)
            # A_LU, pivots, info = torch.linalg.lu_factor_ex(A, check_errors=False)
            # A_LU_solvable = A_LU[info != 0]
            # pivots_solvable = A_LU[info != 0]
            # X = torch.linalg.lu_solve(B, A_LU_solvable, pivots_solvable)

            T_s_lu = bk.linalg.lu_factor_ex(T_s, check_errors=False)
            # T_s_lu = bk.linalg.lu_factor(T_s)
            u = bk.linalg.lu_solve(*T_s_lu[:2], bk.stack([w]).T).flatten()
        else:
            T_s_lu = la.lu_factor(T_s)
            u = la.lu_solve(T_s_lu, w)
            # u = la.solve(T_s, dot(T_ds, x_s))

        # if known_vects is supplied, we should take this into account when
        # finding v
        if weight.lower() == "max element":
            v_s = bk.zeros_like(x_s)
            v_s[bk.argmax(bk.abs(x_s))] = 1.0
        elif weight.lower() == "rayleigh":
            v_s = dot(T_s.T, x_s.conj())
        elif weight.lower() == "rayleigh symmetric":
            v_s = dot(T_s.T, x_s)
        elif weight.lower() == "rayleigh asymmetric":
            w = dot(T_ds.T, y_s)

            if get_backend() == "torch":
                y_s = bk.linalg.lu_solve(
                    *T_s_lu[:2], bk.stack([w]), left=False
                ).flatten()
            else:
                y_s = la.lu_solve(T_s_lu, w, trans=1)
            # y_s = bk.linalg.lu_solve(T_s_lu, dot(T_ds.T, y_s), trans=1)
            y_s /= bk.sqrt(bk.sum(bk.abs(y_s) ** 2))
            v_s = dot(T_s.T, y_s)
        else:
            raise ValueError("Unknown weighting method %s" % weight)

        delta_lambda_abs = dot(v_s, x_s) / dot(v_s, u)

        delta_lambda = bk.abs(delta_lambda_abs / lambda_s)
        converged = delta_lambda < tol

        if converged:
            break

        lambda_s1 = lambda_s - delta_lambda_abs
        x_s1 = u / bk.sqrt(bk.sum(bk.abs(u) ** 2))
        x_s1 = u / (u @ T_ds @ u)

        # update variables for next iteration
        if not func_gives_der:
            lambda_sm = lambda_s
            T_sm = T_s

        lambda_s = lambda_s1
        x_s = x_s1

        # if not has_x_0 and iter_count==0:
        #     _u = null(T_s)[:, 0]
        #     x_s +=  _u/ bk.sqrt(bk.sum(bk.abs(u) ** 2))
        #     x_s /= 2
        logging.debug("%+.4e %+.4ej" % (lambda_s.real, lambda_s.imag))

        # if iter_count>0:
        #     line_cur[0].remove()
        #     pnt_cur[0].remove()
    #     lambda_history.append(lambda_s)
    #     pnt_cur = plt.plot(lambda_s.real,lambda_s.imag,"ob",ms=5)
    #     hst = np.array(lambda_history)
    #     line_cur = plt.plot(hst.real,hst.imag,"-ob",alpha=0.2,ms=5)
    #     # plt.pause(0.2)

    # t += time.time()
    # print(f"elapsed time {t:0.4f}s")
    # print(f"{iter_count} iterations")
    # print(lambda_s)
    # # case = "random" if x_0 is not None else "eig"
    # # np.savez(f"/home/bench/doc/data/tmp_{case}.npz",lambda_history=lambda_history)

    if not converged:
        # return
        raise ConvergenceError("maximum iterations reached, no convergence")
        return

    res = {
        "eigval": lambda_s,
        "iter_count": iter_count + 1,
        "delta_lambda": delta_lambda,
    }

    if weight.lower() == "rayleigh asymmetric":
        # Scale both the left and right eigenvector identically first
        y_s /= bk.sqrt(bk.vdot(y_s, y_s) / bk.vdot(x_s, x_s))

        # Then scale both to match the eigenvalue derivative
        dz_ds = dot(y_s, dot(T_ds, x_s))
        y_s /= bk.sqrt(dz_ds)
        res["eigvec_left"] = y_s

        x_s /= bk.sqrt(dz_ds)
        res["eigvec"] = x_s

    else:
        # scale the eigenvector so that the eigenvalue derivative is 1
        dz_ds = dot(x_s, dot(T_ds, x_s))
        x_s /= bk.sqrt(dz_ds)
        res["eigvec"] = x_s
        res["eigvec_left"] = x_s

    return res


_bounds_im = None


def _nonlinear_eigensolver(
    func,
    omega0,
    omega1,
    dfunc=None,
    guesses=None,
    weight="max element",
    init_vect="eig",
    strategy="peaks",
    peaks_estimate="det",
    tol=1e-6,
    max_iter=100,
    N_grid=(10, 10),
    N_guess_loc=0,
    Rloc=0.01,
    plot_solver=False,
    peak_ref=10,
    verbose=False,
    filter=True,
    scale=1,
    dim=None,
    **kwargs,
):
    return_left = kwargs["return_left"]
    if return_left:
        weight = "rayleigh asymmetric"
    global _bounds_im
    tnlevp = -time.time()
    Nguess_re, Nguess_im = N_grid

    Nre, Nim = int(Nguess_re * peak_ref), int(Nguess_im * peak_ref)

    func_gives_der = dfunc is not None

    if func_gives_der:

        def func_eig(omega):
            return func(omega), dfunc(omega)

    else:

        def func_eig(omega):
            return func(omega)

    if guesses is None:
        guesses_re = bk.linspace(omega0.real, omega1.real, Nre)
        guesses_im = bk.linspace(omega0.imag, omega1.imag, Nim)
        guesses_re, guesses_im = bk.meshgrid(guesses_re, guesses_im, indexing="ij")
        guesses = guesses_re + 1j * guesses_im
        guesses0 = guesses.flatten()
        if strategy == "grid":
            guesses = guesses0
        elif strategy == "peaks":
            omegas_re = bk.linspace(omega0.real, omega1.real, Nre)
            omegas_im = bk.linspace(omega1.imag, omega0.imag, Nim)
            omegas_re_, omegas_im_ = bk.meshgrid(omegas_re, omegas_im, indexing="ij")

            #################################################################
            # Compute complex plane quantities

            omegas_complex = omegas_re_ + 1j * omegas_im_
            Mc = func(omegas_complex)
            if get_backend() == "torch":
                Mc = bk.permute(Mc, (2, 3, 0, 1))
            else:
                Mc = bk.transpose(Mc, axes=(2, 3, 0, 1))
            if peaks_estimate == "eig":
                evs = bk.linalg.eigvals(Mc)
                srt = bk.argsort(bk.abs(evs), axis=-1)
                if get_backend() == "torch":
                    min_evs = bk.gather(evs, -1, srt)[:, :, 0]
                else:
                    min_evs = bk.take_along_axis(evs, srt, axis=-1)[:, :, 0]

                im = -bk.log10(bk.abs(min_evs))
            else:
                im = -bk.log10(bk.abs(bk.linalg.det(Mc)))
            if plot_solver:
                if _bounds_im is None:
                    # vmin = bk.min(im)
                    vmax = bk.max(im)
                    # vmin = bk.mean(im) - 1
                    vmin = bk.min(im)
                    _bounds_im = vmin, vmax
                else:
                    vmin, vmax = _bounds_im
                cmap = plt.gca().pcolormesh(
                    omegas_re / scale,
                    omegas_im / scale,
                    im.T,
                    cmap="inferno",
                    # vmin=vmin,
                    # vmax=vmax,
                )

                # plt.pause(2)
                # plt.colorbar(cmap)

            if get_backend() == "torch":
                im = im.numpy()
            coordinates = peak_local_max(im, min_distance=1)

            guess_peak = bk.array(
                [omegas_complex[coord[0], coord[1]] for coord in coordinates]
            )
            tloc = bk.linspace(0, 2 * bk.pi, N_guess_loc + 1)[:-1]
            guesses = []
            for guess_loc in guess_peak:
                guesses_ = (
                    guess_loc.real
                    + Rloc * bk.cos(tloc)
                    + 1j * (guess_loc.imag + Rloc * bk.sin(tloc))
                )
                guesses_ = bk.hstack([guesses_, guess_loc])
                guesses.append(guesses_)
            if len(guesses) > 0:
                guesses = bk.stack(guesses).flatten()
        elif strategy == "random":
            import random

            rand_re = bk.array([random.random() for i in range(N_grid[0] * N_grid[1])])
            rand_im = bk.array([random.random() for i in range(N_grid[0] * N_grid[1])])
            rand_re = (omega1.real - omega0.real) * rand_re + omega0.real
            rand_im = (omega1.imag - omega0.imag) * rand_im + omega0.imag
            guesses = rand_re + 1j * rand_im
            guesses = bk.stack(guesses).flatten()

        else:
            raise ValueError(f"Wrong strategy {strategy}")
            # ## linearize
            # guesses = []
            # for g in guesses0:
            #     M = func(g)
            #     e = bk.linalg.eigvals(M)
            #     srt = bk.argsort(bk.abs(e))
            #     guesses.append(e[srt][0])
            # guesses = bk.stack(guesses).flatten()

    guesses = bk.array(guesses)
    if plot_solver:
        if len(guesses) > 0:
            guess_plot = plt.gca().plot(
                guesses.real / scale, guesses.imag / scale, "xk"
            )
            plt.gca().set_xlabel(r"Re $\omega/\omega_p$")
            plt.gca().set_ylabel(r"Im $\omega/\omega_p$")
            plt.pause(0.01)

    def compute_eigenvalues(guesses, tol, max_iter):
        evs = []
        modes = []
        modes_left = []
        residuals = []
        ttot = 0
        for guess in guesses:
            if init_vect == "eig":
                vect_init = None
            elif init_vect == "random":
                if dim is None:
                    raise ValueError(
                        f"Please provide the dimension of your matrix with the keyword argument dim if init_vect = random"
                    )

                vect_init = bk.random.rand(dim) + 1j * bk.random.rand(dim)
                vect_init /= (vect_init.conj() @ vect_init) ** 0.5
            else:
                raise ValueError(
                    f"Wrong eigenvector initialization init_vect {init_vect}"
                )
            t0 = -time.time()
            try:
                res = eig_newton(
                    func_eig,
                    guess,
                    vect_init,
                    tol=tol,
                    max_iter=max_iter,
                    func_gives_der=func_gives_der,
                    weight=weight,
                )
            except ConvergenceError as e:
                logging.warning(e)
                res = None
            t0 += time.time()
            ttot += t0
            if res is not None:
                ev = res["eigval"]
                # print(res["iter_count"],res["delta_lambda"])
                if (
                    ev.real > omega0.real
                    and ev.real < omega1.real
                    and ev.imag < omega1.imag
                    and ev.imag > omega0.imag
                ):
                    # if True:
                    evs.append(ev)
                    modes.append(res["eigvec"])
                    modes_left.append(res["eigvec_left"])
                    residuals.append(res["delta_lambda"])
                    if plot_solver:
                        plt.gca().plot(ev.real / scale, ev.imag / scale, ".b")
                        # plt.xlim(omega0.real, omega1.real)
                        # plt.ylim(omega0.imag, omega1.imag)
                        plt.pause(0.001)

        if evs == []:
            # if plot_solver:
            #     try:
            #         [_.remove() for _ in guess_plot]
            #         plt.pause(0.01)
            #     except:
            #         pass
            raise EigenvalueError("No eigenvalues found")
        evs = bk.array(evs)
        modes = bk.stack(modes).T
        modes_left = bk.stack(modes_left).T
        residuals = bk.array(residuals)
        if filter:
            unique_indices = unique(evs, precision=tol * 100)
            modes = modes[:, unique_indices]
            modes_left = modes_left[:, unique_indices]
            evs = evs[unique_indices]
            residuals = residuals[unique_indices]

        if verbose:
            print_results(evs, residuals)

        srt = bk.argsort(bk.real(evs))
        if return_left:
            return evs[srt], modes[:, srt], modes_left[:, srt], residuals[srt]
        return evs[srt], modes[:, srt], residuals[srt]

    if len(guesses) > 0:
        if return_left:
            evs, modes, modes_left, residuals = compute_eigenvalues(
                guesses, tol, max_iter
            )
        else:
            evs, modes, residuals = compute_eigenvalues(guesses, tol, max_iter)

    else:
        raise EigenvalueError("No eigenvalues found")
    if plot_solver:
        try:
            [_.remove() for _ in guess_plot]
            plt.pause(0.01)
        except:
            pass
    tnlevp += time.time()
    if verbose:
        print(f"total time: {tnlevp:.4f}s")
    if return_left:
        return evs, modes, modes_left
    return evs, modes


def print_results(evs0, res0, message=None):
    print("")
    print("----------------------------------------------------")
    if message is not None:
        print(message)
    print(f"Found {len(evs0)} eigenvalues")
    print("----------------------------------------------------")
    print("              eigenvalue                 residual")
    print("----------------------------------------------------")
    for z0, r0 in zip(evs0, res0):
        space = " " if z0.real > 0 else ""
        print(f"{space}{z0:.14f}     {r0:.3e}")


def unique(evs, precision):
    evs_unique = []
    unique_indices = []
    for i, ev in enumerate(evs):
        evfloor = (
            bk.floor(ev.real / precision) + 1j * bk.floor(ev.imag / precision)
        ) * precision
        if not (evfloor in evs_unique):
            evs_unique.append(evfloor)
            unique_indices.append(i)
    return unique_indices


def gram_schmidt(A, dM):
    """
    Perform the Gram-Schmidt process on the columns of matrix A.

    A is a matrix whose columns are the vectors to be orthogonalized.

    dM is a list of n matrices, where n is the number of columns in A,
    and dM[i] is the metric for the i'th column of A.

    The function returns the matrix A, modified so that its columns are
    mutually orthogonal with respect to the metric dM.

    """
    # Get the number of vectors.
    n = A.shape[1]
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        for k in range(j):
            A[:, j] -= bk.dot(A[:, k], dM[k] @ A[:, j]) * A[:, k]
        A[:, j] /= bk.dot(A[:, j], dM[j] @ A[:, j]) ** 0.5
    return A


def _nleigsolve(func, *args, refine=False, **kwargs):
    """
    Try to solve the nonlinear eigenvalue problem using _nonlinear_eigensolver.
    If it fails, return empty arrays. If refine is True, refine the guesses
    and try again.

    Parameters
    ----------
    func : callable
        Function giving the matrix
    dfunc : callable
        Function giving the matrix derivative wrt the eigenvalue
    *args : arguments
        Arguments to be passed to _nonlinear_eigensolver
    refine : bool, optional
        Refine the guesses and try again if _nonlinear_eigensolver fails?
    **kwargs : keyword arguments
        Keyword arguments to be passed to _nonlinear_eigensolver

    Returns
    -------
    evs : array
        The eigenvalues
    eigenvectors : array
        The eigenvectors
    eigenvectors_left : array, optional
        The left eigenvectors
    """
    return_left = kwargs["return_left"]
    try:
        out = _nonlinear_eigensolver(func, *args, **kwargs)
        if return_left:
            evs0, eigenvectors0, eigenvectors_left0 = out
        else:
            evs0, eigenvectors0 = out
        if refine:
            kwargs["guesses"] = evs0
            kwargs["tol"] = max(kwargs["tol"] / 10, 1e-14)
            kwargs["max_iter"] *= 2
            if return_left:
                evs, eigenvectors, eigenvectors_left = _nonlinear_eigensolver(
                    func, *args, **kwargs
                )
            else:
                evs, eigenvectors = _nonlinear_eigensolver(func, *args, **kwargs)
        else:
            if return_left:
                return evs0, eigenvectors0, eigenvectors_left0
            return evs0, eigenvectors0
    except EigenvalueError:
        if return_left:
            return [], [], []
        return [], []
    if return_left:
        return evs, eigenvectors, eigenvectors_left
    return evs, eigenvectors


def plot_rectangle(ax, z0, z1, **kwargs):
    """
    Plot a rectangle on a given axes.

    Parameters
    ----------
    ax : Axes
        The axes to plot on
    z0, z1 : complex
        The corners of the rectangle, given in complex coordinates
    **kwargs : optional
        Any additional keyword arguments are passed to `patches.Rectangle`

    Returns
    -------
    patch : patches.Rectangle
        The plotted rectangle
    """
    patch = patches.Rectangle(
        (z0.real, z0.imag),  # (x,y)
        (z1 - z0).real,  # width
        (z1 - z0).imag,  # height
        **kwargs,
    )
    ax.add_patch(patch)
    plt.pause(0.01)
    return patch


def _nleigsolve_recursive(
    func,
    omega0,
    omega1,
    evs_=None,
    eigenvectors_=None,
    eigenvectors_left_=None,
    Ncutmax=50,
    Ncut=0,
    **kwargs,
):
    """
    Recursive version of _nleigsolve.

    Parameters
    ----------
    func : callable
        Function giving the matrix
    dfunc : callable
        Function giving the matrix derivative wrt the eigenvalue
    omega0, omega1 : complex
        The lower left and upper right corners of the rectangle
    evs_ : list of array, optional
        The eigenvalues found so far
    eigenvectors_ : list of array, optional
        The eigenvectors found so far
    eigenvectors_left_ : list of array, optional
        The left eigenvectors found so far
    Ncutmax : int, optional
        The maximum number of recursive calls
    Ncut : int, optional
        The current number of recursive calls
    **kwargs : keyword arguments
        Keyword arguments to be passed to _nleigsolve

    Returns
    -------
    evs_ : list of array
        The eigenvalues found
    eigenvectors_ : list of array
        The eigenvectors found
    eigenvectors_left_ : list of array, optional
        The left eigenvectors found
    """
    if evs_ == None:
        evs_ = []
    if eigenvectors_ == None:
        eigenvectors_ = []
    if eigenvectors_left_ == None:
        eigenvectors_left_ = []
    Ncut += 1

    return_left = kwargs["return_left"]

    peak_ref0 = kwargs["peak_ref"]
    evs_old = evs_.copy()
    if evs_old != []:
        _evs0 = bk.hstack(evs_old) if evs_old != [] else evs_old
        unique_indices0 = unique(_evs0, precision=kwargs["tol"] * 100)
        Nmodes0 = len(unique_indices0)
        Nmodes_in_region = 0
        for e in _evs0[unique_indices0]:
            if (
                e.real > omega0.real
                and e.imag > omega0.imag
                and e.real < omega1.real
                and e.imag < omega1.imag
            ):
                Nmodes_in_region += 1
        N_grid_ = max(kwargs["N_grid"])
        peak_ref1 = 1 / (Nmodes_in_region + 1) * peak_ref0

        peak_ref1 = max(peak_ref0, peak_ref1)
    else:
        Nmodes0 = 0
        Nmodes_in_region = 0
        peak_ref1 = peak_ref0
    peak_ref1 = peak_ref0  # max(peak_ref1, 4*peak_ref0)

    if kwargs["plot_solver"]:
        rectplot = plot_rectangle(
            plt.gca(),
            omega0 / kwargs["scale"],
            omega1 / kwargs["scale"],
            fill=True,
            linewidth=1,
            facecolor="#d8ff3d",
            alpha=0.1,
            edgecolor="#ff7c24",
        )
        rectplot1 = plot_rectangle(
            plt.gca(),
            omega0 / kwargs["scale"],
            omega1 / kwargs["scale"],
            fill=True,
            linewidth=1,
            facecolor="none",
            edgecolor="#ff7c24",
        )
        plt.pause(0.001)

    kwargs["peak_ref"] = peak_ref1
    if return_left:
        evs, eigenvectors, eigenvectors_left = _nleigsolve(
            func, omega0, omega1, **kwargs
        )

    else:
        evs, eigenvectors = _nleigsolve(func, omega0, omega1, **kwargs)

    kwargs["peak_ref"] = peak_ref0
    if len(evs) > 0:
        evs_.append(evs)
        eigenvectors_.append(eigenvectors)
        if return_left:
            eigenvectors_left_.append(eigenvectors_left)

    if evs_old != []:
        _evs1 = bk.hstack(evs_)
        unique_indices1 = unique(_evs1, precision=kwargs["tol"] * 100)
        cond = len(unique_indices0) == len(unique_indices1)
        # evs_ = [_evs1[i] for i in unique_indices1]
    else:
        cond = False

    # cond = bk.all([e in bk.hstack(evs_old) for e in evs]) if evs_old !=[] else False
    if kwargs["plot_solver"]:
        rectplot.remove()
    if len(evs) > 1 and not cond:
        cut = cut_in_four(omega0, omega1, 0.5, 0.5)
        for bounds in cut:
            omega0, omega1 = bounds
            out = _nleigsolve_recursive(
                func,
                omega0,
                omega1,
                evs_,
                eigenvectors_,
                eigenvectors_left_,
                Ncutmax=Ncutmax,
                Ncut=Ncut,
                **kwargs,
            )
            if return_left:
                evs_, eigenvectors_, eigenvectors_left_ = out
            else:
                evs_, eigenvectors_ = out

    if Ncut > Ncutmax:
        if return_left:
            return evs_, eigenvectors_, eigenvectors_left_
        return evs_, eigenvectors_
    if return_left:
        return evs_, eigenvectors_, eigenvectors_left_
    return evs_, eigenvectors_


def nonlinear_eigensolver(
    func,
    omega0,
    omega1,
    **kwargs,
):
    """
    Find eigenvalues and eigenvectors of a nonlinear eigenvalue problem.

    Parameters
    ----------
    func : callable
        Function giving the matrix
    omega0 : complex
        Lower bound of the frequency interval
    omega1 : complex
        Upper bound of the frequency interval
    **kwargs : keyword arguments
        See `defkwargs` below.

    Returns
    -------
    evs : array
        The eigenvalues
    eigenvectors : array
        The eigenvectors
    eigenvectors_left : array, optional
        The left eigenvectors.

    Notes
    -----
    The function uses a combination of grid search and Newton iterations to
    find the eigenvalues and eigenvectors of the nonlinear eigenvalue problem.
    If `recursive` is `True`, the function refines the guesses by dividing the
    frequency interval in four parts and solving the problem recursively.
    """
    defkwargs = dict(
        dfunc=None,
        guesses=None,
        recursive=False,
        weight="max element",
        init_vect="eig",
        strategy="peaks",
        peaks_estimate="det",
        tol=1e-6,
        max_iter=100,
        N_grid=(10, 10),
        N_guess_loc=0,
        Rloc=0.01,
        plot_solver=False,
        peak_ref=10,
        verbose=False,
        filter=True,
        scale=1,
        return_left=False,
    )
    for k, v in defkwargs.items():
        if k not in kwargs.keys():
            kwargs[k] = v

    return_left = kwargs["return_left"]
    recursive = kwargs["recursive"]
    # dfunc = kwargs["dfunc"]
    if recursive:
        if return_left:
            evs_, eigenvectors_, eigenvectors_left_ = _nleigsolve_recursive(
                func, omega0, omega1, **kwargs
            )
        else:
            evs_, eigenvectors_ = _nleigsolve_recursive(func, omega0, omega1, **kwargs)

    else:
        if return_left:
            evs_, eigenvectors_, eigenvectors_left_ = _nleigsolve(
                func, omega0, omega1, **kwargs
            )
        else:
            evs_, eigenvectors_ = _nleigsolve(func, omega0, omega1, **kwargs)
        if len(evs_) == 0:
            if return_left:
                return evs_, eigenvectors_, eigenvectors_left_
            return evs_, eigenvectors_
        # eigenvectors_ = [eigenvectors_]
        # evs_ = [evs_]
    if len(evs_) > 0:
        if not recursive:
            eigenvectors_ = [eigenvectors_]
            if return_left:
                eigenvectors_left_ = [eigenvectors_left_]
            evs_ = [evs_]
        evs = bk.hstack(evs_)
        unique_indices = unique(evs, precision=kwargs["tol"] * 100)
        evs = evs[unique_indices]
        isort = bk.argsort(evs.real)
        evs = evs[isort]
        eigenvectors = bk.hstack(eigenvectors_)[:, unique_indices][:, isort]

        if return_left:
            eigenvectors_left = bk.hstack(eigenvectors_left_)[:, unique_indices][
                :, isort
            ]

        if return_left:
            return evs, eigenvectors, eigenvectors_left
        return evs, eigenvectors
    if return_left:
        return evs_, eigenvectors_, eigenvectors_left_
    return evs_, eigenvectors_


def polyeig(A):
    """Solve the polynomial eigenvalue problem: (A0 + e A1 +...+  e**p Ap)x=0

    Parameters
    ----------
    A : list of arrays
        The arrays for each power of the polynomial (in increasing order)

    Returns
    -------
    tuple of arrays
        eigenvalues e and eigenvectors x

    """
    if len(A) <= 0:
        raise Exception("Provide at least one matrix")
    for Ai in A:
        if Ai.shape[0] != Ai.shape[1]:
            raise Exception("Matrices must be square")
        if Ai.shape != A[0].shape:
            raise Exception("All matrices must have the same shapes")

    n = A[0].shape[0]
    l = len(A) - 1
    # Assemble matrices for generalized problem
    C = block(
        [[bk.zeros((n * (l - 1), n)), bk.eye(n * (l - 1))], [-bk.column_stack(A[0:-1])]]
    )
    D = block(
        [
            [bk.eye(n * (l - 1)), bk.zeros((n * (l - 1), n))],
            [bk.zeros((n, n * (l - 1))), A[-1]],
        ]
    )
    # Solve generalized eigenvalue problem
    e, X = eig(C, D)
    if bk.all(bk.isreal(e)):
        e = bk.real(e)
    X = X[:n, :]

    # Sort eigenvalues/vectors
    # I = bk.argsort(e)
    # X = X[:,I]
    # e = e[I]

    # Scaling each mode by max
    maxi = bk.max(bk.abs(X), axis=0)

    if get_backend() == "torch":
        maxi = maxi[0]
    X /= bk.tile(maxi, (n, 1))

    return e, X
