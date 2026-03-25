# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod
"""
Boundary continuity check for Floquet-Mie cylinder scattering (TM and TE).

TM polarisation (Psi = E_z):
  BC1 (E_z continuity):
    sum_q [a_q J_n(k_q^ext R) + b_q H_n(k_q^ext R)] e^{-i omega_q t}
    = sum_j c_j J_n(k_j R) [sum_q e_{jq} e^{-i omega_q t}]

  BC2 (H_theta ~ d_r E_z continuity):
    sum_q [a_q k_q J_n'(k_q^ext R) + b_q k_q H_n'(k_q^ext R)] e^{-i omega_q t}
    = sum_j c_j k_j J_n'(k_j R) [sum_q e_{jq} e^{-i omega_q t}]

TE polarisation (Psi = H_z):
  BC1 (H_z continuity): same structure but with h_{jq} = e_{jq}/omega_q
    sum_q [a_q J_n(k_q^ext R) + b_q H_n(k_q^ext R)] e^{-i omega_q t}
    = sum_j c_j J_n(k_j R) [sum_q h_{jq} e^{-i omega_q t}]

  BC2 (E_theta ~ eps^{-1} d_r H_z continuity):
    sum_q (k_q^ext/eps_ext) [a_q J_n'(k_q^ext R) + b_q H_n'(k_q^ext R)] e^{-i omega_q t}
    = sum_j c_j (k_j^3/omega_q^2) J_n'(k_j R) [sum_q h_{jq} e^{-i omega_q t}]

Both are checked:
  (a) in the matrix sense: per Floquet channel q and azimuthal order n
  (b) in the field sense: evaluated on a ring at r = R for a set of angles and times
"""

from __future__ import annotations

import numpy as np
from scipy.special import h1vp, hankel1, jv, jvp


def check_boundary_matrix(
    cyl,
    omegas,
    eigenvalues,
    modes,
    rhs,
    solution,
    inner_coeffs,
    n_range,
    polarization="TM",
    tol=1e-6,
):
    """
    Check the two boundary conditions in matrix/vector form, per Floquet channel.

    For each azimuthal order n and each frequency omega, verifies:
      BC1:  F c = P a + Q b        (field continuity)
      BC2:  G c = P' a + Q' b      (derivative/flux continuity)

    The matrices F, G, P, Q, P', Q' are defined in plans/scattering_matrix.tex and
    returned by cyl.build_sub_matrices for both TM and TE polarizations.

    Parameters
    ----------
    cyl : Cylinder
        The cylinder object.
    omegas : array_like
        Frequencies, shape (nw,).
    eigenvalues : array_like
        Material eigenvalues k_j, shape (nh, nw).
    modes : array_like
        Material eigenvectors e_{jq}, shape (nh, nh, nw).
    rhs : array_like
        Input vector a, shape (n_az, nh, nw).
    solution : array_like
        Scattered amplitudes b, shape (n_az, nh, nw).
    inner_coeffs : array_like
        Interior coefficients c, shape (n_az, nh, nw).
    n_range : list of int
        Azimuthal orders used.
    polarization : str
        'TM' or 'TE'.
    tol : float
        Tolerance for pass/fail.

    Returns
    -------
    dict with keys 'bc1_residuals', 'bc2_residuals', 'passed'
    """
    nw = np.asarray(omegas).size
    nh = cyl.material.nh
    n_az = len(n_range)

    bc1_residuals = np.zeros((n_az, nh, nw))
    bc2_residuals = np.zeros((n_az, nh, nw))

    for ind_n, n in enumerate(n_range):
        P, Q, P_prime, Q_prime, F, G = cyl.build_sub_matrices(
            omegas, eigenvalues, modes, n, polarization
        )
        # shapes after build_sub_matrices with dimhandler:
        # P, Q, P_prime, Q_prime: (nw, nh)  [diagonal entries]
        # F, G: (nw, nh, nh)  [F[w,q,j], G[w,q,j]]

        a = rhs[ind_n]  # (nh, nw)
        b = solution[ind_n]  # (nh, nw)
        c = inner_coeffs[ind_n]  # (nh, nw)

        for iw in range(nw):
            Fw = F[iw]  # (nh, nh)
            Gw = G[iw]  # (nh, nh)
            Pw = P[iw]  # (nh,)
            Qw = Q[iw]
            Ppw = P_prime[iw]
            Qpw = Q_prime[iw]
            aw = a[:, iw]  # (nh,)
            bw = b[:, iw]
            cw = c[:, iw]

            # BC1: F c - (P a + Q b) should be zero  [shape (nh,)]
            bc1 = Fw @ cw - (Pw * aw + Qw * bw)
            bc1_residuals[ind_n, :, iw] = np.abs(bc1)

            # BC2: G c - (P' a + Q' b) should be zero  [shape (nh,)]
            bc2 = Gw @ cw - (Ppw * aw + Qpw * bw)
            bc2_residuals[ind_n, :, iw] = np.abs(bc2)

    max_bc1 = bc1_residuals.max()
    max_bc2 = bc2_residuals.max()
    passed = (max_bc1 < tol) and (max_bc2 < tol)

    print("=" * 60)
    print(f"Matrix boundary check ({polarization})")
    print("=" * 60)
    for ind_n, n in enumerate(n_range):
        print(
            f"  n={n:+d}:  BC1 max residual = {bc1_residuals[ind_n].max():.3e}"
            f"   BC2 max residual = {bc2_residuals[ind_n].max():.3e}"
        )
    print(f"\n  Overall BC1 max = {max_bc1:.3e}  {'PASS' if max_bc1 < tol else 'FAIL'}")
    print(f"  Overall BC2 max = {max_bc2:.3e}  {'PASS' if max_bc2 < tol else 'FAIL'}")
    print("=" * 60)

    return {
        "bc1_residuals": bc1_residuals,
        "bc2_residuals": bc2_residuals,
        "passed": passed,
    }


def check_boundary_field(
    cyl,
    omegas,
    eigenvalues,
    modes,
    rhs,
    solution,
    inner_coeffs,
    n_range,
    polarization="TM",
    t=0.0,
    n_theta=360,
    tol=1e-6,
):
    """
    Check boundary continuity by evaluating the physical fields on a ring at r = R.

    TM: checks continuity of E_z and of k_q J_n'(k_q R) (proportional to H_theta).
    TE: checks continuity of H_z and of (k_q/eps_ext) J_n'(k_q R) (proportional to E_theta).

    Computes at r = R for all angles theta and the given time t:
      exterior (TM/TE BC1):
        sum_n sum_q [a_q J_n(k_q R) + b_q H_n(k_q R)] e^{i n theta} e^{-i omega_q t}
      interior (TM BC1):
        sum_n sum_j c_j J_n(k_j R) [sum_q e_{jq} e^{-i omega_q t}] e^{i n theta}
      interior (TE BC1):
        sum_n sum_j c_j J_n(k_j R) [sum_q h_{jq} e^{-i omega_q t}] e^{i n theta}
        where h_{jq} = e_{jq} / omega_q

    and for BC2:
      exterior (TM):  sum_q k_q [a_q J_n'(k_q R) + b_q H_n'(k_q R)] e^{i n theta} e^{-i omega_q t}
      interior (TM):  sum_j c_j k_j J_n'(k_j R) [sum_q e_{jq} e^{-i omega_q t}] e^{i n theta}
      exterior (TE):  sum_q (k_q/eps_ext) [a_q J_n'(k_q R) + b_q H_n'(k_q R)] e^{i n theta} e^{-i omega_q t}
      interior (TE):  sum_j c_j (k_j^3/omega_q^2) J_n'(k_j R) [sum_q h_{jq} e^{-i omega_q t}] e^{i n theta}
                    = sum_j c_j (k_j^3/omega_q^3) J_n'(k_j R) [sum_q e_{jq} e^{-i omega_q t}] e^{i n theta}

    Parameters
    ----------
    cyl : Cylinder
    omegas : array_like, shape (nw,)
    eigenvalues : array_like, shape (nh, nw)
    modes : array_like, shape (nh, nh, nw)
    rhs : array_like, shape (n_az, nh, nw)
    solution : array_like, shape (n_az, nh, nw)
    inner_coeffs : array_like, shape (n_az, nh, nw)
    n_range : list of int
    polarization : str
        'TM' or 'TE'.
    t : float
        Time at which to evaluate.
    n_theta : int
        Number of angles on the ring.
    tol : float

    Returns
    -------
    dict with keys 'max_discontinuity', 'passed', and per-omega results
    """
    R = cyl.radius
    Omega = cyl.material.modulation_frequency
    Nh = cyl.material.Nh
    nh = cyl.material.nh
    eps_ext = cyl.eps_ext

    omegas_arr = np.asarray(omegas).reshape(-1)
    nw = omegas_arr.size

    # eigenvalues[j, w] and modes[q, j, w]
    eigenvalues_f = np.asarray(eigenvalues).reshape(nh, nw)  # (nh, nw)  [j, w]
    modes_f = np.asarray(modes).reshape(nh, nh, nw)  # (nh, nh, nw)  [q, j, w]

    rhs_f = np.asarray(rhs).reshape(len(n_range), nh, nw)
    solution_f = np.asarray(solution).reshape(len(n_range), nh, nw)
    inner_coeffs_f = np.asarray(inner_coeffs).reshape(len(n_range), nh, nw)

    q_vals = np.arange(-Nh, Nh + 1)  # (nh,)
    n_vals = np.asarray(n_range)
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    # omega_q[q, w]  and  k_ext_q[q, w]
    omega_q = omegas_arr[None, :] - q_vals[:, None] * Omega  # (nh, nw)
    k_ext_q = omega_q * eps_ext**0.5  # (nh, nw)
    time_phase = np.exp(-1j * omega_q * t)  # (nh, nw)

    # Bessel/Hankel values at r = R for all q and w
    kR_ext = k_ext_q * R  # (nh, nw)
    kR_int = eigenvalues_f * R  # (nh, nw)

    max_disc = np.zeros(nw)

    print("=" * 60)
    print(f"Field boundary check at r=R ({polarization}, t={t})")
    print("=" * 60)

    for iw in range(nw):
        # --- BC1: field continuity ---

        # exterior: sum_n sum_q [a_q J_n(k_q R) + b_q H_n(k_q R)] e^{i n theta} e^{-i omega_q t}
        ext_bc1 = np.zeros(n_theta, dtype=complex)
        for ind_n, n in enumerate(n_vals):
            ang = np.exp(1j * n * thetas)  # (n_theta,)
            for iq in range(nh):
                a_nq = rhs_f[ind_n, iq, iw]
                b_nq = solution_f[ind_n, iq, iw]
                J_val = jv(n, kR_ext[iq, iw])
                H_val = hankel1(n, kR_ext[iq, iw])
                tp = time_phase[iq, iw]
                ext_bc1 += ang * (a_nq * J_val + b_nq * H_val) * tp

        # interior (TM): f_{jq} = e_{jq} = modes_f[q, j, w]
        # interior (TE): f_{jq} = h_{jq} = e_{jq}/omega_q = modes_f[q, j, w] / omega_q[q, w]
        if polarization == "TM":
            # f[q, j, w] = e_{jq}(w)
            f_field = modes_f[:, :, iw]  # (q, j) = (nh, nh)
        else:
            # f[q, j, w] = h_{jq}(w) = e_{jq}/omega_q
            f_field = (
                modes_f[:, :, iw] / omega_q[:, iw, None]
            )  # (q, j) divide by omega_q for each q

        # f_time[j] = sum_q f_{jq}(w) e^{-i omega_q t}
        # f_field[q, j], time_phase[q, w] -> sum over q
        f_time = f_field.T @ time_phase[:, iw]  # (j,)

        int_bc1 = np.zeros(n_theta, dtype=complex)
        for ind_n, n in enumerate(n_vals):
            ang = np.exp(1j * n * thetas)  # (n_theta,)
            for ij in range(nh):
                c_nj = inner_coeffs_f[ind_n, ij, iw]
                J_val = jv(n, kR_int[ij, iw])
                int_bc1 += ang * c_nj * J_val * f_time[ij]

        disc_bc1 = np.abs(ext_bc1 - int_bc1)
        disc = disc_bc1  # BC1 discontinuity is the "total" field check

        # --- BC2: derivative/flux continuity ---

        # exterior (TM):  k_q [a_q J_n'(k_q R) + b_q H_n'(k_q R)]
        # exterior (TE):  (k_q/eps_ext) [a_q J_n'(k_q R) + b_q H_n'(k_q R)]
        ext_bc2 = np.zeros(n_theta, dtype=complex)
        for ind_n, n in enumerate(n_vals):
            ang = np.exp(1j * n * thetas)
            for iq in range(nh):
                a_nq = rhs_f[ind_n, iq, iw]
                b_nq = solution_f[ind_n, iq, iw]
                tp = time_phase[iq, iw]
                kq = k_ext_q[iq, iw]
                if polarization == "TM":
                    prefactor = kq
                else:
                    prefactor = kq / eps_ext
                ext_bc2 += (
                    ang
                    * prefactor
                    * (a_nq * jvp(n, kq * R) + b_nq * h1vp(n, kq * R))
                    * tp
                )

        # interior (TM):  c_j k_j J_n'(k_j R) f_time[j]   (f = e_{jq})
        # interior (TE):  c_j G_j J_n'(k_j R)   where G_j = sum_q (k_j^3/omega_q^2) h_{jq} e^{-i omega_q t}
        #               = sum_q (k_j^3/omega_q^3) e_{jq} e^{-i omega_q t}
        int_bc2 = np.zeros(n_theta, dtype=complex)
        for ind_n, n in enumerate(n_vals):
            ang = np.exp(1j * n * thetas)
            for ij in range(nh):
                c_nj = inner_coeffs_f[ind_n, ij, iw]
                kj = eigenvalues_f[ij, iw]
                Jp_val = jvp(n, kR_int[ij, iw])
                if polarization == "TM":
                    # sum_q k_j e_{jq} e^{-i omega_q t} = k_j * f_time[j]  (f=e)
                    int_bc2 += ang * c_nj * kj * Jp_val * f_time[ij]
                else:
                    # sum_q (k_j^3/omega_q^2) h_{jq} e^{-i omega_q t}
                    # = sum_q (k_j^3/omega_q^3) e_{jq} e^{-i omega_q t}
                    g_te = np.sum(
                        (kj**3 / omega_q[:, iw] ** 3)
                        * modes_f[:, ij, iw]
                        * time_phase[:, iw]
                    )
                    int_bc2 += ang * c_nj * Jp_val * g_te

        disc_bc2 = np.abs(ext_bc2 - int_bc2).max()
        max_disc[iw] = disc.max()

        status = "PASS" if max_disc[iw] < tol else "FAIL"
        print(
            f"  omega={omegas_arr[iw]:.4f}:  "
            f"max|ext - int| = {max_disc[iw]:.3e}  "
            f"mean|ext - int| = {disc.mean():.3e}  {status}"
        )
        print(
            f"           BC1 (field):      {disc_bc1.max():.3e}  "
            f"BC2 (derivative): {disc_bc2:.3e}"
        )

    print(
        f"\n  Overall max discontinuity = {max_disc.max():.3e}  "
        f"{'PASS' if max_disc.max() < tol else 'FAIL'}"
    )
    print("=" * 60)

    return {"max_discontinuity": max_disc, "passed": (max_disc.max() < tol)}


def run_checks(
    cyl,
    omegas,
    eigenvalues,
    modes,
    rhs,
    solution,
    inner_coeffs,
    n_range,
    polarization="TM",
    t=0.0,
    tol=1e-6,
):
    """
    Run both matrix and field boundary checks and print a summary.

    Usage
    -----
    After solving the scattering problem:

        mat_result   = cyl.build_matrix(omegas, eigenvalues, modes, n, polarization)
        rhs          = cyl.build_rhs(omegas, incident_field, incident_angles, n)
        solution     = cyl.solve(mat_result, rhs)
        inner_coeffs = cyl.get_inner_coefficients(
                           omegas, eigenvalues, modes, n, polarization, rhs, solution)

        from check_boundary import run_checks
        run_checks(cyl, omegas, eigenvalues, modes,
                   rhs[None], solution[None], inner_coeffs[None],
                   n_range=[n], polarization="TM")

    Note: rhs, solution, inner_coeffs should have a leading n_az axis,
    so wrap with [None] if solving for a single azimuthal order.
    Works for both TM and TE polarizations.

    Parameters
    ----------
    cyl : Cylinder
    omegas : array_like, shape (nw,)
    eigenvalues : array_like, shape (nh, nw)
    modes : array_like, shape (nh, nh, nw)
    rhs : array_like, shape (n_az, nh, nw)
    solution : array_like, shape (n_az, nh, nw)
    inner_coeffs : array_like, shape (n_az, nh, nw)
    n_range : list of int
    polarization : str
        'TM' or 'TE'.
    t : float
        Time at which to evaluate field check.
    tol : float
        Tolerance for pass/fail.

    Returns
    -------
    res_mat : dict
        Matrix check results.
    res_field : dict
        Field check results.
    """
    res_mat = check_boundary_matrix(
        cyl,
        omegas,
        eigenvalues,
        modes,
        rhs,
        solution,
        inner_coeffs,
        n_range,
        polarization,
        tol,
    )
    res_field = check_boundary_field(
        cyl,
        omegas,
        eigenvalues,
        modes,
        rhs,
        solution,
        inner_coeffs,
        n_range,
        polarization,
        t,
        tol=tol,
    )

    print("\nSUMMARY")
    print("  Matrix check:", "PASS" if res_mat["passed"] else "FAIL")
    print("  Field check: ", "PASS" if res_field["passed"] else "FAIL")

    assert res_mat["passed"], "Boundary conditions not satisfied (matrix)."

    assert res_field["passed"], "Boundary conditions not satisfied (fields)."

    return res_mat, res_field
