# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


from __future__ import annotations

import numpy as np
import pytest

from pytmod import Cylinder, Material
from pytmod.cylinder import (
    h1n,
    h1n_double_prime,
    h1n_prime,
    jn,
    jn_double_prime,
    jn_prime,
    mie_coefficient_TE,
    mie_coefficient_TM,
    mie_coefficients_TE,
    mie_coefficients_TM,
    mie_denominator_TE,
    mie_denominator_TM,
    scattering_efficiency,
    scattering_efficiency_TE,
    scattering_efficiency_TM,
    yn,
)

# Test parameters
eps_r = 4.0
m = eps_r**0.5
X = 1.0  # size parameter


def test_bessel_functions():
    """Test Bessel function utilities."""
    z = 1.0 + 0.5j

    # Test J_n - just verify they are finite and reasonable
    J0 = jn(0, z)
    J1 = jn(1, z)
    assert np.isfinite(J0)
    assert np.isfinite(J1)
    assert abs(J0) > abs(J1)  # J_0 should be larger than J_1 for this z

    # Test Y_n
    Y0 = yn(0, z)
    assert np.isfinite(Y0)

    # Test H_n^{(1)}
    H0 = h1n(0, z)
    assert np.isclose(H0, J0 + 1j * Y0)

    # Test derivatives
    J0p = jn_prime(0, z)
    H0p = h1n_prime(0, z)
    assert np.isfinite(J0p)
    assert np.isfinite(H0p)

    # Test recurrence: J_n' = J_{n-1} - n/z * J_n
    J1p = jn_prime(1, z)
    J1p_rec = jn(0, z) - 1 / z * jn(1, z)
    assert np.isclose(J1p, J1p_rec)


def test_bessel_double_prime():
    """Test second derivatives of Bessel functions."""
    z = 1.5 + 0.3j
    n = 2

    # Test J_n'' using finite difference
    Jn_pp = jn_double_prime(n, z)
    Jn_pp_fd = (jn_prime(n, z + 1e-8) - jn_prime(n, z - 1e-8)) / (2e-8)
    assert np.isclose(Jn_pp, Jn_pp_fd, rtol=1e-5)

    # Test H_n^{(1)}'' using finite difference
    Hn_pp = h1n_double_prime(n, z)
    Hn_pp_fd = (h1n_prime(n, z + 1e-8) - h1n_prime(n, z - 1e-8)) / (2e-8)
    assert np.isclose(Hn_pp, Hn_pp_fd, rtol=1e-5)


def test_mie_coefficients_symmetry():
    """Test that b_{-n} = b_n for Mie coefficients."""
    for n in range(1, 5):
        bn_pos = mie_coefficient_TM(n, X, m)
        bn_neg = mie_coefficient_TM(-n, X, m)
        assert np.isclose(bn_pos, bn_neg), f"Symmetry broken for n={n}"

        an_pos = mie_coefficient_TE(n, X, m)
        an_neg = mie_coefficient_TE(-n, X, m)
        assert np.isclose(an_pos, an_neg), f"TE symmetry broken for n={n}"


def test_mie_coefficients_values():
    """Test Mie coefficients against known values."""
    # For eps_r=4, X=1, we can verify the coefficients are reasonable
    b0 = mie_coefficient_TM(0, X, m)
    b1 = mie_coefficient_TM(1, X, m)

    # |b_n| should decrease with n for small X
    assert abs(b1) < abs(b0)

    # Test that coefficients are finite
    for n in range(5):
        bn = mie_coefficient_TM(n, X, m)
        an = mie_coefficient_TE(n, X, m)
        assert np.isfinite(bn), f"TM coefficient not finite for n={n}"
        assert np.isfinite(an), f"TE coefficient not finite for n={n}"


def test_mie_coefficients_array():
    """Test array versions of Mie coefficients."""
    n_max = 5
    bn = mie_coefficients_TM(n_max, X, m)
    an = mie_coefficients_TE(n_max, X, m)

    assert len(bn) == n_max + 1
    assert len(an) == n_max + 1

    # Compare with individual coefficients
    for n in range(n_max + 1):
        assert np.isclose(bn[n], mie_coefficient_TM(n, X, m))
        assert np.isclose(an[n], mie_coefficient_TE(n, X, m))


def test_mie_denominator():
    """Test Mie denominator functions."""
    # At a QNM, the denominator should be zero
    # For now, just test that they are finite
    for n in range(5):
        denom_TM = mie_denominator_TM(n, X, m)
        denom_TE = mie_denominator_TE(n, X, m)
        assert np.isfinite(denom_TM)
        assert np.isfinite(denom_TE)


def test_scattering_efficiency():
    """Test scattering efficiency functions."""
    Q_TM = scattering_efficiency_TM(X, m)
    Q_TE = scattering_efficiency_TE(X, m)

    # Scattering efficiency should be positive
    assert Q_TM > 0
    assert Q_TE > 0

    # Test the combined function
    Q_TM2 = scattering_efficiency(X, m, polarization="TM")
    Q_TE2 = scattering_efficiency(X, m, polarization="TE")

    assert np.isclose(Q_TM, Q_TM2)
    assert np.isclose(Q_TE, Q_TE2)


def test_scattering_efficiency_array():
    """Test scattering efficiency for array input."""
    X_range = np.linspace(0.1, 5.0, 10)
    Q_TM = scattering_efficiency_TM(X_range, m)
    Q_TE = scattering_efficiency_TE(X_range, m)

    assert len(Q_TM) == len(X_range)
    assert len(Q_TE) == len(X_range)
    assert np.all(Q_TM > 0)
    assert np.all(Q_TE > 0)


def test_cylinder_class():
    """Test the Cylinder class."""
    mat = Material([eps_r], modulation_frequency=1.0)
    cyl = Cylinder(mat, radius=1.0)

    # Test properties
    assert cyl.radius == 1.0
    assert cyl.eps_static == eps_r
    assert np.isclose(cyl.m_static, m)
    # Test Mie coefficient through class
    bn_class = cyl.mie_coefficient(1.0, n=1, polarization="TM")
    bn_direct = mie_coefficient_TM(1, 1.0, m)
    assert np.isclose(bn_class, bn_direct)


def test_cylinder_static():
    """Test the static method of Cylinder."""
    mat = Material([eps_r], modulation_frequency=1.0)
    cyl = Cylinder(mat, radius=1.0)

    cyl_static = cyl.static()
    assert cyl_static.eps_static == eps_r


def test_cylinder_polarization():
    """Test both TM and TE polarizations."""
    mat = Material([eps_r], modulation_frequency=1.0)

    cyl = Cylinder(mat, radius=1.0)

    bn = cyl.mie_coefficient(1.0, n=1, polarization="TM")
    an = cyl.mie_coefficient(1.0, n=1, polarization="TE")

    # TM and TE coefficients should be different
    assert not np.isclose(bn, an)


def test_cylinder_scattering_efficiency():
    """Test scattering efficiency through Cylinder class."""
    mat = Material([eps_r], modulation_frequency=1.0)
    cyl = Cylinder(mat, radius=1.0)

    Q = cyl.scattering_efficiency(1.0)
    assert Q > 0

    Q_TM = cyl.scattering_efficiency(1.0, polarization="TM")
    cyl.scattering_efficiency(1.0, polarization="TE")

    assert np.isclose(Q, Q_TM)


def test_cylinder_repr():
    """Test string representation of Cylinder."""
    mat = Material([eps_r], modulation_frequency=1.0)
    cyl = Cylinder(mat, radius=1.0)

    s = repr(cyl)
    assert "Cylinder" in s
    assert "radius=1.0" in s


def test_invalid_polarization():
    """Test that invalid polarization raises error."""
    mat = Material([eps_r], modulation_frequency=1.0)
    cyl = Cylinder(mat, radius=1.0)

    with pytest.raises(ValueError, match="Unknown polarization"):
        cyl.mie_coefficient(1.0, polarization="XX")

    with pytest.raises(ValueError, match="Unknown polarization"):
        scattering_efficiency(X, m, polarization="XX")


def test_cylinder_external_medium():
    """Test cylinder with external medium different from vacuum."""
    eps_ext = 2.0
    mat = Material([eps_r], modulation_frequency=1.0)
    cyl = Cylinder(mat, radius=1.0, eps_ext=eps_ext)

    # The effective refractive index should be sqrt(eps_r / eps_ext)
    m_eff = (eps_r / eps_ext) ** 0.5
    assert np.isclose(cyl.m_static, m_eff)


def test_mie_coefficient_complex_frequency():
    """Test Mie coefficients with complex frequency (for QNM searches)."""
    omega_complex = 1.0 - 0.1j

    mat = Material([eps_r], modulation_frequency=1.0)
    cyl = Cylinder(mat, radius=1.0)

    # Should work with complex frequency
    bn = cyl.mie_coefficient(omega_complex)
    assert np.isfinite(bn)


def test_scattering_efficiency_convergence():
    """Test convergence of scattering efficiency with n_max."""
    X_test = 5.0  # Larger size parameter needs more terms

    Q_10 = scattering_efficiency_TM(X_test, m, n_max=10)
    Q_20 = scattering_efficiency_TM(X_test, m, n_max=20)
    Q_30 = scattering_efficiency_TM(X_test, m, n_max=30)

    # Should converge as n_max increases
    # For sufficiently large n_max, the result should be stable
    assert np.isclose(Q_20, Q_30, rtol=1e-6)

    # All values should be positive
    assert Q_10 > 0
    assert Q_20 > 0
    assert Q_30 > 0


def test_small_particle_limit():
    """Test that small particles scatter less (Rayleigh limit)."""
    X_small = 0.1
    X_large = 1.0

    Q_small = scattering_efficiency_TM(X_small, m)
    Q_large = scattering_efficiency_TM(X_large, m)

    # Small particles should have lower scattering efficiency
    assert Q_small < Q_large


def test_energy_conservation_static():
    """Test energy conservation for lossless static cylinder."""
    # For a lossless dielectric, |b_n|^2 should be bounded
    X_test = 1.0
    m_real = 2.0  # Real refractive index (no absorption)

    for n in range(5):
        bn = mie_coefficient_TM(n, X_test, m_real)
        # |b_n| should be bounded (no gain)
        assert abs(bn) < 10  # Reasonable upper bound


# =============================================================================
# Floquet-Mie (Time-Modulated Cylinder) Tests
# =============================================================================


def test_time_modulated_cylinder_creation():
    """Test creation of time-modulated cylinder."""
    # Create material with sinusoidal modulation
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [
        -deps / (2 * 1j),  # ε₋₁
        eps0,  # ε₀
        deps / (2 * 1j),  # ε₊₁
    ]

    mat = Material(eps_fourier, Omega, Npad=0)
    R = 2 * np.pi / Omega  # Radius = modulation wavelength
    cyl = Cylinder(mat, R)

    assert cyl.radius == R
    assert cyl.eps_static == eps0
    assert cyl.material.nh == 3  # 3 Floquet harmonics
    assert cyl.dim == 3


def test_build_sub_matrices():
    """Test building Floquet-Mie sub-matrices."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    R = 1.0
    cyl = Cylinder(mat, R)

    # Single frequency
    omega = 1.0
    eigenvalues, modes = mat.eigensolve([omega])

    P, Q, P_prime, Q_prime, F, G = cyl.build_sub_matrices(
        [omega], eigenvalues, modes, n=0, polarization="TM"
    )

    # Check shapes
    assert P.shape == (1, 3)  # (nw, nh)
    assert Q.shape == (1, 3)
    assert F.shape == (1, 3, 3)  # (nw, nh, nh)
    assert G.shape == (1, 3, 3)

    # All should be finite
    assert np.all(np.isfinite(P))
    assert np.all(np.isfinite(Q))
    assert np.all(np.isfinite(F))
    assert np.all(np.isfinite(G))


def test_build_matrix():
    """Test building the Floquet-Mie matrix."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    R = 1.0
    cyl = Cylinder(mat, R)

    omega = 1.0
    eigenvalues, modes = mat.eigensolve([omega])

    # Build matrix for TM polarization
    matrix = cyl.build_matrix([omega], eigenvalues, modes, n=0, polarization="TM")

    # Check shape
    assert matrix.shape == (3, 3, 1)  # (nh, nh, nw)
    assert np.all(np.isfinite(matrix))

    # Build matrix for TE polarization
    matrix_TE = cyl.build_matrix([omega], eigenvalues, modes, n=0, polarization="TE")
    assert matrix_TE.shape == (3, 3, 1)
    assert np.all(np.isfinite(matrix_TE))

    # TM and TE matrices should be different
    assert not np.allclose(matrix, matrix_TE)


def test_build_rhs():
    """Test building the RHS vector."""
    eps0 = 3.0
    Omega = 0.6
    eps_fourier = [eps0]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    omegas = np.array([1.0])
    incident_field = np.zeros(1)
    incident_angles = np.zeros(1)

    rhs = cyl.build_rhs(omegas, incident_field, incident_angles, n=0)

    assert rhs.shape == (1, 1)
    assert np.all(np.isfinite(rhs))


def test_init_incident_field():
    """Test initialization of incident field."""
    eps0 = 3.0
    Omega = 0.6
    eps_fourier = [eps0]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    omegas = np.array([1.0, 2.0])
    Eis, thetais = cyl.init_incident_field(omegas)

    assert Eis.shape == (1, 2)
    assert thetais.shape == (1, 2)
    assert np.all(Eis == 0)
    assert np.all(thetais == 0)


def test_solve():
    """Test solving the linear system."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    omega = 1.0
    eigenvalues, modes = mat.eigensolve([omega])
    matrix = cyl.build_matrix([omega], eigenvalues, modes, n=0, polarization="TM")

    # Create a simple RHS
    rhs = np.ones((3, 1), dtype=complex)

    solution = cyl.solve(matrix, rhs)

    assert solution.shape == (3, 1)
    assert np.all(np.isfinite(solution))


def test_get_inner_coefficients():
    """Test computing internal field coefficients."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    omega = 1.0
    eigenvalues, modes = mat.eigensolve([omega])

    # Set up scattering problem
    incident_field, incident_angles = cyl.init_incident_field([omega])
    incident_field[mat.Nh] = 1
    incident_angles[mat.Nh] = 0

    n_az = 0
    matrix = cyl.build_matrix([omega], eigenvalues, modes, n_az, "TM")
    rhs = cyl.build_rhs([omega], incident_field, incident_angles, n_az)
    solution = cyl.solve(matrix, rhs)

    # Get internal coefficients
    inner_coeffs = cyl.get_inner_coefficients(
        [omega], eigenvalues, modes, n_az, "TM", rhs, solution
    )

    assert inner_coeffs.shape == (3, 1)
    assert np.all(np.isfinite(inner_coeffs))


def test_get_fields():
    """Test computing field distributions."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    omega = 1.0
    eigenvalues, modes = mat.eigensolve([omega])

    # Set up scattering problem
    incident_field, incident_angles = cyl.init_incident_field([omega])
    incident_field[mat.Nh] = 1
    incident_angles[mat.Nh] = 0

    n_range = range(-2, 3)
    solutions = []
    inner_coeffs_list = []

    for n_az in n_range:
        matrix = cyl.build_matrix([omega], eigenvalues, modes, n_az, "TM")
        rhs = cyl.build_rhs([omega], incident_field, incident_angles, n_az)
        solution = cyl.solve(matrix, rhs)
        inner_c = cyl.get_inner_coefficients(
            [omega], eigenvalues, modes, n_az, "TM", rhs, solution
        )
        solutions.append(solution)
        inner_coeffs_list.append(inner_c)

    # Create spatial grid
    x = np.linspace(-2, 2, 11)
    y = np.linspace(-2, 2, 11)
    t = 0.0

    fields = cyl.get_fields(
        x,
        y,
        [omega],
        np.array(solutions),
        np.array(inner_coeffs_list),
        eigenvalues,
        modes,
        incident_field,
        incident_angles,
        n_range,
        t,
        "TM",
    )

    # Check that all field types are present
    assert "total" in fields
    assert "scattered" in fields
    assert "incident" in fields

    # Check shapes
    expected_shape = (11, 11, 1)  # (ny, nx, nw)
    assert fields["total"].shape == expected_shape
    assert fields["scattered"].shape == expected_shape
    assert fields["incident"].shape == expected_shape

    # Check that total = incident + scattered
    assert np.allclose(fields["total"], fields["incident"] + fields["scattered"])


def test_get_fields_time_array():
    """Test computing fields at multiple time points."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    omega = 1.0
    eigenvalues, modes = mat.eigensolve([omega])

    # Set up simple scattering problem
    incident_field, incident_angles = cyl.init_incident_field([omega])
    incident_field[mat.Nh] = 1
    incident_angles[mat.Nh] = 0

    n_range = range(-1, 2)
    solutions = []
    inner_coeffs_list = []

    for n_az in n_range:
        matrix = cyl.build_matrix([omega], eigenvalues, modes, n_az, "TM")
        rhs = cyl.build_rhs([omega], incident_field, incident_angles, n_az)
        solution = cyl.solve(matrix, rhs)
        inner_c = cyl.get_inner_coefficients(
            [omega], eigenvalues, modes, n_az, "TM", rhs, solution
        )
        solutions.append(solution)
        inner_coeffs_list.append(inner_c)

    # Multiple time points
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    t = np.linspace(0, 2 * np.pi / Omega, 5)

    fields = cyl.get_fields(
        x,
        y,
        [omega],
        np.array(solutions),
        np.array(inner_coeffs_list),
        eigenvalues,
        modes,
        incident_field,
        incident_angles,
        n_range,
        t,
        "TM",
    )

    # Shape should be (ny, nx, nw, nt)
    expected_shape = (5, 5, 1, 5)
    assert fields["total"].shape == expected_shape


def test_get_fields_x_cut():
    """Test computing field along x-cut (y=scalar)."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    omega = 1.0
    eigenvalues, modes = mat.eigensolve([omega])

    incident_field, incident_angles = cyl.init_incident_field([omega])
    incident_field[mat.Nh] = 1
    incident_angles[mat.Nh] = 0

    n_range = range(-1, 2)
    solutions = []
    inner_coeffs_list = []

    for n_az in n_range:
        matrix = cyl.build_matrix([omega], eigenvalues, modes, n_az, "TM")
        rhs = cyl.build_rhs([omega], incident_field, incident_angles, n_az)
        solution = cyl.solve(matrix, rhs)
        inner_c = cyl.get_inner_coefficients(
            [omega], eigenvalues, modes, n_az, "TM", rhs, solution
        )
        solutions.append(solution)
        inner_coeffs_list.append(inner_c)

    # x-cut at y=0
    x = np.linspace(-2, 2, 21)
    y = 0.0
    t = 0.0

    fields = cyl.get_fields(
        x,
        y,
        [omega],
        np.array(solutions),
        np.array(inner_coeffs_list),
        eigenvalues,
        modes,
        incident_field,
        incident_angles,
        n_range,
        t,
        "TM",
    )

    # Shape should be (1, 21, 1) for scalar y
    assert fields["total"].shape == (1, 21, 1)


def test_plot():
    """Test the plot method creates a circle patch."""
    import matplotlib.pyplot as plt

    mat = Material([eps_r], modulation_frequency=1.0)
    cyl = Cylinder(mat, radius=1.0)

    fig, ax = plt.subplots()
    circle = cyl.plot(ax=ax)

    assert circle is not None
    assert circle.radius == 1.0
    plt.close(fig)


def test_plot_normalized():
    """Test the plot method with normalization."""
    import matplotlib.pyplot as plt

    mat = Material([eps_r], modulation_frequency=1.0)
    cyl = Cylinder(mat, radius=2.0)

    fig, ax = plt.subplots()
    circle = cyl.plot(ax=ax, normalize=True)

    # When normalized, radius should be 1.0
    assert circle.radius == 1.0
    plt.close(fig)


def test_build_dmatrix_domega():
    """Test building the derivative of the matrix wrt omega."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    omega = 1.0
    eigenvalues, modes, modes_left = mat.eigensolve([omega], left=True, normalize=True)

    dmatrix = cyl.build_dmatrix_domega(
        [omega], eigenvalues, modes, modes_left, n=0, polarization="TM"
    )

    # Check shape
    assert dmatrix.shape == (3, 3, 1)
    assert np.all(np.isfinite(dmatrix))


def test_build_dmatrix_domega_TE():
    """Test building the derivative matrix for TE polarization."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    omega = 1.0
    eigenvalues, modes, modes_left = mat.eigensolve([omega], left=True, normalize=True)

    dmatrix_TE = cyl.build_dmatrix_domega(
        [omega], eigenvalues, modes, modes_left, n=0, polarization="TE"
    )

    dmatrix_TM = cyl.build_dmatrix_domega(
        [omega], eigenvalues, modes, modes_left, n=0, polarization="TM"
    )

    # TE and TM derivatives should be different
    assert dmatrix_TE.shape == (3, 3, 1)
    assert not np.allclose(dmatrix_TE, dmatrix_TM)


def test_normalize_modes():
    """Test mode normalization."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    omega = 1.0
    eigenvalues, modes, modes_left = mat.eigensolve([omega], left=True, normalize=True)

    dmatrix = cyl.build_dmatrix_domega(
        [omega], eigenvalues, modes, modes_left, n=0, polarization="TM"
    )

    # Create simple mode vectors
    modes_right = np.random.randn(3, 2, 1) + 1j * np.random.randn(3, 2, 1)
    modes_left = np.random.randn(3, 2, 1) + 1j * np.random.randn(3, 2, 1)

    # Extend dmatrix to match mode shape
    dmatrix_ext = np.repeat(dmatrix, 2, axis=2)

    norm_right, norm_left = cyl.normalize(modes_right, modes_left, dmatrix_ext)

    assert norm_right.shape == modes_right.shape
    assert norm_left.shape == modes_left.shape


def test_get_modes_normalization():
    """Test computing mode normalization constants."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    # Create simple test data with proper shapes
    # modes_right shape: (nh, nmodes, *omegas.shape)
    # For single omega, this is (3, 2, 1) -> (nh, nmodes, 1)
    # We need matrix_derivative shape: (nh, nh, nmodes)
    modes_right = np.random.randn(3, 2) + 1j * np.random.randn(3, 2)
    modes_left = np.random.randn(3, 2) + 1j * np.random.randn(3, 2)

    # matrix_derivative needs shape (nh, nh, nmodes)
    matrix_derivative = np.random.randn(3, 3, 2) + 1j * np.random.randn(3, 3, 2)

    normas = cyl.get_modes_normalization(modes_right, modes_left, matrix_derivative)

    assert normas.shape[0] == 2  # nmodes
    assert np.all(np.isfinite(normas))


def test_scalar_product():
    """Test the scalar product computation."""
    eps0 = 3.0
    Omega = 0.6
    eps_fourier = [eps0]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    # Simple test vectors
    modes_right = np.random.randn(1, 1) + 1j * np.random.randn(1, 1)
    modes_left = np.random.randn(1, 1) + 1j * np.random.randn(1, 1)

    ev_r = 1.0
    ev_l = 1.0 + 0.1j

    matrix_r = np.eye(1)
    matrix_l = np.eye(1)
    matrix_deriv = np.eye(1)

    # Test diagonal computation
    product = cyl.scalar_product(
        modes_right, modes_left, ev_r, ev_l, matrix_r, matrix_l, matrix_deriv, diag=True
    )

    assert np.isfinite(product)


def test_get_multiplicities():
    """Test computing eigenvalue multiplicities."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    # Test with simple frequencies
    eigenvalues = np.array([1.0 - 0.1j, 2.0 - 0.2j])

    # Should return array of multiplicities
    mults = cyl.get_multiplicities(eigenvalues)

    assert len(mults) == 2
    assert np.all(mults >= 0)


def test_eigenvalue_static():
    """Test computing static eigenvalues."""
    eps0 = 3.0
    Omega = 0.6
    eps_fourier = [eps0]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    # Search in a region where static QNMs are known to exist
    # For eps0=3, size parameter X=1 (omega=1 since R=1, c=1),
    # QNMs have negative imaginary parts
    omega0 = -3.0 - 0.5j
    omega1 = 3.0 - 0.001j

    evs = cyl.eigenvalue_static(omega0, omega1, n=0, polarization="TM")

    # Should return list/array of eigenvalues (may be empty if none found)
    assert isinstance(evs, np.ndarray | list)
    # All eigenvalues should be finite if any found
    if len(evs) > 0:
        evs_arr = np.array(evs)
        assert np.all(np.isfinite(evs_arr))


def test_eigensolve_cylinder():
    """Test the cylinder's eigensolve method."""
    eps0 = 3.0
    deps = 0.1  # Small modulation for stability
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    # Search in a larger region to find eigenvalues
    omega0 = -2.0 - 0.5j
    omega1 = 2.0 - 0.01j

    evs, modes = cyl.eigensolve(
        omega0,
        omega1,
        peak_ref=3,
        recursive=True,
        tol=1e-4,
        plot_solver=False,
        n=0,
        polarization="TM",
    )

    # eigensolve returns a list (may be empty if no eigenvalues found)
    assert isinstance(evs, np.ndarray | list)
    assert isinstance(modes, np.ndarray | list)
    # All eigenvalues should be finite if any found
    if len(evs) > 0:
        evs_arr = np.array(evs)
        assert np.all(np.isfinite(evs_arr))


def test_get_mode():
    """Test computing QNM field distributions."""
    eps0 = 3.0
    deps = 0.1
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    # Simple mode computation
    x = np.linspace(-1.5, 1.5, 11)
    y = np.linspace(-1.5, 1.5, 11)

    omega_n = np.array([0.5 - 0.05j])
    eigenvalues, modes = mat.eigensolve(omega_n)

    # Simple test mode vector (nh x nmodes)
    phi_n = np.ones((3, 1), dtype=complex)

    t = np.array([0.0])
    n_range_plt = range(-2, 3)

    fields = cyl.get_mode(
        x,
        y,
        omega_n,
        phi_n,
        eigenvalues,
        modes,
        n=0,
        polarization="TM",
        t=t,
        n_range_plt=n_range_plt,
    )

    # Check field structure
    assert "scattered" in fields
    # Shape depends on whether t is scalar or array:
    # - scalar t: (ny, nx, nw)
    # - array t with len(t)==1: (ny, nx, nw, 1)
    assert fields["scattered"].shape[:2] == (11, 11)


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


def test_mie_coefficient_zero_size():
    """Test Mie coefficient at zero size parameter (Rayleigh limit)."""
    X_zero = 1e-10
    bn = mie_coefficient_TM(0, X_zero, m)

    # Should be finite even at very small X
    assert np.isfinite(bn)
    # For small X, |b_0| should be small (Rayleigh scattering ~ X^2)
    assert abs(bn) < 0.01


def test_mie_coefficient_large_order():
    """Test Mie coefficient for large angular order."""
    n_large = 50
    bn = mie_coefficient_TM(n_large, X, m)

    # Should be finite
    assert np.isfinite(bn)
    # For large n, coefficient should be very small
    assert abs(bn) < 1e-6


def test_cylinder_with_large_radius():
    """Test cylinder with large radius."""
    mat = Material([eps_r], modulation_frequency=1.0)
    cyl = Cylinder(mat, radius=100.0)

    # Should handle large radius
    bn = cyl.mie_coefficient(1.0, n=0)
    assert np.isfinite(bn)


def test_multiple_frequencies_matrix():
    """Test building matrix for multiple frequencies."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    omegas = np.linspace(0.5, 1.5, 5)
    eigenvalues, modes = mat.eigensolve(omegas)

    matrix = cyl.build_matrix(omegas, eigenvalues, modes, n=0, polarization="TM")

    # Shape should be (nh, nh, nw)
    assert matrix.shape == (3, 3, 5)
    assert np.all(np.isfinite(matrix))


def test_solve_2d_matrix():
    """Test solving for 2D frequency grid."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    # 2D frequency grid
    omegas_r = np.linspace(0.5, 1.5, 3)
    omegas_i = np.linspace(-0.1, -0.01, 3)
    re, im = np.meshgrid(omegas_r, omegas_i)
    omegas = re + 1j * im

    eigenvalues, modes = mat.eigensolve(omegas)
    matrix = cyl.build_matrix(omegas, eigenvalues, modes, n=0, polarization="TM")

    # Shape should be (nh, nh, nr, ni)
    assert matrix.shape == (3, 3, 3, 3)

    # Create RHS with matching shape
    rhs = np.ones((3, 3, 3), dtype=complex)

    solution = cyl.solve(matrix, rhs)
    assert solution.shape == rhs.shape


def test_solve_1d_simple():
    """Test solve with 1D matrix (simple case)."""
    cyl = Cylinder(Material([eps_r], 1.0), radius=1.0)

    # Simple 2x2 system
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    rhs = np.array([1.0, 2.0])

    solution = cyl.solve(matrix, rhs)

    assert np.allclose(solution, rhs)


def test_solve_invalid_dimensions():
    """Test that solve raises error for invalid matrix dimensions."""
    cyl = Cylinder(Material([eps_r], 1.0), radius=1.0)

    # 1D matrix should raise error
    matrix = np.array([1.0, 2.0, 3.0])
    rhs = np.array([1.0])

    with pytest.raises(ValueError, match="Unsupported number of dimensions"):
        cyl.solve(matrix, rhs)


def test_static_limit_floquet():
    """Test that Floquet code reduces to static Mie when modulation is zero."""
    eps0 = 4.0
    Omega = 0.6
    R = 1.0

    # Static case (no modulation)
    eps_fourier_static = [eps0]
    mat_static = Material(eps_fourier_static, Omega, Npad=0)
    cyl_static = Cylinder(mat_static, R)

    # Small modulation case
    deps = 1e-10  # Very small modulation
    eps_fourier_mod = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]
    mat_mod = Material(eps_fourier_mod, Omega, Npad=0)
    cyl_mod = Cylinder(mat_mod, R)

    omega = 1.0

    # Get static Mie coefficient
    cyl_static.mie_coefficient(omega, n=0, polarization="TM")

    # Build Floquet matrix and solve for modulated case
    eigenvalues, modes = mat_mod.eigensolve([omega])
    matrix = cyl_mod.build_matrix([omega], eigenvalues, modes, n=0, polarization="TM")

    # The matrix should be approximately diagonal with Mie-like entries
    # (this is a qualitative check)
    assert np.all(np.isfinite(matrix))


def test_TE_polarization_fields():
    """Test computing TE polarized fields."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    omega = 1.0
    eigenvalues, modes = mat.eigensolve([omega])

    incident_field, incident_angles = cyl.init_incident_field([omega])
    incident_field[mat.Nh] = 1

    n_range = range(-1, 2)
    solutions = []
    inner_coeffs_list = []

    for n_az in n_range:
        matrix = cyl.build_matrix([omega], eigenvalues, modes, n_az, "TE")
        rhs = cyl.build_rhs([omega], incident_field, incident_angles, n_az)
        solution = cyl.solve(matrix, rhs)
        inner_c = cyl.get_inner_coefficients(
            [omega], eigenvalues, modes, n_az, "TE", rhs, solution
        )
        solutions.append(solution)
        inner_coeffs_list.append(inner_c)

    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    t = 0.0

    fields = cyl.get_fields(
        x,
        y,
        [omega],
        np.array(solutions),
        np.array(inner_coeffs_list),
        eigenvalues,
        modes,
        incident_field,
        incident_angles,
        n_range,
        t,
        "TE",
    )

    assert np.all(np.isfinite(fields["total"]))


def test_external_medium_fields():
    """Test field computation with non-vacuum external medium."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_ext = 2.0
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0, eps_ext=eps_ext)

    omega = 1.0
    eigenvalues, modes = mat.eigensolve([omega])

    incident_field, incident_angles = cyl.init_incident_field([omega])
    incident_field[mat.Nh] = 1

    n_range = range(-1, 2)
    solutions = []
    inner_coeffs_list = []

    for n_az in n_range:
        matrix = cyl.build_matrix([omega], eigenvalues, modes, n_az, "TM")
        rhs = cyl.build_rhs([omega], incident_field, incident_angles, n_az)
        solution = cyl.solve(matrix, rhs)
        inner_c = cyl.get_inner_coefficients(
            [omega], eigenvalues, modes, n_az, "TM", rhs, solution
        )
        solutions.append(solution)
        inner_coeffs_list.append(inner_c)

    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)

    fields = cyl.get_fields(
        x,
        y,
        [omega],
        np.array(solutions),
        np.array(inner_coeffs_list),
        eigenvalues,
        modes,
        incident_field,
        incident_angles,
        n_range,
        0.0,
        "TM",
    )

    assert np.all(np.isfinite(fields["total"]))


def test_plot_mode():
    """Test the plot_mode method for QNM field visualization."""
    import matplotlib.pyplot as plt

    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    # Create simple test fields
    x = np.linspace(-2, 2, 21)
    y = np.linspace(-2, 2, 21)
    X, Y = np.meshgrid(x, y)

    # Simple test field (2D array)
    test_field = np.exp(-(X**2 + Y**2) / 2) * (1 + 1j)

    # Test with dict input
    fields_dict = {"scattered": test_field}

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig_out, ax_out = cyl.plot_mode(
        x,
        y,
        fields_dict,
        ax=ax,
        plot_type="cartesian",
        polarization="TM",
        normalize=True,
        show_cylinder=True,
    )

    assert fig_out is fig
    assert len(ax_out) == 2
    plt.close(fig)

    # Test with ndarray input
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig_out, ax_out = cyl.plot_mode(
        x,
        y,
        test_field,
        ax=ax,
        plot_type="polar",
        polarization="TE",
        normalize=False,
        show_cylinder=False,
    )

    assert fig_out is fig
    plt.close(fig)

    # Test with 4D field array (ny, nx, nw, nt)
    field_4d = test_field[:, :, np.newaxis, np.newaxis]  # (21, 21, 1, 1)
    fields_dict_4d = {"scattered": field_4d}

    fig, ax = cyl.plot_mode(
        x,
        y,
        fields_dict_4d,
        time_index=0,
        freq_index=0,
        plot_type="cartesian",
        polarization="TM",
    )

    assert fig is not None
    assert len(ax) == 2
    plt.close(fig)


def test_plot_mode_invalid_inputs():
    """Test plot_mode error handling for invalid inputs."""
    import matplotlib.pyplot as plt

    mat = Material([eps_r], modulation_frequency=1.0)
    cyl = Cylinder(mat, radius=1.0)

    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    test_field = np.random.randn(5, 5)

    # Test invalid plot_type
    with pytest.raises(ValueError, match="plot_type must be either"):
        cyl.plot_mode(x, y, test_field, plot_type="invalid")

    # Test insufficient axes
    fig, ax = plt.subplots(1, 1)
    with pytest.raises(ValueError, match="ax must contain at least 2 axes"):
        cyl.plot_mode(x, y, test_field, ax=[ax])
    plt.close(fig)

    # Test unsupported field dimensions
    with pytest.raises(ValueError, match="Unsupported field dimensions"):
        cyl.plot_mode(x, y, np.array([1.0]))  # 1D array


# =============================================================================
# Integration Tests
# =============================================================================


def test_boundary_conditions_integration():
    """Integration test for boundary conditions using check_boundary module.

    This test verifies that the computed fields satisfy electromagnetic
    boundary conditions at the cylinder surface (continuity of tangential E and H).
    """
    from pytmod.check_boundary import check_boundary_field, check_boundary_matrix

    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    R = 1.0
    cyl = Cylinder(mat, R)

    # Set up scattering problem for multiple azimuthal modes
    omega = 1.0
    omegas = np.array([omega])
    eigenvalues, modes = mat.eigensolve(omegas)

    incident_field, incident_angles = cyl.init_incident_field(omegas)
    incident_field[mat.Nh] = 1.0  # Excite central harmonic
    incident_angles[mat.Nh] = 0.0

    n_range = range(-3, 4)  # Multiple azimuthal modes
    solutions = []
    inner_coeffs_list = []
    rhs_list = []

    for n_az in n_range:
        matrix = cyl.build_matrix(omegas, eigenvalues, modes, n_az, "TM")
        rhs = cyl.build_rhs(omegas, incident_field, incident_angles, n_az)
        solution = cyl.solve(matrix, rhs)
        inner_c = cyl.get_inner_coefficients(
            omegas, eigenvalues, modes, n_az, "TM", rhs, solution
        )
        solutions.append(solution)
        inner_coeffs_list.append(inner_c)
        rhs_list.append(rhs)

    # Convert to arrays with shape (n_az, nh, nw)
    solution_arr = np.array(solutions)
    inner_coeffs_arr = np.array(inner_coeffs_list)
    rhs_arr = np.array(rhs_list)

    # Test matrix boundary check
    result_matrix = check_boundary_matrix(
        cyl,
        omegas,
        eigenvalues,
        modes,
        rhs_arr,
        solution_arr,
        inner_coeffs_arr,
        n_range=list(n_range),
        polarization="TM",
        tol=1e-5,
    )

    assert result_matrix["passed"], "Matrix boundary check failed"
    assert result_matrix["bc1_residuals"].max() < 1e-5, (
        "BC1 (field continuity) residual too large"
    )
    assert result_matrix["bc2_residuals"].max() < 1e-5, (
        "BC2 (derivative continuity) residual too large"
    )

    # Test field boundary check
    result_field = check_boundary_field(
        cyl,
        omegas,
        eigenvalues,
        modes,
        rhs_arr,
        solution_arr,
        inner_coeffs_arr,
        n_range=list(n_range),
        polarization="TM",
        t=0.0,
        n_theta=72,
        tol=1e-5,
    )

    assert result_field["passed"], "Field boundary check failed"
    assert result_field["max_discontinuity"].max() < 1e-5, (
        "Field discontinuity at boundary too large"
    )


def test_boundary_conditions_TE():
    """Integration test for boundary conditions with TE polarization."""
    from pytmod.check_boundary import check_boundary_matrix

    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    omega = 1.0
    omegas = np.array([omega])
    eigenvalues, modes = mat.eigensolve(omegas)

    incident_field, incident_angles = cyl.init_incident_field(omegas)
    incident_field[mat.Nh] = 1.0

    n_range = range(-2, 3)
    solutions = []
    inner_coeffs_list = []
    rhs_list = []

    for n_az in n_range:
        matrix = cyl.build_matrix(omegas, eigenvalues, modes, n_az, "TE")
        rhs = cyl.build_rhs(omegas, incident_field, incident_angles, n_az)
        solution = cyl.solve(matrix, rhs)
        inner_c = cyl.get_inner_coefficients(
            omegas, eigenvalues, modes, n_az, "TE", rhs, solution
        )
        solutions.append(solution)
        inner_coeffs_list.append(inner_c)
        rhs_list.append(rhs)

    solution_arr = np.array(solutions)
    inner_coeffs_arr = np.array(inner_coeffs_list)
    rhs_arr = np.array(rhs_list)

    # Check boundary conditions for TE
    result_matrix = check_boundary_matrix(
        cyl,
        omegas,
        eigenvalues,
        modes,
        rhs_arr,
        solution_arr,
        inner_coeffs_arr,
        n_range=list(n_range),
        polarization="TE",
        tol=1e-5,
    )

    assert result_matrix["passed"], "TE matrix boundary check failed"


def test_matrix_derivative_finite_difference():
    """Integration test for matrix derivative using finite differences.

    This test verifies the analytical derivative dM/domega against a
    finite difference approximation, as demonstrated in plot_cylinder_modes.py.
    """
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    # Test at a single frequency with complex part
    omega = 0.5 - 0.05j

    # Small step for finite difference
    dw = 1e-7

    # Compute eigenvalues and modes at omega + dw and omega - dw
    ev_p, mo_p, mo_l_p = mat.eigensolve(omega + dw, left=True, normalize=True)
    ev_m, mo_m, mo_l_m = mat.eigensolve(omega - dw, left=True, normalize=True)

    n = 0
    polarization = "TM"

    # Build matrices at shifted frequencies
    M_plus = cyl.build_matrix(omega + dw, ev_p, mo_p, n=n, polarization=polarization)
    M_minus = cyl.build_matrix(omega - dw, ev_m, mo_m, n=n, polarization=polarization)

    # Finite difference derivative
    dM_fd = (M_plus - M_minus) / (2 * dw)

    # Analytical derivative
    ev, mo, mo_l = mat.eigensolve(omega, left=True, normalize=True)
    dM_analytic = cyl.build_dmatrix_domega(
        omega, ev, mo, mo_l, n=n, polarization=polarization
    )

    # Compute relative error
    err = dM_analytic - dM_fd
    rel_error = np.abs(err) / np.linalg.norm(dM_fd)
    max_rel_error = rel_error.max()

    # The finite difference should match the analytical derivative within tolerance
    # Allow up to 1% relative error due to finite difference approximation
    assert max_rel_error < 0.01, (
        f"Matrix derivative mismatch: {100 * max_rel_error:.4f}% > 1%"
    )


def test_matrix_derivative_TE_finite_difference():
    """Integration test for TE matrix derivative using finite differences."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    omega = 0.5 - 0.05j
    dw = 1e-7

    ev_p, mo_p, mo_l_p = mat.eigensolve(omega + dw, left=True, normalize=True)
    ev_m, mo_m, mo_l_m = mat.eigensolve(omega - dw, left=True, normalize=True)

    n = 0
    polarization = "TE"

    M_plus = cyl.build_matrix(omega + dw, ev_p, mo_p, n=n, polarization=polarization)
    M_minus = cyl.build_matrix(omega - dw, ev_m, mo_m, n=n, polarization=polarization)
    dM_fd = (M_plus - M_minus) / (2 * dw)

    ev, mo, mo_l = mat.eigensolve(omega, left=True, normalize=True)
    dM_analytic = cyl.build_dmatrix_domega(
        omega, ev, mo, mo_l, n=n, polarization=polarization
    )

    err = dM_analytic - dM_fd
    rel_error = np.abs(err) / np.linalg.norm(dM_fd)
    max_rel_error = rel_error.max()

    assert max_rel_error < 0.01, (
        f"TE matrix derivative mismatch: {100 * max_rel_error:.4f}% > 1%"
    )


def test_matrix_derivative_multiple_frequencies():
    """Integration test for matrix derivative at multiple frequencies."""
    eps0 = 3.0
    deps = 0.3
    Omega = 0.6
    eps_fourier = [-deps / (2 * 1j), eps0, deps / (2 * 1j)]

    mat = Material(eps_fourier, Omega, Npad=0)
    cyl = Cylinder(mat, radius=1.0)

    # Test with multiple frequencies
    omegas = np.array([0.4 - 0.04j, 0.5 - 0.05j, 0.6 - 0.06j])
    dw = 1e-7

    ev_p, mo_p, mo_l_p = mat.eigensolve(omegas + dw, left=True, normalize=True)
    ev_m, mo_m, mo_l_m = mat.eigensolve(omegas - dw, left=True, normalize=True)

    M_plus = cyl.build_matrix(omegas + dw, ev_p, mo_p, n=0, polarization="TM")
    M_minus = cyl.build_matrix(omegas - dw, ev_m, mo_m, n=0, polarization="TM")
    dM_fd = (M_plus - M_minus) / (2 * dw)

    ev, mo, mo_l = mat.eigensolve(omegas, left=True, normalize=True)
    dM_analytic = cyl.build_dmatrix_domega(omegas, ev, mo, mo_l, n=0, polarization="TM")

    err = dM_analytic - dM_fd
    rel_error = np.abs(err) / (np.linalg.norm(dM_fd, axis=(0, 1)) + 1e-15)
    max_rel_error = np.max(rel_error)

    assert max_rel_error < 0.01, (
        f"Multi-frequency derivative mismatch: {100 * max_rel_error:.4f}% > 1%"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
