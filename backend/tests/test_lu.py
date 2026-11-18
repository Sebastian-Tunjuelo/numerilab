import numpy as np
from app.solvers.lu import solve_lu


def test_lu_doolittle_crout_partial_pivot_standard():
    A = [[10.0, -1.0, 2.0], [-1.0, 11.0, -1.0], [2.0, -1.0, 10.0]]
    b = [6.0, 22.0, -10.0]
    expected = np.array([1.0, 2.0, -1.0])

    for strategy in ("doolittle", "crout", "partial_pivot"):
        res = solve_lu(A, b, strategy=strategy, decimals=8)
        assert res["status"] == "converged"
        assert np.allclose(res["x"], expected, atol=1e-7)
        assert res["residual_norm"] < 1e-12

        P = np.array(res["P"], dtype=float)
        L = np.array(res["L"], dtype=float)
        U = np.array(res["U"], dtype=float)
        assert np.allclose(L @ U, P @ np.array(A), atol=1e-7)
        assert len(res["forward_equations"]) == 3
        assert len(res["backward_equations"]) == 3


def test_lu_zero_pivot_requires_pivoting():
    A = [[0.0, 2.0, 3.0], [4.0, 6.0, 7.0], [2.0, -3.0, 6.0]]
    b = [8.0, -3.0, 5.0]

    res_doolittle = solve_lu(A, b, strategy="doolittle")
    assert res_doolittle["status"] == "zero_pivot_requires_pivoting"
    assert "U(1,1)" in res_doolittle["message"]
    assert "_{" not in res_doolittle["message"]

    res_crout = solve_lu(A, b, strategy="crout")
    assert res_crout["status"] == "zero_pivot_requires_pivoting"
    assert "L(1,1)" in res_crout["message"]
    assert "_{" not in res_crout["message"]

    res_pivot = solve_lu(A, b, strategy="partial_pivot", decimals=8)
    assert res_pivot["status"] == "converged"
    assert res_pivot["residual_norm"] < 1e-10
    assert any(step["pivot_swap"] is not None for step in res_pivot["steps"])


def test_lu_singular_matrix():
    A = [[1.0, 2.0], [2.0, 4.0]]
    b = [3.0, 6.0]
    res = solve_lu(A, b, strategy="partial_pivot")
    assert res["status"] == "singular_matrix"


def test_lu_keeps_small_factors_and_residuals_in_full_precision():
    A = [[1e14, 1], [1, 1]]
    for strategy in ("partial_pivot", "doolittle", "crout"):
        result = solve_lu(A, [1e14, 1], strategy=strategy)
        assert result["status"] == "converged"
        assert result["x_raw"] == [1.0, 0.0]
        assert result["residual_norm"] == 0.0
        assert result["backward_equations"][0].startswith("x_{2}")
        assert result["backward_equations"][-1].startswith("x_{1}")

    result = solve_lu([[4, -1], [-1, 4]], [1e-14, 2e-14])
    assert all(0 < value < 1e-13 for value in result["x_raw"])
    assert result["residual_norm"] < 1e-28


def test_lu_reports_arithmetic_overflow():
    for strategy in ("partial_pivot", "doolittle", "crout"):
        result = solve_lu([[1e308, 1e308], [1e308, -1e308]], [1, 1], strategy=strategy)
        assert result["status"] == "diverged"
        assert "x" not in result


def test_partial_pivoting_preserves_a_solution_under_uniform_scaling():
    for scale in (1e-15, 1e-310):
        result = solve_lu([[0, 2 * scale], [scale, 0]], [4 * scale, scale])
        assert result["status"] == "converged"
        assert result["x_raw"] == [1, 2]
        assert result["residual_norm"] == 0


def test_lu_singular_matrix_includes_factors_and_supports_6x6():
    singular = solve_lu([[1.0, 2.0], [2.0, 4.0]], [3.0, 6.0], strategy="partial_pivot")
    assert singular["status"] == "singular_matrix"
    for key in ("L", "U", "P", "L_latex", "U_latex", "P_latex"):
        assert key in singular

    # Sistema tridiagonal 6x6 con solución exacta [1, 1, 1, 1, 1, 1]
    n = 6
    A = [
        [4.0 if i == j else (-1.0 if abs(i - j) == 1 else 0.0) for j in range(n)]
        for i in range(n)
    ]
    b = [3.0 if i in (0, n - 1) else 2.0 for i in range(n)]
    for strategy in ("doolittle", "crout", "partial_pivot"):
        res = solve_lu(A, b, strategy=strategy, decimals=8)
        assert res["status"] == "converged"
        assert np.allclose(res["x_raw"], np.ones(n), atol=1e-12)
        assert res["residual_norm"] < 1e-12


def test_lu_doolittle_and_crout_require_pivoting_below_absolute_threshold():
    for strategy in ("doolittle", "crout"):
        res = solve_lu([[1e-15, 0.0], [0.0, 1e-15]], [1e-15, 2e-15], strategy=strategy, decimals=6)
        assert res["status"] == "zero_pivot_requires_pivoting"
        res = solve_lu([[1e-14, 0.0], [0.0, 1e-14]], [1e-14, 2e-14], strategy=strategy)
        assert res["status"] == "converged"
        assert res["x_raw"] == [1.0, 2.0]


def test_lu_preserves_full_latex_decimals():
    res_prec = solve_lu([[3.0, 1.0], [1.0, 3.0]], [1.0, 1.0], strategy="doolittle", decimals=6)
    assert res_prec["status"] == "converged"
    assert "0.333333" in res_prec["L_latex"]
    assert "0.666667" in res_prec["y_latex"]
    assert r"L_{2,1} = \frac{1}{3}" in res_prec["steps"][0]["latex_operation"]
    assert "3 - (0.333333)(1)" in res_prec["steps"][1]["latex_operation"]
    crout = solve_lu([[3.0, 1.0], [1.0, 3.0]], [1.0, 1.0], strategy="crout")
    assert r"U_{1,2} = \frac{1}{3}" in crout["steps"][0]["latex_operation"]


def test_lu_zero_row_handling_and_2x2_to_6x6():
    zero_row = [[0.0, 0.0], [1.0, 2.0]]
    assert solve_lu(zero_row, [0.0, 1.0], strategy="partial_pivot")["status"] == "singular_matrix"
    assert solve_lu(zero_row, [0.0, 1.0], strategy="doolittle")["status"] == "zero_pivot_requires_pivoting"
    assert solve_lu(zero_row, [0.0, 1.0], strategy="crout")["status"] == "zero_pivot_requires_pivoting"


def test_lu_partial_pivot_mixed_scale_off_diagonal():
    res_tri = solve_lu([[1.0, 1e17], [0.0, 1.0]], [1e17, 1.0], strategy="partial_pivot")
    assert res_tri["status"] == "converged"
    assert res_tri["x_raw"] == [0.0, 1.0]
    assert res_tri["residual_norm"] == 0.0

    res_tie = solve_lu([[1.0, 1e17], [1.0, 0.0]], [1e17, 1.0], strategy="partial_pivot", decimals=12)
    assert res_tie["status"] == "converged"
    P = np.array(res_tie["P"], dtype=float)
    L = np.array(res_tie["L"], dtype=float)
    U = np.array(res_tie["U"], dtype=float)
    assert np.allclose(L @ U, P @ np.array([[1.0, 1e17], [1.0, 0.0]]), atol=1e-9)

    res_small = solve_lu([[1e-16, 1.0], [1.0, 1.0]], [1.0, 2.0], strategy="partial_pivot")
    assert res_small["status"] == "converged"
    assert np.allclose(res_small["x_raw"], [1.0, 1.0], atol=1e-12)


def test_lu_last_stage_zero_pivot_reports_singular_in_doolittle_and_crout():
    A = [[1.0, 2.0], [2.0, 4.0]]
    b = [3.0, 6.0]
    for strategy in ("doolittle", "crout"):
        res = solve_lu(A, b, strategy=strategy)
        assert res["status"] == "singular_matrix"
        assert "L" in res and "U" in res and "P" in res


def test_lu_partial_pivot_multi_swap_4x4_and_5x5_pa_equals_lu():
    A4 = [
        [0.0, 2.0, -1.0, 3.0],
        [0.0, 0.0, 4.0, 1.0],
        [5.0, 1.0, 2.0, -1.0],
        [1.0, 6.0, 0.0, 2.0],
    ]
    b4 = [5.0, 15.0, 9.0, 21.0]
    res4 = solve_lu(A4, b4, strategy="partial_pivot", decimals=12)
    assert res4["status"] == "converged"
    assert np.allclose(
        np.array(res4["L"], dtype=float) @ np.array(res4["U"], dtype=float),
        np.array(res4["P"], dtype=float) @ np.array(A4, dtype=float),
        atol=1e-10,
    )
    assert res4["residual_norm"] < 1e-12

    A5 = [
        [0.0, 1.0, 0.0, 2.0, 1.0],
        [1.0, 0.0, 3.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 4.0, 1.0],
        [2.0, 1.0, 1.0, 0.0, 3.0],
        [0.0, 3.0, 2.0, 1.0, 0.0],
    ]
    x_true = np.array([1.0, -1.0, 2.0, 0.5, 3.0])
    b5 = (np.array(A5) @ x_true).tolist()
    res5 = solve_lu(A5, b5, strategy="partial_pivot", decimals=12)
    assert res5["status"] == "converged"
    assert np.allclose(res5["x_raw"], x_true, atol=1e-11)
    assert np.allclose(
        np.array(res5["L"], dtype=float) @ np.array(res5["U"], dtype=float),
        np.array(res5["P"], dtype=float) @ np.array(A5, dtype=float),
        atol=1e-10,
    )


def test_lu_operation_count_and_finite_arithmetic():
    from app.solvers.matrix_utils import apply_finite_arithmetic

    assert apply_finite_arithmetic(0.299, 2, "truncate") == 0.29
    assert apply_finite_arithmetic(-0.299, 2, "truncate") == -0.29
    assert apply_finite_arithmetic(0.299, 2, "round") == 0.3

    A = [[10.0, -1.0, 2.0], [-1.0, 11.0, -1.0], [2.0, -1.0, 10.0]]
    b = [6.0, 22.0, -10.0]
    for strategy in ("doolittle", "crout", "partial_pivot"):
        res = solve_lu(A, b, strategy=strategy, decimals=6)
        ops = res["operation_count"]
        assert ops["mul_div"] == ops["factorization_mul_div"] + ops["substitution_mul_div"]
        assert ops["add_sub"] == ops["factorization_add_sub"] + ops["substitution_add_sub"]
        assert ops["total_flops"] == ops["mul_div"] + ops["add_sub"] == 28
        assert ops["theoretical_flops"] == 36
        assert res["rounding_mode"] == "none"
        assert res["finite_arithmetic_diff"] is None

    res_round = solve_lu([[3.0, 1.0], [1.0, 3.0]], [1.0, 1.0], strategy="doolittle", decimals=2, rounding_mode="round")
    res_trunc = solve_lu([[3.0, 1.0], [1.0, 3.0]], [1.0, 1.0], strategy="doolittle", decimals=2, rounding_mode="truncate")
    assert res_round["status"] == "converged"
    assert res_trunc["status"] == "converged"
    assert res_round["rounding_mode"] == "round"
    assert res_trunc["rounding_mode"] == "truncate"
    assert res_round["finite_arithmetic_diff"] > 0.0
    assert res_trunc["finite_arithmetic_diff"] > 0.0
    assert res_round["x"] != res_trunc["x"]


def test_lu_partial_pivot_applies_finite_arithmetic_to_initial_u():
    A = [[1.239, 2.349], [3.459, 4.569]]
    b = [10.0, 3.0]
    res_trunc = solve_lu(A, b, strategy="partial_pivot", decimals=2, rounding_mode="truncate")
    assert res_trunc["status"] == "converged"
    # Fila pivote 2 (3.459, 4.569) queda truncada a (3.45, 4.56) antes de eliminar
    assert res_trunc["U"][0] == [3.45, 4.56]
    assert r"4.56" in res_trunc["backward_equations"][1]

    res_round = solve_lu(A, b, strategy="partial_pivot", decimals=2, rounding_mode="round")
    res_none = solve_lu(A, b, strategy="partial_pivot", decimals=2, rounding_mode="none")
    assert res_round["status"] == "converged"
    assert res_round["U"][0] == [3.46, 4.57]
    assert res_round["x"] != res_trunc["x"]
    assert res_round["x"] != res_none["x"]


def test_lu_finite_rounding_induced_zero_pivot():
    res_sing = solve_lu([[1.0, 1.0], [1.0, 1.004]], [2.0, 2.004], strategy="doolittle", decimals=2, rounding_mode="round")
    assert res_sing["status"] == "singular_matrix"

    res_pivot_req = solve_lu(
        [[1.0, 1.0, 0.0], [1.0, 1.004, 1.0], [0.0, 1.0, 2.0]],
        [2.0, 3.004, 3.0],
        strategy="doolittle",
        decimals=2,
        rounding_mode="round",
    )
    assert res_pivot_req["status"] == "zero_pivot_requires_pivoting"





