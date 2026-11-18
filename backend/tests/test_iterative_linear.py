import json
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.solvers.iterative_linear import solve_iterative_linear, sweep_sor_omega
from app.solvers.matrix_utils import check_diagonal_dominance, compute_iteration_matrices, mat_vec_mul


def test_singular_zero_diagonal_verdict_matches_lu_status():
    with TestClient(app) as client:
        response = client.post("/api/linear/analyze", json={"A": [[0, 1], [0, 2]], "b": [1, 2]})

    assert response.status_code == 200
    data = response.json()
    assert data["lu"]["status"] == "singular_matrix"
    assert data["jacobi"]["status"] == "zero_diagonal"
    assert any("matriz singular" in note for note in data["verdict"]["notes"])
    assert all("resuelve el sistema" not in note for note in data["verdict"]["notes"])


def test_relative_error_overflow_does_not_invalidate_a_finite_residual():
    result = solve_iterative_linear(
        [[1.0, 0.0], [0.0, 1.0]], [1e-10, 0.0], [1e308, 0.0],
        method="jacobi", criterion="residual",
    )

    assert result["status"] == "converged"
    assert result["steps"][0]["abs_error"] == 1e308
    assert result["steps"][0]["rel_error"] is None
    assert result["steps"][0]["rel_error_pct"] is None
    assert result["residual_norm"] == 0.0


def test_iterative_methods_converge_diagonally_dominant():
    A = [[10.0, -1.0, 2.0], [-1.0, 11.0, -1.0], [2.0, -1.0, 10.0]]
    b = [6.0, 22.0, -10.0]
    expected = np.array([1.0, 2.0, -1.0])

    dom = check_diagonal_dominance(A)
    assert dom["is_strictly_dominant"] is True

    mats = compute_iteration_matrices(A, b, omega=1.15)
    assert mats["jacobi"]["spectral_radius"] < 1.0
    assert mats["gauss_seidel"]["spectral_radius"] < 1.0
    assert mats["sor"]["spectral_radius"] < 1.0

    for method, w in (("jacobi", 1.0), ("gauss_seidel", 1.0), ("sor", 1.1)):
        res = solve_iterative_linear(A, b, [0.0, 0.0, 0.0], method=method, omega=w, tol=1e-8)
        assert res["status"] == "converged"
        assert np.allclose(res["solution"], expected, atol=1e-6)
        assert len(res["steps"][0]["variable_steps"]) == 3


def test_non_dominant_divergent_system():
    A = [[1.0, 3.0, -2.0], [3.0, 2.0, 1.0], [2.0, -1.0, 1.0]]
    b = [5.0, 10.0, 3.0]

    dom = check_diagonal_dominance(A)
    assert dom["is_strictly_dominant"] is False

    mats = compute_iteration_matrices(A, b, omega=1.25)
    assert mats["jacobi"]["spectral_radius"] > 1.0

    res_j = solve_iterative_linear(A, b, [0.0, 0.0, 0.0], method="jacobi", tol=1e-6, max_iter=60)
    assert res_j["status"] in ("diverged", "max_iterations")


def test_sor_sweep_improves_tridiagonal():
    A = [
        [4.0, -1.0, 0.0, 0.0],
        [-1.0, 4.0, -1.0, 0.0],
        [0.0, -1.0, 4.0, -1.0],
        [0.0, 0.0, -1.0, 3.0],
    ]
    b = [15.0, 10.0, 10.0, 10.0]

    gs = solve_iterative_linear(A, b, None, method="gauss_seidel", tol=1e-7, max_iter=100)
    sweep = sweep_sor_omega(A, b, None, tol=1e-7, max_iter=100)
    assert sweep["available"] is True
    assert sweep["optimal_omega"] is not None
    assert sweep["optimal_iterations"] <= gs["iterations"]


def test_zero_diagonal_handling():
    A = [[0.0, 2.0], [3.0, 4.0]]
    b = [1.0, 2.0]
    res = solve_iterative_linear(A, b, None, method="jacobi")
    assert res["status"] == "zero_diagonal"
    assert "a(1,1)" in res["message"]
    assert "_{" not in res["message"]


def test_large_finite_solution_is_not_classified_as_divergence():
    identity = [[1, 0], [0, 1]]
    target = [2e12, -3e12]
    for method in ("jacobi", "gauss_seidel", "sor"):
        result = solve_iterative_linear(identity, target, None, method, omega=1.0)
        assert result["status"] == "converged"
        assert result["solution_raw"] == target
        assert result["residual_norm"] == 0


def test_small_nonzero_diagonal_keeps_iterative_diagnostics_available():
    for scale in (1e-15, 1e-310):
        A = [[scale, 0], [0, 2 * scale]]
        b = [scale, 4 * scale]
        assert not check_diagonal_dominance(A)["has_zero_diagonal"]
        matrices = compute_iteration_matrices(A, b, omega=1.2)
        for method in ("jacobi", "gauss_seidel", "sor"):
            result = solve_iterative_linear(A, b, None, method, omega=1.2)
            assert result["status"] == "converged"
            assert np.allclose(result["solution"], [1, 2], atol=1e-6)
            assert matrices[method]["available"]
        assert sweep_sor_omega(A, b, None)["available"]


def test_raw_iterate_matches_reported_residual_and_rounding_stagnation_is_visible():
    A, b = [[1, 0], [0, 1]], [1e-5, 1]
    result = solve_iterative_linear(A, b, None, "jacobi", decimals=4)
    assert result["solution"] == [0, 1]
    assert result["solution_raw"] == b
    assert result["residual_norm"] == 0
    for criterion in ("abs_diff", "rel_diff", "rel_pct", "residual"):
        rounded = solve_iterative_linear(
            A, b, None, "jacobi", decimals=4, criterion=criterion, apply_rounding=True,
        )
        assert rounded["status"] == "stagnated"
        assert rounded["residual_norm"] == 1e-5


def test_inspector_keeps_equations_after_iteration_40():
    result = solve_iterative_linear(
        [[100, -99], [-99, 100]], [1, 1], None, "jacobi", tol=1e-10, max_iter=55,
    )
    assert result["iterations"] == 55
    assert all(len(step["variable_steps"]) == 2 for step in result["steps"])


def test_gauss_seidel_and_sor_round_after_computing_all_components():
    for method, omega, expected in (
        ("gauss_seidel", 1.0, [0.0, 0.05]),
        ("sor", 0.8, [0.0, 0.03]),
    ):
        result = solve_iterative_linear(
            [[1, 0], [-10, 1]], [0.0049, 0], None, method, omega=omega,
            max_iter=1, decimals=2, apply_rounding=True,
        )
        assert result["solution_raw"] == expected
        assert result["steps"][0]["x"] == expected
        assert r"\operatorname{redondear}" in result["steps"][0]["variable_steps"][0]


def test_sweep_matches_sor_under_the_same_stopping_and_rounding_settings():
    A = [[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]]
    b = [15, 10, 10, 10]
    options = {"criterion": "residual", "decimals": 4, "apply_rounding": True}
    sweep = sweep_sor_omega(A, b, None, **options)
    assert len(sweep["points"]) == 38
    for point in sweep["points"]:
        full = solve_iterative_linear(A, b, None, "sor", omega=point["omega"], **options)
        assert point["iterations"] == full["iterations"]
        assert point["converged"] == (full["status"] == "converged")
    assert sweep["optimal_iterations"] == 6


def test_matrix_iteration_form_matches_pointwise_step():
    A = [[10.0, -1.0, 2.0], [-1.0, 11.0, -1.0], [2.0, -1.0, 10.0]]
    b = [6.0, 22.0, -10.0]
    x0 = [1.0, -2.0, 0.5]
    omega = 1.255
    mats = compute_iteration_matrices(A, b, omega=omega, decimals=12)
    for method in ("jacobi", "gauss_seidel", "sor"):
        step1 = solve_iterative_linear(A, b, x0, method=method, omega=omega, max_iter=1, decimals=12)
        expected = [v + ci for v, ci in zip(mat_vec_mul(mats[method]["T"], x0), mats[method]["c"])]
        assert np.allclose(step1["solution_raw"], expected, atol=1e-11)
    assert "1.255" in solve_iterative_linear(A, b, x0, method="sor", omega=omega, max_iter=1)["steps"][0]["variable_steps"][0]


def test_linear_api_handles_extreme_overflow_and_zero_matrix_without_nan():
    client = TestClient(app)
    for payload in (
        {"A": [[1e308, 1e308, 1e308], [1e308, 1e308, 1e308], [1e308, 1e308, 1e308]], "b": [1, 1, 1]},
        {"A": [[1e-310, 1.0], [1.0, 1e-310]], "b": [1.0, 1.0]},
        {"A": [[0.0, 0.0], [0.0, 0.0]], "b": [0.0, 0.0]},
    ):
        r_analyze = client.post("/api/linear/analyze", json=payload)
        assert r_analyze.status_code == 200
        json.dumps(r_analyze.json(), allow_nan=False)

        r_sweep = client.post("/api/linear/sor-sweep", json={"A": payload["A"], "b": payload["b"]})
        assert r_sweep.status_code == 200
        json.dumps(r_sweep.json(), allow_nan=False)


def test_zero_diagonal_keys_and_diagnostics_decimals_consistency():
    res = solve_iterative_linear([[0.0, 1.0], [1.0, 2.0]], [1.0, 2.0], None, method="jacobi")
    assert res["status"] == "zero_diagonal"
    assert "solution_latex" in res and res["solution_latex"] is None

    sweep = sweep_sor_omega([[0.0, 1.0], [1.0, 2.0]], [1.0, 2.0], None)
    assert sweep["available"] is False
    assert "optimal_spectral_radius" in sweep and sweep["optimal_spectral_radius"] is None

    client = TestClient(app)
    r_analyze = client.post(
        "/api/linear/analyze",
        json={"A": [[3.0, 1.0], [1.0, 3.0]], "b": [1.0, 1.0], "decimals": 6},
    )
    assert r_analyze.status_code == 200
    jacobi_mat = r_analyze.json()["diagnostics"]["matrices"]["jacobi"]
    assert jacobi_mat["c"][0] == 0.333333
    assert "0.333333" in jacobi_mat["T_latex"]


def test_diagonal_dominance_weak_vs_strict_and_all_stopping_criteria():
    weak_dom = check_diagonal_dominance([[2.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 4.0]])
    assert weak_dom["is_strictly_dominant"] is False
    assert weak_dom["is_weakly_dominant"] is True

    equal_dom = check_diagonal_dominance([[1.0, 1.0], [1.0, 1.0]])
    assert equal_dom["is_strictly_dominant"] is False
    assert equal_dom["is_weakly_dominant"] is False

    A = [[10.0, -1.0, 2.0], [-1.0, 11.0, -1.0], [2.0, -1.0, 10.0]]
    b = [6.0, 22.0, -10.0]
    for criterion in ("abs_diff", "rel_diff", "rel_pct", "residual"):
        res = solve_iterative_linear(A, b, [0.0, 0.0, 0.0], method="gauss_seidel", tol=1e-7, criterion=criterion)
        assert res["status"] == "converged"
        assert np.allclose(res["solution_raw"], [1.0, 2.0, -1.0], atol=1e-6)

    # Sistema homogéneo cuyo iterado es cero: rel_error == 0.0 sin división por cero
    zero_sol = solve_iterative_linear([[4.0, 1.0], [1.0, 4.0]], [0.0, 0.0], [0.0, 0.0], method="jacobi", criterion="rel_diff")
    assert zero_sol["status"] == "converged"
    assert zero_sol["steps"][0]["rel_error"] == 0.0
    assert zero_sol["steps"][0]["rel_error_pct"] == 0.0


def test_linear_api_validates_dimensions_and_parameter_bounds():
    client = TestClient(app)
    invalid_payloads = [
        {"A": [[1.0]], "b": [1.0]},
        {"A": [[float(i == j) for j in range(7)] for i in range(7)], "b": [1.0] * 7},
        {"A": [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], "b": [1.0, 1.0, 1.0]},
        {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [1.0]},
        {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [1.0, 1.0], "x0": [0.0]},
        {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [1.0, 1.0], "omega": 2.0},
        {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [1.0, 1.0], "omega": 0.0},
        {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [1.0, 1.0], "tol": 1.5},
        {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [1.0, 1.0], "decimals": 1},
        {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [1.0, 1.0], "max_iter": 501},
    ]
    for payload in invalid_payloads:
        assert client.post("/api/linear/analyze", json=payload).status_code == 422

    for strategy in ("doolittle", "crout", "partial_pivot"):
        r = client.post(
            "/api/linear/analyze",
            json={"A": [[0.0, 2.0], [1.0, 3.0]], "b": [2.0, 4.0], "lu_strategy": strategy},
        )
        assert r.status_code == 200
        expected_status = "converged" if strategy == "partial_pivot" else "zero_pivot_requires_pivoting"
        assert r.json()["lu"]["status"] == expected_status


def test_iterative_rel_diff_tiny_scale_and_landing_on_zero():
    A = [[10.0, 1.0], [1.0, 10.0]]
    b = [1e-18, 1e-18]
    x_exact = 1e-18 / 11.0
    res = solve_iterative_linear(A, b, [1e-17, 1e-17], method="jacobi", tol=1e-6, criterion="rel_diff")
    assert res["status"] == "converged"
    assert abs(res["solution_raw"][0] - x_exact) / x_exact < 1e-6

    res_land = solve_iterative_linear([[4.0, 0.0], [0.0, 4.0]], [0.0, 0.0], [1.0, 1.0], method="jacobi", tol=1.0, criterion="rel_diff")
    assert res_land["status"] == "converged"
    assert res_land["iterations"] == 2
    assert res_land["steps"][0]["rel_error"] is None
    assert res_land["steps"][1]["rel_error"] == 0.0


def test_sor_sweep_all_divergent_validation_and_cors():
    sweep = sweep_sor_omega([[1.0, 3.0], [3.0, 1.0]], [1.0, 1.0], [1.0, 1.0], tol=1e-8, max_iter=5)
    assert sweep["available"] is True
    assert len(sweep["points"]) == 38
    assert sweep["optimal_omega"] is None
    assert sweep["optimal_iterations"] is None
    assert sweep["optimal_spectral_radius"] is None

    client = TestClient(app)
    for bad in (
        {"A": [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], "b": [1.0, 1.0, 1.0]},
        {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [1.0]},
        {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [1.0, 1.0], "max_iter": 0},
    ):
        assert client.post("/api/linear/sor-sweep", json=bad).status_code == 422

    r_cors = client.options(
        "/api/health",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert r_cors.headers.get("access-control-allow-credentials") != "true"


def test_suggest_row_reordering_improves_dominance_and_removes_zero_diagonal():
    from app.solvers.matrix_utils import suggest_row_reordering

    # Matriz 2x2 reordenable a estrictamente dominante
    reord_2x2 = suggest_row_reordering([[1.0, 5.0], [6.0, 1.0]], [11.0, 7.0])
    assert reord_2x2 is not None
    assert reord_2x2["improves"] is True
    assert reord_2x2["permutation"] == [2, 1]
    assert reord_2x2["is_strictly_dominant"] is True
    assert reord_2x2["has_zero_diagonal"] is False
    assert reord_2x2["A_reordered"] == [[6.0, 1.0], [1.0, 5.0]]
    assert reord_2x2["b_reordered"] == [7.0, 11.0]

    # Matriz con cero en la diagonal (preset PA=LU)
    reord_zero = suggest_row_reordering([[0.0, 2.0, 3.0], [4.0, 6.0, 7.0], [2.0, -3.0, 6.0]], [8.0, -3.0, 5.0])
    assert reord_zero is not None
    assert reord_zero["improves"] is True
    assert reord_zero["has_zero_diagonal"] is False

    # Matriz ya estrictamente dominante o imposible de mejorar
    assert suggest_row_reordering([[10.0, 1.0], [1.0, 10.0]], [1.0, 1.0]) is None
    assert suggest_row_reordering([[1.0, 3.0], [1.0, 3.0]], [1.0, 1.0]) is None


def test_iterative_flops_verdict_and_truncate_mode():
    A = [[10.0, -1.0, 2.0], [-1.0, 11.0, -1.0], [2.0, -1.0, 10.0]]
    b = [6.0, 22.0, -10.0]

    gs = solve_iterative_linear(A, b, [0.0, 0.0, 0.0], method="gauss_seidel", tol=1e-6)
    sor = solve_iterative_linear(A, b, [0.0, 0.0, 0.0], method="sor", omega=1.15, tol=1e-6)
    assert gs["operation_count"]["per_iteration_flops"] == 3 * 3 + 3 * 2
    assert sor["operation_count"]["per_iteration_flops"] == 3 * 5 + 3 * 4
    assert gs["operation_count"]["total_flops"] == gs["iterations"] * gs["operation_count"]["per_iteration_flops"]

    # Diferencia entre redondeo y truncamiento a 2 decimales
    res_round = solve_iterative_linear([[1.0, 0.0], [0.0, 1.0]], [0.009, 1.0], None, "jacobi", decimals=2, rounding_mode="round")
    res_trunc = solve_iterative_linear([[1.0, 0.0], [0.0, 1.0]], [0.009, 1.0], None, "jacobi", decimals=2, rounding_mode="truncate")
    assert res_round["solution"] == [0.01, 1.0]
    assert res_trunc["solution"] == [0.0, 1.0]
    assert res_trunc["status"] == "stagnated"
    assert r"\operatorname{truncar}" in res_trunc["steps"][0]["variable_steps"][0]

    client = TestClient(app)
    resp = client.post(
        "/api/linear/analyze",
        json={"A": [[1.0, 5.0], [6.0, 1.0]], "b": [11.0, 7.0], "rounding_mode": "truncate", "decimals": 3},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["diagnostics"]["dominance"]["reordering_suggestion"]["permutation"] == [2, 1]
    assert payload["verdict"]["lowest_flops_method"] == "lu"
    assert "FLOPs" in payload["verdict"]["summary"]
    assert payload["lu"]["rounding_mode"] == "truncate"

    resp_sweep = client.post(
        "/api/linear/sor-sweep",
        json={"A": [[6.0, 1.0], [1.0, 5.0]], "b": [7.0, 11.0], "rounding_mode": "truncate", "decimals": 3},
    )
    assert resp_sweep.status_code == 200
    assert resp_sweep.json()["available"] is True


def test_iterative_rel_pct_steps_stopping_and_tol_validation():
    A = [[10.0, -1.0, 2.0], [-1.0, 11.0, -1.0], [2.0, -1.0, 10.0]]
    b = [6.0, 22.0, -10.0]
    res_pct = solve_iterative_linear(A, b, [0.0, 0.0, 0.0], method="jacobi", tol=1.0, criterion="rel_pct")
    assert res_pct["status"] == "converged"
    assert "error relativo porcentual" in res_pct["message"]
    assert res_pct["steps"][-1]["rel_error_pct"] < 1.0
    if len(res_pct["steps"]) >= 2:
        assert res_pct["steps"][-2]["rel_error_pct"] >= 1.0
    for step in res_pct["steps"]:
        assert abs(step["rel_error_pct"] - step["rel_error"] * 100.0) < 1e-12

    # Aterrizar en x = 0 desde x0 != 0 no dispara parada por rel_pct en la iteración 1
    res_land = solve_iterative_linear(
        [[4.0, 0.0], [0.0, 4.0]], [0.0, 0.0], [1.0, 1.0], method="jacobi", tol=50.0, criterion="rel_pct",
    )
    assert res_land["status"] == "converged"
    assert res_land["iterations"] == 2
    assert res_land["steps"][0]["rel_error"] is None
    assert res_land["steps"][0]["rel_error_pct"] is None
    assert res_land["steps"][1]["rel_error_pct"] == 0.0

    sweep_pct = sweep_sor_omega(A, b, [0.0, 0.0, 0.0], tol=1.0, criterion="rel_pct")
    assert sweep_pct["available"] is True
    assert sweep_pct["optimal_omega"] is not None

    client = TestClient(app)
    for endpoint in ("/api/linear/analyze", "/api/linear/sor-sweep"):
        r_ok = client.post(endpoint, json={"A": A, "b": b, "tol": 5.0, "criterion": "rel_pct"})
        assert r_ok.status_code == 200
        if endpoint == "/api/linear/analyze":
            assert r_ok.json()["jacobi"]["steps"][-1]["rel_error_pct"] < 5.0
        r_edge_ok = client.post(endpoint, json={"A": A, "b": b, "tol": 99.9, "criterion": "rel_pct"})
        assert r_edge_ok.status_code == 200
        for bad_tol in (0.0, -1.0, 100.0, 150.0):
            assert client.post(endpoint, json={"A": A, "b": b, "tol": bad_tol, "criterion": "rel_pct"}).status_code == 422
        for crit in ("abs_diff", "rel_diff", "residual"):
            assert client.post(endpoint, json={"A": A, "b": b, "tol": 1.5, "criterion": crit}).status_code == 422


def test_finite_arithmetic_stagnation_under_rel_pct_and_zero_normalization():
    import math
    from app.solvers.matrix_utils import format_omega, round_matrix, round_vector

    # Caso 1: tol=0.5 (%) -> res_norm=0.009 >= 0.005 debe estancarse, no converger falsamente (antes 0.009 < 0.5 pasaba)
    stag_pct = solve_iterative_linear(
        [[1.0, 0.0], [0.0, 1.0]],
        [0.049, 1.0],
        [0.0, 0.0],
        method="jacobi",
        tol=0.5,
        criterion="rel_pct",
        decimals=2,
        rounding_mode="truncate",
    )
    assert stag_pct["status"] == "stagnated"

    # Caso 2: solución pequeña truncada a [0, 0] con error relativo sin redondear de 100%
    stag_zero = solve_iterative_linear(
        [[1.0, 0.0], [0.0, 1.0]],
        [0.009, 0.009],
        [0.0, 0.0],
        method="jacobi",
        tol=1.0,
        criterion="rel_pct",
        decimals=2,
        rounding_mode="truncate",
    )
    assert stag_zero["status"] == "stagnated"

    # Caso 3: convergencia legítima exacta en aritmética finita y tolerancia holgada válida
    exact_hit = solve_iterative_linear(
        [[4.0, 1.0], [1.0, 4.0]],
        [5.0, 5.0],
        [1.0, 1.0],
        method="jacobi",
        tol=1.0,
        criterion="rel_pct",
        decimals=2,
        rounding_mode="round",
    )
    assert exact_hit["status"] == "converged"
    assert exact_hit["residual_norm"] == 0.0

    valid_tol = solve_iterative_linear(
        [[3.0, 0.0], [0.0, 3.0]],
        [1.0, 1.0],
        [0.0, 0.0],
        method="jacobi",
        tol=5.0,
        criterion="rel_pct",
        decimals=2,
        rounding_mode="round",
    )
    assert valid_tol["status"] == "converged"

    # Normalización de -0.0 y formato de omega
    rv = round_vector([-1e-10, 1.0], 4)
    rm = round_matrix([[-1e-10, 0.0], [0.0, -0.0]], 4)
    assert rv[0] == 0.0 and math.copysign(1.0, rv[0]) == 1.0
    assert rm[0][0] == 0.0 and math.copysign(1.0, rm[0][0]) == 1.0
    assert rm[1][1] == 0.0 and math.copysign(1.0, rm[1][1]) == 1.0
    assert format_omega(1.2) == "1.20"
    assert format_omega(1.25) == "1.25"
    assert format_omega(1.255) == "1.255"





