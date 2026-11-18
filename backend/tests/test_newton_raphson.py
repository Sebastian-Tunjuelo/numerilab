import json
import math
from fastapi.testclient import TestClient
from app.main import app
from app.solvers.newton_raphson import solve_newton_1d, solve_newton_system


def test_newton_1d_polynomial_and_transcendental():
    res_poly = solve_newton_1d("x^3 - 2*x - 5", x0=2.0, tol=1e-9, decimals=8)
    assert res_poly["status"] == "converged"
    assert abs(res_poly["root_raw"] - 2.0945514815) < 1e-7
    assert "3 x^{2}" in res_poly["df_latex"] or "3x^{2}" in res_poly["df_latex"].replace(" ", "")

    res_trig = solve_newton_1d("cos(x) - x", x0=0.5, tol=1e-9, decimals=8)
    assert res_trig["status"] == "converged"
    assert abs(res_trig["root_raw"] - 0.7390851332) < 1e-7

    res_exp = solve_newton_1d("e^(-x) - x", x0=0.0, tol=1e-8, decimals=8)
    assert res_exp["status"] == "converged"
    assert abs(res_exp["root_raw"] - 0.56714329) < 1e-6


def test_newton_1d_zero_derivative():
    res = solve_newton_1d("x^2 + 1", x0=0.0)
    assert res["status"] == "zero_derivative"
    assert "f'(x₀)" in res["message"]
    assert "x_0" not in res["message"]


def test_newton_system_2x2_and_lu_bridge():
    res = solve_newton_system(
        equations=["x^2 + y^2 - 4", "exp(x) + y - 1"],
        variables=["x", "y"],
        x0=[1.0, -1.5],
        tol=1e-8,
        decimals=7,
    )
    assert res["status"] == "converged"
    x_sol, y_sol = res["solution"]
    assert abs(x_sol**2 + y_sol**2 - 4.0) < 1e-6
    assert abs(math.exp(x_sol) + y_sol - 1.0) < 1e-6
    assert "lu_detail" in res["steps"][0]
    assert res["contour_2d"] is not None
    assert len(res["contour_2d"]["f1_segments"]) > 10
    assert len(res["contour_2d"]["f2_segments"]) > 10


def test_api_endpoints_integration():
    client = TestClient(app)
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_lin = client.post(
        "/api/linear/analyze",
        json={
            "A": [[10, -1, 2], [-1, 11, -1], [2, -1, 10]],
            "b": [6, 25, -11],
            "omega": 1.1,
        },
    )
    assert r_lin.status_code == 200
    data = r_lin.json()
    assert data["lu"]["status"] == "converged"
    assert data["jacobi"]["status"] == "converged"
    assert data["gauss_seidel"]["status"] == "converged"
    assert data["sor"]["status"] == "converged"

    r_sweep = client.post(
        "/api/linear/sor-sweep",
        json={
            "A": [[4, -1, 0], [-1, 4, -1], [0, -1, 4]],
            "b": [15, 10, 10],
        },
    )
    assert r_sweep.status_code == 200
    assert r_sweep.json()["optimal_omega"] is not None

    r_nl1 = client.post("/api/nonlinear/newton-1d", json={"expression": "x^3 - 2*x - 5", "x0": 2.0})
    assert r_nl1.status_code == 200
    assert r_nl1.json()["status"] == "converged"

    r_nl2 = client.post(
        "/api/nonlinear/newton-system",
        json={"equations": ["x^2 + y^2 - 4", "exp(x) + y - 1"], "variables": ["x", "y"], "x0": [1.0, -1.5]},
    )
    assert r_nl2.status_code == 200
    assert r_nl2.json()["status"] == "converged"


def test_newton_requires_a_small_residual_in_both_modes():
    expression = "1e18*(x-1)^2 + 1"
    scalar = solve_newton_1d(expression, x0=1.000000001)
    system = solve_newton_system([expression, "y - 2"], ["x", "y"], [1.000000001, 2])
    assert scalar["status"] != "converged"
    assert scalar["root_raw"] is None
    assert system["status"] != "converged"
    assert system["solution"] is None


def test_initial_roots_with_singular_derivatives_are_recognized():
    scalar = solve_newton_1d("x^2", x0=0)
    system = solve_newton_system(["x^2", "y^2"], ["x", "y"], [0, 0])
    assert (scalar["status"], scalar["root_raw"], scalar["iterations"]) == ("converged", 0, 0)
    assert (system["status"], system["solution"], system["iterations"]) == ("converged", [0, 0], 0)


def test_newton_does_not_accept_a_small_residual_while_steps_grow():
    expression = "1/(x-0.13)"
    for initial in (1, 2e7):
        scalar = solve_newton_1d(expression, initial)
        system = solve_newton_system([expression, "y"], ["x", "y"], [initial, 0])
        assert scalar["status"] != "converged"
        assert scalar["root_raw"] is None
        assert system["status"] != "converged"
        assert system["solution"] is None


def test_plot_contains_tangent_for_large_abscissa():
    result = solve_newton_1d("x-600", 590)
    assert result["root_raw"] == 600
    assert result["plot_bounds"]["x_min"] < 590 < 600 < result["plot_bounds"]["x_max"]


def test_finite_newton_steps_can_exceed_previous_magnitude_limit():
    scalar = solve_newton_1d("x - 2e10", 0)
    system = solve_newton_system(["x - 2e10", "y - 2"], ["x", "y"], [0, 0])
    assert scalar["root_raw"] == 2e10
    assert scalar["plot_bounds"]["x_max"] > 2e10
    assert system["solution"] == [2e10, 2]


def test_contour_window_includes_large_finite_trajectory():
    result = solve_newton_system(["x - 600", "y - 2"], ["x", "y"], [0, 0])
    assert result["solution"] == [600, 2]
    contour = result["contour_2d"]
    assert contour["bounds"]["x_min"] < 0 < 600 < contour["bounds"]["x_max"]
    assert contour["f1_segments"]
    assert all(abs(x - 600) < 1e-6 for segment in contour["f1_segments"] for x, _ in segment)


def test_newton_does_not_treat_residual_underflow_as_an_exact_root():
    scalar = solve_newton_1d("exp(-x)", 1000)
    system = solve_newton_system(["exp(-x)", "y"], ["x", "y"], [1000, 0])
    assert scalar["status"] == system["status"] == "diverged"
    assert scalar["root_raw"] is None
    assert system["solution"] is None


def test_newton_does_not_accept_a_zero_caused_by_cancellation():
    expression = "(x + 1) / x - 1"
    scalar = solve_newton_1d(expression, 1e17)
    system = solve_newton_system([expression, "y"], ["x", "y"], [1e17, 0])
    assert scalar["status"] == "zero_derivative"
    assert scalar["root_raw"] is None
    assert system["status"] == "singular_jacobian"
    assert system["solution"] is None


def test_small_derivative_allows_acceptance_when_no_step_is_needed():
    x0 = 1.00000001
    result = solve_newton_1d("1e-15*(x-1)", x0)
    assert result["status"] == "converged"
    assert result["root_raw"] == x0
    assert result["iterations"] == 0


def test_contours_reject_a_pole_and_find_the_base_of_an_even_power():
    bounds = {"x_min": -2, "x_max": 2, "y_min": -2, "y_max": 2}
    for expression, has_zero_curve in (("1/(x-0.13)", False), ("(x-0.13)^2", True)):
        result = solve_newton_system([expression, "y"], ["x", "y"], [1, 1], bounds=bounds)
        segments = result["contour_2d"]["f1_segments"]
        assert bool(segments) == has_zero_curve
        assert all(abs(x - 0.13) < 1e-6 for segment in segments for x, _ in segment)
        assert result["contour_2d"]["f2_segments"]


def test_expression_grammar_validates_equations_and_symbols():
    valid = solve_newton_1d("2x + sen(0) = 4", x0=0)
    assert valid["root_raw"] == 2

    for invalid in ("x + y", "x - 2 =", "x = 2 = 3", "7 8", "x[1]"):
        error = solve_newton_1d(invalid, x0=1)
        assert error["status"] == "parse_error"
        assert error["steps"] == []

    for variables in (["x", "x"], ["x", "sin"]):
        error = solve_newton_system(["x - 2", "x - 1"], variables, [0, 0])
        assert error["status"] == "parse_error"

    assert solve_newton_system(["x + z", "y"], ["x", "y"], [0, 0])["status"] == "parse_error"
    client = TestClient(app)
    response = client.post("/api/nonlinear/newton-system", json={
        "equations": [" " * 301 + "x", "y"], "x0": [0, 0],
    })
    assert response.status_code == 422


def test_newton_3d_reference_still_uses_lu():
    result = solve_newton_system(
        ["3*x - cos(y*z) - 0.5", "x^2 - 81*(y + 0.1)^2 + sin(z) + 1.06",
         "exp(-x*y) + 20*z + (10*pi - 3)/3"],
        ["x", "y", "z"], [0.1, 0.1, -0.1],
    )
    assert result["status"] == "converged"
    assert result["solution"] == [0.5, 0.0, -0.523599]
    assert all("P_latex" in step["lu_detail"] for step in result["steps"])


def test_newton_step_exposes_the_evaluated_system_and_row_permutation():
    result = solve_newton_system(["y - 2", "x - 3"], ["x", "y"], [0, 0])
    assert result["solution"] == [3, 2]
    [step] = result["steps"]
    assert step["J_eval_latex"] == step["lu_detail"]["P_latex"]
    assert step["neg_F_latex"] == r"\begin{bmatrix} 2 \\ 3 \end{bmatrix}"
    assert step["dx"] == [3, 2]

def test_api_rejects_non_finite_inputs_with_serializable_detail():
    client = TestClient(app)
    response = client.post(
        "/api/linear/analyze",
        content='{"A":[[1e400,1],[1,2]],"b":[1,2]}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "A", 0, 0]
    assert error["type"] == "finite_number"


def test_newton_reports_overflow_and_domain_errors():
    assert solve_newton_1d("exp(x)", x0=1000)["status"] == "diverged"
    assert solve_newton_1d("sqrt(x)", x0=-1)["status"] == "domain_error"
    assert solve_newton_1d("log(x, 0)", x0=2.0)["status"] == "parse_error"

    res_err_ovf = solve_newton_1d("x - 1e308", x0=-1e308)
    assert res_err_ovf["status"] == "diverged"
    json.dumps(res_err_ovf, allow_nan=False)

    res_sys_ovf = solve_newton_system(["x - 1e308", "y - 1e308"], ["x", "y"], [-1e308, -1e308])
    assert res_sys_ovf["status"] == "diverged"
    json.dumps(res_sys_ovf, allow_nan=False)


def test_api_validates_bounds_and_rejects_non_finite_across_all_endpoints():
    client = TestClient(app)

    bad_bounds = client.post(
        "/api/nonlinear/newton-system",
        json={
            "equations": ["x", "y"],
            "variables": ["x", "y"],
            "x0": [0.0, 0.0],
            "bounds": {"x_min": 2.0, "x_max": 2.0, "y_min": -1.0, "y_max": 1.0},
        },
    )
    assert bad_bounds.status_code == 422

    span_overflow = client.post(
        "/api/nonlinear/newton-system",
        json={
            "equations": ["x", "y"],
            "variables": ["x", "y"],
            "x0": [0.0, 0.0],
            "bounds": {"x_min": -1e308, "x_max": 1e308, "y_min": -1.0, "y_max": 1.0},
        },
    )
    assert span_overflow.status_code == 422

    for url, body in (
        ("/api/linear/analyze", '{"A":[[1,0],[0,1]],"b":[1,2],"omega":NaN}'),
        ("/api/linear/sor-sweep", '{"A":[[1,0],[0,1]],"b":[1,1e400]}'),
        ("/api/nonlinear/newton-1d", '{"expression":"x","x0":1e400}'),
        ("/api/nonlinear/newton-system", '{"equations":["x","y"],"x0":[0,1e400]}'),
    ):
        resp = client.post(url, content=body, headers={"Content-Type": "application/json"})
        assert resp.status_code == 422
        json.dumps(resp.json(), allow_nan=False)


def test_underflow_near_multiple_roots_converges_while_asymptotic_tail_diverges():
    for expr, x0 in (("x^2", 1e-200), ("x^3", 1e-120)):
        res = solve_newton_1d(expr, x0)
        assert res["status"] == "converged"
        assert res["root_raw"] == x0

    sys_res = solve_newton_system(["x^2", "y - 1"], ["x", "y"], [1e-200, 1.0])
    assert sys_res["status"] == "converged"
    assert sys_res["solution"] == [0.0, 1.0]


def test_log_constant_base_validation_and_trailing_syntax_errors():
    for invalid in ("log(x, 1-1)", "log(x, 2-1)", "ln(x, -2)", "log(x, 1/0)"):
        err = solve_newton_1d(invalid, x0=2.0)
        assert err["status"] == "parse_error"

    unmatched_paren = solve_newton_1d("x + 1)", x0=1.0)
    assert unmatched_paren["status"] == "parse_error"
    assert "igualdad" not in unmatched_paren["message"]

    unexpected_comma = solve_newton_1d("sin(x, 2)", x0=1.0)
    assert unexpected_comma["status"] == "parse_error"
    assert "igualdad" not in unexpected_comma["message"]


def test_newton_system_j_eval_respects_decimals():
    res = solve_newton_system(["x^3 - y", "y - 1"], ["x", "y"], [1 / 3, 1.0], decimals=6, max_iter=1)
    assert res["steps"][0]["J_eval"][0][0] == round(3 * (1 / 3) ** 2, 6)


def test_newton_stopping_criteria_rel_diff_rel_pct_and_validation():
    res_pct = solve_newton_1d("exp(-x) - ln(x)", x0=1.0, tol=1.0, criterion="rel_pct")
    assert res_pct["status"] == "converged"
    assert res_pct["criterion"] == "rel_pct"
    assert res_pct["steps"][-1]["rel_error_pct"] <= 1.0
    assert abs(res_pct["steps"][-1]["rel_error_pct"] - res_pct["steps"][-1]["rel_error"] * 100.0) < 1e-12

    res_rel = solve_newton_1d("exp(-x) - ln(x)", x0=1.0, tol=1e-4, criterion="rel_diff")
    assert res_rel["status"] == "converged"
    assert res_rel["criterion"] == "rel_diff"
    assert res_rel["steps"][-1]["rel_error"] <= 1e-4

    # Paso que cae exactamente en x_next = 0 desde x0 != 0 debe reportar rel_error = None
    res_zero = solve_newton_1d("x", x0=2.0, criterion="rel_diff")
    assert res_zero["status"] == "converged"
    assert res_zero["steps"][0]["rel_error"] is None
    assert res_zero["steps"][0]["rel_error_pct"] is None

    sys_pct = solve_newton_system(
        ["x^2 + y^2 - 4", "exp(x) + y - 1"],
        ["x", "y"],
        [1.0, -1.5],
        tol=1.0,
        criterion="rel_pct",
    )
    assert sys_pct["status"] == "converged"
    assert sys_pct["criterion"] == "rel_pct"
    assert sys_pct["steps"][-1]["rel_error_pct"] <= 1.0

    client = TestClient(app)
    for bad_crit in ("residual_and_step", "rel_diff"):
        r_bad = client.post(
            "/api/nonlinear/newton-1d",
            json={"expression": "x - 1", "x0": 2.0, "tol": 2.0, "criterion": bad_crit},
        )
        assert r_bad.status_code == 422

    r_ok = client.post(
        "/api/nonlinear/newton-1d",
        json={"expression": "exp(-x) - ln(x)", "x0": 1.0, "tol": 1.0, "criterion": "rel_pct"},
    )
    assert r_ok.status_code == 200
    assert r_ok.json()["criterion"] == "rel_pct"

    r_bad_pct = client.post(
        "/api/nonlinear/newton-system",
        json={"equations": ["x - 1", "y - 1"], "x0": [0.0, 0.0], "tol": 100.0, "criterion": "rel_pct"},
    )
    assert r_bad_pct.status_code == 422


def test_newton_1d_estimated_order_distinguishes_double_and_simple_roots():
    double_root = solve_newton_1d("x^3 - 5*x^2 + 7*x - 3", x0=0.5, tol=1e-7)
    simple_root = solve_newton_1d("x^3 - 5*x^2 + 7*x - 3", x0=4.0, tol=1e-7)
    assert double_root["status"] == "converged"
    assert simple_root["status"] == "converged"
    assert 0.85 <= double_root["estimated_order"] <= 1.15
    assert 1.85 <= simple_root["estimated_order"] <= 2.15

    too_few = solve_newton_1d("x - 2", x0=0.0)
    assert too_few["estimated_order"] is None


def test_newton_stagnated_and_singular_jacobian():
    # x0 = 1e16 tiene ulp(1e16) = 2.0, por lo que la corrección de 0.1 deja x_next == x_curr
    stag_1d = solve_newton_1d("x - 1e16 - 0.1", x0=1e16, tol=1e-6)
    assert stag_1d["status"] == "stagnated"
    assert stag_1d["root"] is None

    stag_sys = solve_newton_system(["x - 1e16 - 0.1", "y - 1"], ["x", "y"], [1e16, 1.0], tol=1e-6)
    assert stag_sys["status"] == "stagnated"
    assert stag_sys["solution"] is None

    sing_sys = solve_newton_system(["x^2 - 1", "y^2 - 1"], ["x", "y"], [0.0, 2.0])
    assert sing_sys["status"] == "singular_jacobian"
    assert sing_sys["solution"] is None


def test_expression_parser_security_and_exponent_bounds():
    import pytest
    from app.solvers.math_expression import parse_math_expression
    from app.solvers.newton_raphson import _estimate_convergence_order

    for unsafe in (
        "__import__('os').system('echo 1')",
        "eval('1+1')",
        "x + 1e-100000000",
        "x + 0e100000000",
        "x^(1/0)",
        "x^sqrt(-1)",
        "sqrt(-1)^2",
        "(" * 26 + "x" + ")" * 26,
        " + ".join(["x"] * 70),
    ):
        with pytest.raises(ValueError):
            parse_math_expression(unsafe, ["x"])
        res = solve_newton_1d(unsafe, x0=1.0)
        assert res["status"] == "parse_error"

    # Orden de convergencia no debe retornar 0.0 si d1/d0 desborda
    overflow_steps = [
        {"abs_error": 1e-300},
        {"abs_error": 1e100},
        {"abs_error": 1e200},
    ]
    assert _estimate_convergence_order(overflow_steps) is None


def test_course_presets_and_system_rel_diff_zero_target():
    res_exp = solve_newton_1d("e^(3*x) - 4", x0=0.5, tol=1e-8)
    assert res_exp["status"] == "converged"
    assert abs(res_exp["root_raw"] - math.log(4) / 3) < 1e-7

    res_sys_zero = solve_newton_system(["x", "y"], ["x", "y"], [1.0, -2.0], criterion="rel_diff")
    assert res_sys_zero["status"] == "converged"
    assert res_sys_zero["steps"][0]["rel_error"] is None
    assert res_sys_zero["steps"][0]["rel_error_pct"] is None


def test_contour_extracts_asymmetric_nonlinear_crossing_and_rejects_poles():
    bounds = {"x_min": -2.0, "x_max": 2.0, "y_min": -2.0, "y_max": 2.0}
    res = solve_newton_system(["exp(10*x) - 1 - 1e-10", "y"], ["x", "y"], [1.0, 1.0], bounds=bounds)
    segments = res["contour_2d"]["f1_segments"]
    assert len(segments) > 10
    assert all(abs(x) < 1e-5 for segment in segments for x, _ in segment)

    pole = solve_newton_system(["1/(x - 0.13)", "y"], ["x", "y"], [1.0, 1.0], bounds=bounds)
    assert pole["contour_2d"]["f1_segments"] == []


def test_newton_system_solution_raw_and_3d_edge_cases():
    res = solve_newton_system(["x^2 - 2", "y - 1"], ["x", "y"], [1.5, 1.0], decimals=4)
    assert res["status"] == "converged"
    assert res["solution"] == [1.4142, 1.0]
    assert abs(res["solution_raw"][0] - math.sqrt(2)) < 1e-8

    assert solve_newton_1d("x^2 - 2", x0=10.0, max_iter=1)["status"] == "max_iterations"
    assert solve_newton_system(["x^2 - 1", "y^2 - 1", "z^2 - 1"], ["x", "y", "z"], [5.0, 5.0, 5.0], max_iter=1)["status"] == "max_iterations"

    sing_3d = solve_newton_system(["x + y + z", "2*x + 2*y + 2*z", "z - 1"], ["x", "y", "z"], [1.0, 1.0, 1.0])
    assert sing_3d["status"] == "singular_jacobian"
    assert sing_3d["solution_raw"] is None

    dom_3d = solve_newton_system(["sqrt(x)", "y", "z"], ["x", "y", "z"], [-1.0, 0.0, 0.0])
    assert dom_3d["status"] == "domain_error"
    assert dom_3d["solution_raw"] is None


def test_expression_parser_arity_and_keyword_variables():
    import pytest
    from app.solvers.math_expression import parse_math_expression

    for bad_arity in ("sin()", "log()", "sin(x, 2)", "log(x, 2, 3)", "abs(x, y)"):
        with pytest.raises(ValueError):
            parse_math_expression(bad_arity, ["x", "y"])
        assert solve_newton_1d(bad_arity, x0=1.0)["status"] == "parse_error"

    kw_res = solve_newton_system(["lambda - 2", "class - 3"], ["lambda", "class"], [0.0, 0.0])
    assert kw_res["status"] == "converged"
    assert kw_res["solution"] == [2.0, 3.0]


def test_newton_1d_d2f_fourier_and_convergence_warnings():
    res_simple = solve_newton_1d("x^3 - 5*x^2 + 7*x - 3", x0=4.0, tol=1e-7)
    assert res_simple["status"] == "converged"
    assert "6 x - 10" in res_simple["d2f_latex"] or "6x - 10" in res_simple["d2f_latex"].replace(" ", "")
    assert res_simple["fourier_condition"]["satisfied"] is True
    assert res_simple["convergence_warnings"] == []
    assert all(step["d2fx_k"] is not None for step in res_simple["steps"])

    res_double = solve_newton_1d("x^3 - 5*x^2 + 7*x - 3", x0=0.5, tol=1e-7)
    assert res_double["status"] == "converged"
    assert any("raíz múltiple" in msg for msg in res_double["convergence_warnings"])

    res_fourier_neg = solve_newton_1d("x^3 - 3*x + 1", x0=0.99, tol=1e-7)
    assert res_fourier_neg["fourier_condition"]["satisfied"] is False
    assert any("Fourier" in msg for msg in res_fourier_neg["convergence_warnings"])

    res_inflection = solve_newton_1d("x^3 + x - 1", x0=1e-10, tol=1e-8)
    assert any("inflexión" in msg for msg in res_inflection["convergence_warnings"])

    # Abs(x) produce derivada de orden superior no elemental en lambdify(math) pero no debe romper solve_newton_1d
    res_abs = solve_newton_1d("abs(x) - 1", x0=2.0)
    assert res_abs["status"] == "converged"
    assert res_abs["root_raw"] == 1.0


def test_newton_finite_arithmetic_round_and_truncate():
    res_r = solve_newton_1d("x^2 - 2", x0=1.5, decimals=3, tol=1e-6, rounding_mode="round")
    res_t = solve_newton_1d("x^2 - 2", x0=1.5, decimals=3, tol=1e-6, rounding_mode="truncate")
    assert res_r["rounding_mode"] == "round"
    assert res_t["rounding_mode"] == "truncate"
    assert res_r["steps"][0]["x_next"] == 1.417
    assert res_t["steps"][0]["x_next"] == 1.416
    assert r"\operatorname{truncar}" in res_t["steps"][0]["step_latex"]

    sys_t = solve_newton_system(
        ["x^2 + y^2 - 4", "exp(x) + y - 1"],
        ["x", "y"],
        [1.0, -1.5],
        decimals=3,
        tol=1e-6,
        rounding_mode="truncate",
    )
    assert sys_t["rounding_mode"] == "truncate"
    assert sys_t["status"] in ("converged", "stagnated")

    client = TestClient(app)
    r1 = client.post(
        "/api/nonlinear/newton-1d",
        json={"expression": "x^3 - 2*x - 5", "x0": 2.0, "decimals": 3, "rounding_mode": "truncate"},
    )
    assert r1.status_code == 200
    assert r1.json()["rounding_mode"] == "truncate"
    json.dumps(r1.json(), allow_nan=False)

    r2 = client.post(
        "/api/nonlinear/newton-system",
        json={
            "equations": ["x^2 + y^2 - 4", "exp(x) + y - 1"],
            "variables": ["x", "y"],
            "x0": [1.0, -1.5],
            "decimals": 3,
            "rounding_mode": "round",
        },
    )
    assert r2.status_code == 200
    assert r2.json()["rounding_mode"] == "round"
    json.dumps(r2.json(), allow_nan=False)


def test_newton_identities_not_treated_as_nonzero_evalf_artifacts():
    for expr, x0 in (
        ("x - x", 1.0),
        ("x = x", 1.0),
        ("sin(x)^2 + cos(x)^2 - 1", 1.0),
        ("log(x, x) - 1", 2.0),
        ("cos(2*x) - (cos(x)^2 - sin(x)^2)", 0.0),
    ):
        res = solve_newton_1d(expr, x0=x0)
        assert res["status"] == "converged"
        assert res["iterations"] == 0
        assert res["root_raw"] == x0


def test_newton_system_abs_and_rel_error_use_actual_step_and_parse_error_includes_variables():
    # Con redondeo a 2 decimales, x0=[1.414, 1.732] salta ~0.004 en el primer paso aunque dx sea pequeño.
    res = solve_newton_system(
        equations=["x^2 - 2", "y^2 - 3"],
        variables=["x", "y"],
        x0=[1.414, 1.732],
        decimals=2,
        rounding_mode="round",
        criterion="rel_diff",
        tol=1e-4,
    )
    assert abs(res["steps"][0]["abs_error"] - 0.004) < 1e-12
    assert res["steps"][0]["rel_error"] > 1e-4
    # No debe converger falsamente en la iteración 1 cuando ||x_1 - x_0||_inf = 0.004 > tol
    assert res["iterations"] >= 2 or res["status"] == "stagnated"

    err_2d = solve_newton_system(["x +", "y"], ["x", "y"], [0.0, 0.0])
    assert err_2d["status"] == "parse_error"
    assert err_2d["variables"] == ["x", "y"]

    err_3d = solve_newton_system(["x +", "y", "z"], ["x", "y", "z"], [0.0, 0.0, 0.0])
    assert err_3d["status"] == "parse_error"
    assert err_3d["variables"] == ["x", "y", "z"]


def test_newton_1d_divergent_geometric_step_does_not_emit_multiple_root_warning():
    # Para f(x) = 1/x con x0 = 1.0, Newton genera x_{k+1} = 2*x_k (orden aparente p = 1.0 pero divergiendo).
    res = solve_newton_1d("1/x", x0=1.0, max_iter=5)
    assert res["status"] == "max_iterations"
    assert res["estimated_order"] == 1.0
    assert not any("raíz múltiple" in msg for msg in res["convergence_warnings"])


def test_newton_spanish_eval_errors_rounding_stagnation_and_3d_http():
    # Mensajes en español sin fugas de textos en inglés de Python
    for expr, x0, expected_status in (
        ("sqrt(x)", -1.0, "domain_error"),
        ("1/0", 1.0, "domain_error"),
        ("exp(x)", 1000.0, "diverged"),
    ):
        res = solve_newton_1d(expr, x0=x0)
        assert res["status"] == expected_status
        for english in ("math domain error", "division by zero", "math range error"):
            assert english not in res["message"].lower()

    sys_dom = solve_newton_system(["sqrt(x)", "y - 1"], ["x", "y"], [-1.0, 1.0])
    assert sys_dom["status"] == "domain_error"
    assert "math domain error" not in sys_dom["message"].lower()

    # Estancamiento por redondeo bajo rel_diff y rel_pct no debe reportar converged falsamente
    for crit, tol in (("rel_diff", 1e-4), ("rel_pct", 1.0)):
        stag_1d = solve_newton_1d("x^2 - 2", x0=1.4, decimals=1, tol=tol, criterion=crit, rounding_mode="round")
        assert stag_1d["status"] == "stagnated"
        assert stag_1d["root"] is None

        stag_sys = solve_newton_system(
            ["x^2 - 2", "y - 1"],
            ["x", "y"],
            [1.4, 1.0],
            decimals=1,
            tol=tol,
            criterion=crit,
            rounding_mode="round",
        )
        assert stag_sys["status"] == "stagnated"
        assert stag_sys["solution"] is None

    # Convergencia válida en aritmética finita cuando residual y paso sin redondear cumplen tolerancia
    ok_1d = solve_newton_1d("x^2 - 2", x0=1.414, decimals=3, tol=1.0, criterion="rel_pct", rounding_mode="round")
    assert ok_1d["status"] == "converged"
    assert ok_1d["root"] == 1.414

    # Integración HTTP para sistema no lineal 3x3 con puente LU
    client = TestClient(app)
    r3 = client.post(
        "/api/nonlinear/newton-system",
        json={
            "equations": [
                "3*x - cos(y*z) - 0.5",
                "x^2 - 81*(y + 0.1)^2 + sin(z) + 1.06",
                "exp(-x*y) + 20*z + (10*pi - 3)/3",
            ],
            "variables": ["x", "y", "z"],
            "x0": [0.1, 0.1, -0.1],
        },
    )
    assert r3.status_code == 200
    payload = r3.json()
    json.dumps(payload, allow_nan=False)
    assert payload["status"] == "converged"
    assert payload["contour_2d"] is None
    assert all(
        all(k in step["lu_detail"] for k in ("P_latex", "L_latex", "U_latex", "y_latex", "dx_latex"))
        for step in payload["steps"]
    )





