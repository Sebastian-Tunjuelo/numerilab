import math
from collections.abc import Iterable
from copy import deepcopy
from typing import Literal
from .matrix_utils import (
    apply_finite_arithmetic,
    fmt_num,
    mat_vec_mul,
    matrix_to_latex,
    resolve_rounding_mode,
    round_matrix,
    round_vector,
    vec_diff_inf_norm,
    vec_inf_norm,
    vector_to_latex,
)

UNPIVOTED_PIVOT_TOL = 1e-14


def _subtract_products(value: float, products: Iterable[tuple[float, float]], decimals: int) -> str:
    terms = [fmt_num(value, decimals)]
    terms.extend(
        f"({fmt_num(left, decimals)})({fmt_num(right, decimals)})"
        for left, right in products
        if left != 0 and right != 0
    )
    return " - ".join(terms)


def _snapshot(
    step_index: int,
    title: str,
    description: str,
    latex_operation: str,
    L: list[list[float]],
    U: list[list[float]],
    P: list[list[float]],
    decimals: int,
    pivot_swap: list[int] | None = None,
) -> dict:
    return {
        "step": step_index,
        "title": title,
        "description": description,
        "latex_operation": latex_operation,
        "pivot_swap": pivot_swap,
        "L": round_matrix(L, decimals),
        "U": round_matrix(U, decimals),
        "P": round_matrix(P, 0),
        "L_latex": matrix_to_latex(L, decimals),
        "U_latex": matrix_to_latex(U, decimals),
        "P_latex": matrix_to_latex(P, 0),
    }


def _factor_payload(
    L: list[list[float]],
    U: list[list[float]],
    P: list[list[float]],
    decimals: int,
) -> dict:
    return {
        "L": round_matrix(L, decimals),
        "U": round_matrix(U, decimals),
        "P": round_matrix(P, 0),
        "L_latex": matrix_to_latex(L, decimals),
        "U_latex": matrix_to_latex(U, decimals),
        "P_latex": matrix_to_latex(P, 0),
    }


def solve_lu(
    A: list[list[float]],
    b: list[float],
    strategy: str = "partial_pivot",
    decimals: int = 6,
    apply_rounding: bool = False,
    rounding_mode: Literal["none", "round", "truncate"] = "none",
    *,
    _compute_reference: bool = True,
) -> dict:
    """Factorización LU paso a paso (Doolittle, Crout o Pivoteo Parcial PA=LU)."""
    n = len(A)
    mode = resolve_rounding_mode(apply_rounding, rounding_mode)
    A_work = [[float(val) for val in row] for row in A]
    b_work = [float(val) for val in b]

    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]
    P = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    steps: list[dict] = []
    step_counter = 1
    work_scales = [[abs(val) for val in row] for row in A_work]
    relative_pivot_tol = n * math.ulp(1.0)

    fact_mul_div = 0
    fact_add_sub = 0
    sub_mul_div = 0
    sub_add_sub = 0

    def build_op_count() -> dict:
        return {
            "factorization_mul_div": fact_mul_div,
            "factorization_add_sub": fact_add_sub,
            "substitution_mul_div": sub_mul_div,
            "substitution_add_sub": sub_add_sub,
            "mul_div": fact_mul_div + sub_mul_div,
            "add_sub": fact_add_sub + sub_add_sub,
            "total_flops": fact_mul_div + fact_add_sub + sub_mul_div + sub_add_sub,
            "theoretical_flops": round((2.0 / 3.0) * (n ** 3) + 2.0 * (n ** 2)),
            "complexity_latex": r"\mathcal{O}\!\left(\frac{2}{3}n^3 + 2n^2\right)",
        }

    def numeric_failure() -> dict:
        return {
            "strategy": strategy,
            "status": "diverged",
            "message": "La factorización o la sustitución excedió el rango numérico.",
            "rounding_mode": mode,
            "operation_count": build_op_count(),
            "steps": steps,
        }

    if strategy == "doolittle":
        for k in range(n):
            u_terms = []
            for j in range(k, n):
                s = sum(L[k][m] * U[m][j] for m in range(k))
                fact_mul_div += k
                fact_add_sub += k
                u_val = A_work[k][j] - s
                if not math.isfinite(u_val):
                    return numeric_failure()
                U[k][j] = apply_finite_arithmetic(u_val, decimals, mode)
                numerator = _subtract_products(
                    A_work[k][j], ((L[k][m], U[m][j]) for m in range(k)), decimals
                )
                u_terms.append(f"U_{{{k+1},{j+1}}} = {numerator} = {fmt_num(U[k][j], decimals)}")

            if abs(U[k][k]) < UNPIVOTED_PIVOT_TOL:
                if k == n - 1:
                    return {
                        "strategy": strategy,
                        "status": "singular_matrix",
                        "message": (
                            f"El último pivote U({k+1},{k+1}) tiene magnitud inferior a 10⁻¹⁴, "
                            "por lo que la matriz es singular o casi singular."
                        ),
                        "rounding_mode": mode,
                        "operation_count": build_op_count(),
                        "steps": steps,
                        **_factor_payload(L, U, P, decimals),
                    }
                return {
                    "strategy": strategy,
                    "status": "zero_pivot_requires_pivoting",
                    "message": (
                        f"El pivote U({k+1},{k+1}) tiene magnitud inferior a 10⁻¹⁴ en Doolittle sin pivoteo. "
                        "Activa la estrategia 'Pivoteo Parcial (PA = LU)' para intercambiar filas."
                    ),
                    "rounding_mode": mode,
                    "operation_count": build_op_count(),
                    "steps": steps,
                    **_factor_payload(L, U, P, decimals),
                }

            l_terms = []
            for i in range(k + 1, n):
                s = sum(L[i][m] * U[m][k] for m in range(k))
                fact_mul_div += k + 1
                fact_add_sub += k
                l_val = (A_work[i][k] - s) / U[k][k]
                if not math.isfinite(l_val):
                    return numeric_failure()
                L[i][k] = apply_finite_arithmetic(l_val, decimals, mode)
                numerator = _subtract_products(
                    A_work[i][k], ((L[i][m], U[m][k]) for m in range(k)), decimals
                )
                l_terms.append(
                    rf"L_{{{i+1},{k+1}}} = \frac{{{numerator}}}{{{fmt_num(U[k][k], decimals)}}}"
                    rf" = {fmt_num(L[i][k], decimals)}"
                )

            op_tex = r"\begin{gathered}" + r" \\ ".join(u_terms + l_terms) + r"\end{gathered}"
            steps.append(
                _snapshot(
                    step_counter,
                    f"Etapa k = {k + 1} (Doolittle)",
                    f"Cálculo de la fila {k + 1} de U y columna {k + 1} de L.",
                    op_tex,
                    L,
                    U,
                    P,
                    decimals,
                )
            )
            step_counter += 1

    elif strategy == "crout":
        U = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        L = [[0.0 for _ in range(n)] for _ in range(n)]
        for k in range(n):
            l_terms = []
            for i in range(k, n):
                s = sum(L[i][m] * U[m][k] for m in range(k))
                fact_mul_div += k
                fact_add_sub += k
                l_val = A_work[i][k] - s
                if not math.isfinite(l_val):
                    return numeric_failure()
                L[i][k] = apply_finite_arithmetic(l_val, decimals, mode)
                numerator = _subtract_products(
                    A_work[i][k], ((L[i][m], U[m][k]) for m in range(k)), decimals
                )
                l_terms.append(f"L_{{{i+1},{k+1}}} = {numerator} = {fmt_num(L[i][k], decimals)}")

            if abs(L[k][k]) < UNPIVOTED_PIVOT_TOL:
                if k == n - 1:
                    return {
                        "strategy": strategy,
                        "status": "singular_matrix",
                        "message": (
                            f"El último pivote L({k+1},{k+1}) tiene magnitud inferior a 10⁻¹⁴, "
                            "por lo que la matriz es singular o casi singular."
                        ),
                        "rounding_mode": mode,
                        "operation_count": build_op_count(),
                        "steps": steps,
                        **_factor_payload(L, U, P, decimals),
                    }
                return {
                    "strategy": strategy,
                    "status": "zero_pivot_requires_pivoting",
                    "message": (
                        f"El pivote L({k+1},{k+1}) tiene magnitud inferior a 10⁻¹⁴ en Crout sin pivoteo. "
                        "Activa 'Pivoteo Parcial (PA = LU)' para reordenar filas."
                    ),
                    "rounding_mode": mode,
                    "operation_count": build_op_count(),
                    "steps": steps,
                    **_factor_payload(L, U, P, decimals),
                }

            u_terms = []
            for j in range(k + 1, n):
                s = sum(L[k][m] * U[m][j] for m in range(k))
                fact_mul_div += k + 1
                fact_add_sub += k
                u_val = (A_work[k][j] - s) / L[k][k]
                if not math.isfinite(u_val):
                    return numeric_failure()
                U[k][j] = apply_finite_arithmetic(u_val, decimals, mode)
                numerator = _subtract_products(
                    A_work[k][j], ((L[k][m], U[m][j]) for m in range(k)), decimals
                )
                u_terms.append(
                    rf"U_{{{k+1},{j+1}}} = \frac{{{numerator}}}{{{fmt_num(L[k][k], decimals)}}}"
                    rf" = {fmt_num(U[k][j], decimals)}"
                )

            op_tex = r"\begin{gathered}" + r" \\ ".join(l_terms + u_terms) + r"\end{gathered}"
            steps.append(
                _snapshot(
                    step_counter,
                    f"Etapa k = {k + 1} (Crout)",
                    f"Cálculo de la columna {k + 1} de L y fila {k + 1} de U.",
                    op_tex,
                    L,
                    U,
                    P,
                    decimals,
                )
            )
            step_counter += 1

    else:  # partial_pivot (PA = LU)
        U = (
            [[apply_finite_arithmetic(val, decimals, mode) for val in row] for row in A_work]
            if mode != "none"
            else deepcopy(A_work)
        )
        for k in range(n - 1):
            pivot_row = max(range(k, n), key=lambda r: abs(U[r][k]))
            scale = work_scales[pivot_row][k]
            if scale == 0.0 or abs(U[pivot_row][k]) / scale <= relative_pivot_tol:
                return {
                    "strategy": strategy,
                    "status": "singular_matrix",
                    "message": f"La matriz es singular o casi singular en la columna {k + 1} (pivote máximo ≈ 0).",
                    "rounding_mode": mode,
                    "operation_count": build_op_count(),
                    "steps": steps,
                    **_factor_payload(L, U, P, decimals),
                }

            if pivot_row != k:
                U[k], U[pivot_row] = U[pivot_row], U[k]
                P[k], P[pivot_row] = P[pivot_row], P[k]
                work_scales[k], work_scales[pivot_row] = work_scales[pivot_row], work_scales[k]
                for col_l in range(k):
                    L[k][col_l], L[pivot_row][col_l] = L[pivot_row][col_l], L[k][col_l]

                steps.append(
                    _snapshot(
                        step_counter,
                        f"Pivoteo Parcial (columna {k + 1})",
                        f"Se intercambia la fila {k + 1} con la fila {pivot_row + 1} para maximizar el pivote.",
                        rf"F_{{{k+1}}} \leftrightarrow F_{{{pivot_row+1}}}",
                        L,
                        U,
                        P,
                        decimals,
                        pivot_swap=[k + 1, pivot_row + 1],
                    )
                )
                step_counter += 1

            elim_ops = []
            for i in range(k + 1, n):
                factor = U[i][k] / U[k][k]
                fact_mul_div += 1
                if not math.isfinite(factor):
                    return numeric_failure()
                factor = apply_finite_arithmetic(factor, decimals, mode)
                L[i][k] = factor
                U[i][k] = 0.0
                for j in range(k + 1, n):
                    u_val = U[i][j] - factor * U[k][j]
                    fact_mul_div += 1
                    fact_add_sub += 1
                    if not math.isfinite(u_val):
                        return numeric_failure()
                    U[i][j] = apply_finite_arithmetic(u_val, decimals, mode)
                    work_scales[i][j] = max(work_scales[i][j], abs(factor) * work_scales[k][j])
                elim_ops.append(
                    rf"m_{{{i+1},{k+1}}} = {fmt_num(factor, decimals)},\; F_{{{i+1}}} \leftarrow F_{{{i+1}}} - ({fmt_num(factor, decimals)}) F_{{{k+1}}}"
                )

            steps.append(
                _snapshot(
                    step_counter,
                    f"Eliminación bajo pivote U({k + 1},{k + 1})",
                    f"Cero debajo de la diagonal en la columna {k + 1}.",
                    r"\begin{gathered}" + r" \\ ".join(elim_ops) + r"\end{gathered}",
                    L,
                    U,
                    P,
                    decimals,
                )
            )
            step_counter += 1

        last_scale = work_scales[-1][-1]
        if last_scale == 0.0 or abs(U[-1][-1]) / last_scale <= relative_pivot_tol:
            return {
                "strategy": strategy,
                "status": "singular_matrix",
                "message": "El último pivote es nulo o demasiado pequeño respecto a su escala.",
                "rounding_mode": mode,
                "operation_count": build_op_count(),
                "steps": steps,
                **_factor_payload(L, U, P, decimals),
            }

    # Sustitución progresiva: L y = P b
    pb = mat_vec_mul(P, b_work)
    y = [0.0] * n
    forward_equations: list[str] = []
    for i in range(n):
        known_sum = sum(L[i][j] * y[j] for j in range(i))
        sub_mul_div += i + (0 if L[i][i] == 1.0 else 1)
        sub_add_sub += i
        y_val = (pb[i] - known_sum) / L[i][i]

        if not math.isfinite(y_val):
            return numeric_failure()
        y[i] = apply_finite_arithmetic(y_val, decimals, mode)
        rhs_expr = _subtract_products(pb[i], ((L[i][j], y[j]) for j in range(i)), decimals)
        if L[i][i] != 1.0:
            eq_tex = rf"y_{{{i+1}}} = \frac{{{rhs_expr}}}{{{fmt_num(L[i][i], decimals)}}} = {fmt_num(y[i], decimals)}"
        else:
            eq_tex = rf"y_{{{i+1}}} = {rhs_expr} = {fmt_num(y[i], decimals)}"
        forward_equations.append(eq_tex)

    # Sustitución regresiva: U x = y
    x = [0.0] * n
    backward_equations: list[str] = []
    for i in range(n - 1, -1, -1):
        if U[i][i] == 0.0:
            return {
                "strategy": strategy,
                "status": "singular_matrix",
                "message": f"Elemento diagonal U({i+1},{i+1}) = 0 en sustitución regresiva.",
                "rounding_mode": mode,
                "operation_count": build_op_count(),
                "steps": steps,
                **_factor_payload(L, U, P, decimals),
            }
        known_sum = sum(U[i][j] * x[j] for j in range(i + 1, n))
        terms_count = n - 1 - i
        sub_mul_div += terms_count + (0 if U[i][i] == 1.0 else 1)
        sub_add_sub += terms_count
        x_val = (y[i] - known_sum) / U[i][i]

        if not math.isfinite(x_val):
            return numeric_failure()
        x[i] = apply_finite_arithmetic(x_val, decimals, mode)
        rhs_expr = _subtract_products(y[i], ((U[i][j], x[j]) for j in range(i + 1, n)), decimals)
        if U[i][i] != 1.0:
            eq_tex = rf"x_{{{i+1}}} = \frac{{{rhs_expr}}}{{{fmt_num(U[i][i], decimals)}}} = {fmt_num(x[i], decimals)}"
        else:
            eq_tex = rf"x_{{{i+1}}} = {rhs_expr} = {fmt_num(x[i], decimals)}"
        backward_equations.append(eq_tex)

    Ax = mat_vec_mul(A_work, x)
    residual_vec = [Ax[i] - b_work[i] for i in range(n)]
    if not all(math.isfinite(value) for value in residual_vec):
        return numeric_failure()
    residual_norm = vec_inf_norm(residual_vec)

    finite_arithmetic_diff = None
    if mode != "none" and _compute_reference:
        ref = solve_lu(
            A,
            b,
            strategy=strategy,
            decimals=decimals,
            apply_rounding=False,
            rounding_mode="none",
            _compute_reference=False,
        )
        if ref["status"] == "converged":
            diff = vec_diff_inf_norm(x, ref["x_raw"])
            if math.isfinite(diff):
                finite_arithmetic_diff = diff

    return {
        "strategy": strategy,
        "status": "converged",
        "message": "Factorización y resolución completadas correctamente.",
        "rounding_mode": mode,
        "finite_arithmetic_diff": finite_arithmetic_diff,
        "operation_count": build_op_count(),
        **_factor_payload(L, U, P, decimals),
        "pb": round_vector(pb, decimals),
        "pb_latex": vector_to_latex(pb, decimals),
        "y": round_vector(y, decimals),
        "y_latex": vector_to_latex(y, decimals),
        "x": round_vector(x, decimals),
        "x_raw": x,
        "x_latex": vector_to_latex(x, decimals),
        "forward_equations": forward_equations,
        "backward_equations": backward_equations,
        "steps": steps,
        "residual_norm": residual_norm,
    }
