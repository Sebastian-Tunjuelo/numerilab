import math
from typing import Literal
from .matrix_utils import (
    apply_finite_vector,
    compute_spectral_radius,
    fmt_num,
    compute_sor_form,
    mat_vec_mul,
    resolve_rounding_mode,
    round_vector,
    vec_diff_inf_norm,
    vec_inf_norm,
    vector_to_latex,
)


_CRITERION_LABELS = {
    "abs_diff": "diferencia absoluta",
    "rel_diff": "error relativo",
    "rel_pct": "error relativo porcentual",
    "residual": "residual del sistema",
}


def _format_variable_step(
    method: Literal["jacobi", "gauss_seidel", "sor"],
    i: int,
    k: int,
    A: list[list[float]],
    b: list[float],
    x_source_before: list[float],
    x_source_after: list[float],
    x_final: float,
    omega: float,
    decimals: int,
) -> str:
    n = len(A)
    terms: list[str] = []
    for j in range(n):
        if j == i:
            continue
        val_x = x_source_before[j] if (method == "jacobi" or j > i) else x_source_after[j]
        coeff = A[i][j]
        if coeff == 0:
            continue
        terms.append(f"({fmt_num(coeff, decimals)})({fmt_num(val_x, decimals)})")

    numerator = fmt_num(b[i], decimals)
    if terms:
        numerator += " - " + " - ".join(terms)
    gs_expr = rf"\frac{{{numerator}}}{{{fmt_num(A[i][i], decimals)}}}"

    if method == "sor" and omega != 1.0:
        return (
            rf"x_{{{i+1}}}^{{({k})}} = (1 - {fmt_num(omega, decimals)})({fmt_num(x_source_before[i], decimals)}) "
            rf"+ {fmt_num(omega, decimals)} \left[{gs_expr}\right] = {fmt_num(x_final, decimals)}"
        )
    return rf"x_{{{i+1}}}^{{({k})}} = {gs_expr} = {fmt_num(x_final, decimals)}"


def solve_iterative_linear(
    A: list[list[float]],
    b: list[float],
    x0: list[float] | None,
    method: Literal["jacobi", "gauss_seidel", "sor"],
    omega: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 100,
    criterion: Literal["abs_diff", "rel_diff", "rel_pct", "residual"] = "abs_diff",
    decimals: int = 6,
    apply_rounding: bool = False,
    rounding_mode: Literal["none", "round", "truncate"] = "none",
    *,
    record_steps: bool = True,
) -> dict:
    """Ejecuta Jacobi, Gauss-Seidel (w=1) o SOR (0<w<2) con trazabilidad paso a paso."""
    n = len(A)
    effective_omega = 1.0 if method == "gauss_seidel" else float(omega)
    relaxed = method == "sor" and effective_omega != 1.0
    mode = resolve_rounding_mode(apply_rounding, rounding_mode)
    is_finite_arith = mode != "none"
    x_curr = [0.0] * n if x0 is None else [float(v) for v in x0]

    if relaxed:
        iter_mul_div = n * (n + 2)
        iter_add_sub = n * (n + 1)
    else:
        iter_mul_div = n * n
        iter_add_sub = n * (n - 1)

    for i in range(n):
        if A[i][i] == 0.0:
            return {
                "method": method,
                "omega": effective_omega,
                "status": "zero_diagonal",
                "message": (
                    f"El elemento diagonal a({i+1},{i+1}) = 0 impide despejar x({i+1}). "
                    "Reordena las filas del sistema o utiliza Factorización LU con pivoteo parcial."
                ),
                "rounding_mode": mode,
                "operation_count": {
                    "per_iteration_flops": iter_mul_div + iter_add_sub,
                    "mul_div": 0,
                    "add_sub": 0,
                    "total_flops": 0,
                    "complexity_latex": r"\mathcal{O}(k \cdot n^2)",
                },
                "iterations": 0,
                "steps": [],
                "solution": None,
                "solution_raw": None,
                "solution_latex": None,
                "residual_norm": None,
            }

    steps: list[dict] = []
    iterations = 0
    attempts = 0
    final_res = None
    status = "max_iterations"
    message = f"Se alcanzó el máximo de {max_iter} iteraciones sin cumplir la tolerancia {tol:.1e}."

    for k in range(1, max_iter + 1):
        attempts = k
        x_prev = list(x_curr)
        x_next = list(x_curr)
        variable_steps: list[str] = []

        for i in range(n):
            off_diagonal_products = (
                A[i][j] * (x_next[j] if method != "jacobi" and j < i else x_prev[j])
                for j in range(n)
                if j != i
            )
            # Inicia con el primer producto para sumar los n-1 términos con n-2 sumas.
            sigma = sum(off_diagonal_products, next(off_diagonal_products))
            x_val = (b[i] - sigma) / A[i][i]
            if relaxed:
                x_val = (1.0 - effective_omega) * x_prev[i] + effective_omega * x_val
            x_next[i] = x_val

        if any(not math.isfinite(val) for val in x_next):
            status = "diverged"
            message = (
                f"El iterado excedió el rango numérico en la iteración {k}. "
                "Verifica la dominancia diagonal o el radio espectral ρ(T) < 1."
            )
            break

        if record_steps:
            variable_steps = [
                _format_variable_step(
                    method, i, k, A, b, x_prev, x_next, value, effective_omega,
                    max(8, decimals + 2) if is_finite_arith else decimals,
                )
                for i, value in enumerate(x_next)
            ]
        x_unrounded = list(x_next)
        if is_finite_arith:
            rounded = apply_finite_vector(x_next, decimals, mode)
            if record_steps:
                op_label = "truncar" if mode == "truncate" else "redondear"
                for i, (before, after) in enumerate(zip(x_next, rounded)):
                    if before != after:
                        variable_steps[i] += (
                            rf"\quad \xrightarrow{{\operatorname{{{op_label}}}}} {fmt_num(after, decimals)}"
                        )
            x_next = rounded

        abs_err = vec_diff_inf_norm(x_next, x_prev)
        norm_next = vec_inf_norm(x_next)
        if abs_err == 0.0:
            rel_err = 0.0
        elif norm_next > 0.0:
            rel_err = abs_err / norm_next
        else:
            rel_err = None
        rel_err_pct = rel_err * 100.0 if rel_err is not None else None
        Ax = mat_vec_mul(A, x_next)
        residual = [Ax[i] - b[i] for i in range(n)]
        if not all(math.isfinite(val) for val in [abs_err, *residual]):
            status = "diverged"
            message = f"El error o el residual excedió el rango numérico en la iteración {k}."
            break
        if rel_err is not None and not math.isfinite(rel_err):
            rel_err = None
        if rel_err_pct is not None and not math.isfinite(rel_err_pct):
            rel_err_pct = None
        res_norm = vec_inf_norm(residual)

        iterations = k
        final_res = res_norm
        if record_steps:
            steps.append(
                {
                    "iteration": k,
                    "x": round_vector(x_next, decimals),
                    "abs_error": abs_err,
                    "rel_error": rel_err,
                    "rel_error_pct": rel_err_pct,
                    "residual": res_norm,
                    "variable_steps": variable_steps,
                }
            )

        x_curr = x_next

        if criterion == "abs_diff":
            metric = abs_err
        elif criterion == "rel_diff":
            metric = rel_err
        elif criterion == "rel_pct":
            metric = rel_err_pct
        else:
            metric = res_norm

        residual_tol = tol / 100.0 if criterion == "rel_pct" else tol
        if abs_err == 0.0:
            unrounded_stagnated = False
            if is_finite_arith and criterion != "residual":
                u_abs_err = vec_diff_inf_norm(x_unrounded, x_prev)
                u_norm = vec_inf_norm(x_unrounded)
                if u_abs_err == 0.0:
                    u_rel_err = 0.0
                elif u_norm > 0.0:
                    u_rel_err = u_abs_err / u_norm
                else:
                    u_rel_err = math.inf
                u_metric = (
                    u_abs_err
                    if criterion == "abs_diff"
                    else (u_rel_err * 100.0 if criterion == "rel_pct" else u_rel_err)
                )
                unrounded_stagnated = u_metric >= tol

            if res_norm >= residual_tol or unrounded_stagnated:
                status = "stagnated"
                if is_finite_arith:
                    op_name = "truncamiento" if mode == "truncate" else "redondeo"
                    if res_norm >= residual_tol:
                        message = f"El {op_name} dejó el vector sin cambios y el residual sigue fuera de tolerancia."
                    else:
                        message = f"El {op_name} dejó el vector sin cambios antes de cumplir el criterio con el paso sin aproximar."
                else:
                    message = "La precisión numérica dejó el vector sin cambios y el residual sigue por encima de la tolerancia."
                break

        if metric is not None and metric < tol:
            status = "converged"
            criterion_label = _CRITERION_LABELS.get(criterion, criterion)
            message = f"Convergió en {k} iteraciones bajo el criterio de {criterion_label}."
            break

    total_mul_div = attempts * iter_mul_div
    total_add_sub = attempts * iter_add_sub

    return {
        "method": method,
        "omega": effective_omega,
        "status": status,
        "message": message,
        "rounding_mode": mode,
        "operation_count": {
            "per_iteration_flops": iter_mul_div + iter_add_sub,
            "mul_div": total_mul_div,
            "add_sub": total_add_sub,
            "total_flops": total_mul_div + total_add_sub,
            "complexity_latex": r"\mathcal{O}(k \cdot n^2)",
        },
        "iterations": iterations,
        "steps": steps,
        "solution": round_vector(x_curr, decimals) if status != "diverged" else None,
        "solution_raw": x_curr if status != "diverged" else None,
        "solution_latex": vector_to_latex(x_curr, decimals) if status != "diverged" else None,
        "residual_norm": final_res,
    }


def sweep_sor_omega(
    A: list[list[float]],
    b: list[float],
    x0: list[float] | None,
    tol: float = 1e-6,
    max_iter: int = 100,
    criterion: Literal["abs_diff", "rel_diff", "rel_pct", "residual"] = "abs_diff",
    decimals: int = 6,
    apply_rounding: bool = False,
    rounding_mode: Literal["none", "round", "truncate"] = "none",
) -> dict:
    """Evalúa valores de ω ∈ [0.10, 1.95] para encontrar el parámetro óptimo empírico."""
    n = len(A)
    mode = resolve_rounding_mode(apply_rounding, rounding_mode)
    if any(A[i][i] == 0.0 for i in range(n)):
        return {
            "available": False,
            "reason": "Diagonal principal con ceros; no es posible evaluar el barrido SOR.",
            "points": [],
            "optimal_omega": None,
            "optimal_iterations": None,
            "optimal_spectral_radius": None,
        }

    points: list[dict] = []
    best_point: dict | None = None

    for idx in range(38):
        w = round(0.10 + 0.05 * idx, 2)
        res = solve_iterative_linear(
            A=A,
            b=b,
            x0=x0,
            method="sor",
            omega=w,
            tol=tol,
            max_iter=max_iter,
            criterion=criterion,
            decimals=decimals,
            apply_rounding=mode != "none",
            rounding_mode=mode,
            record_steps=False,
        )

        T_w, _ = compute_sor_form(A, b, w)
        rho = compute_spectral_radius(T_w)

        converged = res["status"] == "converged"
        pt = {
            "omega": w,
            "converged": converged,
            "iterations": res["iterations"],
            "spectral_radius": rho,
        }
        points.append(pt)

        if converged:
            if best_point is None:
                best_point = pt
            elif pt["iterations"] < best_point["iterations"]:
                best_point = pt
            elif pt["iterations"] == best_point["iterations"]:
                pt_rho = pt["spectral_radius"] if pt["spectral_radius"] is not None else math.inf
                best_rho = best_point["spectral_radius"] if best_point["spectral_radius"] is not None else math.inf
                if pt_rho < best_rho:
                    best_point = pt

    return {
        "available": True,
        "points": points,
        "optimal_omega": best_point["omega"] if best_point else None,
        "optimal_iterations": best_point["iterations"] if best_point else None,
        "optimal_spectral_radius": best_point["spectral_radius"] if best_point else None,
    }
