import math
import sympy as sp
from .lu import solve_lu
from .math_expression import parse_math_expression
from .matrix_utils import (
    apply_finite_arithmetic,
    apply_finite_vector,
    fmt_num,
    matrix_to_latex,
    resolve_rounding_mode,
    round_matrix,
    round_vector,
    vec_diff_inf_norm,
    vec_inf_norm,
    vector_to_latex,
)


PLOT_LIMIT = 1e15
_SUB_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUP_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _sub(n: int) -> str:
    return str(n).translate(_SUB_DIGITS)


def _sup(n: int) -> str:
    return str(n).translate(_SUP_DIGITS)


def _eval_scalar(fn, args: list[float]) -> float:
    val = fn(*args)
    if isinstance(val, complex):
        if val.imag != 0.0:
            raise ValueError("La evaluación produjo un número complejo fuera del dominio real.")
        val = val.real
    val_f = float(val)
    if not math.isfinite(val_f):
        raise OverflowError("La evaluación produjo un valor no finito.")
    return val_f


def _safe_eval_optional(fn, args: list[float]) -> float | None:
    if fn is None:
        return None
    try:
        val = _eval_scalar(fn, args)
        return val if math.isfinite(val) else None
    except Exception:
        return None


def _small_correction(size: float, x_norm: float, tol: float) -> bool:
    return size <= max(tol, 4 * math.ulp(x_norm))


def _compute_rel_errors(abs_err: float, x_next_norm: float) -> tuple[float | None, float | None, bool]:
    """Calcula (rel_error, rel_error_pct, is_finite) evitando división por cero."""
    if not math.isfinite(abs_err):
        return None, None, False
    if abs_err == 0.0:
        return 0.0, 0.0, True
    if x_next_norm <= 0.0:
        return None, None, True
    rel_err = abs_err / x_next_norm
    if not math.isfinite(rel_err):
        return None, None, False
    rel_err_pct = rel_err * 100.0
    if not math.isfinite(rel_err_pct):
        return None, None, False
    return rel_err, rel_err_pct, True


def _format_eval_error(exc: Exception) -> str:
    msg = str(exc)
    lower = msg.lower()
    if isinstance(exc, ZeroDivisionError) or "division by zero" in lower:
        return "división por cero."
    if "math domain error" in lower:
        return "valor fuera del dominio real de la función."
    if "math range error" in lower:
        return "desbordamiento numérico (valor fuera de rango)."
    if msg.startswith(("La ", "El ")):
        return msg
    if isinstance(exc, OverflowError):
        return "desbordamiento numérico (valor fuera de rango)."
    return "error numérico durante la evaluación."


def _zero_step_converged(
    criterion: str,
    residual_norm: float,
    correction_norm: float,
    x_next_norm: float,
    u_abs_norm: float,
    u_next_norm: float,
    rel_err: float | None,
    rel_err_pct: float | None,
    tol: float,
    mode: str,
) -> bool:
    res_tol = (tol / 100.0) if criterion == "rel_pct" else tol
    if residual_norm > res_tol:
        return False
    if mode != "none":
        u_rel, u_rel_pct, u_fin = _compute_rel_errors(u_abs_norm, u_next_norm)
        if not u_fin or not _criterion_satisfied(
            criterion, residual_norm, correction_norm, u_next_norm, u_rel, u_rel_pct, tol
        ):
            return False
    return _criterion_satisfied(
        criterion, residual_norm, correction_norm, x_next_norm, rel_err, rel_err_pct, tol
    )


def _criterion_satisfied(
    criterion: str,
    res_norm: float,
    correction_norm: float,
    x_norm: float,
    rel_err: float | None,
    rel_err_pct: float | None,
    tol: float,
) -> bool:
    if criterion == "rel_diff":
        return rel_err is not None and rel_err <= tol
    if criterion == "rel_pct":
        return rel_err_pct is not None and rel_err_pct <= tol
    return res_norm <= tol and _small_correction(correction_norm, x_norm, tol)


def _last_order_triple(steps: list[dict]) -> tuple[float, float, float, float] | None:
    """Devuelve (orden, d0, d1, d2) para las últimas 3 iteraciones consecutivas válidas."""
    for idx in range(len(steps) - 3, -1, -1):
        d0 = steps[idx]["abs_error"]
        d1 = steps[idx + 1]["abs_error"]
        d2 = steps[idx + 2]["abs_error"]
        if not (d0 > 0.0 and d1 > 0.0 and d2 > 0.0):
            continue
        if not (math.isfinite(d0) and math.isfinite(d1) and math.isfinite(d2)):
            continue
        r1 = d1 / d0
        r2 = d2 / d1
        if not (math.isfinite(r1) and math.isfinite(r2)) or r1 <= 0.0 or r2 <= 0.0 or r1 == 1.0:
            continue
        denom = math.log(r1)
        if abs(denom) <= 1e-15:
            continue
        order = math.log(r2) / denom
        if math.isfinite(order):
            return round(order, 4), d0, d1, d2
    return None


def _estimate_convergence_order(steps: list[dict]) -> float | None:
    """Estima p ≈ ln(|dx_{k+1}/dx_k|) / ln(|dx_k/dx_{k-1}|) con las últimas 3 iteraciones válidas."""
    triple = _last_order_triple(steps)
    return triple[0] if triple is not None else None


def _is_sympy_evalf_zero(value) -> bool:
    if value == 0:
        return True
    if isinstance(value, sp.Float):
        # evalf devuelve Float('0.e-...') con _prec <= 1 cuando la cancelación anula toda la mantisa.
        return value._prec <= 1 or value._mpf_[1] == 0
    return False


def _eval_residual(fn, expression, symbols, args: list[float]) -> tuple[float, bool]:
    value = _eval_scalar(fn, args)
    if value == 0.0:
        precise = expression.evalf(32, subs=dict(zip(symbols, args)))
        if not _is_sympy_evalf_zero(precise):
            value = float(precise)
            if not math.isfinite(value):
                raise OverflowError("El residual excedió el rango numérico.")
            if value == 0.0:
                return 0.0, True
    return value, False


def _verify_underflow_1d(f_expr, df_expr, x_sym, x_val: float, tol: float) -> None:
    subs = {x_sym: sp.Float(x_val)}
    f_p = f_expr.evalf(32, subs=subs)
    df_p = df_expr.evalf(32, subs=subs)
    if df_p == 0:
        raise OverflowError("El residual es demasiado pequeño para representarlo en coma flotante.")
    try:
        step_f = float(abs(f_p / df_p))
    except OverflowError as exc:
        raise OverflowError("El residual es demasiado pequeño para representarlo en coma flotante.") from exc
    if not math.isfinite(step_f) or not _small_correction(step_f, abs(x_val), tol):
        raise OverflowError("El residual es demasiado pequeño para representarlo en coma flotante.")


def _verify_underflow_system(
    f_exprs,
    J_exprs,
    symbols,
    x_val: list[float],
    tol: float,
    decimals: int,
) -> None:
    n = len(x_val)
    subs = dict(zip(symbols, [sp.Float(v) for v in x_val]))
    f_p = [expr.evalf(32, subs=subs) for expr in f_exprs]
    j_p = [[expr.evalf(32, subs=subs) for expr in row] for row in J_exprs]

    j_scaled: list[list[float]] = []
    neg_f_scaled: list[float] = []
    for i in range(n):
        row_max = max([abs(f_p[i]), *(abs(j_p[i][j]) for j in range(n))])
        if row_max == 0:
            raise OverflowError("El residual es demasiado pequeño para representarlo en coma flotante.")
        inv_scale = sp.Integer(1) / row_max
        try:
            neg_f_scaled.append(float(-f_p[i] * inv_scale))
            j_scaled.append([float(j_p[i][j] * inv_scale) for j in range(n)])
        except (TypeError, OverflowError) as exc:
            raise OverflowError("El residual es demasiado pequeño para representarlo en coma flotante.") from exc
        if not math.isfinite(neg_f_scaled[-1]) or any(not math.isfinite(val) for val in j_scaled[-1]):
            raise OverflowError("El residual es demasiado pequeño para representarlo en coma flotante.")

    lu_res = solve_lu(j_scaled, neg_f_scaled, strategy="partial_pivot", decimals=decimals)
    if lu_res["status"] != "converged" or not _small_correction(
        vec_inf_norm(lu_res["x_raw"]), vec_inf_norm(x_val), tol
    ):
        raise OverflowError("El residual es demasiado pequeño para representarlo en coma flotante.")


def solve_newton_1d(
    expression: str,
    x0: float,
    tol: float = 1e-7,
    max_iter: int = 50,
    decimals: int = 6,
    criterion: str = "residual_and_step",
    apply_rounding: bool = False,
    rounding_mode: str = "none",
) -> dict:
    """Newton-Raphson escalar f(x) = 0 con derivada simbólica y geometría de tangentes."""
    mode = resolve_rounding_mode(apply_rounding, rounding_mode)
    try:
        f_expr, [x_sym] = parse_math_expression(expression, ["x"])
        df_expr = sp.diff(f_expr, x_sym)
        d2f_expr = sp.diff(df_expr, x_sym)
        f_fn = sp.lambdify(x_sym, f_expr, modules=["math"], dummify=True)
        df_fn = sp.lambdify(x_sym, df_expr, modules=["math"], dummify=True)
        try:
            d2f_fn = sp.lambdify(x_sym, d2f_expr, modules=["math"], dummify=True)
        except Exception:
            d2f_fn = None
        f_latex = sp.latex(f_expr)
        df_latex = sp.latex(df_expr)
        d2f_latex = sp.latex(d2f_expr)
    except Exception as exc:
        return {
            "status": "parse_error",
            "criterion": criterion,
            "message": f"No se pudo interpretar la expresión '{expression}': {exc}",
            "f_latex": "",
            "df_latex": "",
            "d2f_latex": "",
            "fourier_condition": None,
            "convergence_warnings": [],
            "rounding_mode": mode,
            "steps": [],
            "root": None,
            "iterations": 0,
            "estimated_order": None,
        }

    x0_val = float(x0)
    fx_0 = _safe_eval_optional(f_fn, [x0_val])
    d2fx_0 = _safe_eval_optional(d2f_fn, [x0_val])
    fourier_prod = (
        fx_0 * d2fx_0
        if (fx_0 is not None and d2fx_0 is not None and math.isfinite(fx_0 * d2fx_0))
        else None
    )
    fourier_condition = {
        "x0": round(x0_val, decimals),
        "fx_0": round(fx_0, decimals) if fx_0 is not None else None,
        "d2fx_0": round(d2fx_0, decimals) if d2fx_0 is not None else None,
        "product": round(fourier_prod, decimals) if fourier_prod is not None else None,
        "satisfied": bool(fourier_prod is not None and fourier_prod > 0.0),
    }

    steps: list[dict] = []
    x_curr = x0_val
    status = "max_iterations"
    message = f"No se alcanzó tolerancia {tol:.1e} en {max_iter} iteraciones."

    for k in range(1, max_iter + 1):
        try:
            fx, fx_underflowed = _eval_residual(f_fn, f_expr, [x_sym], [x_curr])
            if fx == 0.0:
                if fx_underflowed:
                    _verify_underflow_1d(f_expr, df_expr, x_sym, x_curr, tol)
                status = "converged"
                message = "La función vale cero en el punto actual."
                break
            dfx = _eval_scalar(df_fn, [x_curr])
        except Exception as exc:
            status = "diverged" if isinstance(exc, OverflowError) else "domain_error"
            message = f"Error de evaluación en iteración {k} (x = {fmt_num(x_curr, decimals)}): {_format_eval_error(exc)}"
            break

        d2fx_k = _safe_eval_optional(d2f_fn, [x_curr])

        correction = fx / dfx if dfx != 0 else math.inf
        if criterion == "residual_and_step" and abs(fx) <= tol and _small_correction(abs(correction), abs(x_curr), tol):
            status = "converged"
            message = "El punto actual satisface las tolerancias del residual y de la corrección."
            break

        if abs(dfx) < 1e-14:
            status = "zero_derivative"
            message = (
                f"Derivada nula f'(x{_sub(k - 1)}) ≈ 0 en x = {fmt_num(x_curr, decimals)}. "
                "La recta tangente es horizontal; cambia el punto inicial x₀."
            )
            break

        x_next_calc = x_curr - correction
        if not math.isfinite(x_next_calc):
            status = "diverged"
            message = f"El iterado excedió el rango numérico en la iteración {k}."
            break

        x_next = apply_finite_arithmetic(x_next_calc, decimals, mode) if mode != "none" else x_next_calc
        if not math.isfinite(x_next):
            status = "diverged"
            message = f"El iterado excedió el rango numérico en la iteración {k}."
            break

        abs_err = abs(x_next - x_curr)
        rel_err, rel_err_pct, err_finite = _compute_rel_errors(abs_err, abs(x_next))
        if not err_finite:
            status = "diverged"
            message = "El error relativo excedió el rango numérico."
            break

        evaluation_error = None
        try:
            fx_next, fx_next_underflowed = _eval_residual(f_fn, f_expr, [x_sym], [x_next])
            if fx_next == 0.0 and fx_next_underflowed:
                _verify_underflow_1d(f_expr, df_expr, x_sym, x_next, tol)
        except Exception as exc:
            evaluation_error = exc
            fx_next = math.inf

        if mode != "none" and x_next != x_next_calc:
            op_label = "truncar" if mode == "truncate" else "redondear"
            step_latex = (
                rf"x_{{{k}}} = {fmt_num(x_curr, decimals)} - \frac{{{fmt_num(fx, decimals)}}}{{{fmt_num(dfx, decimals)}}} "
                rf"= {fmt_num(x_next_calc, max(8, decimals + 2))}"
                rf"\quad \xrightarrow{{\operatorname{{{op_label}}}}} {fmt_num(x_next, decimals)}"
            )
        else:
            step_latex = (
                rf"x_{{{k}}} = {fmt_num(x_curr, decimals)} - \frac{{{fmt_num(fx, decimals)}}}{{{fmt_num(dfx, decimals)}}} "
                rf"= {fmt_num(x_next, decimals)}"
            )

        steps.append(
            {
                "iteration": k,
                "x_k": round(x_curr, decimals),
                "fx_k": fx,
                "dfx_k": dfx,
                "d2fx_k": d2fx_k,
                "fx_next": fx_next if math.isfinite(fx_next) else None,
                "x_next": round(x_next, decimals),
                "x_next_raw": x_next,
                "x_k_raw": x_curr,
                "abs_error": abs_err,
                "rel_error": rel_err,
                "rel_error_pct": rel_err_pct,
                "step_latex": step_latex,
            }
        )

        x_curr = x_next
        if evaluation_error is not None:
            status = "diverged" if isinstance(evaluation_error, OverflowError) else "domain_error"
            message = f"No se pudo evaluar el nuevo punto de la iteración {k}: {_format_eval_error(evaluation_error)}"
            break
        if fx_next == 0.0:
            status = "converged"
            message = f"Raíz aproximada en {k} iteraciones."
            break
        if abs_err == 0.0:
            if _zero_step_converged(
                criterion,
                abs(fx_next),
                abs(correction),
                abs(x_next),
                abs(x_next_calc - x_curr),
                abs(x_next_calc),
                rel_err,
                rel_err_pct,
                tol,
                mode,
            ):
                status = "converged"
                message = f"Raíz aproximada en {k} iteraciones."
            else:
                status = "stagnated"
                if mode != "none":
                    op_name = "truncamiento" if mode == "truncate" else "redondeo"
                    message = f"El {op_name} dejó el punto sin cambios sin cumplir la tolerancia del residual."
                else:
                    message = "El punto dejó de cambiar sin cumplir la tolerancia del residual."
            break
        if _criterion_satisfied(
            criterion, abs(fx_next), abs(correction), abs(x_next), rel_err, rel_err_pct, tol
        ):
            status = "converged"
            message = f"Raíz aproximada en {k} iteraciones."
            break

    order_info = _last_order_triple(steps)
    estimated_order = order_info[0] if order_info is not None else None
    convergence_warnings: list[str] = []
    if fourier_prod is not None and fourier_prod <= 0.0 and fx_0 is not None and abs(fx_0) > tol:
        convergence_warnings.append(
            "Condición de Fourier f(x₀)·f''(x₀) ≤ 0: no se garantiza convergencia monótona inicial desde x₀."
        )
    if not bool(d2f_expr.is_zero) and any(
        s["d2fx_k"] is not None and abs(s["d2fx_k"]) < 1e-8 for s in steps
    ):
        convergence_warnings.append(
            "Se detectó |f''(xₖ)| < 10⁻⁸ en la trayectoria (proximidad a un punto de inflexión)."
        )
    deltas = [s["x_next_raw"] - s["x_k_raw"] for s in steps]
    sign_flips = sum(1 for i in range(len(deltas) - 1) if deltas[i] * deltas[i + 1] < 0.0)
    if sign_flips >= 2:
        convergence_warnings.append(
            "Se observa alternancia de signo en los pasos Δxₖ (oscilación alrededor de la raíz o de un extremo local)."
        )
    has_contraction = status == "converged" or (
        status not in ("diverged", "zero_derivative", "domain_error")
        and order_info is not None
        and order_info[3] < order_info[2] < order_info[1]
    )
    if estimated_order is not None and estimated_order < 1.3 and has_contraction:
        convergence_warnings.append(
            f"Orden empírico estimado p ≈ {estimated_order:.2f} < 1.3: convergencia lineal característica de una raíz múltiple (m > 1)."
        )

    # Calcular ventana de gráfica inteligente cubriendo x0 y los pasos acotados
    visited_x = [x for x in [float(x0), *(s["x_next_raw"] for s in steps)] if abs(x) <= PLOT_LIMIT] or [0.0]
    min_x = min(visited_x)
    max_x = max(visited_x)
    span = max(max_x - min_x, 4.0)
    pad = span * 0.45
    plot_x_min = min_x - pad
    plot_x_max = max_x + pad

    sample_points: list[dict] = []
    num_samples = 160
    for i in range(num_samples):
        xv = plot_x_min + (plot_x_max - plot_x_min) * i / (num_samples - 1)
        try:
            yv = _eval_scalar(f_fn, [xv])
            if abs(yv) <= 1e4:
                sample_points.append({"x": round(xv, 5), "y": round(yv, 5)})
            else:
                sample_points.append({"x": round(xv, 5), "y": None})
        except Exception:
            sample_points.append({"x": round(xv, 5), "y": None})

    finite_y = [pt["y"] for pt in sample_points if pt["y"] is not None]
    finite_y.extend(s["fx_k"] for s in steps if abs(s["fx_k"]) <= PLOT_LIMIT)
    y_min = min([0.0, *finite_y]) if finite_y else -5.0
    y_max = max([0.0, *finite_y]) if finite_y else 5.0
    y_span = max(y_max - y_min, 2.0)

    return {
        "status": status,
        "criterion": criterion,
        "message": message,
        "expression_normalized": str(f_expr),
        "f_latex": f_latex,
        "df_latex": df_latex,
        "d2f_latex": d2f_latex,
        "fourier_condition": fourier_condition,
        "convergence_warnings": convergence_warnings,
        "rounding_mode": mode,
        "root": round(x_curr, decimals) if status == "converged" else None,
        "root_raw": x_curr if status == "converged" else None,
        "iterations": len(steps),
        "estimated_order": estimated_order,
        "steps": steps,
        "plot_bounds": {
            "x_min": round(plot_x_min, 4),
            "x_max": round(plot_x_max, 4),
            "y_min": round(y_min - y_span * 0.15, 4),
            "y_max": round(y_max + y_span * 0.15, 4),
        },
        "sample_points": sample_points,
    }


def _extract_zero_contours(
    xs: list[float],
    ys: list[float],
    grid: list[list[float | None]],
    fn,
) -> list[list[list[float]]]:
    """Extrae segmentos de curva implícita f(x, y) = 0 mediante Marching Squares."""
    segments: list[list[list[float]]] = []
    ny = len(ys)
    nx = len(xs)

    def crossing(p1, p2, v1: float, v2: float) -> list[float] | None:
        if v1 == 0.0:
            return list(p1)
        if v2 == 0.0:
            return list(p2)
        reference = min(abs(v1), abs(v2))
        left, right = 0.0, 1.0
        f_left, f_right = v1, v2
        # Un polo también cambia de signo, pero no reduce el residual hacia cero.
        for step_idx in range(32):
            if step_idx % 2 == 0:
                scale = max(abs(f_left), abs(f_right))
                w_left = abs(f_left) / scale
                w_right = abs(f_right) / scale
                denom = w_left + w_right
                t = left + (right - left) * (w_left / denom) if denom > 0.0 else (left + right) / 2.0
                if not (left < t < right):
                    t = (left + right) / 2.0
            else:
                t = (left + right) / 2.0
            point = [start + t * (end - start) for start, end in zip(p1, p2)]
            try:
                value = _eval_scalar(fn, point)
            except (ArithmeticError, ValueError, TypeError):
                return None
            if abs(value) <= 1e-5 * reference or ((right - left) <= 1e-13 and abs(value) < reference):
                return point
            if (value < 0) == (f_left < 0):
                left, f_left = t, value
            else:
                right, f_right = t, value
        return None

    # Tabla estándar de aristas por caso (0..15): aristas 0=bottom, 1=right, 2=top, 3=left
    edge_table = {
        0: [],
        1: [(3, 0)],
        2: [(0, 1)],
        3: [(3, 1)],
        4: [(1, 2)],
        5: [(3, 2), (0, 1)],
        6: [(0, 2)],
        7: [(3, 2)],
        8: [(2, 3)],
        9: [(0, 2)],
        10: [(0, 3), (1, 2)],
        11: [(1, 2)],
        12: [(1, 3)],
        13: [(0, 1)],
        14: [(3, 0)],
        15: [],
    }

    for j in range(ny - 1):
        for i in range(nx - 1):
            v0 = grid[j][i]         # bottom-left
            v1 = grid[j][i + 1]     # bottom-right
            v2 = grid[j + 1][i + 1] # top-right
            v3 = grid[j + 1][i]     # top-left
            if v0 is None or v1 is None or v2 is None or v3 is None:
                continue

            mask = (1 if v0 >= 0 else 0) | (2 if v1 >= 0 else 0) | (4 if v2 >= 0 else 0) | (8 if v3 >= 0 else 0)
            pairs = edge_table[mask]
            if not pairs:
                continue

            p0 = (xs[i], ys[j])
            p1 = (xs[i + 1], ys[j])
            p2 = (xs[i + 1], ys[j + 1])
            p3 = (xs[i], ys[j + 1])

            def edge_point(edge_idx: int) -> list[float] | None:
                if edge_idx == 0:
                    return crossing(p0, p1, v0, v1)
                if edge_idx == 1:
                    return crossing(p1, p2, v1, v2)
                if edge_idx == 2:
                    return crossing(p3, p2, v3, v2)
                return crossing(p0, p3, v0, v3)

            for e_a, e_b in pairs:
                start, end = edge_point(e_a), edge_point(e_b)
                if start is not None and end is not None and start != end:
                    segments.append([start, end])

    return segments


def _contour_function(expr, symbols, fn):
    base = expr
    # Las potencias enteras positivas conservan los ceros y ocultan el signo si son pares.
    while isinstance(base, sp.Pow) and base.exp.is_Integer and base.exp > 0:
        base = base.base
    return fn if base is expr else sp.lambdify(symbols, base, modules=["math"], dummify=True)


def solve_newton_system(
    equations: list[str],
    variables: list[str],
    x0: list[float],
    tol: float = 1e-7,
    max_iter: int = 50,
    decimals: int = 6,
    bounds: dict[str, float] | None = None,
    criterion: str = "residual_and_step",
    apply_rounding: bool = False,
    rounding_mode: str = "none",
) -> dict:
    """Newton-Raphson multivariable F(x)=0 resolviendo J(x_k) dx = -F(x_k) con LU propio."""
    n = len(equations)
    mode = resolve_rounding_mode(apply_rounding, rounding_mode)
    try:
        parsed = [parse_math_expression(eq, variables) for eq in equations]
        symbols = parsed[0][1]
        f_exprs = [expr for expr, _ in parsed]
        J_exprs = [[sp.diff(expr, symbol) for symbol in symbols] for expr in f_exprs]
        f_fns = [sp.lambdify(symbols, expr, modules=["math"], dummify=True) for expr in f_exprs]
        if n == 2:
            contour_fns = [_contour_function(expr, symbols, fn) for expr, fn in zip(f_exprs, f_fns)]
        J_fns = [
            [sp.lambdify(symbols, expr, modules=["math"], dummify=True) for expr in row]
            for row in J_exprs
        ]
        f_latex_list = [sp.latex(f) + " = 0" for f in f_exprs]
        j_rows_tex = [" & ".join(sp.latex(expr) for expr in row) for row in J_exprs]
        jacobian_latex = r"\begin{bmatrix} " + r" \\ ".join(j_rows_tex) + r" \end{bmatrix}"
    except Exception as exc:
        return {
            "status": "parse_error",
            "criterion": criterion,
            "message": f"Error interpretando el sistema de ecuaciones: {exc}",
            "variables": variables,
            "f_latex_list": [],
            "jacobian_latex": "",
            "estimated_order": None,
            "rounding_mode": mode,
            "steps": [],
            "solution": None,
            "solution_raw": None,
            "iterations": 0,
        }

    steps: list[dict] = []
    x_curr = [float(v) for v in x0]
    status = "max_iterations"
    message = f"No se alcanzó la tolerancia {tol:.1e} en {max_iter} iteraciones."

    for k in range(1, max_iter + 1):
        try:
            evals = [_eval_residual(fn, expr, symbols, x_curr) for fn, expr in zip(f_fns, f_exprs)]
            F_val = [val for val, _ in evals]
            f_norm = vec_inf_norm(F_val)
            if f_norm == 0.0:
                if any(flag for _, flag in evals):
                    _verify_underflow_system(f_exprs, J_exprs, symbols, x_curr, tol, decimals)
                status = "converged"
                message = "El vector actual anula el sistema."
                break
            J_val = [[_eval_scalar(J_fns[i][j], x_curr) for j in range(n)] for i in range(n)]
        except Exception as exc:
            status = "diverged" if isinstance(exc, OverflowError) else "domain_error"
            message = f"Error evaluando F o J en la iteración {k}: {_format_eval_error(exc)}"
            break

        neg_F = [-v for v in F_val]
        # Puente directo S.L. <-> N.L.: resolver J(x_k) * dx = -F(x_k) con nuestra factorización LU
        lu_res = solve_lu(
            J_val,
            neg_F,
            strategy="partial_pivot",
            decimals=decimals,
            apply_rounding=(mode != "none"),
            rounding_mode=mode,
        )
        if lu_res["status"] == "diverged":
            status = "diverged"
            message = f"El paso lineal de la iteración {k} excedió el rango numérico."
            break
        if lu_res["status"] != "converged":
            status = "singular_jacobian"
            message = (
                f"El Jacobiano J(x⁽{_sup(k - 1)}⁾) es singular o no invertible en la iteración {k}. "
                "Prueba otro vector inicial x₀."
            )
            break

        dx = lu_res["x_raw"]
        correction_norm = vec_inf_norm(dx)
        if criterion == "residual_and_step" and f_norm <= tol and _small_correction(
            correction_norm, vec_inf_norm(x_curr), tol
        ):
            status = "converged"
            message = "El vector actual satisface las tolerancias del residual y de la corrección."
            break
        x_next_calc = [x_curr[i] + dx[i] for i in range(n)]

        if any(not math.isfinite(val) for val in x_next_calc):
            status = "diverged"
            message = f"El iterado excedió el rango numérico en la iteración {k}."
            break

        x_next = apply_finite_vector(x_next_calc, decimals, mode) if mode != "none" else x_next_calc
        if mode != "none" and any(not math.isfinite(val) for val in x_next):
            status = "diverged"
            message = f"El iterado excedió el rango numérico en la iteración {k}."
            break

        abs_err = vec_diff_inf_norm(x_next, x_curr)
        x_next_norm = vec_inf_norm(x_next)
        rel_err, rel_err_pct, err_finite = _compute_rel_errors(abs_err, x_next_norm)
        if not math.isfinite(abs_err) or not err_finite:
            status = "diverged"
            message = "El error relativo excedió el rango numérico."
            break
        evaluation_error = None
        try:
            next_evals = [_eval_residual(fn, expr, symbols, x_next) for fn, expr in zip(f_fns, f_exprs)]
            F_next = [val for val, _ in next_evals]
            res_norm = vec_inf_norm(F_next)
            if res_norm == 0.0 and any(flag for _, flag in next_evals):
                _verify_underflow_system(f_exprs, J_exprs, symbols, x_next, tol, decimals)
        except Exception as exc:
            evaluation_error = exc
            res_norm = math.inf

        steps.append(
            {
                "iteration": k,
                "x_k": round_vector(x_curr, decimals),
                "x_next": round_vector(x_next, decimals),
                "x_next_raw": x_next,
                "F_eval": round_vector(F_val, decimals),
                "J_eval": round_matrix(J_val, decimals),
                "J_eval_latex": matrix_to_latex(J_val, decimals),
                "neg_F_latex": vector_to_latex(neg_F, decimals),
                "dx": round_vector(dx, decimals),
                "abs_error": abs_err,
                "rel_error": rel_err,
                "rel_error_pct": rel_err_pct,
                "residual": res_norm if math.isfinite(res_norm) else None,
                "lu_detail": {
                    "P_latex": lu_res["P_latex"],
                    "L_latex": lu_res["L_latex"],
                    "U_latex": lu_res["U_latex"],
                    "y_latex": lu_res["y_latex"],
                    "dx_latex": lu_res["x_latex"],
                },
            }
        )

        x_curr = x_next
        if evaluation_error is not None:
            status = "diverged" if isinstance(evaluation_error, OverflowError) else "domain_error"
            message = f"No se pudo evaluar el nuevo vector de la iteración {k}: {_format_eval_error(evaluation_error)}"
            break
        if res_norm == 0.0:
            status = "converged"
            message = f"Sistema resuelto en {k} iteraciones (J·Δx = -F resuelto con PA = LU)."
            break
        if abs_err == 0.0:
            if _zero_step_converged(
                criterion,
                res_norm,
                correction_norm,
                x_next_norm,
                correction_norm,
                vec_inf_norm(x_next_calc),
                rel_err,
                rel_err_pct,
                tol,
                mode,
            ):
                status = "converged"
                message = f"Sistema resuelto en {k} iteraciones (J·Δx = -F resuelto con PA = LU)."
            else:
                status = "stagnated"
                if mode != "none":
                    op_name = "truncamiento" if mode == "truncate" else "redondeo"
                    message = f"El {op_name} dejó el vector sin cambios sin cumplir la tolerancia del residual."
                else:
                    message = "El vector dejó de cambiar sin cumplir la tolerancia del residual."
            break
        if _criterion_satisfied(
            criterion, res_norm, correction_norm, x_next_norm, rel_err, rel_err_pct, tol
        ):
            status = "converged"
            message = f"Sistema resuelto en {k} iteraciones (J·Δx = -F resuelto con PA = LU)."
            break

    contour_2d = None
    if n == 2 and vec_inf_norm(x0) <= PLOT_LIMIT:
        if bounds and all(k in bounds for k in ("x_min", "x_max", "y_min", "y_max")):
            x_min = float(bounds["x_min"])
            x_max = float(bounds["x_max"])
            y_min = float(bounds["y_min"])
            y_max = float(bounds["y_max"])
        else:
            points = [x0, *(s["x_next_raw"] for s in steps if vec_inf_norm(s["x_next_raw"]) <= PLOT_LIMIT)]
            pts_x, pts_y = zip(*points)
            cx = (min(pts_x) + max(pts_x)) / 2.0
            cy = (min(pts_y) + max(pts_y)) / 2.0
            half = max(max(pts_x) - min(pts_x), max(pts_y) - min(pts_y), 5.0) * 0.75
            x_min, x_max = cx - half, cx + half
            y_min, y_max = cy - half, cy + half

        grid_size = 65
        xs = [x_min + (x_max - x_min) * (i / (grid_size - 1)) for i in range(grid_size)]
        ys = [y_min + (y_max - y_min) * (j / (grid_size - 1)) for j in range(grid_size)]

        g1: list[list[float | None]] = []
        g2: list[list[float | None]] = []
        for yv in ys:
            row1: list[float | None] = []
            row2: list[float | None] = []
            for xv in xs:
                try:
                    row1.append(_eval_scalar(contour_fns[0], [xv, yv]))
                except Exception:
                    row1.append(None)
                try:
                    row2.append(_eval_scalar(contour_fns[1], [xv, yv]))
                except Exception:
                    row2.append(None)
            g1.append(row1)
            g2.append(row2)

        contour_2d = {
            "bounds": {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            },
            "f1_segments": _extract_zero_contours(xs, ys, g1, contour_fns[0]),
            "f2_segments": _extract_zero_contours(xs, ys, g2, contour_fns[1]),
        }

    return {
        "status": status,
        "criterion": criterion,
        "message": message,
        "variables": variables,
        "f_latex_list": f_latex_list,
        "jacobian_latex": jacobian_latex,
        "estimated_order": _estimate_convergence_order(steps),
        "rounding_mode": mode,
        "solution": round_vector(x_curr, decimals) if status == "converged" else None,
        "solution_raw": x_curr if status == "converged" else None,
        "iterations": len(steps),
        "steps": steps,
        "contour_2d": contour_2d,
    }
