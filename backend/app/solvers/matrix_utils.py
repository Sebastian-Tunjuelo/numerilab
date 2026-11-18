import math
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_EVEN
from itertools import permutations
from typing import Literal
import numpy as np


def resolve_rounding_mode(
    apply_rounding: bool = False,
    rounding_mode: str = "none",
) -> Literal["none", "round", "truncate"]:
    if rounding_mode == "truncate":
        return "truncate"
    if rounding_mode == "round":
        return "round"
    if apply_rounding:
        return "round"
    return "none"


def apply_finite_arithmetic(val: float, decimals: int, mode: str = "none") -> float:
    if not math.isfinite(val):
        return float(val)
    if val == 0.0:
        return 0.0
    if mode == "none":
        return float(val)
    if mode == "round":
        res = round(float(val), decimals)
        return 0.0 if res == 0.0 else res
    if mode == "truncate":
        try:
            q = Decimal(f"1e-{decimals}")
            res = float(Decimal(str(float(val))).quantize(q, rounding=ROUND_DOWN))
        except (InvalidOperation, ValueError, OverflowError):
            factor = 10 ** decimals
            scaled = float(val) * factor
            if not math.isfinite(scaled):
                return float(val)
            res = math.trunc(scaled) / factor
        return 0.0 if res == 0.0 else res
    return float(val)


def apply_finite_vector(v: list[float], decimals: int, mode: str = "none") -> list[float]:
    return [apply_finite_arithmetic(float(x), decimals, mode) for x in v]


def apply_finite_matrix(M: list[list[float]], decimals: int, mode: str = "none") -> list[list[float]]:
    return [[apply_finite_arithmetic(float(x), decimals, mode) for x in row] for row in M]


def fmt_num(val: float, decimals: int = 4) -> str:
    if val == 0.0:
        return "0"
    if abs(val) >= 1e6 or (0 < abs(val) < 10 ** (-decimals)):
        return f"{val:.{max(2, decimals - 2)}e}"
    text = f"{val:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def round_vector(v: list[float], decimals: int) -> list[float]:
    return [0.0 if (rounded := round(float(x), decimals)) == 0.0 else rounded for x in v]


def round_matrix(M: list[list[float]], decimals: int) -> list[list[float]]:
    return [round_vector(row, decimals) for row in M]


def format_omega(omega: float) -> str:
    return f"{omega:.2f}" if round(omega, 2) == omega else f"{omega:.4g}"


def matrix_to_latex(M: list[list[float]], decimals: int = 4) -> str:
    rows_tex = [" & ".join(fmt_num(val, decimals) for val in row) for row in M]
    return r"\begin{bmatrix} " + r" \\ ".join(rows_tex) + r" \end{bmatrix}"


def vector_to_latex(v: list[float], decimals: int = 4) -> str:
    rows_tex = [fmt_num(val, decimals) for val in v]
    return r"\begin{bmatrix} " + r" \\ ".join(rows_tex) + r" \end{bmatrix}"


def mat_vec_mul(M: list[list[float]], v: list[float]) -> list[float]:
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def vec_inf_norm(v: list[float]) -> float:
    if not v:
        return 0.0
    return max(abs(x) for x in v)


def vec_diff_inf_norm(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def mat_norm_1(M: list[list[float]]) -> float:
    if not M:
        return 0.0
    cols = len(M[0])
    return max(sum(abs(M[i][j]) for i in range(len(M))) for j in range(cols))


def mat_norm_inf(M: list[list[float]]) -> float:
    if not M:
        return 0.0
    return max(sum(abs(val) for val in row) for row in M)


def compute_sor_form(A: list[list[float]], b: list[float], omega: float) -> tuple[list[list[float]], list[float]]:
    """Obtiene T y c por sustitución progresiva, normalizando cada fila por su diagonal."""
    n = len(A)
    T: list[list[float]] = []
    c: list[float] = []
    for i, row in enumerate(A):
        diagonal = row[i]
        lower = [omega * (row[j] / diagonal) for j in range(i)]
        right = [0.0] * i + [1.0 - omega] + [-omega * (row[j] / diagonal) for j in range(i + 1, n)]
        T.append([
            right[col] - sum(coefficient * T[j][col] for j, coefficient in enumerate(lower))
            for col in range(n)
        ])
        c.append(omega * (b[i] / diagonal) - sum(coefficient * c[j] for j, coefficient in enumerate(lower)))
    return T, c


def compute_spectral_radius(T: list[list[float]]) -> float | None:
    try:
        arr = np.array(T, dtype=float)
        if not np.all(np.isfinite(arr)):
            return None
        eigvals = np.linalg.eigvals(arr)
        radius = float(np.max(np.abs(eigvals)))
        return radius if math.isfinite(radius) else None
    except Exception:
        return None


def check_diagonal_dominance(A: list[list[float]], b: list[float] | None = None) -> dict:
    n = len(A)
    rows_analysis = []
    strict_all = True
    weak_all = True
    strict_at_least_one = False
    has_zero_diagonal = False

    for i in range(n):
        diag_abs = abs(A[i][i])
        if diag_abs == 0.0:
            has_zero_diagonal = True
        off_diag_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        is_strict = diag_abs > off_diag_sum
        is_weak = diag_abs >= off_diag_sum
        if not is_strict:
            strict_all = False
        if not is_weak:
            weak_all = False
        if is_strict:
            strict_at_least_one = True

        rows_analysis.append(
            {
                "row": i + 1,
                "diag_abs": diag_abs,
                "off_diag_sum": off_diag_sum if math.isfinite(off_diag_sum) else None,
                "is_strictly_dominant": is_strict,
                "is_weakly_dominant": is_weak,
            }
        )

    result = {
        "is_strictly_dominant": strict_all,
        "is_weakly_dominant": weak_all and strict_at_least_one,
        "has_zero_diagonal": has_zero_diagonal,
        "rows_analysis": rows_analysis,
    }
    if b is not None:
        result["reordering_suggestion"] = suggest_row_reordering(A, b)
    return result


def suggest_row_reordering(A: list[list[float]], b: list[float]) -> dict | None:
    """Busca una permutación de filas que elimine ceros diagonales o maximice la dominancia diagonal."""
    n = len(A)
    if n < 2 or n > 6 or len(b) != n:
        return None
    if not all(math.isfinite(val) for row in A for val in row) or not all(math.isfinite(val) for val in b):
        return None

    curr_dom = check_diagonal_dominance(A)
    if curr_dom["is_strictly_dominant"] and not curr_dom["has_zero_diagonal"]:
        return None

    def _perm_score(perm: tuple[int, ...]) -> tuple[bool, bool, bool, int, int, float, float] | None:
        strict_count = 0
        weak_count = 0
        no_zero_diag = True
        min_margin = math.inf
        diag_sum = 0.0
        for i in range(n):
            row = A[perm[i]]
            diag = abs(row[i])
            off = sum(abs(row[j]) for j in range(n) if j != i)
            if not math.isfinite(diag) or not math.isfinite(off):
                return None
            if diag == 0.0:
                no_zero_diag = False
            if diag > off:
                strict_count += 1
            if diag >= off:
                weak_count += 1
            margin = diag - off
            if not math.isfinite(margin):
                return None
            if margin < min_margin:
                min_margin = margin
            diag_sum += diag
        if not math.isfinite(min_margin) or not math.isfinite(diag_sum):
            return None
        is_strict = strict_count == n
        is_weak = weak_count == n and strict_count >= 1
        return (
            no_zero_diag,
            is_strict,
            is_weak,
            strict_count,
            weak_count,
            round(min_margin, 12),
            round(diag_sum, 12),
        )

    id_perm = tuple(range(n))
    id_score = _perm_score(id_perm)
    if id_score is None:
        return None

    best_perm = id_perm
    best_score = id_score
    for perm in permutations(range(n)):
        score = _perm_score(perm)
        if score is not None and score > best_score:
            best_score = score
            best_perm = perm

    if best_perm != id_perm and best_score > id_score and best_score[0]:
        A_reordered = [[float(v) for v in A[r]] for r in best_perm]
        b_reordered = [float(b[r]) for r in best_perm]
        new_dom = check_diagonal_dominance(A_reordered)
        return {
            "improves": True,
            "permutation": [r + 1 for r in best_perm],
            "is_strictly_dominant": new_dom["is_strictly_dominant"],
            "is_weakly_dominant": new_dom["is_weakly_dominant"],
            "has_zero_diagonal": new_dom["has_zero_diagonal"],
            "A_reordered": A_reordered,
            "b_reordered": b_reordered,
        }
    return None


def build_iteration_matrix_info(
    name: str,
    T: list[list[float]],
    c: list[float],
    decimals: int = 4,
) -> dict:
    if not all(math.isfinite(value) for row in [*T, c] for value in row):
        return {"available": False, "reason": "La forma matricial excedió el rango numérico."}
    rho = compute_spectral_radius(T)
    n1 = mat_norm_1(T)
    ninf = mat_norm_inf(T)
    converges_by_rho = rho is not None and rho < 1.0
    return {
        "available": True,
        "name": name,
        "T": round_matrix(T, decimals),
        "c": round_vector(c, decimals),
        "T_latex": matrix_to_latex(T, decimals),
        "c_latex": vector_to_latex(c, decimals),
        "norm_1": n1 if math.isfinite(n1) else None,
        "norm_inf": ninf if math.isfinite(ninf) else None,
        "spectral_radius": rho,
        "guaranteed_convergence": converges_by_rho,
    }


def compute_iteration_matrices(
    A: list[list[float]],
    b: list[float],
    omega: float,
    decimals: int = 4,
) -> dict:
    """Calcula D, L_estricta, U_estricta y las formas matriciales de Jacobi, GS y SOR."""
    n = len(A)
    D = [[A[i][j] if i == j else 0.0 for j in range(n)] for i in range(n)]
    L_strict = [[-A[i][j] if i > j else 0.0 for j in range(n)] for i in range(n)]
    U_strict = [[-A[i][j] if i < j else 0.0 for j in range(n)] for i in range(n)]

    split = {
        "D_latex": matrix_to_latex(D, decimals),
        "L_latex": matrix_to_latex(L_strict, decimals),
        "U_latex": matrix_to_latex(U_strict, decimals),
    }

    if any(A[i][i] == 0.0 for i in range(n)):
        unavailable = {
            "available": False,
            "reason": "La matriz tiene al menos un elemento nulo en la diagonal principal (aᵢᵢ = 0), por lo que D no es invertible.",
        }
        return {
            "split": split,
            "jacobi": unavailable,
            "gauss_seidel": unavailable,
            "sor": unavailable,
        }

    # Jacobi: T_j = D^{-1}(L + U), c_j = D^{-1} b
    T_j = [[-A[i][j] / A[i][i] if i != j else 0.0 for j in range(n)] for i in range(n)]
    c_j = [b[i] / A[i][i] for i in range(n)]

    def build_sor_matrices(w: float, label: str) -> dict:
        T_w, c_w = compute_sor_form(A, b, w)
        return build_iteration_matrix_info(label, T_w, c_w, decimals=decimals)

    return {
        "split": split,
        "jacobi": build_iteration_matrix_info("Jacobi", T_j, c_j, decimals=decimals),
        "gauss_seidel": build_sor_matrices(1.0, "Gauss-Seidel (ω = 1)"),
        "sor": build_sor_matrices(omega, f"SOR (ω = {format_omega(omega)})"),
    }
