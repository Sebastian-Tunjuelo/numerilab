from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    LinearAnalyzeRequest,
    Newton1DRequest,
    NewtonSystemRequest,
    SORSweepRequest,
)
from .solvers.iterative_linear import solve_iterative_linear, sweep_sor_omega
from .solvers.lu import solve_lu
from .solvers.matrix_utils import check_diagonal_dominance, compute_iteration_matrices, format_omega
from .solvers.newton_raphson import solve_newton_1d, solve_newton_system

app = FastAPI(
    title="NumériLab API",
    description="Backend numérico paso a paso para LU, Jacobi, Gauss-Seidel (SOR) y Newton-Raphson.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
    # El valor rechazado puede ser infinito y no se puede serializar en JSON.
    errors = [{key: error[key] for key in ("type", "loc", "msg")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok", "service": "numerilab-backend"}


def _build_linear_verdict(
    lu_res: dict,
    jacobi_res: dict,
    gs_res: dict,
    sor_res: dict,
    dominance: dict,
    iter_matrices: dict,
) -> dict:
    iterative_candidates = [
        ("jacobi", "Jacobi", jacobi_res),
        ("gauss_seidel", "Gauss-Seidel", gs_res),
        ("sor", f"SOR (ω={format_omega(sor_res.get('omega', 1.0))})", sor_res),
    ]
    all_candidates = [
        ("lu", "Factorización L.U.", lu_res),
        *iterative_candidates,
    ]

    converged_iter = [
        (key, label, res)
        for key, label, res in iterative_candidates
        if res.get("status") == "converged"
    ]
    converged_all = [
        (key, label, res)
        for key, label, res in all_candidates
        if res.get("status") == "converged"
    ]

    best_iterative = None
    if converged_iter:
        best_iter_entry = min(
            converged_iter,
            key=lambda item: (
                item[2].get("iterations", 10**9),
                item[2].get("operation_count", {}).get("total_flops", 10**9),
            ),
        )
        best_iterative = best_iter_entry[0]

    lowest_flops_method = None
    if converged_all:
        lowest_entry = min(
            converged_all,
            key=lambda item: (
                item[2].get("operation_count", {}).get("total_flops", 10**9),
                item[2].get("iterations", 10**9),
            ),
        )
        lowest_flops_method = lowest_entry[0]

    notes: list[str] = []
    if dominance.get("has_zero_diagonal"):
        notes.append(
            "La matriz contiene ceros en la diagonal principal. Jacobi, Gauss-Seidel y SOR requieren una diagonal sin ceros."
        )
        if lu_res["status"] == "singular_matrix":
            notes.append("L.U. reportó una matriz singular o casi singular con la estrategia y la precisión actuales.")
        elif lu_res["status"] == "zero_pivot_requires_pivoting":
            notes.append("Prueba L.U. con pivoteo parcial para intentar continuar la factorización.")
    elif dominance.get("is_strictly_dominant"):
        notes.append(
            "La matriz es estrictamente diagonalmente dominante, lo que garantiza convergencia de Jacobi y Gauss-Seidel desde cualquier x⁽⁰⁾."
        )
    else:
        notes.append(
            "La matriz no es estrictamente diagonalmente dominante; la convergencia iterativa depende del radio espectral ρ(T) < 1."
        )

    for key, label in (("jacobi", "Jacobi"), ("gauss_seidel", "Gauss-Seidel"), ("sor", "SOR")):
        mat_info = iter_matrices.get(key, {})
        rho = mat_info.get("spectral_radius")
        if mat_info.get("available") and rho is not None:
            notes.append(
                f"{label}: ρ(T) = {rho:.4f} ({'converge' if rho < 1.0 else 'no garantiza convergencia'})."
            )

    if not converged_all:
        summary = "Ninguno de los métodos logró converger bajo los parámetros actuales; revisa la singularidad o la tolerancia."
    elif not converged_iter and lu_res.get("status") == "converged":
        lu_flops = lu_res.get("operation_count", {}).get("total_flops", 0)
        summary = (
            f"Factorización L.U. obtuvo una solución con {lu_flops} FLOPs. Los métodos iterativos no convergieron con los parámetros actuales."
        )
    else:
        best_label = next(lbl for k, lbl, _ in all_candidates if k == lowest_flops_method)
        best_res = next(res for k, _, res in all_candidates if k == lowest_flops_method)
        best_flops = best_res.get("operation_count", {}).get("total_flops", 0)
        lu_flops = lu_res.get("operation_count", {}).get("total_flops")
        best_iter_label = next(lbl for k, lbl, _ in iterative_candidates if k == best_iterative)
        best_iter_res = next(res for k, _, res in iterative_candidates if k == best_iterative)
        best_iter_its = best_iter_res.get("iterations", 0)
        best_iter_flops = best_iter_res.get("operation_count", {}).get("total_flops", 0)

        if lu_res.get("status") == "converged" and lu_flops is not None:
            summary = (
                f"Menor costo aritmético total: {best_label} ({best_flops} FLOPs). "
                f"Entre los iterativos, {best_iter_label} destacó con {best_iter_its} iteraciones ({best_iter_flops} FLOPs) "
                f"frente a {lu_flops} FLOPs de L.U. directo."
            )
        else:
            summary = (
                f"Menor costo aritmético total: {best_label} ({best_flops} FLOPs, {best_res['iterations']} iteraciones)."
            )

    return {
        "best_iterative": best_iterative,
        "lowest_flops_method": lowest_flops_method,
        "summary": summary,
        "notes": notes,
    }


@app.post("/api/linear/analyze")
def analyze_linear_system(req: LinearAnalyzeRequest) -> dict:
    dominance = check_diagonal_dominance(req.A, req.b)
    iter_matrices = compute_iteration_matrices(req.A, req.b, req.omega, decimals=req.decimals)

    lu_res = solve_lu(
        req.A,
        req.b,
        strategy=req.lu_strategy,
        decimals=req.decimals,
        apply_rounding=req.apply_rounding,
        rounding_mode=req.rounding_mode,
    )
    jacobi_res = solve_iterative_linear(
        A=req.A,
        b=req.b,
        x0=req.x0,
        method="jacobi",
        omega=1.0,
        tol=req.tol,
        max_iter=req.max_iter,
        criterion=req.criterion,
        decimals=req.decimals,
        apply_rounding=req.apply_rounding,
        rounding_mode=req.rounding_mode,
    )
    gs_res = solve_iterative_linear(
        A=req.A,
        b=req.b,
        x0=req.x0,
        method="gauss_seidel",
        omega=1.0,
        tol=req.tol,
        max_iter=req.max_iter,
        criterion=req.criterion,
        decimals=req.decimals,
        apply_rounding=req.apply_rounding,
        rounding_mode=req.rounding_mode,
    )
    sor_res = solve_iterative_linear(
        A=req.A,
        b=req.b,
        x0=req.x0,
        method="sor",
        omega=req.omega,
        tol=req.tol,
        max_iter=req.max_iter,
        criterion=req.criterion,
        decimals=req.decimals,
        apply_rounding=req.apply_rounding,
        rounding_mode=req.rounding_mode,
    )

    verdict = _build_linear_verdict(lu_res, jacobi_res, gs_res, sor_res, dominance, iter_matrices)

    return {
        "diagnostics": {
            "dominance": dominance,
            "matrices": iter_matrices,
        },
        "lu": lu_res,
        "jacobi": jacobi_res,
        "gauss_seidel": gs_res,
        "sor": sor_res,
        "verdict": verdict,
    }


@app.post("/api/linear/sor-sweep")
def sor_sweep_endpoint(req: SORSweepRequest) -> dict:
    return sweep_sor_omega(
        A=req.A,
        b=req.b,
        x0=req.x0,
        tol=req.tol,
        max_iter=req.max_iter,
        criterion=req.criterion,
        decimals=req.decimals,
        apply_rounding=req.apply_rounding,
        rounding_mode=req.rounding_mode,
    )


@app.post("/api/nonlinear/newton-1d")
def newton_1d_endpoint(req: Newton1DRequest) -> dict:
    return solve_newton_1d(
        expression=req.expression,
        x0=req.x0,
        tol=req.tol,
        max_iter=req.max_iter,
        decimals=req.decimals,
        criterion=req.criterion,
        apply_rounding=req.apply_rounding,
        rounding_mode=req.rounding_mode,
    )


@app.post("/api/nonlinear/newton-system")
def newton_system_endpoint(req: NewtonSystemRequest) -> dict:
    return solve_newton_system(
        equations=req.equations,
        variables=req.variables,
        x0=req.x0,
        tol=req.tol,
        max_iter=req.max_iter,
        decimals=req.decimals,
        bounds=req.bounds,
        criterion=req.criterion,
        apply_rounding=req.apply_rounding,
        rounding_mode=req.rounding_mode,
    )
