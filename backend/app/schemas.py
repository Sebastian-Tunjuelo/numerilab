import math
from typing import Annotated, Literal
from pydantic import BaseModel, Field, FiniteFloat, field_validator, model_validator

from .solvers.math_expression import MAX_EXPRESSION_LENGTH


StoppingCriterion = Literal["abs_diff", "rel_diff", "rel_pct", "residual"]
NewtonStoppingCriterion = Literal["residual_and_step", "rel_diff", "rel_pct"]
RoundingMode = Literal["none", "round", "truncate"]
MathExpression = Annotated[str, Field(min_length=1, max_length=MAX_EXPRESSION_LENGTH)]


def _sync_rounding_mode(apply_rounding: bool, rounding_mode: RoundingMode) -> tuple[bool, RoundingMode]:
    if rounding_mode == "none" and apply_rounding:
        return True, "round"
    if rounding_mode != "none":
        return True, rounding_mode
    return False, "none"


def _validate_tol_for_criterion(criterion: str, tol: float) -> None:
    if criterion == "rel_pct":
        if not (0.0 < tol < 100.0):
            raise ValueError("Para criterio porcentual, la tolerancia debe ser mayor que 0 y menor que 100.")
    elif tol > 1.0:
        raise ValueError("Para este criterio, la tolerancia debe ser menor o igual que 1.")


def _validate_linear_tol(criterion: StoppingCriterion, tol: float) -> None:
    _validate_tol_for_criterion(criterion, tol)


def _validate_newton_tol(criterion: NewtonStoppingCriterion, tol: float) -> None:
    _validate_tol_for_criterion(criterion, tol)


class LinearAnalyzeRequest(BaseModel):
    A: list[list[FiniteFloat]] = Field(..., min_length=2, max_length=6)
    b: list[FiniteFloat] = Field(..., min_length=2, max_length=6)
    x0: list[FiniteFloat] | None = None
    omega: float = Field(default=1.25, gt=0.0, lt=2.0)
    tol: float = Field(default=1e-6, gt=0.0, le=100.0)
    max_iter: int = Field(default=100, ge=1, le=500)
    criterion: StoppingCriterion = "abs_diff"
    lu_strategy: Literal["doolittle", "crout", "partial_pivot"] = "partial_pivot"
    decimals: int = Field(default=6, ge=2, le=12)
    apply_rounding: bool = False
    rounding_mode: RoundingMode = "none"

    @model_validator(mode="after")
    def validate_dimensions(self) -> "LinearAnalyzeRequest":
        _validate_linear_tol(self.criterion, self.tol)
        n = len(self.A)
        if any(len(row) != n for row in self.A):
            raise ValueError("La matriz A debe ser cuadrada (n x n).")
        if len(self.b) != n:
            raise ValueError("El vector b debe tener la misma dimensión que A.")
        if self.x0 is not None and len(self.x0) != n:
            raise ValueError("El vector inicial x0 debe tener la misma dimensión que A.")
        self.apply_rounding, self.rounding_mode = _sync_rounding_mode(self.apply_rounding, self.rounding_mode)
        return self


class SORSweepRequest(BaseModel):
    A: list[list[FiniteFloat]] = Field(..., min_length=2, max_length=6)
    b: list[FiniteFloat] = Field(..., min_length=2, max_length=6)
    x0: list[FiniteFloat] | None = None
    tol: float = Field(default=1e-6, gt=0.0, le=100.0)
    max_iter: int = Field(default=100, ge=1, le=500)
    criterion: StoppingCriterion = "abs_diff"
    decimals: int = Field(default=6, ge=2, le=12)
    apply_rounding: bool = False
    rounding_mode: RoundingMode = "none"

    @model_validator(mode="after")
    def validate_dimensions(self) -> "SORSweepRequest":
        _validate_linear_tol(self.criterion, self.tol)
        n = len(self.A)
        if any(len(row) != n for row in self.A):
            raise ValueError("La matriz A debe ser cuadrada (n x n).")
        if len(self.b) != n:
            raise ValueError("El vector b debe tener la misma dimensión que A.")
        if self.x0 is not None and len(self.x0) != n:
            raise ValueError("El vector inicial x0 debe tener la misma dimensión que A.")
        self.apply_rounding, self.rounding_mode = _sync_rounding_mode(self.apply_rounding, self.rounding_mode)
        return self


class Newton1DRequest(BaseModel):
    expression: MathExpression
    x0: FiniteFloat
    tol: float = Field(default=1e-7, gt=0.0, le=100.0)
    max_iter: int = Field(default=50, ge=1, le=200)
    decimals: int = Field(default=6, ge=2, le=12)
    criterion: NewtonStoppingCriterion = "residual_and_step"
    apply_rounding: bool = False
    rounding_mode: RoundingMode = "none"

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("La expresión f(x) no puede estar vacía.")
        return cleaned

    @model_validator(mode="after")
    def validate_tol_for_criterion(self) -> "Newton1DRequest":
        _validate_newton_tol(self.criterion, self.tol)
        self.apply_rounding, self.rounding_mode = _sync_rounding_mode(self.apply_rounding, self.rounding_mode)
        return self


class NewtonSystemRequest(BaseModel):
    equations: list[MathExpression] = Field(..., min_length=2, max_length=3)
    variables: list[Annotated[str, Field(min_length=1, max_length=32)]] = Field(
        default_factory=lambda: ["x", "y"], min_length=2, max_length=3
    )
    x0: list[FiniteFloat] = Field(..., min_length=2, max_length=3)
    tol: float = Field(default=1e-7, gt=0.0, le=100.0)
    max_iter: int = Field(default=50, ge=1, le=200)
    decimals: int = Field(default=6, ge=2, le=12)
    criterion: NewtonStoppingCriterion = "residual_and_step"
    apply_rounding: bool = False
    rounding_mode: RoundingMode = "none"
    bounds: dict[str, FiniteFloat] | None = None

    @field_validator("bounds")
    @classmethod
    def validate_bounds(cls, bounds: dict[str, float] | None) -> dict[str, float] | None:
        if bounds is None:
            return None
        for axis in ("x", "y"):
            start, end = bounds.get(f"{axis}_min"), bounds.get(f"{axis}_max")
            if start is None or end is None or start >= end or not math.isfinite(end - start):
                raise ValueError("La ventana gráfica requiere límites finitos y crecientes para ambos ejes.")
        return bounds

    @model_validator(mode="after")
    def validate_system(self) -> "NewtonSystemRequest":
        _validate_newton_tol(self.criterion, self.tol)
        n = len(self.equations)
        if len(self.variables) != n:
            raise ValueError("La cantidad de variables debe coincidir con la cantidad de ecuaciones.")
        if len(self.x0) != n:
            raise ValueError("El vector inicial x0 debe coincidir con la dimensión del sistema.")
        if any(not eq.strip() for eq in self.equations):
            raise ValueError("Ninguna ecuación del sistema puede estar vacía.")
        self.apply_rounding, self.rounding_mode = _sync_rounding_mode(self.apply_rounding, self.rounding_mode)
        return self
