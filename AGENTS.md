# AGENTS.md — Contexto y Reglas de Desarrollo (NumériLab)

Este archivo define las convenciones técnicas, pedagógicas y de arquitectura para mantener el proyecto bajo metodología **SDD (Spec-Driven Development)**.

## 1. Propósito del proyecto
**NumériLab** es una aplicación web académica para el curso **Análisis Numérico (`ING01199`)** del Politécnico Colombiano Jaime Isaza Cadavid. Debe mostrar **todo el procedimiento matemático paso a paso** (no solo la respuesta final), permitiendo analizar convergencia, estabilidad, errores y limitaciones de cada método.

## 2. Reglas del stack
- **Backend (`backend/`)**: Python 3.11+ administrado exclusivamente con **`uv`** (`uv sync`, `uv add`, `uv run pytest`, `uv run uvicorn`). Prohibido usar `pip install` global.
- **Frontend (`frontend/`)**: React + TypeScript + Vite administrado con **`pnpm`**. Proxy en `vite.config.ts` para reenviar `/api` hacia `http://127.0.0.1:8000`.
- **Idioma de UI y documentación**: Español claro y consistente con terminología del curso (*Factorización L.U.*, *Pivoteo parcial*, *Dominancia diagonal*, *Radio espectral*, *Sobrerrelajación sucesiva SOR*, *Jacobiano*).

## 3. Invariantes matemáticas obligatorias
1. **Algoritmos paso a paso desde cero**:
   - `lu.py`, `iterative_linear.py` y `newton_raphson.py` deben implementar manualmente los bucles numéricos de factorización, sustitución progresiva/regresiva, iteraciones de Jacobi/Gauss-Seidel/SOR y pasos de Newton-Raphson.
   - `SymPy` se utiliza exclusivamente para parseo seguro de expresiones simbólicas, diferenciación exacta ($f'(x)$ y Jacobiano $J(x)$) y generación de LaTeX.
   - `NumPy` se utiliza únicamente para cálculo de autovalores / radio espectral $\rho(T)$ y validación de dimensiones numéricas auxiliares.
2. **Puente `S.L.` $\leftrightarrow$ `N.L.`**:
   - En sistemas no lineales (`solve_newton_system`), cada iteración $k$ debe resolver el sistema lineal $J(x^{(k)})\Delta x^{(k)} = -F(x^{(k)})$ llamando a nuestra función `solve_lu(..., strategy="partial_pivot")`, exponiendo los factores $P, L, U$ de ese paso al frontend.
3. **Manejo seguro de casos límite y divergencia**:
   - Ningún método puede caer en excepción no controlada (HTTP 500) por pivote cero, diagonal nula, matriz singular, derivada nula o desbordamiento por divergencia.
   - Siempre devolver estados estructurados (`zero_pivot_requires_pivoting`, `zero_diagonal`, `singular_matrix`, `zero_derivative`, `diverged`, `max_iterations`, `converged`) y proteger la serialización JSON contra `NaN` o `Infinity`.

## 4. Estándares de UI/UX y accesibilidad
- Mantener la distinción visual de dos modos principales: **`S.L.` (Sistemas Lineales)** y **`N.L.` (No Lineales)**.
- Tablas iterativas dentro de contenedores desplazables accesibles (`tabIndex={0}`, `role="region"`, encabezados `thead` pegajosos y diseño adaptable con `@container`).
- Pestañas implementadas con semántica ARIA (`role="tablist"`, `role="tab"`, `aria-selected`, `role="tabpanel"`).
- Validación de formularios diferida tras interacción (`:user-invalid`), mostrando pistas de formato **encima** de los campos de ecuaciones.
- Soporte completo para tema **Oscuro** y **Claro** con números tabulares (`tabular-nums`) en tablas y matrices.

## 5. Archivos fuente originales (No modificar)
- `FD-GC71.pdf`
- `Captura de pantalla 2026-11-17 213107.png`
