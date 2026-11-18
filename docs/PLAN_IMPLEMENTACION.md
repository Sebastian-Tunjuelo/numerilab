# Plan de Implementación — Mejoras Pedagógicas de NumériLab

Este plan cubre únicamente los **4 bloques oficiales del proyecto** (`L.U.`, `Jacobi`, `Gauss-Seidel (SOR)` y `Newton-Raphson`), respetando todas las invariantes de [AGENTS.md](../AGENTS.md) (algoritmos desde cero en Python, SymPy solo para parseo/derivación/LaTeX, NumPy solo para autovalores/radio espectral y puente `S.L. ↔ N.L.`).

---

## Bloque 1: Segunda derivada $f''(x)$ y diagnóstico de convergencia en Newton-Raphson 1D
**Objetivo académico:** Cubrir explícitamente las condiciones y limitaciones de [NEWTON-RAPHSON.pdf](../NEWTON-RAPHSON.pdf) (condición de Fourier $f(x_0)f''(x_0) > 0$, puntos de inflexión donde $f''(x) \approx 0$ y raíces múltiples con convergencia lineal).

### Backend (`backend/app/solvers/newton_raphson.py`)
- En `solve_newton_1d`, calcular simbólicamente `d2f_expr = sp.diff(df_expr, x_sym)`, `d2f_latex = sp.latex(d2f_expr)` y `d2f_fn = sp.lambdify(...)`.
- Evaluar de forma segura `d2fx_0` en $x_0$ (y `d2fx_k` en cada iteración $k$ dentro de `steps`, sin lanzar excepción si sale del dominio).
- Calcular el indicador inicial `fourier_product` ($f(x_0) \cdot f''(x_0)$) y generar `convergence_warnings: list[str]` cuando:
  - $f(x_0)f''(x_0) \le 0$ (no garantiza convergencia monótona inicial según criterio de Fourier).
  - $|f''(x_k)| < 10^{-8}$ en algún iterado cercano a un punto de inflexión.
  - Se detecta cambio de signo alternante en los pasos o orden estimado $p < 1.3$ (posible raíz de multiplicidad $m > 1$).

### Frontend (`frontend/src/types/numerical.ts`, `frontend/src/components/labs/NonlinearLab.tsx`)
- Extender `Newton1DResponse` y `Newton1DStep` con `d2f_latex`, `d2fx_k`, `fourier_condition` y `convergence_warnings`.
- Mostrar $f''(x)$ en LaTeX junto a $f(x)$ y $f'(x)$ y renderizar alertas pedagógicas claras cuando existan advertencias de divergencia, oscilación o raíz múltiple.

---

## Bloque 2: Gráfica de convergencia en `N.L.` (1D y Sistemas $2\times 2 / 3\times 3$) + Comparador en 1D
**Objetivo académico:** Permitir visualizar la velocidad de convergencia (cuadrática vs. lineal) en `Newton-Raphson` y darle representación gráfica al modo de sistemas $3\times 3$.

### Frontend (`frontend/src/components/charts/NewtonConvergenceChart.tsx`, `frontend/src/components/labs/NonlinearLab.tsx`)
- Crear `NewtonConvergenceChart.tsx` (SVG responsivo en escala $\log_{10}$ análogo a [ConvergenceChart.tsx](../frontend/src/components/charts/ConvergenceChart.tsx)) que grafique:
  - Error paso a paso ($|x_{k+1}-x_k|$ o $\|\Delta x^{(k)}\|_\infty$) y residual ($|f(x_k)|$ o $\|F(x_k)\|_\infty$) frente a la iteración $k$.
  - Soporte uniforme para **Newton 1D**, **Sistemas $2\times 2$** y **Sistemas $3\times 3$**.
- Agregar en Newton 1D un modo **"Comparar con otro $x_0$"** (segundo $x_0$ opcional en la misma ecuación):
  - Ejecuta `solveNewton1D` para ambos valores iniciales y superpone sus curvas en `NewtonConvergenceChart`, mostrando lado a lado el número de iteraciones y el orden empírico $p$ (ideal para el ejercicio de $x^3-5x^2+7x-3$ con raíz doble vs. raíz simple).

---

## Bloque 3: Reordenamiento automático de filas para Jacobi / Gauss-Seidel / SOR (`S.L.`)
**Objetivo académico:** Enseñar que una matriz con ceros en la diagonal o no dominante puede volverse convergente mediante permutación de filas $PAx = Pb$, equivalente al botón de *Activar Pivoteo Parcial* en L.U.

### Backend (`backend/app/solvers/matrix_utils.py`, `backend/app/main.py`)
- Implementar `suggest_row_reordering(A, b)` usando búsqueda exacta sobre permutaciones (viable porque `LinearAnalyzeRequest` acota $2 \le n \le 6$, es decir $\le 720$ permutaciones):
  - Priorizar permutaciones sin ceros en la diagonal (`has_zero_diagonal == False`).
  - Maximizar primero dominancia diagonal estricta, luego dominancia débil y finalmente el margen mínimo $\min_i \left(|a_{ii}| - \sum_{j \neq i} |a_{ij}|\right)$.
- Exponer `reordering_suggestion` dentro de `diagnostics.dominance` en `/api/linear/analyze` solo cuando la matriz actual tenga diagonal nula o no sea estrictamente dominante y exista una permutación que mejore la condición (`permutation`, `improves`, `is_strictly_dominant`, `is_weakly_dominant`, `has_zero_diagonal`, `A_reordered`, `b_reordered`).

### Frontend (`frontend/src/types/numerical.ts`, `frontend/src/components/labs/LinearSystemsLab.tsx`)
- Cuando `reordering_suggestion?.improves` sea verdadero, mostrar el botón **"Reordenar filas (mejorar dominancia)"** en el panel de diagnóstico y en las alertas de diagonal nula, aplicando `A_reordered` y `b_reordered` al formulario y recalculando automáticamente.

---

## Bloque 4: Complejidad computacional (FLOPs) y Veredicto Pedagógico en `S.L.`
**Objetivo académico:** Responder a los objetivos de eficiencia y comparación de [FD-GC71.pdf](../FD-GC71.pdf) comparando métodos directos vs. iterativos no solo en iteraciones y residual, sino en costo aritmético real.

### Backend (`backend/app/solvers/lu.py`, `backend/app/solvers/iterative_linear.py`)
- Contar las operaciones aritméticas escalares reales (`mul_div`, `add_sub`, `total_flops`) durante la ejecución manual:
  - En `solve_lu`: conteo durante factorización ($L, U$), sustitución progresiva ($Ly=Pb$) y sustitución regresiva ($Ux=y$), junto a la fórmula teórica $\approx \frac{2}{3}n^3 + 2n^2$.
  - En `solve_iterative_linear`: conteo por iteración y acumulado total ($k$ iteraciones), junto a la cota $O(k \cdot n^2)$.

### Frontend (`frontend/src/types/numerical.ts`, `frontend/src/components/labs/LinearSystemsLab.tsx`)
- En la pestaña *Comparador y gráficas*, añadir una columna/tarjeta de **Operaciones (FLOPs)** y un **Veredicto Pedagógico** automático que compare L.U. frente a Jacobi, Gauss-Seidel y SOR ($\omega$), explicando cuál convergió con menos costo y por qué.

---

## Bloque 5: Aritmética finita de *n* decimales extendida y actualización SDD
**Objetivo académico:** Alinear completamente el proyecto con aritmética de máquina finita (Unidad 1, Sesión 6 de [FD-GC71.pdf](../FD-GC71.pdf)) y mantener la documentación SDD sincronizada.

### Backend (`backend/app/schemas.py`, `backend/app/solvers/matrix_utils.py`, `lu.py`, `iterative_linear.py`, `newton_raphson.py`)
- Crear utilidades compartidas `apply_finite_arithmetic(val, decimals, mode)` con soporte para `rounding_mode: Literal["none", "round", "truncate"] = "none"` (manteniendo retrocompatibilidad con `apply_rounding: bool`).
- Permitir aplicar redondeo o truncamiento a *n* decimales paso a paso también en `L.U.` y `Newton-Raphson` (1D y Sistemas) cuando el usuario lo active, registrando las diferencias respecto al cálculo en precisión completa.

### Frontend (`frontend/src/components/labs/LinearSystemsLab.tsx`, `frontend/src/components/labs/NonlinearLab.tsx`)
- Ampliar el selector de decimales en la interfaz a `2, 3, 4, 6, 8` en ambos laboratorios.
- Agregar selector compacto de aritmética finita (`Sin redondeo por paso` / `Redondeo por paso` / `Truncamiento por paso`) en `S.L.` y `N.L.`.

### Documentación SDD (`docs/SPEC.md`, `docs/TASKS.md`)
- Actualizar `docs/SPEC.md` y `docs/TASKS.md` con las nuevas capacidades una vez implementadas.

---

## Estrategia de Verificación
1. **Pruebas unitarias (`backend/tests/`)**: Ampliar `test_lu.py`, `test_iterative_linear.py` y `test_newton_raphson.py` para cubrir `d2f_latex` y advertencias de Fourier/inflexión, permutaciones de dominancia diagonal, conteo exacto de FLOPs y modos `round`/`truncate`; validar con `uv run pytest -v`.
2. **Compilación y tipos (`frontend/`)**: Ejecutar `pnpm --dir frontend build` (`tsc --noEmit && vite build`) con 0 errores.
3. **Pruebas E2E en navegador real (Playwright)**: Verificar los nuevos paneles ($f''(x)$, gráfica de convergencia `N.L.` en 1D y $3\times 3$, comparador de dos $x_0$, botón de reordenamiento de filas, FLOPs y aritmética finita) sin errores de consola ni regresiones en escritorio y móvil.
