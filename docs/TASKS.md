# TASKS.md — Plan de Tareas (SDD)

## Fase 1: Descubrimiento de Skills y Documentación SDD
- [x] Instalar skill `find-skills`, `vercel-react-best-practices` y `frontend-design` en `.agents/skills/`.
- [x] Crear `README.md` en español con contexto del curso y guía de arranque.
- [x] Crear `AGENTS.md` con reglas e invariantes matemáticas.
- [x] Crear `docs/SPEC.md` (`RF-01` a `RF-07`), `docs/ARCHITECTURE.md` y `docs/TASKS.md`.

## Fase 2: Backend Numérico (`backend/`)
- [x] Inicializar proyecto Python con `uv` (`pyproject.toml`, FastAPI, SymPy, NumPy, Pytest).
- [x] Implementar `app/schemas.py` con validaciones Pydantic.
- [x] Implementar `app/solvers/matrix_utils.py` (dominancia diagonal, $D, L, U$, matrices de iteración, normas y $\rho(T)$).
- [x] Implementar `app/solvers/lu.py` (Doolittle, Crout, Pivoteo Parcial $PA=LU$, pasos, $Ly=Pb$, $Ux=y$).
- [x] Implementar `app/solvers/iterative_linear.py` (Jacobi, Gauss-Seidel, SOR y barrido $\omega$).
- [x] Implementar `app/solvers/newton_raphson.py` (1D + Sistemas $2\times 2 / 3\times 3$ usando `solve_lu` + marching squares 2D).
- [x] Implementar `app/main.py` con endpoints REST y CORS.
- [x] Escribir y aprobar pruebas unitarias y de endpoints (`test_lu.py`, `test_iterative_linear.py`, `test_newton_raphson.py`) con `uv run pytest`.

## Fase 3: Frontend Interactivo (`frontend/`)
- [x] Inicializar React + TypeScript + Vite (`pnpm`) con Tailwind CSS, Motion, Mafs, KaTeX, Lucide React y mathjs.
- [x] Configurar proxy `/api` en `vite.config.ts` y sistema de diseño (tema Oscuro/Claro, tipografía técnica y estilos accesibles de tablas/tabs/formularios).
- [x] Implementar cliente API tipado (`src/api/client.ts`) y presets del curso (`src/data/presets.ts`).
- [x] Implementar componentes matemáticos (`LatexRenderer.tsx`, `MatrixDisplay.tsx`).
- [x] Implementar gráficas (`ConvergenceChart.tsx`, `OmegaSweepChart.tsx`, `Newton1DPlot.tsx` con Mafs + punto $x_0$ arrastrable, `Newton2DPlot.tsx` con curvas implícitas y clic para fijar $(x_0, y_0)$).
- [x] Implementar `LinearSystemsLab.tsx` (`Comparador`, `L.U.`, `Jacobi`, `Gauss-Seidel / SOR`, inspector de fórmula por fila y exportación CSV/LaTeX).
- [x] Implementar `NonlinearLab.tsx` (1D y Sistemas $2\times 2 / 3\times 3$ con reproductor paso a paso y detalle del puente LU).
- [x] Implementar `App.tsx`, `Header.tsx` y `TheoryDrawer.tsx` (Guía Teórica de sustentación).
- [x] Verificar compilación limpia (`pnpm build`).

## Fase 4: Verificación End-to-End y Cierre
- [x] Crear script de arranque `start-dev.ps1`.
- [x] Levantar backend + frontend y validar visualmente y funcionalmente todos los módulos en navegador real sin errores de consola.

## Fase 5: Mejoras Pedagógicas (`docs/PLAN_IMPLEMENTACION.md`)
- [x] **Bloque 1:** Segunda derivada $f''(x)$, criterio de Fourier y advertencias de convergencia (inflexión, oscilación, raíz múltiple) en Newton 1D.
- [x] **Bloque 2:** Gráfica de convergencia $\log_{10}$ en `N.L.` (`NewtonConvergenceChart.tsx`) para 1D y sistemas $2\times 2 / 3\times 3$, más comparador de dos $x_0$ en 1D.
- [x] **Bloque 3:** Sugerencia de reordenamiento automático de filas ($PAx = Pb$) para mejorar dominancia diagonal en `S.L.`.
- [x] **Bloque 4:** Conteo de operaciones aritméticas (FLOPs) en `L.U.`, `Jacobi`, `Gauss-Seidel` y `SOR`, con tabla comparativa y veredicto pedagógico automático.
- [x] **Bloque 5:** Aritmética finita extendida (`none` / `round` / `truncate`) en `S.L.` y `N.L.`, selector ampliado de decimales (`2, 3, 4, 6, 8`) y actualización SDD.
