# NumériLab — Laboratorio Visual de Análisis Numérico

Aplicación web interactiva desarrollada bajo metodología **SDD (Spec-Driven Development)** para el taller de **Análisis Numérico (`ING01199`)** del **Politécnico Colombiano Jaime Isaza Cadavid** (Programa de Ingeniería Informática, periodo `2026-2`).

Alineada con la concertación de evaluación ([FD-GC71.pdf](FD-GC71.pdf)) y los temas definidos en clase ([Captura de pantalla 2026-11-17 213107.png](Captura%20de%20pantalla%202026-11-17%20213107.png)), la aplicación integra paso a paso los métodos de los bloques **S.L. (Sistemas Lineales)** y **N.L. (No Lineales)**:

---

## Métodos implementados

### 1. Bloque `S.L.` — Sistemas de Ecuaciones Lineales ($Ax = b$)
- **Factorización L.U.** (Unidad 3, Sesión 14):
  - Estrategias **Doolittle** ($L_{ii} = 1$), **Crout** ($U_{ii} = 1$) y **Pivoteo Parcial ($PA = LU$)**.
  - Detalle de cada etapa de eliminación (multiplicadores $m_{ik}$ e intercambios de fila), matrices $P, L, U$, sustitución progresiva ($Ly = Pb$), sustitución regresiva ($Ux = y$) y residual $\|Ax - b\|_\infty$.
- **Método de Jacobi** (Unidad 3, Sesiones 15 y 18):
  - Forma puntual y forma matricial $x^{(k+1)} = T_j x^{(k)} + c_j$.
- **Método de Gauss-Seidel y SOR** (Unidad 2 Sesión 10 + Unidad 3 Sesiones 16 y 18):
  - Gauss-Seidel estándar ($\omega = 1$) y Sobrerrelajación Sucesiva **SOR** ($0 < \omega < 2$) en forma puntual y matricial $x^{(k+1)} = T_\omega x^{(k)} + c_\omega$.
  - Deslizador en vivo de $\omega$ y **optimizador automático** mediante barrido $\omega \in [0.10, 1.95]$ para descubrir $\omega^*$.
- **Diagnóstico y Comparador S.L.**:
  - Verificación de dominancia diagonal estricta/débil fila por fila y **sugerencia automática de reordenamiento de filas ($PAx = Pb$)** cuando mejora la dominancia o elimina ceros diagonales.
  - Normas $\|T\|_1, \|T\|_\infty$ y **radio espectral** $\rho(T) = \max|\lambda_i(T)|$.
  - Cuatro criterios de parada configurables (`abs_diff`, `rel_diff`, `rel_pct`, `residual`) y **aritmética finita por paso** (`none`, `round`, `truncate`) en `2, 3, 4, 6, 8` decimales.
  - Conteo real de operaciones aritméticas (**FLOPs**), **veredicto pedagógico automático**, gráfica comparativa en escala $\log_{10}$ y exportación en **CSV** y **LaTeX**.

### 2. Bloque `N.L.` — Ecuaciones y Sistemas No Lineales
- **Newton-Raphson 1D ($f(x) = 0$)** (Unidad 2, Sesión 11):
  - Primera y segunda derivada simbólica exacta ($f'(x)$ y $f''(x)$) calculadas con **SymPy**.
  - Evaluación de la **condición de Fourier** $f(x_0)f''(x_0)$, orden empírico de convergencia $p$ y alertas pedagógicas (puntos de inflexión, oscilación, raíces múltiples).
  - Plano cartesiano interactivo con **Mafs** (zoom, pan, rectas tangentes animadas paso a paso y **punto inicial $x_0$ arrastrable con el mouse**).
  - Gráfica de convergencia en escala $\log_{10}$ y **comparador opcional con un segundo $x_0$** sobre la misma ecuación.
- **Newton-Raphson Sistemas No Lineales $2\times 2$ y $3\times 3$ ($F(x) = 0$)** (Unidad 3, Sesión 19):
  - Matriz Jacobiana simbólica $J(x)$ generada con **SymPy**.
  - En cada iteración $k$, resuelve el sistema lineal $J(x^{(k)})\Delta x^{(k)} = -F(x^{(k)})$ **utilizando nuestro propio módulo de Factorización L.U. ($PA = LU$)**, conectando ambos bloques del curso.
  - Plano 2D interactivo con curvas implícitas $f_1(x,y) = 0$ y $f_2(x,y) = 0$, trayectoria paso a paso, selección de $(x_0, y_0)$ con clic directo y gráfica de convergencia $\log_{10}$ para $2\times 2$ y $3\times 3$.
- **Control numérico compartido en `N.L.`**:
  - Tres criterios de parada (`residual_and_step`, `rel_diff`, `rel_pct`), aritmética finita configurable (`none`, `round`, `truncate` a `2, 3, 4, 6, 8` decimales) y exportación en **CSV** / **LaTeX**.

---

## Arquitectura rápida

```text
analisis/
├── AGENTS.md                 # Contexto, reglas de ingeniería y estándares del proyecto
├── README.md                 # Guía general y comandos de arranque
├── start-dev.ps1             # Script para iniciar Backend + Frontend simultáneamente
├── docs/
│   ├── SPEC.md               # Especificación funcional y matemática (SDD)
│   ├── ARCHITECTURE.md       # Contratos REST, flujo de datos y módulos
│   ├── PLAN_IMPLEMENTACION.md # Plan de mejoras pedagógicas (Bloques 1-5)
│   ├── AUDITORIA-FUNCIONAL-ACADEMICA.md # Dictamen y evidencia de verificación
│   └── TASKS.md              # Checklist trazable de implementación y pruebas
├── backend/                  # Python 3.11+ / FastAPI + SymPy + NumPy (gestionado con uv)
│   ├── app/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── solvers/          # lu.py, iterative_linear.py, matrix_utils.py, math_expression.py, newton_raphson.py
│   └── tests/                # Pruebas unitarias y de API con pytest (76 tests)
└── frontend/                 # React + TypeScript + Vite + Tailwind + Motion + Mafs + KaTeX
    └── src/
        ├── components/       # Labs (S.L. / N.L.), gráficas interactivas, KaTeX, modales
        ├── data/             # Casos precargados del curso (Chapra / Burden)
        └── api/              # Cliente tipado hacia /api/*
```

---

## Requisitos y ejecución local

- **Python 3.11+** (probado en `3.12.13`) con **`uv`**
- **Node.js 20+** (probado en `v24.16.0`) con **`pnpm`** (`11.8.0`)

### Opción 1: Arranque automático (PowerShell en Windows)
```powershell
.\start-dev.ps1
```
Luego abre `http://localhost:5173`.

### Opción 2: Arranque manual en dos terminales
**Terminal 1 — Backend (`http://127.0.0.1:8000`):**
```powershell
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — Frontend (`http://localhost:5173`):**
```powershell
cd frontend
pnpm install
pnpm dev
```

---

## Pruebas y verificación

**Pruebas unitarias y de API del backend:**
```powershell
cd backend
uv run pytest -v
```

**Validación de tipos y build de producción del frontend:**
```powershell
cd frontend
pnpm build
```

---

## Documentación SDD
- [AGENTS.md](AGENTS.md) — Reglas del repositorio e invariantes matemáticas.
- [docs/SPEC.md](docs/SPEC.md) — Requisitos funcionales (`RF-01` a `RF-07`) y definiciones matemáticas.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Contratos de endpoints REST y estructura de módulos.
- [docs/PLAN_IMPLEMENTACION.md](docs/PLAN_IMPLEMENTACION.md) — Plan de mejoras pedagógicas (Bloques 1-5).
- [docs/AUDITORIA-FUNCIONAL-ACADEMICA.md](docs/AUDITORIA-FUNCIONAL-ACADEMICA.md) — Dictamen y evidencia de verificación para sustentación.
- [docs/TASKS.md](docs/TASKS.md) — Plan de tareas verificables.
