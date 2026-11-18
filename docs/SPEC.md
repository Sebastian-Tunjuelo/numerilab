# SPEC.md — Especificación Funcional y Matemática (SDD)

## 1. Alcance y Referencias
- **Fuente académica:** [FD-GC71.pdf](../FD-GC71.pdf) (Unidades 1, 2 y 3) y [Captura de pantalla 2026-11-17 213107.png](../Captura%20de%20pantalla%202026-11-17%20213107.png).
- **Dimensiones soportadas:** Sistemas lineales $n \times n$ con $2 \le n \le 6$; ecuaciones no lineales 1D y sistemas no lineales $2 \times 2$ y $3 \times 3$.

---

## 2. Requisitos Funcionales

### RF-01: Factorización L.U. (`S.L.`)
Dado un sistema $Ax = b$:
- Debe soportar tres variantes seleccionables:
  1. **Doolittle:** $L$ triangular inferior con $L_{ii} = 1$, $U$ triangular superior.
  2. **Crout:** $L$ triangular inferior, $U$ triangular superior con $U_{ii} = 1$.
  3. **Pivoteo Parcial ($PA = LU$):** intercambio de filas para ubicar el mayor pivote en valor absoluto $|a_{rk}|$ y evitar divisiones por cero.
- **Salidas requeridas:**
  - Matrices $P, L, U$ y su representación LaTeX.
  - Lista cronológica de pasos de eliminación (intercambios de fila, multiplicadores $m_{ik}$ y estados parciales).
  - Vector intermedio $y$ (sustitución progresiva $Ly = Pb$) y ecuaciones paso a paso.
  - Vector solución $x$ (sustitución regresiva $Ux = y$) y ecuaciones paso a paso.
  - Residual final $\|Ax - b\|_\infty$.
  - Conteo real de operaciones aritméticas escalares (`operation_count`: multiplicaciones/divisiones, sumas/restas, `total_flops` y cota teórica $\approx \frac{2}{3}n^3 + 2n^2$) y soporte opcional de aritmética finita (`round` / `truncate` a $n$ decimales) registrando la diferencia frente a precisión completa (`finite_arithmetic_diff`).
- **Casos límite.** Si en Doolittle o Crout sin pivoteo aparece un pivote $|p| < 10^{-14}$, retornar `zero_pivot_requires_pivoting`. En pivoteo parcial, comparar el pivote con la escala de su fila original. Retornar `singular_matrix` si la fila es nula o el cociente entre el pivote absoluto y el máximo absoluto de esa fila es inferior o igual a $n\epsilon_{\rm máquina}$.

### RF-02: Método de Jacobi (`S.L.`)
Dado $Ax = b$, vector inicial $x^{(0)}$, tolerancia $\varepsilon$ y máximo de iteraciones $M$:
- Ecuación puntual por componente:
  $$x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j \ne i} a_{ij} x_j^{(k)} \right)$$
- Forma matricial explícita (Sesión 18):
  $$A = D - L_{\text{estricta}} - U_{\text{estricta}}, \quad T_j = D^{-1}(L_{\text{estricta}} + U_{\text{estricta}}), \quad c_j = D^{-1} b$$
- Debe registrar en cada iteración $k$: $x^{(k)}$, error absoluto $\|x^{(k)} - x^{(k-1)}\|_\infty$, error relativo (`rel_error`), error relativo porcentual (`rel_error_pct`), residual $\|Ax^{(k)} - b\|_\infty$ y las ecuaciones sustituidas por variable para el inspector paso a paso.

### RF-03: Método de Gauss-Seidel y SOR (`S.L.`)
Dado $Ax = b$, $x^{(0)}$, parámetro de relajación $\omega \in (0, 2)$, tolerancia $\varepsilon$ y máximo de iteraciones $M$:
- Ecuación puntual:
  $$\tilde{x}_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)} \right)$$
  $$x_i^{(k+1)} = (1 - \omega) x_i^{(k)} + \omega \tilde{x}_i^{(k+1)}$$
  *(Cuando $\omega = 1$ equivale exactamente al método de Gauss-Seidel clásico).*
- Forma matricial explícita:
  $$T_\omega = (D - \omega L_{\text{estricta}})^{-1} \left[(1 - \omega)D + \omega U_{\text{estricta}}\right], \quad c_\omega = \omega (D - \omega L_{\text{estricta}})^{-1} b$$
- El barrido usa las mismas opciones de parada y redondeo por paso que el análisis actual.
- **Optimizador de $\omega$:** debe existir un endpoint que barra $\omega \in [0.10, 1.95]$ en pasos de $0.05$, reportando para cada valor si converge, el número de iteraciones y $\rho(T_\omega)$, destacando el $\omega^*$ óptimo.

### RF-04: Diagnóstico Matricial y Criterios de Parada (`S.L.`)
- Evaluar dominancia diagonal estricta por fila: $|a_{ii}| > \sum_{j \ne i} |a_{ij}|$, y sugerir automáticamente un reordenamiento de filas $PAx = Pb$ (`reordering_suggestion`) cuando la matriz tenga ceros diagonales o no sea estrictamente dominante y exista una permutación que mejore la condición sin dejar ceros en la diagonal.
- Calcular $\|T\|_1$, $\|T\|_\infty$ y radio espectral $\rho(T) = \max_i |\lambda_i(T)|$ para $T_j$, $T_g$ y $T_\omega$.
- Soportar 4 criterios de parada seleccionables:
  1. `abs_diff` — Diferencia absoluta máxima: $\|x^{(k)} - x^{(k-1)}\|_\infty < \varepsilon$ ($0 < \varepsilon \le 1$).
  2. `rel_diff` — Error relativo aproximado: $\frac{\|x^{(k)} - x^{(k-1)}\|_\infty}{\|x^{(k)}\|_\infty} < \varepsilon$ para $\|x^{(k)}\|_\infty > 0$ (resguardado sin parada cuando $\|x^{(k)}\|_\infty = 0$ y el paso es no nulo; $0$ si el paso es nulo), con $0 < \varepsilon \le 1$.
  3. `rel_pct` — Error relativo porcentual: $e_{\text{rel}}(\%) = 100 \cdot e_{\text{rel}} < \varepsilon$ (con el mismo resguardo cuando $\|x^{(k)}\|_\infty = 0$), con $0 < \varepsilon < 100$.
  4. `residual` — Residual del sistema: $\|Ax^{(k)} - b\|_\infty < \varepsilon$ ($0 < \varepsilon \le 1$).
- Soportar aritmética finita configurable a $n \in \{2, 3, 4, 6, 8\}$ decimales (Unidad 1, Sesión 6) en modos `none`, `round` o `truncate`, aplicado por paso cuando está activo. El comparador muestra el vector usado para calcular el residual. Si el redondeo/truncamiento por paso deja fijo el iterado y el residual sigue fuera de tolerancia, retornar `stagnated`.
- Registrar `operation_count` ($\mathcal{O}(k \cdot n^2)$ FLOPs reales) por método y generar un **Veredicto Pedagógico** (`verdict`) que compare L.U. frente a Jacobi, Gauss-Seidel y SOR.

### RF-05: Método de Newton-Raphson 1D (`N.L.`)
Dada una expresión $f(x)$ y un valor inicial $x_0$:
- Calcular simbólicamente $f'(x)$ y $f''(x)$ con `SymPy` y generar su LaTeX (`df_latex`, `d2f_latex`).
- Evaluar la condición inicial de Fourier $f(x_0)f''(x_0)$ (`fourier_condition`) y generar alertas pedagógicas (`convergence_warnings`) ante $f(x_0)f''(x_0) \le 0$, cercanía a puntos de inflexión ($|f''(x_k)| < 10^{-8}$), alternancia oscilatoria en $\Delta x_k$ o convergencia lineal ($p < 1.3$).
- Iterar mediante:
  $$x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}$$
  con soporte opcional de aritmética finita (`round` / `truncate` a $n$ decimales).
- Soportar 3 criterios de parada seleccionables (`criterion`):
  1. `residual_and_step` *(por defecto)*: acepta un residual evaluado igual a cero tras descartar subdesbordamiento; para residual no nulo exige $|f(x)| \le \varepsilon$ y una corrección $|\Delta x| \le \max(\varepsilon, 4\,\text{ulp}(x))$, con $0 < \varepsilon \le 1$.
  2. `rel_diff`: error relativo aproximado $|e_{\text{rel}}| = \frac{|x_{k+1} - x_k|}{|x_{k+1}|} \le \varepsilon$ (con $0 < \varepsilon \le 1$).
  3. `rel_pct`: error relativo porcentual $e_{\text{rel}}(\%) = 100\,|e_{\text{rel}}| \le \varepsilon$ (con $0 < \varepsilon < 100$).
- Si todavía se requiere una corrección y $|f'(x_k)| < 10^{-14}$, retornar `zero_derivative`.
- Registrar `d2fx_k`, `rel_error` y `rel_error_pct` en cada paso, y estimar el orden empírico de convergencia `estimated_order` ($p \approx \frac{\ln|\Delta x_{k+1}/\Delta x_k|}{\ln|\Delta x_k/\Delta x_{k-1}|}$) con las últimas 3 iteraciones consecutivas válidas (para contrastar convergencia cuadrática $p \approx 2$ en raíces simples frente a lineal $p \approx 1$ en raíces múltiples).
- Retornar los puntos de cada recta tangente $(x_k, f(x_k)) \to (x_{k+1}, 0)$ para visualización geométrica interactiva en `Mafs` con $x_0$ arrastrable, gráfica de convergencia en escala $\log_{10}$ y comparador opcional con un segundo punto inicial $x_0^{(B)}$.

### RF-06: Newton-Raphson para Sistemas No Lineales $2\times 2$ y $3\times 3$ (`N.L.`)
Dado un vector de ecuaciones $F(x) = [f_1, \dots, f_n]^T = 0$ y $x^{(0)}$:
- Calcular simbólicamente la matriz Jacobiana $J_{ij}(x) = \frac{\partial f_i}{\partial x_j}$ con `SymPy`.
- En cada iteración $k$:
  1. Evaluar $F(x^{(k)})$ y $J(x^{(k)})$.
  2. Resolver $J(x^{(k)}) \Delta x^{(k)} = -F(x^{(k)})$ llamando a `solve_lu(..., strategy="partial_pivot")` (propagando el modo de aritmética finita si está activo).
  3. Actualizar $x^{(k+1)} = x^{(k)} + \Delta x^{(k)}$.
  4. Exponer los factores $P, L, U$ y $\Delta x^{(k)}$ dentro del detalle de la iteración $k$, junto con `rel_error` y `rel_error_pct` (calculados con $\|\Delta x^{(k)}\|_\infty / \|x^{(k+1)}\|_\infty$).
- Soportar los mismos 3 criterios de parada de Newton 1D (`residual_and_step`, `rel_diff`, `rel_pct`) aplicados a las normas infinito del residual y la corrección, incluido el punto inicial en `residual_and_step`, así como la gráfica de convergencia $\log_{10}$ para sistemas $2\times 2$ y $3\times 3$.
- Para sistemas $2\times 2$, aproximar las curvas $f_1(x,y)=0$ y $f_2(x,y)=0$ mediante *marching squares*. Verificar que el residual disminuya hacia cero en los cruces. Para una potencia entera positiva explícita de toda la expresión, usar su base para buscar los ceros. La malla puede omitir otros ceros sin cambio de signo o detalles menores que su resolución.

### RF-07: Comparador, Exportación y Guía Teórica
- Gráfica en escala logarítmica ($\log_{10}(\text{error})$ vs. iteración) comparando Jacobi, Gauss-Seidel ($\omega=1$) y SOR ($\omega$ actual), tabla comparativa de FLOPs y veredicto pedagógico automático en `S.L.`, y gráfica de convergencia $\log_{10}$ (con comparador opcional de dos $x_0$ en 1D) en `N.L.`.
- Copiado rápido en formato **CSV** y **LaTeX** para tablas iterativas (`S.L.`), **Factorización L.U.** (matrices $P, L, U$, vectores y etapas) y **Newton-Raphson** (`N.L.` 1D y Sistemas).
- Casos precargados del curso (convergentes, pivoteo requerido, SOR eficiente, divergencia por falta de dominancia diagonal, 1D trascendentales, ejercicios de `NEWTON-RAPHSON.pdf` como $x^3 - 5x^2 + 7x - 3$ en raíz doble/simple, $e^{-x} - \ln(x)$, $e^{3x} - 4$, y sistemas $2\times 2 / 3\times 3$).
- Panel/modal de **Guía Teórica** con las fórmulas de consulta rápida para sustentación, incluyendo la condición de convergencia $f'(x) \neq 0$, $f''(x) \neq 0$ (sin puntos de inflexión cercanos), degradación a orden lineal en raíces múltiples y la aclaración de redondeo por paso (Sesión 6).
