import type { LUStrategy, NewtonStoppingCriterion } from '../types/numerical';

export interface LinearPreset {
  id: string;
  title: string;
  subtitle: string;
  tag: string;
  A: number[][];
  b: number[];
  x0: number[];
  omega: number;
  luStrategy: LUStrategy;
}

export interface Newton1DPreset {
  id: string;
  title: string;
  subtitle: string;
  expression: string;
  x0: number;
  compareX0?: number;
  stopping?: { criterion: NewtonStoppingCriterion; tol: number };
}

export interface NewtonSystemPreset {
  id: string;
  title: string;
  subtitle: string;
  variables: string[];
  equations: string[];
  x0: number[];
}

export const LINEAR_PRESETS: LinearPreset[] = [
  {
    id: 'chapra-dom-3x3',
    title: 'Sistema 3×3 Diagonalmente Dominante',
    subtitle: 'Solución exacta [1, 2, -1]. Todos los métodos convergen limpiamente.',
    tag: 'Convergente',
    A: [
      [10, -1, 2],
      [-1, 11, -1],
      [2, -1, 10],
    ],
    b: [6, 22, -10],
    x0: [0, 0, 0],
    omega: 1.1,
    luStrategy: 'partial_pivot',
  },
  {
    id: 'sor-tridiag-4x4',
    title: 'Red Tridiagonal 4×4 (Ventaja SOR)',
    subtitle: 'Compara las iteraciones y prueba el barrido para ver cómo cambia el resultado con ω.',
    tag: 'Ventaja SOR',
    A: [
      [4, -1, 0, 0],
      [-1, 4, -1, 0],
      [0, -1, 4, -1],
      [0, 0, -1, 3],
    ],
    b: [15, 10, 10, 10],
    x0: [0, 0, 0, 0],
    omega: 1.05,
    luStrategy: 'partial_pivot',
  },
  {
    id: 'lu-zero-pivot',
    title: 'Pivote Nulo a₁₁ = 0 (Demostración PA = LU)',
    subtitle: 'Falla en Doolittle/Crout puros y en Jacobi sin reordenar; PA = LU lo resuelve.',
    tag: 'Pivoteo L.U.',
    A: [
      [0, 2, 3],
      [4, 6, 7],
      [2, -3, 6],
    ],
    b: [8, -3, 5],
    x0: [0, 0, 0],
    omega: 1.0,
    luStrategy: 'doolittle',
  },
  {
    id: 'divergent-non-dom',
    title: 'Matriz No Dominante (Límite Iterativo)',
    subtitle: 'Radio espectral ρ(T) > 1: Jacobi y Gauss-Seidel divergen mientras L.U. halla la solución.',
    tag: 'Divergencia',
    A: [
      [1, 3, -2],
      [3, 2, 1],
      [2, -1, 1],
    ],
    b: [5, 10, 3],
    x0: [0, 0, 0],
    omega: 1.2,
    luStrategy: 'partial_pivot',
  },
];

export const NEWTON_1D_PRESETS: Newton1DPreset[] = [
  {
    id: 'cubic-classic',
    title: 'Polinomio cúbico (x³ - 2x - 5)',
    subtitle: 'Raíz real clásica de referencia (~2.09455).',
    expression: 'x^3 - 2*x - 5',
    x0: 2.0,
  },
  {
    id: 'cos-fixed',
    title: 'Trascendental (cos(x) - x)',
    subtitle: 'Punto fijo del coseno (~0.739085).',
    expression: 'cos(x) - x',
    x0: 0.5,
  },
  {
    id: 'exp-chapra',
    title: 'Exponencial (e^(-x) - x)',
    subtitle: 'Ejemplo clásico de Chapra y Canale (~0.567143).',
    expression: 'e^(-x) - x',
    x0: 0.0,
  },
  {
    id: 'near-flat-tangent',
    title: 'Tangente casi horizontal (Caso crítico)',
    subtitle: 'Arranca cerca de un mínimo local (x₀ = 0.99) y salta lejos antes de estabilizarse.',
    expression: 'x^3 - 3*x + 1',
    x0: 0.99,
  },
  {
    id: 'prof-poly-double',
    title: 'Raíz doble: x³ - 5x² + 7x - 3',
    subtitle: 'NEWTON-RAPHSON.pdf: desde x₀ = 0.5 converge a la raíz doble x = 1 con orden lineal (p ≈ 1).',
    expression: 'x^3 - 5*x^2 + 7*x - 3',
    x0: 0.5,
    compareX0: 4,
  },
  {
    id: 'prof-poly-simple',
    title: 'Raíz simple: x³ - 5x² + 7x - 3',
    subtitle: 'NEWTON-RAPHSON.pdf: desde x₀ = 4 converge a la raíz simple x = 3 con orden cuadrático (p ≈ 2).',
    expression: 'x^3 - 5*x^2 + 7*x - 3',
    x0: 4,
    compareX0: 0.5,
  },
  {
    id: 'prof-exp-ln',
    title: 'Exponencial y logaritmo (e⁻ˣ - ln(x))',
    subtitle: 'Ejercicio de NEWTON-RAPHSON.pdf evaluado con error relativo porcentual e_rel < 1%.',
    expression: 'exp(-x) - ln(x)',
    x0: 1,
    stopping: { criterion: 'rel_pct', tol: 1 },
  },
  {
    id: 'prof-exp3x',
    title: 'Exponencial (e³ˣ - 4)',
    subtitle: 'NEWTON-RAPHSON.pdf en [0, 1] (tol = 0.001): prueba de Fourier en [0, 1] (x₀ = 1 cumple, x₀ = 0 no; compara x₀ = 0.5 vs x₀ = 0).',
    expression: 'exp(3*x) - 4',
    x0: 0.5,
    compareX0: 0,
    stopping: { criterion: 'rel_diff', tol: 0.001 },
  },
];

export const NEWTON_SYSTEM_PRESETS: NewtonSystemPreset[] = [
  {
    id: 'circle-exp-2x2',
    title: 'Círculo y Exponencial (2×2)',
    subtitle: 'Intersección de x² + y² = 4 con eˣ + y = 1.',
    variables: ['x', 'y'],
    equations: ['x^2 + y^2 - 4', 'exp(x) + y - 1'],
    x0: [1.0, -1.5],
  },
  {
    id: 'burden-2x2',
    title: 'Sistema Cuadrático (2×2 · Burden)',
    subtitle: 'Solución exacta en (2, 3) con convergencia cuadrática.',
    variables: ['x', 'y'],
    equations: ['x^2 + x*y - 10', 'y + 3*x*y^2 - 57'],
    x0: [1.5, 3.5],
  },
  {
    id: 'burden-3x3',
    title: 'Sistema No Lineal 3×3 (Burden Cap. 10)',
    subtitle: 'Sistema acoplado trigonométrico y exponencial resuelto con Jacobiano + L.U.',
    variables: ['x', 'y', 'z'],
    equations: [
      '3*x - cos(y*z) - 0.5',
      'x^2 - 81*(y + 0.1)^2 + sin(z) + 1.06',
      'exp(-x*y) + 20*z + (10*pi - 3)/3',
    ],
    x0: [0.1, 0.1, -0.1],
  },
];
