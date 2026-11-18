export type StoppingCriterion = 'abs_diff' | 'rel_diff' | 'rel_pct' | 'residual';
export type NewtonStoppingCriterion = 'residual_and_step' | 'rel_diff' | 'rel_pct';
export type LUStrategy = 'partial_pivot' | 'doolittle' | 'crout';
export type RoundingMode = 'none' | 'round' | 'truncate';

export interface DominanceRowAnalysis {
  row: number;
  diag_abs: number;
  off_diag_sum: number | null;
  is_strictly_dominant: boolean;
  is_weakly_dominant: boolean;
}

export interface ReorderingSuggestion {
  improves: boolean;
  permutation: number[];
  is_strictly_dominant: boolean;
  is_weakly_dominant: boolean;
  has_zero_diagonal: boolean;
  A_reordered: number[][];
  b_reordered: number[];
}

export interface DominanceInfo {
  is_strictly_dominant: boolean;
  is_weakly_dominant: boolean;
  has_zero_diagonal: boolean;
  rows_analysis: DominanceRowAnalysis[];
  reordering_suggestion?: ReorderingSuggestion | null;
}

export interface OperationCount {
  factorization_mul_div?: number;
  factorization_add_sub?: number;
  substitution_mul_div?: number;
  substitution_add_sub?: number;
  per_iteration_flops?: number;
  mul_div: number;
  add_sub: number;
  total_flops: number;
  theoretical_flops?: number;
  complexity_latex: string;
}

export interface LinearPedagogicalVerdict {
  best_iterative: string | null;
  lowest_flops_method: string | null;
  summary: string;
  notes: string[];
}

export interface IterationMatrixDetail {
  available: boolean;
  name?: string;
  reason?: string;
  T?: number[][];
  c?: number[];
  T_latex?: string;
  c_latex?: string;
  norm_1?: number | null;
  norm_inf?: number | null;
  spectral_radius?: number | null;
  guaranteed_convergence?: boolean;
}

export interface LinearDiagnostics {
  dominance: DominanceInfo;
  matrices: {
    split: {
      D_latex: string;
      L_latex: string;
      U_latex: string;
    };
    jacobi: IterationMatrixDetail;
    gauss_seidel: IterationMatrixDetail;
    sor: IterationMatrixDetail;
  };
}

export interface LUStep {
  step: number;
  title: string;
  description: string;
  latex_operation: string;
  pivot_swap: number[] | null;
  L: number[][];
  U: number[][];
  P: number[][];
  L_latex: string;
  U_latex: string;
  P_latex: string;
}

export interface LUResult {
  strategy: LUStrategy;
  status: string;
  message: string;
  rounding_mode?: RoundingMode;
  finite_arithmetic_diff?: number | null;
  operation_count?: OperationCount;
  L?: number[][];
  U?: number[][];
  P?: number[][];
  L_latex?: string;
  U_latex?: string;
  P_latex?: string;
  pb?: number[];
  pb_latex?: string;
  y?: number[];
  y_latex?: string;
  x?: number[];
  x_raw?: number[];
  x_latex?: string;
  forward_equations?: string[];
  backward_equations?: string[];
  steps: LUStep[];
  residual_norm?: number;
}

export interface IterativeStep {
  iteration: number;
  x: number[];
  abs_error: number;
  rel_error: number | null;
  rel_error_pct: number | null;
  residual: number;
  variable_steps: string[];
}

export interface IterativeLinearResult {
  method: 'jacobi' | 'gauss_seidel' | 'sor';
  omega: number;
  status: string;
  message: string;
  rounding_mode?: RoundingMode;
  operation_count?: OperationCount;
  iterations: number;
  steps: IterativeStep[];
  solution: number[] | null;
  solution_raw?: number[] | null;
  solution_latex?: string | null;
  residual_norm: number | null;
}

export interface LinearAnalyzeResponse {
  diagnostics: LinearDiagnostics;
  lu: LUResult;
  jacobi: IterativeLinearResult;
  gauss_seidel: IterativeLinearResult;
  sor: IterativeLinearResult;
  verdict?: LinearPedagogicalVerdict;
}

export interface SORSweepPoint {
  omega: number;
  converged: boolean;
  iterations: number;
  spectral_radius: number | null;
}

export interface SORSweepResponse {
  available: boolean;
  reason?: string;
  points: SORSweepPoint[];
  optimal_omega: number | null;
  optimal_iterations: number | null;
  optimal_spectral_radius?: number | null;
}

export interface FourierCondition {
  x0: number;
  fx_0: number | null;
  d2fx_0: number | null;
  product: number | null;
  satisfied: boolean;
}

export interface Newton1DStep {
  iteration: number;
  x_k: number;
  fx_k: number;
  dfx_k: number;
  d2fx_k?: number | null;
  fx_next?: number | null;
  x_next: number;
  x_next_raw: number;
  x_k_raw: number;
  abs_error: number;
  rel_error: number | null;
  rel_error_pct?: number | null;
  step_latex: string;
}

export interface Newton1DResponse {
  status: string;
  message: string;
  criterion?: NewtonStoppingCriterion;
  estimated_order?: number | null;
  expression_normalized?: string;
  f_latex: string;
  df_latex: string;
  d2f_latex?: string;
  fourier_condition?: FourierCondition | null;
  convergence_warnings?: string[];
  rounding_mode?: RoundingMode;
  root: number | null;
  root_raw?: number | null;
  iterations: number;
  steps: Newton1DStep[];
  plot_bounds?: {
    x_min: number;
    x_max: number;
    y_min: number;
    y_max: number;
  };
  sample_points?: Array<{ x: number; y: number | null }>;
}

export interface NewtonSystemStep {
  iteration: number;
  x_k: number[];
  x_next: number[];
  x_next_raw: number[];
  F_eval: number[];
  J_eval: number[][];
  J_eval_latex: string;
  neg_F_latex: string;
  dx: number[];
  abs_error: number;
  rel_error: number | null;
  rel_error_pct?: number | null;
  residual: number | null;
  lu_detail: {
    P_latex: string;
    L_latex: string;
    U_latex: string;
    y_latex: string;
    dx_latex: string;
  };
}

export interface Contour2DData {
  bounds: {
    x_min: number;
    x_max: number;
    y_min: number;
    y_max: number;
  };
  f1_segments: Array<[number[], number[]]>;
  f2_segments: Array<[number[], number[]]>;
}

export interface NewtonSystemResponse {
  status: string;
  message: string;
  criterion?: NewtonStoppingCriterion;
  estimated_order?: number | null;
  rounding_mode?: RoundingMode;
  variables?: string[];
  f_latex_list: string[];
  jacobian_latex: string;
  solution: number[] | null;
  iterations: number;
  steps: NewtonSystemStep[];
  contour_2d?: Contour2DData | null;
}
