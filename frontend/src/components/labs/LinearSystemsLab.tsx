import { useCallback, useEffect, useId, useRef, useState } from 'react';
import {
  AlertTriangle,
  ArrowUpDown,
  Check,
  CheckCircle2,
  Copy,
  Play,
  Shuffle,
  SlidersHorizontal,
  Sparkles,
} from 'lucide-react';
import { analyzeLinearSystem, sweepSorOmega } from '../../api/client';
import { LINEAR_PRESETS } from '../../data/presets';
import type {
  IterativeLinearResult,
  IterationMatrixDetail,
  LinearAnalyzeResponse,
  LUResult,
  LUStrategy,
  ReorderingSuggestion,
  RoundingMode,
  SORSweepResponse,
  StoppingCriterion,
} from '../../types/numerical';
import { ConvergenceChart } from '../charts/ConvergenceChart';
import { OmegaSweepChart } from '../charts/OmegaSweepChart';
import { LatexRenderer } from '../math/LatexRenderer';
import { MatrixDisplay } from '../math/MatrixDisplay';

type SLTab = 'compare' | 'lu' | 'jacobi' | 'gauss_seidel' | 'sor';

function formatLUStrategyLabel(strategy: LUStrategy, short = false): string {
  if (strategy === 'partial_pivot') return short ? 'PA = LU' : 'Pivoteo Parcial PA = LU';
  if (strategy === 'doolittle') return 'Doolittle';
  return 'Crout';
}

function formatIterativeMessage(message: string): string {
  return message
    .replace(/'abs_diff'/g, 'diferencia absoluta ||x⁽ᵏ⁾ − x⁽ᵏ⁻¹⁾||∞')
    .replace(/'rel_diff'/g, 'error relativo aproximado')
    .replace(/'rel_pct'/g, 'error relativo porcentual e_rel (%)')
    .replace(/'residual'/g, 'residual ||Ax⁽ᵏ⁾ − b||∞');
}

function formatRelPct(pct?: number | null): string {
  if (pct === null || pct === undefined || !Number.isFinite(pct)) return '—';
  if (pct === 0) return '0%';
  if (pct < 0.001) return `${pct.toExponential(2)}%`;
  return `${pct.toFixed(4)}%`;
}

function formatRelPctLatex(pct?: number | null): string {
  if (pct === null || pct === undefined || !Number.isFinite(pct)) return '\\text{—}';
  if (pct === 0) return '0\\%';
  if (pct < 0.001) return `${pct.toExponential(2)}\\%`;
  return `${pct.toFixed(4)}\\%`;
}

function formatStatusBadge(status: string) {
  if (status === 'converged') {
    return {
      label: 'Convergió',
      cls: 'bg-emerald-500/15 text-[var(--text-ok)] border-emerald-500/30',
    };
  }
  if (status === 'zero_pivot_requires_pivoting') {
    return {
      label: 'Requiere pivoteo',
      cls: 'bg-amber-500/15 text-[var(--text-nl)] border-amber-500/30',
    };
  }
  if (status === 'stagnated') {
    return {
      label: 'Estancado',
      cls: 'bg-amber-500/15 text-[var(--text-nl)] border-amber-500/30',
    };
  }
  if (status === 'max_iterations') {
    return {
      label: 'Máx. iteraciones',
      cls: 'bg-amber-500/15 text-[var(--text-nl)] border-amber-500/30',
    };
  }
  if (status === 'singular_matrix') {
    return {
      label: 'Matriz singular',
      cls: 'bg-rose-500/15 text-[var(--text-err)] border-rose-500/30',
    };
  }
  return {
    label: status === 'diverged' ? 'Divergió' : status === 'zero_diagonal' ? 'Diagonal nula' : 'No convergió',
    cls: 'bg-rose-500/15 text-[var(--text-err)] border-rose-500/30',
  };
}

function buildCsv(result: IterativeLinearResult): string {
  if (!result.steps.length) return '';
  const n = result.steps[0].x.length;
  const headers = ['Iteracion', ...Array.from({ length: n }, (_, i) => `x${i + 1}`), 'Error_Abs', 'Error_Rel', 'e_rel_%', 'Residual'];
  const rows = result.steps.map((s) =>
    [s.iteration, ...s.x, s.abs_error, s.rel_error, s.rel_error_pct, s.residual].join(','),
  );
  return [headers.join(','), ...rows].join('\n');
}

function buildLatexTable(result: IterativeLinearResult): string {
  if (!result.steps.length) return '';
  const n = result.steps[0].x.length;
  const cols = 'c|' + 'r'.repeat(n) + '|r|r|r|r';
  const headerVars = Array.from({ length: n }, (_, i) => `x_${i + 1}`).join(' & ');
  const lines = [
    `\\begin{array}{${cols}}`,
    `k & ${headerVars} & \\|x^{(k)}-x^{(k-1)}\\|_\\infty & \\text{Error relativo} & e_{\\text{rel}}\\,(\\%) & \\|Ax^{(k)}-b\\|_\\infty \\\\ \\hline`,
    ...result.steps.map(
      (s) =>
        `${s.iteration} & ${s.x.join(' & ')} & ${s.abs_error.toExponential(2)} & ${s.rel_error?.toExponential(2) ?? '\\text{—}'} & ${formatRelPctLatex(s.rel_error_pct)} & ${s.residual.toExponential(2)} \\\\`,
    ),
    '\\end{array}',
  ];
  return lines.join('\n');
}

function csvCell(value: string | number): string {
  const str = String(value);
  return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
}

function buildLUCsv(lu: LUResult): string {
  const width = lu.P?.length ?? lu.steps[0]?.P.length ?? 1;
  const rows: (string | number)[][] = [
    ['Seccion', 'Detalle', ...Array.from({ length: width }, (_, i) => 'Valor_' + (i + 1))],
    ['Resumen', 'Estrategia', formatLUStrategyLabel(lu.strategy)],
    ['Resumen', 'Estado', lu.status],
  ];
  if (lu.residual_norm !== undefined) {
    rows.push(['Resumen', 'Residual_inf', lu.residual_norm]);
  }
  const pushMatrix = (name: string, mat?: number[][]) => {
    if (!mat) return;
    mat.forEach((row, i) => rows.push([name, `Fila_${i + 1}`, ...row]));
  };
  pushMatrix('P', lu.P);
  pushMatrix('L', lu.L);
  pushMatrix('U', lu.U);
  if (lu.pb) rows.push(['Vector', 'Pb', ...lu.pb]);
  if (lu.y) rows.push(['Vector', 'y', ...lu.y]);
  if (lu.x) rows.push(['Vector', 'x', ...lu.x]);
  lu.steps.forEach((step) => {
    const section = `Paso_${step.step}`;
    rows.push([section, step.title, step.description], [section, 'Operacion LaTeX', step.latex_operation]);
  });
  lu.forward_equations?.forEach((equation, i) => rows.push(['Ly=Pb', i + 1, equation]));
  lu.backward_equations?.forEach((equation, i) => rows.push(['Ux=y', i + 1, equation]));
  return rows
    .map((row) => [...row.map(csvCell), ...Array(Math.max(0, width + 2 - row.length)).fill('')].join(','))
    .join('\n');
}

function buildLULatex(lu: LUResult): string {
  const blocks: string[] = [];
  if (lu.P_latex && lu.L_latex && lu.U_latex) {
    blocks.push(`P = ${lu.P_latex},\\quad L = ${lu.L_latex},\\quad U = ${lu.U_latex}`);
  } else if (lu.L_latex && lu.U_latex) {
    blocks.push(`L = ${lu.L_latex},\\quad U = ${lu.U_latex}`);
  }
  if (lu.pb_latex && lu.y_latex && lu.x_latex) {
    blocks.push(`Pb = ${lu.pb_latex},\\quad y = ${lu.y_latex},\\quad x = ${lu.x_latex}`);
  }
  if (lu.steps.length > 0) {
    blocks.push('% Etapas de eliminación');
    lu.steps.forEach((step) => {
      blocks.push(`% Paso ${step.step}: ${step.title}\n${step.latex_operation}`);
    });
  }
  if (lu.forward_equations?.length) {
    blocks.push('% Sustitución progresiva (Ly = Pb)\n' + lu.forward_equations.join(' \\\\\n'));
  }
  if (lu.backward_equations?.length) {
    blocks.push('% Sustitución regresiva (Ux = y)\n' + lu.backward_equations.join(' \\\\\n'));
  }
  if (lu.residual_norm !== undefined) {
    blocks.push(`% Residual ||Ax - b||_inf = ${lu.residual_norm.toExponential(3)}`);
  }
  return blocks.join('\n\n');
}

interface LUPanelProps {
  lu: LUResult;
  onEnablePartialPivot: () => void;
}

function LUPanel({ lu, onEnablePartialPivot }: LUPanelProps) {
  const [copiedType, setCopiedType] = useState<'csv' | 'latex' | null>(null);
  const [copyError, setCopyError] = useState(false);
  const copyTimer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(copyTimer.current), []);

  const handleCopy = async (kind: 'csv' | 'latex') => {
    const content = kind === 'csv' ? buildLUCsv(lu) : buildLULatex(lu);
    if (!content) return;
    setCopyError(false);
    try {
      await navigator.clipboard.writeText(content);
    } catch {
      setCopiedType(null);
      setCopyError(true);
      return;
    }
    window.clearTimeout(copyTimer.current);
    setCopiedType(kind);
    copyTimer.current = window.setTimeout(() => setCopiedType(null), 1800);
  };

  return (
    <div className="space-y-4">
      {copyError ? (
        <p role="alert" className="text-sm text-[var(--text-err)]">
          No se pudo copiar el procedimiento L.U. Comprueba el acceso al portapapeles e inténtalo de nuevo.
        </p>
      ) : null}
      <div className="panel-surface p-4 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-base font-bold">
                Factorización L.U. ({formatLUStrategyLabel(lu.strategy)})
              </h3>
              <span className={`text-xs px-2.5 py-0.5 rounded-full border ${formatStatusBadge(lu.status).cls}`}>
                {formatStatusBadge(lu.status).label}
              </span>
            </div>
            <p className="text-xs text-[var(--text-secondary)] mt-0.5">{lu.message}</p>
          </div>

          <div className="flex flex-wrap gap-2">
            {lu.status === 'zero_pivot_requires_pivoting' ? (
              <button
                type="button"
                onClick={onEnablePartialPivot}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500 text-slate-950 hover:bg-amber-400 cursor-pointer"
              >
                <Sparkles className="h-3.5 w-3.5" />
                Activar Pivoteo Parcial (PA = LU)
              </button>
            ) : null}
            <button
              type="button"
              onClick={() => handleCopy('csv')}
              disabled={!lu.L && !lu.steps.length}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium elevated-surface hover:border-[var(--border-strong)] disabled:opacity-40 cursor-pointer"
            >
              {copiedType === 'csv' ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
              Copiar CSV
            </button>
            <button
              type="button"
              onClick={() => handleCopy('latex')}
              disabled={!lu.L_latex && !lu.steps.length}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium elevated-surface hover:border-[var(--border-strong)] disabled:opacity-40 cursor-pointer"
            >
              {copiedType === 'latex' ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
              Copiar LaTeX
            </button>
          </div>
        </div>

        {lu.L_latex && lu.U_latex ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {lu.P_latex ? (
              <MatrixDisplay label="Matriz de permutación (P)" latex={`P = ${lu.P_latex}`} />
            ) : null}
            <MatrixDisplay label="Matriz triangular inferior (L)" latex={`L = ${lu.L_latex}`} accent="sl" />
            <MatrixDisplay label="Matriz triangular superior (U)" latex={`U = ${lu.U_latex}`} accent="sl" />
          </div>
        ) : null}

        {lu.status === 'converged' && lu.pb_latex && lu.y_latex && lu.x_latex ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <MatrixDisplay label="Vector permutado (Pb)" latex={`Pb = ${lu.pb_latex}`} />
            <MatrixDisplay label="Vector intermedio (y)" latex={`y = ${lu.y_latex}`} accent="sl" />
            <MatrixDisplay label="Vector solución (x)" latex={`x = ${lu.x_latex}`} accent="ok" />
          </div>
        ) : null}

        {lu.operation_count ? (
          <div className="flex flex-wrap items-center gap-2.5 text-xs font-mono-num pt-1">
            <span className="elevated-surface px-2.5 py-1">
              Operaciones reales: <strong>{lu.operation_count.total_flops} FLOPs</strong>{' '}
              ({lu.operation_count.mul_div} mult./div. + {lu.operation_count.add_sub} sum./rest.)
            </span>
            {lu.operation_count.theoretical_flops !== undefined ? (
              <span className="elevated-surface px-2.5 py-1 text-[var(--text-secondary)]">
                Cota teórica (2n³/3 + 2n²): ≈ {lu.operation_count.theoretical_flops} FLOPs
              </span>
            ) : null}
            {lu.rounding_mode && lu.rounding_mode !== 'none' && lu.finite_arithmetic_diff != null ? (
              <span className="elevated-surface px-2.5 py-1 border border-amber-500/35 text-[var(--text-nl)]">
                Diferencia vs. precisión completa: ||x_finito − x_exacto||∞ = {lu.finite_arithmetic_diff.toExponential(3)}
              </span>
            ) : null}
          </div>
        ) : null}
      </div>

      {/* Elimination Steps */}
      {lu.steps.length > 0 ? (
        <div className="panel-surface p-4 space-y-3">
          <h4 className="text-sm font-semibold">Procedimiento de eliminación paso a paso</h4>
          <div className="space-y-3">
            {lu.steps.map((step) => (
              <div key={step.step} className="elevated-surface p-3.5 space-y-2.5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs font-semibold text-[var(--text-sl)]">
                    Paso {step.step}: {step.title}
                  </span>
                  {step.pivot_swap ? (
                    <span className="text-[11px] px-2 py-0.5 rounded-full bg-amber-500/15 text-[var(--text-nl)] border border-amber-500/30">
                      Intercambio F{step.pivot_swap[0]} ↔ F{step.pivot_swap[1]}
                    </span>
                  ) : null}
                </div>
                <p className="text-xs text-[var(--text-secondary)]">{step.description}</p>
                <div className="text-sm overflow-x-auto">
                  <LatexRenderer math={step.latex_operation} displayMode />
                </div>
                <div
                  className={`grid grid-cols-1 ${
                    lu.strategy === 'partial_pivot' || step.pivot_swap ? 'md:grid-cols-3' : 'md:grid-cols-2'
                  } gap-2.5`}
                >
                  {lu.strategy === 'partial_pivot' || step.pivot_swap ? (
                    <MatrixDisplay label="Estado P" latex={`P = ${step.P_latex}`} />
                  ) : null}
                  <MatrixDisplay label="Estado L" latex={`L = ${step.L_latex}`} />
                  <MatrixDisplay label="Estado U" latex={`U = ${step.U_latex}`} />
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {/* Forward & Backward Substitutions */}
      {lu.status === 'converged' && lu.forward_equations && lu.backward_equations ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="panel-surface p-4 space-y-2">
            <h4 className="text-sm font-semibold text-[var(--text-sl)]">1. Sustitución progresiva (Ly = Pb)</h4>
            <div className="space-y-1.5 overflow-x-auto">
              {lu.forward_equations.map((eq, idx) => (
                <div key={idx} className="elevated-surface px-3 py-1.5 text-sm">
                  <LatexRenderer math={eq} displayMode />
                </div>
              ))}
            </div>
          </div>

          <div className="panel-surface p-4 space-y-2">
            <h4 className="text-sm font-semibold text-[var(--text-ok)]">2. Sustitución regresiva (Ux = y)</h4>
            <div className="space-y-1.5 overflow-x-auto">
              {lu.backward_equations.map((eq, idx) => (
                <div key={idx} className="elevated-surface px-3 py-1.5 text-sm">
                  <LatexRenderer math={eq} displayMode />
                </div>
              ))}
            </div>
            {lu.residual_norm !== undefined ? (
              <p className="text-xs font-mono-num text-[var(--text-secondary)] pt-1">
                Residual de verificación ||Ax - b||∞ = {lu.residual_norm.toExponential(3)}
              </p>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}

interface IterativePanelProps {
  title: string;
  subtitle: string;
  result: IterativeLinearResult;
  matrixInfo: IterationMatrixDetail;
  splitInfo: { D_latex: string; L_latex: string; U_latex: string };
  reorderingSuggestion?: ReorderingSuggestion | null;
  onApplyReordering?: (suggestion: ReorderingSuggestion) => void;
}

function IterativeMethodPanel({
  title,
  subtitle,
  result,
  matrixInfo,
  splitInfo,
  reorderingSuggestion,
  onApplyReordering,
}: IterativePanelProps) {
  const [selectedStepIdx, setSelectedStepIdx] = useState(0);
  const [showSplit, setShowSplit] = useState(false);
  const [copiedType, setCopiedType] = useState<'csv' | 'latex' | null>(null);
  const [copyError, setCopyError] = useState(false);
  const copyTimer = useRef<number | undefined>(undefined);
  const tableCaptionId = useId();

  useEffect(() => () => window.clearTimeout(copyTimer.current), []);

  useEffect(() => {
    setSelectedStepIdx(Math.max(0, result.steps.length - 1));
  }, [result]);

  const badge = formatStatusBadge(result.status);
  const selectedStep = result.steps[selectedStepIdx] ?? null;

  const handleCopy = async (kind: 'csv' | 'latex') => {
    const content = kind === 'csv' ? buildCsv(result) : buildLatexTable(result);
    if (!content) return;
    setCopyError(false);
    try {
      await navigator.clipboard.writeText(content);
    } catch {
      setCopiedType(null);
      setCopyError(true);
      return;
    }
    window.clearTimeout(copyTimer.current);
    setCopiedType(kind);
    copyTimer.current = window.setTimeout(() => setCopiedType(null), 1800);
  };

  return (
    <div className="space-y-4">
      {copyError ? <p role="alert" className="text-sm text-[var(--text-err)]">No se pudo copiar la tabla. Comprueba el acceso al portapapeles e inténtalo de nuevo.</p> : null}
      {/* Header & Matrix Form */}
      <div className="panel-surface p-4 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2.5">
              <h3 className="text-base font-bold">{title}</h3>
              <span className={`text-xs px-2.5 py-0.5 rounded-full border ${badge.cls}`}>
                {badge.label}
              </span>
            </div>
            <p className="text-xs text-[var(--text-secondary)] mt-0.5">{subtitle}</p>
            <p className="text-xs text-[var(--text-primary)] mt-1.5">{formatIterativeMessage(result.message)}</p>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setShowSplit((v) => !v)}
              className="px-3 py-1.5 rounded-lg text-xs font-medium elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
            >
              {showSplit ? 'Ocultar descomposición D, L, U' : 'Ver matrices D, L, U'}
            </button>
            <button
              type="button"
              onClick={() => handleCopy('csv')}
              disabled={!result.steps.length}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium elevated-surface hover:border-[var(--border-strong)] disabled:opacity-40 cursor-pointer"
            >
              {copiedType === 'csv' ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
              Copiar CSV
            </button>
            <button
              type="button"
              onClick={() => handleCopy('latex')}
              disabled={!result.steps.length}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium elevated-surface hover:border-[var(--border-strong)] disabled:opacity-40 cursor-pointer"
            >
              {copiedType === 'latex' ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
              Copiar LaTeX
            </button>
          </div>
        </div>

        {showSplit ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <MatrixDisplay label="Diagonal (D)" latex={`D = ${splitInfo.D_latex}`} />
            <MatrixDisplay label="Triangular inferior estricta (L, A = D − L − U)" latex={`L = ${splitInfo.L_latex}`} />
            <MatrixDisplay label="Triangular superior estricta (U, A = D − L − U)" latex={`U = ${splitInfo.U_latex}`} />
          </div>
        ) : null}

        {matrixInfo.available && matrixInfo.T_latex && matrixInfo.c_latex ? (
          <div className="space-y-3">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <MatrixDisplay
                label="Matriz de iteración (Forma matricial Sesión 18)"
                latex={`T = ${matrixInfo.T_latex}`}
                accent="sl"
              />
              <MatrixDisplay
                label="Vector constante c"
                latex={`c = ${matrixInfo.c_latex}`}
                accent="sl"
              />
            </div>
            <div className="flex flex-wrap gap-3 text-xs font-mono-num">
              <span className="elevated-surface px-2.5 py-1">
                ‖T‖₁ = {matrixInfo.norm_1?.toFixed(4) ?? '—'}
              </span>
              <span className="elevated-surface px-2.5 py-1">
                ‖T‖∞ = {matrixInfo.norm_inf?.toFixed(4) ?? '—'}
              </span>
              <span
                className={`elevated-surface px-2.5 py-1 border ${
                  matrixInfo.guaranteed_convergence
                    ? 'border-emerald-500/35 text-[var(--text-ok)]'
                    : 'border-rose-500/35 text-[var(--text-err)]'
                }`}
              >
                ρ(T) = {matrixInfo.spectral_radius !== null && matrixInfo.spectral_radius !== undefined
                  ? matrixInfo.spectral_radius.toFixed(5)
                  : '—'}{' '}
                {matrixInfo.spectral_radius != null && (
                  matrixInfo.guaranteed_convergence ? '(< 1, converge en aritmética exacta)' : '(≥ 1, no garantizado)'
                )}
              </span>
              {result.operation_count ? (
                <span className="elevated-surface px-2.5 py-1">
                  Costo: <strong>{result.operation_count.total_flops} FLOPs</strong>
                  {result.operation_count.per_iteration_flops
                    ? ` (${result.operation_count.per_iteration_flops}/it.)`
                    : ''}
                </span>
              ) : null}
            </div>
          </div>
        ) : (
          <div className="elevated-surface p-3 text-xs text-[var(--text-err)] flex flex-wrap items-center justify-between gap-2">
            <span>{matrixInfo.reason || 'Forma matricial no disponible.'}</span>
            {reorderingSuggestion?.improves && onApplyReordering ? (
              <button
                type="button"
                onClick={() => onApplyReordering(reorderingSuggestion)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500 text-slate-950 hover:bg-amber-400 cursor-pointer"
              >
                <ArrowUpDown className="h-3.5 w-3.5" />
                Reordenar filas (mejorar dominancia)
              </button>
            ) : null}
          </div>
        )}
      </div>

      {/* Iteration Table */}
      {result.steps.length > 0 ? (
        <div className="panel-surface p-4 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h4 id={tableCaptionId} className="text-sm font-semibold">
                Tabla de iteraciones ({result.iterations} pasos)
              </h4>
              <p className="text-xs text-[var(--text-secondary)]">
                Selecciona una fila para ver su sustitución. El residual se calcula antes del redondeo de presentación.
              </p>
            </div>
            {result.solution_latex ? (
              <div className="text-xs elevated-surface px-3 py-1.5">
                <span className="text-[var(--text-secondary)] mr-2">
                  {result.status === 'converged' ? 'Solución:' : 'Último iterado:'}
                </span>
                <LatexRenderer math={`x \\approx ${result.solution_latex}`} />
              </div>
            ) : null}
          </div>

          <div
            tabIndex={0}
            role="region"
            aria-label={`Tabla de iteraciones: ${title}`}
            aria-labelledby={tableCaptionId}
            className="table-wrapper"
          >
            <table className="num-table font-mono-num">
              <thead>
                <tr>
                  <th>k</th>
                  {result.steps[0].x.map((_, idx) => (
                    <th key={idx}>x{idx + 1}</th>
                  ))}
                  <th>Error abs.</th>
                  <th>Error rel.</th>
                  <th>e_rel (%)</th>
                  <th>Residual</th>
                </tr>
              </thead>
              <tbody>
                {result.steps.map((step, idx) => (
                  <tr
                    key={step.iteration}
                    tabIndex={0}
                    aria-current={idx === selectedStepIdx ? 'step' : undefined}
                    data-active={idx === selectedStepIdx}
                    onClick={() => setSelectedStepIdx(idx)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        setSelectedStepIdx(idx);
                      }
                    }}
                  >
                    <th>k = {step.iteration}</th>
                    {step.x.map((val, j) => (
                      <td key={j} data-label={`x${j + 1}`}>
                        {val}
                      </td>
                    ))}
                    <td data-label="Error abs.">{step.abs_error.toExponential(3)}</td>
                    <td data-label="Error rel.">{step.rel_error?.toExponential(3) ?? '—'}</td>
                    <td data-label="e_rel (%)">{formatRelPct(step.rel_error_pct)}</td>
                    <td data-label="Residual">{step.residual.toExponential(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Step Inspector */}
          {selectedStep ? (
            <div className="elevated-surface p-4 space-y-2.5 border border-sky-500/25">
              <div className="flex items-center justify-between">
                <h5 className="text-xs font-semibold text-[var(--text-sl)]">
                  Desglose paso a paso — Iteración k = {selectedStep.iteration}
                </h5>
                {selectedStep.variable_steps.length === 0 ? (
                  <span className="text-xs text-[var(--text-secondary)]">
                    No hay desglose simbólico disponible para este paso.
                  </span>
                ) : null}
              </div>
              <div className="space-y-1.5 overflow-x-auto">
                {selectedStep.variable_steps.map((tex, idx) => (
                  <div key={idx} className="text-sm">
                    <LatexRenderer math={tex} displayMode />
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

interface FlopsRow {
  label: string;
  status: string;
  stepsLabel: string;
  ops?: LinearAnalyzeResponse['lu']['operation_count'];
}

function getCompareRows(result: LinearAnalyzeResponse): FlopsRow[] {
  return [
    {
      label: `L.U. (${formatLUStrategyLabel(result.lu.strategy, true)})`,
      status: result.lu.status,
      stepsLabel: `${result.lu.steps.length} etapas`,
      ops: result.lu.operation_count,
    },
    {
      label: 'Jacobi',
      status: result.jacobi.status,
      stepsLabel: `${result.jacobi.iterations} it.`,
      ops: result.jacobi.operation_count,
    },
    {
      label: 'Gauss-Seidel (ω = 1.0)',
      status: result.gauss_seidel.status,
      stepsLabel: `${result.gauss_seidel.iterations} it.`,
      ops: result.gauss_seidel.operation_count,
    },
    {
      label: `SOR (ω = ${result.sor.omega.toFixed(2)})`,
      status: result.sor.status,
      stepsLabel: `${result.sor.iterations} it.`,
      ops: result.sor.operation_count,
    },
  ];
}

function buildCompareCsv(result: LinearAnalyzeResponse): string {
  const headers = ['Método', 'Estado', 'Pasos/It.', 'Mult./Div.', 'Sum./Rest.', 'Total FLOPs', 'Complejidad'];
  const rows = getCompareRows(result).map((row) =>
    [
      row.label,
      formatStatusBadge(row.status).label,
      row.stepsLabel,
      row.ops?.mul_div ?? '',
      row.ops?.add_sub ?? '',
      row.ops?.total_flops ?? '',
      row.ops?.complexity_latex ?? '',
    ]
      .map(csvCell)
      .join(','),
  );
  return [headers.map(csvCell).join(','), ...rows].join('\n');
}

function buildCompareLatex(result: LinearAnalyzeResponse): string {
  const rows = getCompareRows(result).map((row) => {
    const safeLabel = row.label.replace(/ω/g, '$\\omega$');
    const statusLabel = formatStatusBadge(row.status).label;
    const mulDiv = row.ops?.mul_div ?? '\\text{—}';
    const addSub = row.ops?.add_sub ?? '\\text{—}';
    const total = row.ops?.total_flops ?? '\\text{—}';
    const complexity = row.ops?.complexity_latex ?? '\\text{—}';
    return `\\text{${safeLabel}} & \\text{${statusLabel}} & \\text{${row.stepsLabel}} & ${mulDiv} & ${addSub} & ${total} & ${complexity} \\\\`;
  });
  return [
    '\\begin{array}{l|l|c|r|r|r|c}',
    '\\text{Método} & \\text{Estado} & \\text{Pasos/It.} & \\text{Mult./Div.} & \\text{Sum./Rest.} & \\text{Total FLOPs} & \\text{Complejidad} \\\\ \\hline',
    ...rows,
    '\\end{array}',
  ].join('\n');
}

function CompareFlopsCard({ result }: { result: LinearAnalyzeResponse }) {
  const [copiedType, setCopiedType] = useState<'csv' | 'latex' | null>(null);
  const [copyError, setCopyError] = useState(false);
  const copyTimer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(copyTimer.current), []);

  const rows = getCompareRows(result);

  const handleCopy = async (kind: 'csv' | 'latex') => {
    const content = kind === 'csv' ? buildCompareCsv(result) : buildCompareLatex(result);
    if (!content) return;
    setCopyError(false);
    try {
      await navigator.clipboard.writeText(content);
    } catch {
      setCopiedType(null);
      setCopyError(true);
      return;
    }
    window.clearTimeout(copyTimer.current);
    setCopiedType(kind);
    copyTimer.current = window.setTimeout(() => setCopiedType(null), 1800);
  };

  return (
    <div className="panel-surface p-4 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h4 className="text-sm font-semibold">
            Costo computacional (FLOPs) y Veredicto Pedagógico
          </h4>
          <span className="text-xs text-[var(--text-secondary)]">
            Operaciones aritméticas escalares reales ejecutadas por cada algoritmo
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {copiedType ? (
            <span role="status" className="text-xs text-[var(--text-ok)]">
              Copiado
            </span>
          ) : null}
          <button
            type="button"
            onClick={() => handleCopy('csv')}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
          >
            {copiedType === 'csv' ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
            Copiar CSV
          </button>
          <button
            type="button"
            onClick={() => handleCopy('latex')}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
          >
            {copiedType === 'latex' ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
            Copiar LaTeX
          </button>
        </div>
      </div>

      {copyError ? (
        <p role="alert" className="text-xs text-[var(--text-err)]">
          No se pudo copiar la tabla comparativa. Comprueba el acceso al portapapeles e inténtalo de nuevo.
        </p>
      ) : null}

      <div className="table-wrapper" tabIndex={0} role="region" aria-label="Tabla comparativa de operaciones FLOPs">
        <table className="num-table font-mono-num">
          <thead>
            <tr>
              <th>Método</th>
              <th>Estado</th>
              <th>Pasos / It.</th>
              <th>Mult. / Div.</th>
              <th>Sum. / Rest.</th>
              <th>Total FLOPs</th>
              <th>Complejidad</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const badge = formatStatusBadge(row.status);
              return (
                <tr key={row.label}>
                  <th>{row.label}</th>
                  <td data-label="Estado">
                    <span className={`text-[11px] px-2 py-0.5 rounded-full border ${badge.cls}`}>
                      {badge.label}
                    </span>
                  </td>
                  <td data-label="Pasos / It.">{row.stepsLabel}</td>
                  <td data-label="Mult. / Div.">{row.ops?.mul_div ?? '—'}</td>
                  <td data-label="Sum. / Rest.">{row.ops?.add_sub ?? '—'}</td>
                  <td data-label="Total FLOPs" className="font-semibold">
                    {row.ops?.total_flops ?? '—'}
                  </td>
                  <td data-label="Complejidad">
                    {row.ops?.complexity_latex ? <LatexRenderer math={row.ops.complexity_latex} /> : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {result.verdict ? (
        <div className="elevated-surface p-3.5 border border-sky-500/30 space-y-1.5">
          <p className="text-xs font-semibold text-[var(--text-sl)]">
            Veredicto Pedagógico: {result.verdict.summary}
          </p>
          {result.verdict.notes.length > 0 ? (
            <ul className="text-xs text-[var(--text-secondary)] list-disc pl-4 space-y-0.5">
              {result.verdict.notes.map((note, idx) => (
                <li key={idx}>{note}</li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

export function LinearSystemsLab() {
  const [size, setSize] = useState(3);
  const [matrixFields, setA] = useState(() => LINEAR_PRESETS[0].A.map((row) => row.map(String)));
  const [rhsFields, setB] = useState(() => LINEAR_PRESETS[0].b.map(String));
  const [initialFields, setX0] = useState(() => LINEAR_PRESETS[0].x0.map(String));
  const [omega, setOmega] = useState(LINEAR_PRESETS[0].omega);
  const [tol, setTol] = useState(1e-6);
  const [maxIterField, setMaxIter] = useState('100');
  const [decimals, setDecimals] = useState(6);
  const [roundingMode, setRoundingMode] = useState<RoundingMode>('none');
  const applyRounding = roundingMode !== 'none';
  const [criterion, setCriterion] = useState<StoppingCriterion>('abs_diff');
  const [luStrategy, setLuStrategy] = useState<LUStrategy>('partial_pivot');
  const [activePresetId, setActivePresetId] = useState(LINEAR_PRESETS[0].id);

  const [activeTab, setActiveTab] = useState<SLTab>('compare');
  const [chartMetric, setChartMetric] = useState<'abs_error' | 'residual'>('abs_error');
  const [showDominanceRows, setShowDominanceRows] = useState(false);
  const [analysis, setAnalysis] = useState<{ key: string; data: LinearAnalyzeResponse } | null>(null);
  const [sweep, setSweep] = useState<{ key: string; data: SORSweepResponse } | null>(null);
  const [loading, setLoading] = useState(false);
  const [sweepLoading, setSweepLoading] = useState(false);
  const [error, setError] = useState<{ key: string; message: string } | null>(null);
  const [sweepErrorKey, setSweepErrorKey] = useState<string | null>(null);

  const analysisRequest = useRef<AbortController | null>(null);
  const sweepRequest = useRef<{ key: string; controller: AbortController } | null>(null);
  const A = matrixFields.map((row) => row.map(Number));
  const b = rhsFields.map(Number);
  const x0 = initialFields.map(Number);
  const maxIter = Number(maxIterField);
  const validIterations = Number.isInteger(maxIter) && maxIter >= 1 && maxIter <= 500;
  const validEntries = [...matrixFields.flat(), ...rhsFields, ...initialFields]
    .every((value) => value.trim() !== '' && Number.isFinite(Number(value)));

  const runAnalysis = useCallback(
    async (
      matA: number[][],
      vecB: number[],
      vecX0: number[],
      w: number,
      t: number,
      mIter: number,
      crit: StoppingCriterion,
      strat: LUStrategy,
      dec: number,
      rMode: RoundingMode,
    ) => {
      analysisRequest.current?.abort();
      const controller = new AbortController();
      analysisRequest.current = controller;
      setLoading(true);
      setError(null);

      const roundEach = rMode !== 'none';
      const sweepInput = {
        A: matA,
        b: vecB,
        x0: vecX0,
        tol: t,
        max_iter: mIter,
        criterion: crit,
        decimals: dec,
        apply_rounding: roundEach,
        rounding_mode: rMode,
      };
      const sweepKey = JSON.stringify(sweepInput);
      const analysisKey = JSON.stringify([sweepKey, w, strat]);
      if (sweepRequest.current?.key !== sweepKey) {
        sweepRequest.current?.controller.abort();
        const sweepController = new AbortController();
        sweepRequest.current = { key: sweepKey, controller: sweepController };
        setSweep(null);
        setSweepErrorKey(null);
        setSweepLoading(true);
        sweepSorOmega(sweepInput, sweepController.signal)
          .then((data) => {
            if (!sweepController.signal.aborted) setSweep({ key: sweepKey, data });
          })
          .catch(() => {
            if (!sweepController.signal.aborted) {
              sweepRequest.current = null;
              setSweepErrorKey(sweepKey);
            }
          })
          .finally(() => {
            if (!sweepController.signal.aborted) setSweepLoading(false);
          });
      }

      try {
        const data = await analyzeLinearSystem({
          ...sweepInput,
          omega: w,
          lu_strategy: strat,
        }, controller.signal);
        if (!controller.signal.aborted) setAnalysis({ key: analysisKey, data });
      } catch (err) {
        if (!controller.signal.aborted) {
          setError({
            key: analysisKey,
            message: err instanceof Error ? err.message : 'No se pudo conectar con el backend Python.',
          });
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    runAnalysis(A, b, x0, omega, tol, maxIter, criterion, luStrategy, decimals, roundingMode);
    return () => {
      analysisRequest.current?.abort();
      sweepRequest.current?.controller.abort();
      sweepRequest.current = null;
    };
    // Initial run only
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSizeChange = (newSize: number) => {
    const nextA = Array.from({ length: newSize }, (_, i) =>
      Array.from({ length: newSize }, (__, j) => {
        if (i < matrixFields.length && j < matrixFields[i].length) return matrixFields[i][j];
        return i === j ? '5' : '0';
      }),
    );
    const nextB = Array.from({ length: newSize }, (_, i) => rhsFields[i] ?? '1');
    const nextX0 = Array.from({ length: newSize }, (_, i) => initialFields[i] ?? '0');
    setSize(newSize);
    setA(nextA);
    setB(nextB);
    setX0(nextX0);
    setActivePresetId('custom');
  };

  const applyPreset = (presetId: string) => {
    const preset = LINEAR_PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    const nextMaxIter = validIterations ? maxIter : 100;
    if (!validIterations) setMaxIter('100');
    setActivePresetId(preset.id);
    setSize(preset.A.length);
    setA(preset.A.map((row) => row.map(String)));
    setB(preset.b.map(String));
    setX0(preset.x0.map(String));
    setOmega(preset.omega);
    setLuStrategy(preset.luStrategy);
    runAnalysis(
      preset.A,
      preset.b,
      preset.x0,
      preset.omega,
      tol,
      nextMaxIter,
      criterion,
      preset.luStrategy,
      decimals,
      roundingMode,
    );
  };

  const generateDominantSystem = () => {
    const n = size;
    const randA = Array.from({ length: n }, (_, i) => {
      const row = Array.from({ length: n }, (__, j) =>
        i === j ? 0 : Math.floor(Math.random() * 5) - 2,
      );
      const offSum = row.reduce((acc, v) => acc + Math.abs(v), 0);
      row[i] = offSum + Math.floor(Math.random() * 4) + 2;
      return row;
    });
    const exactX = Array.from({ length: n }, () => Math.floor(Math.random() * 5) - 2 || 1);
    const randB = randA.map((row) => row.reduce((acc, val, j) => acc + val * exactX[j], 0));
    const zeroX0 = Array(n).fill(0);
    const nextMaxIter = validIterations ? maxIter : 100;
    if (!validIterations) setMaxIter('100');
    setA(randA.map((row) => row.map(String)));
    setB(randB.map(String));
    setX0(zeroX0.map(String));
    setActivePresetId('custom');
    runAnalysis(randA, randB, zeroX0, omega, tol, nextMaxIter, criterion, luStrategy, decimals, roundingMode);
  };

  const applyRowReordering = (suggestion: ReorderingSuggestion) => {
    const nextMaxIter = validIterations ? maxIter : 100;
    if (!validIterations) setMaxIter('100');
    setA(suggestion.A_reordered.map((row) => row.map(String)));
    setB(suggestion.b_reordered.map(String));
    setActivePresetId('custom');
    runAnalysis(
      suggestion.A_reordered,
      suggestion.b_reordered,
      x0,
      omega,
      tol,
      nextMaxIter,
      criterion,
      luStrategy,
      decimals,
      roundingMode,
    );
  };

  const handleOmegaSlider = (newOmega: number) => {
    setOmega(newOmega);
    if (validEntries && validIterations) runAnalysis(A, b, x0, newOmega, tol, maxIter, criterion, luStrategy, decimals, roundingMode);
  };

  const handleCriterionChange = (next: StoppingCriterion) => {
    setCriterion(next);
    if (next === 'rel_pct') {
      if (![5, 1, 0.5, 0.1, 0.01].includes(tol)) setTol(1);
    } else if (![1e-3, 1e-4, 1e-6, 1e-8, 1e-10].includes(tol)) {
      setTol(1e-6);
    }
  };

  const currentSweepKey = JSON.stringify({
    A, b, x0, tol, max_iter: maxIter, criterion, decimals, apply_rounding: applyRounding, rounding_mode: roundingMode,
  });
  const sweepMatchesForm = sweep?.key === currentSweepKey;
  const currentAnalysisKey = JSON.stringify([currentSweepKey, omega, luStrategy]);
  const lastAnalysis = analysis?.data ?? null;
  const isResultStale = !!lastAnalysis && (!validEntries || !validIterations || analysis?.key !== currentAnalysisKey);
  const result = lastAnalysis;
  const currentError = validEntries && validIterations && error?.key === currentAnalysisKey ? error.message : null;
  const diagnostics = result?.diagnostics;

  return (
    <div className="space-y-6">
      {/* Presets Strip */}
      <section className="panel-surface p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold">Casos de prueba del curso (Chapra · Burden)</h2>
            <p className="text-xs text-[var(--text-secondary)]">
              Carga un sistema de referencia para observar convergencia, pivoteo o divergencia, o edita tu propia matriz abajo.
            </p>
          </div>
          <button
            type="button"
            onClick={generateDominantSystem}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
          >
            <Shuffle className="h-3.5 w-3.5 text-sky-400" />
            Generar matriz dominante aleatoria
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-2.5">
          {LINEAR_PRESETS.map((preset) => {
            const active = activePresetId === preset.id;
            return (
              <button
                key={preset.id}
                type="button"
                onClick={() => applyPreset(preset.id)}
                className={`text-left p-3 rounded-xl border transition-all cursor-pointer ${
                  active
                    ? 'bg-sky-500/12 border-sky-500/50'
                    : 'elevated-surface hover:border-[var(--border-strong)]'
                }`}
              >
                <div className="flex items-center justify-between gap-2 mb-1">
                  <span className="text-xs font-semibold">{preset.title}</span>
                  <span className="text-[11px] px-2 py-0.5 rounded-full bg-sky-500/15 text-[var(--text-sl)] shrink-0">
                    {preset.tag}
                  </span>
                </div>
                <p className="text-xs text-[var(--text-secondary)] line-clamp-2">{preset.subtitle}</p>
              </button>
            );
          })}
        </div>
      </section>

      {/* Main Workbench: Matrix Input + Parameters */}
      <section className="grid grid-cols-1 xl:grid-cols-12 gap-5">
        {/* Matrix Editor */}
        <div className="xl:col-span-7 panel-surface p-4 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-semibold">Sistema lineal Ax = b</h3>
              <p className="text-xs text-[var(--text-secondary)]">
                Edita coeficientes de A, términos independientes b y vector inicial x⁽⁰⁾.
              </p>
            </div>
            <div className="inline-flex items-center gap-1 elevated-surface p-1 rounded-lg">
              {[2, 3, 4, 5, 6].map((dim) => (
                <button
                  key={dim}
                  type="button"
                  aria-pressed={size === dim}
                  onClick={() => handleSizeChange(dim)}
                  className={`px-2.5 py-1 rounded-md text-xs font-mono-num cursor-pointer transition-colors ${
                    size === dim
                      ? 'bg-sky-500 text-slate-950 font-bold'
                      : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                  }`}
                >
                  {dim}×{dim}
                </button>
              ))}
            </div>
          </div>

          <div className="overflow-x-auto pb-1">
            <div className="inline-block min-w-full">
              <div
                className="grid gap-1.5 items-center"
                style={{ gridTemplateColumns: `repeat(${size}, minmax(60px, 1fr)) 16px minmax(64px, 1fr) 16px minmax(64px, 1fr)` }}
              >
                {Array.from({ length: size }, (_, j) => (
                  <div key={`ha-${j}`} className="text-center text-xs font-mono-num text-[var(--text-secondary)]">
                    a·{j + 1}
                  </div>
                ))}
                <div />
                <div className="text-center text-xs font-mono-num text-[var(--text-sl)] font-semibold">b</div>
                <div />
                <div className="text-center text-xs font-mono-num text-[var(--text-nl)] font-semibold">x⁽⁰⁾</div>

                {matrixFields.map((row, i) => (
                  <div key={`row-${i}`} className="contents">
                    {row.map((val, j) => (
                      <input
                        key={`a-${i}-${j}`}
                        type="number"
                        step="any"
                        aria-label={`A fila ${i + 1} columna ${j + 1}`}
                        required
                        value={val}
                        onChange={(e) => {
                          const v = e.target.value;
                          setA((prev) =>
                            prev.map((r, ri) =>
                              ri === i ? r.map((c, ci) => (ci === j ? v : c)) : r,
                            ),
                          );
                          setActivePresetId('custom');
                        }}
                        className={`field-input font-mono-num text-sm text-center py-1.5 px-1.5 ${
                          i === j ? 'border-sky-500/45 font-semibold' : ''
                        }`}
                      />
                    ))}
                    <div className="text-center text-[var(--text-muted)] font-mono-num">|</div>
                    <input
                      type="number"
                      step="any"
                      aria-label={`b fila ${i + 1}`}
                      required
                      value={rhsFields[i]}
                      onChange={(e) => {
                        const v = e.target.value;
                        setB((prev) => prev.map((item, idx) => (idx === i ? v : item)));
                        setActivePresetId('custom');
                      }}
                      className="field-input font-mono-num text-sm text-center py-1.5 px-1.5 border-sky-500/30"
                    />
                    <div className="text-center text-[var(--text-muted)] font-mono-num">·</div>
                    <input
                      type="number"
                      step="any"
                      aria-label={`x0 componente ${i + 1}`}
                      required
                      value={initialFields[i]}
                      onChange={(e) => {
                        const v = e.target.value;
                        setX0((prev) => prev.map((item, idx) => (idx === i ? v : item)));
                        setActivePresetId('custom');
                      }}
                      className="field-input font-mono-num text-sm text-center py-1.5 px-1.5 border-amber-500/30"
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Parameters & Controls */}
        <div className="xl:col-span-5 panel-surface p-4 flex flex-col justify-between gap-4">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="h-4 w-4 text-sky-400" />
              <h3 className="text-sm font-semibold">Parámetros numéricos</h3>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="sl-tol" className="block text-xs text-[var(--text-secondary)] mb-1">
                  {criterion === 'rel_pct' ? 'Tolerancia ε (%)' : 'Tolerancia ε'}
                </label>
                <select
                  id="sl-tol"
                  value={tol}
                  onChange={(e) => setTol(Number(e.target.value))}
                  className="field-input w-full px-2.5 py-1.5 text-sm font-mono-num"
                >
                  {criterion === 'rel_pct' ? (
                    <>
                      <option value={5}>5%</option>
                      <option value={1}>1%</option>
                      <option value={0.5}>0.5%</option>
                      <option value={0.1}>0.1%</option>
                      <option value={0.01}>0.01%</option>
                    </>
                  ) : (
                    <>
                      <option value={1e-3}>10⁻³ (0.001)</option>
                      <option value={1e-4}>10⁻⁴ (0.0001)</option>
                      <option value={1e-6}>10⁻⁶</option>
                      <option value={1e-8}>10⁻⁸</option>
                      <option value={1e-10}>10⁻¹⁰</option>
                    </>
                  )}
                </select>
              </div>

              <div>
                <label htmlFor="sl-maxiter" className="block text-xs text-[var(--text-secondary)] mb-1">
                  Máx. iteraciones
                </label>
                <input
                  id="sl-maxiter"
                  type="number"
                  step={1}
                  min={1}
                  max={500}
                  required
                  value={maxIterField}
                  onChange={(e) => setMaxIter(e.target.value)}
                  className="field-input w-full px-2.5 py-1.5 text-sm font-mono-num"
                />
                <div className="error-msg">Entero entre 1 y 500.</div>
              </div>

              <div>
                <label htmlFor="sl-criterion" className="block text-xs text-[var(--text-secondary)] mb-1">
                  Criterio de parada
                </label>
                <select
                  id="sl-criterion"
                  value={criterion}
                  onChange={(e) => handleCriterionChange(e.target.value as StoppingCriterion)}
                  className="field-input w-full px-2.5 py-1.5 text-xs"
                >
                  <option value="abs_diff">Diferencia ||x⁽ᵏ⁾ - x⁽ᵏ⁻¹⁾||∞</option>
                  <option value="rel_diff">Error relativo aproximado</option>
                  <option value="rel_pct">Error relativo porcentual e_rel (%)</option>
                  <option value="residual">Residual ||Ax⁽ᵏ⁾ - b||∞</option>
                </select>
              </div>

              <div>
                <label htmlFor="sl-lu-strategy" className="block text-xs text-[var(--text-secondary)] mb-1">
                  Estrategia L.U.
                </label>
                <select
                  id="sl-lu-strategy"
                  value={luStrategy}
                  onChange={(e) => setLuStrategy(e.target.value as LUStrategy)}
                  className="field-input w-full px-2.5 py-1.5 text-xs"
                >
                  <option value="partial_pivot">Pivoteo Parcial (PA = LU)</option>
                  <option value="doolittle">Doolittle (Lᵢᵢ = 1)</option>
                  <option value="crout">Crout (Uᵢᵢ = 1)</option>
                </select>
              </div>
            </div>

            {/* Live SOR Omega Slider */}
            <div className="elevated-surface p-3 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <label htmlFor="sor-omega" className="font-medium">
                  Sobrerrelajación sucesiva SOR (ω)
                </label>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => handleOmegaSlider(1.0)}
                    className="px-2 py-0.5 rounded text-[11px] elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
                  >
                    ω = 1 (GS)
                  </button>
                  <span className="font-mono-num font-bold text-[var(--text-ok)]">
                    {omega.toFixed(2)}
                  </span>
                </div>
              </div>
              <input
                id="sor-omega"
                type="range"
                min={0.1}
                max={1.95}
                step={0.05}
                value={omega}
                onChange={(e) => handleOmegaSlider(Number(e.target.value))}
                className="w-full accent-emerald-400 cursor-pointer"
              />
              <div className="flex justify-between text-[11px] text-[var(--text-muted)]">
                <span>0.10 (Subrelajación)</span>
                <span>1.00 (Gauss-Seidel)</span>
                <span>1.95 (Sobrerrelajación)</span>
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
              <label className="inline-flex items-center gap-1.5">
                <span className="text-[var(--text-secondary)]">Aritmética finita (Sesión 6):</span>
                <select
                  aria-label="Modo de aritmética finita por paso"
                  value={roundingMode}
                  onChange={(e) => setRoundingMode(e.target.value as RoundingMode)}
                  className="field-input px-2 py-1 text-xs"
                >
                  <option value="none">Sin corte por paso (precisión completa)</option>
                  <option value="round">Redondeo por paso ({decimals} dec.)</option>
                  <option value="truncate">Truncamiento por paso ({decimals} dec.)</option>
                </select>
              </label>

              <div className="inline-flex items-center gap-1.5">
                <span className="text-[var(--text-secondary)]">Decimales:</span>
                <select
                  aria-label="Decimales a mostrar"
                  value={decimals}
                  onChange={(e) => setDecimals(Number(e.target.value))}
                  className="field-input px-2 py-1 text-xs font-mono-num"
                >
                  <option value={2}>2</option>
                  <option value={3}>3</option>
                  <option value={4}>4</option>
                  <option value={6}>6</option>
                  <option value={8}>8</option>
                </select>
              </div>
            </div>
          </div>

          <button
            type="button"
            disabled={loading || !validEntries || !validIterations}
            onClick={() =>
              runAnalysis(A, b, x0, omega, tol, maxIter, criterion, luStrategy, decimals, roundingMode)
            }
            className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold text-sm transition-colors disabled:opacity-60 cursor-pointer"
          >
            <Play className="h-4 w-4 fill-current" />
            {loading ? 'Calculando métodos...' : 'Resolver y comparar métodos'}
          </button>
        </div>
      </section>

      {currentError ? (
        <div role="alert" className="panel-surface p-4 border-rose-500/40 bg-rose-500/10 text-[var(--text-err)] text-sm flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          <span>{currentError}</span>
        </div>
      ) : (!result || isResultStale) && !loading ? (
        <p role="status" className={`text-sm ${isResultStale ? 'text-[var(--text-nl)]' : 'text-[var(--text-secondary)]'}`}>
          {!validEntries ? 'Completa las entradas de A, b y x⁽⁰⁾ con números finitos.'
            : !validIterations ? 'Ingresa un máximo de iteraciones entero entre 1 y 500.'
              : isResultStale
                ? 'Mostrando el último resultado calculado (desactualizado). Pulsa «Resolver y comparar métodos» para actualizar.'
                : 'Resuelve el sistema para actualizar los resultados.'}
        </p>
      ) : null}

      {/* Diagnostics Banner */}
      {diagnostics ? (
        <section className={`panel-surface p-4 space-y-3 transition-opacity ${isResultStale ? 'opacity-75' : ''}`}>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              {diagnostics.dominance.is_strictly_dominant ? (
                <CheckCircle2 className="h-5 w-5 text-[var(--text-ok)] shrink-0" />
              ) : (
                <AlertTriangle className="h-5 w-5 text-[var(--text-nl)] shrink-0" />
              )}
              <div>
                <p className="text-sm font-semibold">
                  {diagnostics.dominance.is_strictly_dominant
                    ? 'Matriz estrictamente diagonalmente dominante'
                    : diagnostics.dominance.has_zero_diagonal
                      ? 'Diagonal principal contiene ceros (aᵢᵢ = 0)'
                      : 'Matriz NO estrictamente diagonalmente dominante'}
                </p>
                <p className="text-xs text-[var(--text-secondary)]">
                  {diagnostics.dominance.is_strictly_dominant
                    ? 'La dominancia garantiza la convergencia de Jacobi y Gauss-Seidel en aritmética exacta.'
                    : 'Los métodos iterativos no tienen convergencia garantizada por dominancia; verifica el radio espectral ρ(T) < 1.'}
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 text-xs font-mono-num">
              {diagnostics.matrices.jacobi.available ? (
                <span
                  className={`px-2.5 py-1 rounded-lg border ${
                    diagnostics.matrices.jacobi.guaranteed_convergence
                      ? 'bg-emerald-500/10 border-emerald-500/30 text-[var(--text-ok)]'
                      : 'bg-rose-500/10 border-rose-500/30 text-[var(--text-err)]'
                  }`}
                >
                  ρ(T_J) = {diagnostics.matrices.jacobi.spectral_radius?.toFixed(4) ?? '—'}
                </span>
              ) : null}
              {diagnostics.matrices.gauss_seidel.available ? (
                <span
                  className={`px-2.5 py-1 rounded-lg border ${
                    diagnostics.matrices.gauss_seidel.guaranteed_convergence
                      ? 'bg-emerald-500/10 border-emerald-500/30 text-[var(--text-ok)]'
                      : 'bg-rose-500/10 border-rose-500/30 text-[var(--text-err)]'
                  }`}
                >
                  ρ(T_G) = {diagnostics.matrices.gauss_seidel.spectral_radius?.toFixed(4) ?? '—'}
                </span>
              ) : null}
              {diagnostics.matrices.sor.available ? (
                <span
                  className={`px-2.5 py-1 rounded-lg border ${
                    diagnostics.matrices.sor.guaranteed_convergence
                      ? 'bg-emerald-500/10 border-emerald-500/30 text-[var(--text-ok)]'
                      : 'bg-rose-500/10 border-rose-500/30 text-[var(--text-err)]'
                  }`}
                >
                  ρ(T_ω) = {diagnostics.matrices.sor.spectral_radius?.toFixed(4) ?? '—'}
                </span>
              ) : null}
              <button
                type="button"
                aria-expanded={showDominanceRows}
                onClick={() => setShowDominanceRows((v) => !v)}
                className="px-2.5 py-1 rounded-lg elevated-surface hover:border-[var(--border-strong)] text-[var(--text-secondary)] cursor-pointer font-sans"
              >
                {showDominanceRows ? 'Ocultar detalle por fila' : 'Detalle por fila'}
              </button>
              {diagnostics.dominance.reordering_suggestion?.improves ? (
                <button
                  type="button"
                  onClick={() => applyRowReordering(diagnostics.dominance.reordering_suggestion!)}
                  className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold bg-amber-500 text-slate-950 hover:bg-amber-400 cursor-pointer font-sans"
                >
                  <ArrowUpDown className="h-3.5 w-3.5" />
                  Reordenar filas (mejorar dominancia)
                </button>
              ) : null}
            </div>
          </div>

          {showDominanceRows ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 pt-2 border-t border-[var(--border-subtle)] text-xs font-mono-num">
              {diagnostics.dominance.rows_analysis.map((row) => (
                <div
                  key={row.row}
                  className={`elevated-surface px-3 py-1.5 flex items-center justify-between gap-2 border ${
                    row.is_strictly_dominant
                      ? 'border-emerald-500/25'
                      : 'border-amber-500/25'
                  }`}
                >
                  <span>
                    F{row.row}: |a{row.row}{row.row}| = {row.diag_abs} {row.is_strictly_dominant ? '>' : row.is_weakly_dominant ? '≥' : '<'} Σ = {row.off_diag_sum ?? '∞'}
                  </span>
                  <span className={row.is_strictly_dominant ? 'text-[var(--text-ok)]' : 'text-[var(--text-nl)]'}>
                    {row.is_strictly_dominant ? 'Estricta' : row.is_weakly_dominant ? 'Débil' : 'No dom.'}
                  </span>
                </div>
              ))}
            </div>
          ) : null}
        </section>
      ) : null}

      {/* Tabs Navigation */}
      {result ? (
        <section className={`space-y-4 transition-opacity ${isResultStale ? 'opacity-75' : ''}`}>
          <div
            role="tablist"
            aria-label="Vistas del sistema lineal"
            onKeyDown={(event) => {
              const tabs = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]'));
              const current = tabs.findIndex((tab) => tab === event.target);
              if (current < 0) return;

              let next: number;
              switch (event.key) {
                case 'ArrowRight': next = (current + 1) % tabs.length; break;
                case 'ArrowLeft': next = (current + tabs.length - 1) % tabs.length; break;
                case 'Home': next = 0; break;
                case 'End': next = tabs.length - 1; break;
                default: return;
              }
              event.preventDefault();
              tabs[next].focus();
              tabs[next].click();
            }}
            className="flex overflow-x-auto border-b border-[var(--border-subtle)] gap-1"
          >
            {(
              [
                { id: 'compare', label: 'Comparador y gráficas' },
                { id: 'lu', label: 'Factorización L.U.' },
                { id: 'jacobi', label: `Jacobi (${result.jacobi.iterations} it.)` },
                { id: 'gauss_seidel', label: `Gauss-Seidel (${result.gauss_seidel.iterations} it.)` },
                { id: 'sor', label: `SOR (${result.sor.iterations} it.)` },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                type="button"
                role="tab"
                id={`tab-sl-${tab.id}`}
                aria-selected={activeTab === tab.id}
                aria-controls={`panel-sl-${tab.id}`}
                tabIndex={activeTab === tab.id ? 0 : -1}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2.5 text-sm font-medium shrink-0 cursor-pointer transition-colors ${
                  activeTab === tab.id
                    ? 'text-[var(--text-sl)]'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Panel 1: Comparador */}
          <div
            role="tabpanel"
            id="panel-sl-compare" aria-labelledby="tab-sl-compare"
            hidden={activeTab !== 'compare'}
          >
            {activeTab === 'compare' ? (
              <div className="space-y-4">
                {/* Summary KPI Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  {[
                    {
                      name: `L.U. (${formatLUStrategyLabel(result.lu.strategy, true)})`,
                      status: result.lu.status,
                      detail: `Directo (${result.lu.steps.length} etapas)`,
                      flops: result.lu.operation_count?.total_flops,
                      sol: result.lu.x,
                      res: result.lu.residual_norm,
                      onOpen: () => setActiveTab('lu'),
                    },
                    {
                      name: 'Jacobi',
                      status: result.jacobi.status,
                      detail: `${result.jacobi.iterations} iteraciones`,
                      flops: result.jacobi.operation_count?.total_flops,
                      sol: result.jacobi.solution,
                      res: result.jacobi.residual_norm,
                      onOpen: () => setActiveTab('jacobi'),
                    },
                    {
                      name: 'Gauss-Seidel (ω = 1.0)',
                      status: result.gauss_seidel.status,
                      detail: `${result.gauss_seidel.iterations} iteraciones`,
                      flops: result.gauss_seidel.operation_count?.total_flops,
                      sol: result.gauss_seidel.solution,
                      res: result.gauss_seidel.residual_norm,
                      onOpen: () => setActiveTab('gauss_seidel'),
                    },
                    {
                      name: `SOR (ω = ${result.sor.omega.toFixed(2)})`,
                      status: result.sor.status,
                      detail: `${result.sor.iterations} iteraciones`,
                      flops: result.sor.operation_count?.total_flops,
                      sol: result.sor.solution,
                      res: result.sor.residual_norm,
                      onOpen: () => setActiveTab('sor'),
                    },
                  ].map((card) => {
                    const bStyle = formatStatusBadge(card.status);
                    return (
                      <button
                        key={card.name}
                        type="button"
                        onClick={card.onOpen}
                        className="text-left panel-surface p-4 hover:border-[var(--border-strong)] transition-colors space-y-2 cursor-pointer"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-xs font-semibold">{card.name}</span>
                          <span className={`text-[11px] px-2 py-0.5 rounded-full border ${bStyle.cls}`}>
                            {bStyle.label}
                          </span>
                        </div>
                        <div className="text-lg font-bold font-mono-num">{card.detail}</div>
                        <div className="text-xs font-mono-num text-[var(--text-sl)]">
                          Costo: {card.flops !== undefined ? `${card.flops} FLOPs` : '—'}
                        </div>
                        <div className="text-xs font-mono-num text-[var(--text-secondary)] break-words">
                          x = {card.sol ? `[${card.sol.join(', ')}]` : '—'}
                        </div>
                        <div className="text-[11px] font-mono-num text-[var(--text-muted)]">
                          Residual: {card.res !== undefined && card.res !== null ? card.res.toExponential(2) : '—'}
                        </div>
                      </button>
                    );
                  })}
                </div>

                {/* FLOPs Table & Pedagogical Verdict */}
                <CompareFlopsCard result={result} />

                {/* Convergence & Omega Sweep */}
                <div className="space-y-2">
                  <div className="flex justify-end gap-2">
                    <button
                      type="button"
                      aria-pressed={chartMetric === 'abs_error'}
                      onClick={() => setChartMetric('abs_error')}
                      className={`px-3 py-1 rounded-lg text-xs cursor-pointer ${
                        chartMetric === 'abs_error'
                          ? 'bg-sky-500/20 text-[var(--text-sl)] border border-sky-500/40'
                          : 'elevated-surface text-[var(--text-secondary)]'
                      }`}
                    >
                      Error entre iteraciones
                    </button>
                    <button
                      type="button"
                      aria-pressed={chartMetric === 'residual'}
                      onClick={() => setChartMetric('residual')}
                      className={`px-3 py-1 rounded-lg text-xs cursor-pointer ${
                        chartMetric === 'residual'
                          ? 'bg-sky-500/20 text-[var(--text-sl)] border border-sky-500/40'
                          : 'elevated-surface text-[var(--text-secondary)]'
                      }`}
                    >
                      Residual ||Ax - b||∞
                    </button>
                  </div>

                  <ConvergenceChart
                    jacobi={result.jacobi}
                    gaussSeidel={result.gauss_seidel}
                    sor={result.sor}
                    metric={chartMetric}
                  />
                </div>

                <OmegaSweepChart
                  sweep={sweepMatchesForm ? sweep.data : (!sweepLoading && sweep ? sweep.data : null)}
                  loading={sweepLoading}
                  failed={sweepErrorKey === currentSweepKey}
                  isStale={!sweepMatchesForm && Boolean(sweep)}
                  currentOmega={omega}
                  onSelectOmega={handleOmegaSlider}
                />
              </div>
            ) : null}
          </div>

          {/* Tab Panel 2: Factorización L.U. */}
          <div role="tabpanel" id="panel-sl-lu" aria-labelledby="tab-sl-lu" hidden={activeTab !== 'lu'}>
            {activeTab === 'lu' ? (
              <LUPanel
                lu={result.lu}
                onEnablePartialPivot={() => {
                  setLuStrategy('partial_pivot');
                  runAnalysis(A, b, x0, omega, tol, maxIter, criterion, 'partial_pivot', decimals, roundingMode);
                }}
              />
            ) : null}
          </div>

          {/* Tab Panel 3: Jacobi */}
          <div role="tabpanel" id="panel-sl-jacobi" aria-labelledby="tab-sl-jacobi" hidden={activeTab !== 'jacobi'}>
            {activeTab === 'jacobi' ? (
              <IterativeMethodPanel
                title="Método de Jacobi"
                subtitle="Cada componente xᵢ⁽ᵏ⁺¹⁾ se calcula usando únicamente los valores del paso anterior x⁽ᵏ⁾."
                result={result.jacobi}
                matrixInfo={result.diagnostics.matrices.jacobi}
                splitInfo={result.diagnostics.matrices.split}
                reorderingSuggestion={result.diagnostics.dominance.reordering_suggestion}
                onApplyReordering={applyRowReordering}
              />
            ) : null}
          </div>

          <div role="tabpanel" id="panel-sl-gauss_seidel" aria-labelledby="tab-sl-gauss_seidel" hidden={activeTab !== 'gauss_seidel'}>
            {activeTab === 'gauss_seidel' ? (
              <IterativeMethodPanel
                title="Método de Gauss-Seidel (ω = 1)"
                subtitle="Cada componente nuevo se utiliza de inmediato en las siguientes filas."
                result={result.gauss_seidel}
                matrixInfo={result.diagnostics.matrices.gauss_seidel}
                splitInfo={result.diagnostics.matrices.split}
                reorderingSuggestion={result.diagnostics.dominance.reordering_suggestion}
                onApplyReordering={applyRowReordering}
              />
            ) : null}
          </div>

          <div role="tabpanel" id="panel-sl-sor" aria-labelledby="tab-sl-sor" hidden={activeTab !== 'sor'}>
            {activeTab === 'sor' ? (
              <IterativeMethodPanel
                title={`Sobrerrelajación sucesiva SOR (ω = ${result.sor.omega.toFixed(2)})`}
                subtitle={
                  result.sor.omega === 1
                    ? 'Con ω = 1.00 corresponde al método clásico de Gauss-Seidel (actualización inmediata).'
                    : 'El factor ω pondera el nuevo valor de Gauss-Seidel y el valor de la iteración anterior.'
                }
                result={result.sor}
                matrixInfo={result.diagnostics.matrices.sor}
                splitInfo={result.diagnostics.matrices.split}
                reorderingSuggestion={result.diagnostics.dominance.reordering_suggestion}
                onApplyReordering={applyRowReordering}
              />
            ) : null}
          </div>
        </section>
      ) : null}
    </div>
  );
}
