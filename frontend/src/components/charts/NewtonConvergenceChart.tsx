import { useMemo, useState } from 'react';

export interface NewtonConvergenceStepInput {
  iteration: number;
  abs_error: number;
  residual: number | null;
}

interface NewtonConvergenceChartProps {
  mode: '1d' | 'system';
  primaryLabel: string;
  primarySteps: NewtonConvergenceStepInput[];
  primaryOrder?: number | null;
  primaryStatus?: string;
  primaryRootLabel?: string;
  secondaryLabel?: string;
  secondarySteps?: NewtonConvergenceStepInput[];
  secondaryOrder?: number | null;
  secondaryStatus?: string;
  secondaryRootLabel?: string;
}

type MetricView = 'both' | 'abs_error' | 'residual';

const COLORS = {
  primaryError: 'var(--chart-nl-err)',
  primaryResidual: 'var(--chart-nl-res)',
  secondaryError: 'var(--chart-nl2-err)',
  secondaryResidual: 'var(--chart-nl2-res)',
};

function formatExp(val?: number | null): string {
  if (val === null || val === undefined || !Number.isFinite(val)) return '—';
  return val.toExponential(2);
}

function formatStatusShort(status?: string): string {
  if (!status) return '—';
  const map: Record<string, string> = {
    converged: 'Convergió',
    max_iterations: 'Máx. it.',
    stagnated: 'Estancado',
    zero_derivative: "f'(x) = 0",
    singular_jacobian: 'J singular',
    diverged: 'Divergió',
    domain_error: 'Error dominio',
    parse_error: 'Error sintaxis',
  };
  return map[status] ?? status;
}

function classifyOrder(order?: number | null): string {
  if (order === null || order === undefined || !Number.isFinite(order)) return 'N/D';
  if (order >= 1.65) return `p ≈ ${order.toFixed(2)} (Cuadrática)`;
  if (order < 1.35) return `p ≈ ${order.toFixed(2)} (Lineal)`;
  return `p ≈ ${order.toFixed(2)} (Superlineal)`;
}

export function NewtonConvergenceChart({
  mode,
  primaryLabel,
  primarySteps,
  primaryOrder,
  primaryStatus,
  primaryRootLabel,
  secondaryLabel,
  secondarySteps,
  secondaryOrder,
  secondaryStatus,
  secondaryRootLabel,
}: NewtonConvergenceChartProps) {
  const [hoverIter, setHoverIter] = useState<number | null>(null);
  const [metricView, setMetricView] = useState<MetricView>('both');

  const hasSecondary = Boolean(secondarySteps && secondaryLabel);

  const chartData = useMemo(() => {
    const extract = (steps: NewtonConvergenceStepInput[]) =>
      steps.map((s) => {
        const errRaw = s.abs_error;
        const resRaw = s.residual;
        return {
          k: s.iteration,
          errRaw,
          resRaw,
          errZero: errRaw === 0,
          resZero: resRaw === 0,
          errLog: Number.isFinite(errRaw) && errRaw > 0 ? Math.log10(errRaw) : null,
          resLog: resRaw !== null && Number.isFinite(resRaw) && resRaw > 0 ? Math.log10(resRaw) : null,
        };
      });

    const pPts = extract(primarySteps);
    const sPts = secondarySteps ? extract(secondarySteps) : [];
    const allPts = [...pPts, ...sPts];

    const maxK = Math.max(
      1,
      pPts.length ? pPts[pPts.length - 1].k : 1,
      sPts.length ? sPts[sPts.length - 1].k : 1,
    );

    const logValues: number[] = [];
    for (const pt of allPts) {
      if ((metricView === 'both' || metricView === 'abs_error') && pt.errLog !== null) {
        logValues.push(pt.errLog);
      }
      if ((metricView === 'both' || metricView === 'residual') && pt.resLog !== null) {
        logValues.push(pt.resLog);
      }
    }

    const minLog = logValues.length ? Math.floor(Math.min(...logValues, -8)) : -8;
    const maxLog = logValues.length ? Math.ceil(Math.max(...logValues, 1)) : 1;

    return { pPts, sPts, maxK, minLog, maxLog };
  }, [primarySteps, secondarySteps, metricView]);

  const width = 680;
  const height = 250;
  const pad = { top: 22, right: 24, bottom: 38, left: 54 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const { pPts, sPts, maxK, minLog, maxLog } = chartData;
  const logSpan = Math.max(maxLog - minLog, 1);

  const xScale = (k: number) =>
    maxK <= 1
      ? pad.left + plotW / 2
      : pad.left + ((k - 1) / (maxK - 1)) * plotW;
  const yScale = (logVal: number) =>
    pad.top + ((maxLog - logVal) / logSpan) * plotH;

  const buildPath = (pts: Array<{ k: number; val: number | null; isZero: boolean }>) => {
    const valid = pts.filter((p) => p.val !== null || p.isZero);
    if (valid.length === 0) return '';
    return valid
      .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${xScale(p.k).toFixed(2)} ${yScale(p.val ?? minLog).toFixed(2)}`)
      .join(' ');
  };

  const yTicks: number[] = [];
  const stepTick = Math.max(1, Math.ceil(logSpan / 5));
  for (let v = minLog; v <= maxLog; v += stepTick) {
    yTicks.push(v);
  }

  const allEmpty = pPts.length === 0 && sPts.length === 0;
  const errName = mode === '1d' ? '|xₖ₊₁ − xₖ|' : '||Δx⁽ᵏ⁾||∞';
  const resName = mode === '1d' ? '|f(xₖ)|' : '||F(xₖ₊₁)||∞';

  if (allEmpty) {
    return (
      <div className="panel-surface p-4 space-y-1">
        <h4 className="text-sm font-semibold">
          Curva de convergencia (escala log₁₀)
        </h4>
        <p className="text-xs text-[var(--text-secondary)]">
          No hay iteraciones registradas para graficar (por ejemplo, el punto inicial ya satisface la tolerancia o falló antes del paso 1).
        </p>
      </div>
    );
  }

  const findAt = (pts: typeof pPts, k: number) => pts.find((p) => p.k === k);

  return (
    <div className="panel-surface p-4 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h4 className="text-sm font-semibold">
            Convergencia de Newton-Raphson ({mode === '1d' ? '1D' : 'Sistemas'})
          </h4>
          <p className="text-xs text-[var(--text-secondary)]">
            Escala log₁₀ por iteración k. Línea continua: paso {errName}; línea discontinua: residual {resName}.
          </p>
        </div>

        <div className="inline-flex items-center gap-1 elevated-surface p-1 rounded-lg text-xs">
          {(
            [
              { id: 'both', label: 'Ambos' },
              { id: 'abs_error', label: `Paso ${mode === '1d' ? '|Δx|' : '||Δx||∞'}` },
              { id: 'residual', label: `Residual ${mode === '1d' ? '|f|' : '||F||∞'}` },
            ] as const
          ).map((opt) => (
            <button
              key={opt.id}
              type="button"
              aria-pressed={metricView === opt.id}
              onClick={() => setMetricView(opt.id)}
              className={`px-2.5 py-1 rounded-md cursor-pointer transition-colors ${
                metricView === opt.id
                  ? 'bg-amber-500/20 text-[var(--text-nl)] font-semibold'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-4 text-xs">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: COLORS.primaryError }} />
          {primaryLabel} ({primarySteps.length} it.)
        </span>
        {hasSecondary ? (
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: COLORS.secondaryError }} />
            {secondaryLabel} ({secondarySteps?.length ?? 0} it.)
          </span>
        ) : null}
      </div>

      {/* Side-by-side comparison cards when comparing two x0 */}
      {hasSecondary ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs">
          <div className="elevated-surface p-3 border border-amber-500/30 space-y-1">
            <div className="flex items-center justify-between gap-2">
              <span className="font-semibold text-[var(--text-nl)]">{primaryLabel}</span>
              <span className="font-mono-num text-[var(--text-secondary)]">{formatStatusShort(primaryStatus)}</span>
            </div>
            <div className="font-mono-num">
              Iteraciones: <strong>{primarySteps.length}</strong> · Orden: <strong>{classifyOrder(primaryOrder)}</strong>
            </div>
            {primaryRootLabel ? (
              <div className="font-mono-num text-[var(--text-secondary)]">{primaryRootLabel}</div>
            ) : null}
          </div>

          <div className="elevated-surface p-3 border border-sky-500/30 space-y-1">
            <div className="flex items-center justify-between gap-2">
              <span className="font-semibold text-[var(--text-sl)]">{secondaryLabel}</span>
              <span className="font-mono-num text-[var(--text-secondary)]">{formatStatusShort(secondaryStatus)}</span>
            </div>
            <div className="font-mono-num">
              Iteraciones: <strong>{secondarySteps?.length ?? 0}</strong> · Orden: <strong>{classifyOrder(secondaryOrder)}</strong>
            </div>
            {secondaryRootLabel ? (
              <div className="font-mono-num text-[var(--text-secondary)]">{secondaryRootLabel}</div>
            ) : null}
          </div>
        </div>
      ) : null}

      <div className="relative w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto min-w-[420px]"
          onMouseLeave={() => setHoverIter(null)}
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const relX = ((e.clientX - rect.left) / rect.width) * width;
            const norm = (relX - pad.left) / plotW;
            const estK = maxK <= 1
              ? 1
              : Math.round(1 + Math.max(0, Math.min(1, norm)) * (maxK - 1));
            setHoverIter(estK);
          }}
        >
          {yTicks.map((tick) => {
            const y = yScale(tick);
            return (
              <g key={tick}>
                <line
                  x1={pad.left}
                  y1={y}
                  x2={width - pad.right}
                  y2={y}
                  stroke="var(--border-subtle)"
                  strokeDasharray="3 3"
                />
                <text
                  x={pad.left - 8}
                  y={y + 4}
                  textAnchor="end"
                  className="fill-[var(--text-secondary)] font-mono-num"
                  fontSize="10"
                >
                  10^{tick}
                </text>
              </g>
            );
          })}

          <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} stroke="var(--border-strong)" />
          <line x1={pad.left} y1={height - pad.bottom} x2={width - pad.right} y2={height - pad.bottom} stroke="var(--border-strong)" />

          {[1, Math.ceil(maxK / 2), maxK]
            .filter((v, i, arr) => arr.indexOf(v) === i)
            .map((kTick) => (
              <text
                key={kTick}
                x={xScale(kTick)}
                y={height - pad.bottom + 16}
                textAnchor="middle"
                className="fill-[var(--text-secondary)] font-mono-num"
                fontSize="10"
              >
                k={kTick}
              </text>
            ))}

          <text
            x={pad.left + plotW / 2}
            y={height - 5}
            textAnchor="middle"
            className="fill-[var(--text-muted)]"
            fontSize="11"
          >
            Iteración (k)
          </text>

          {/* Primary curves */}
          {(metricView === 'both' || metricView === 'abs_error') && pPts.length > 0 ? (
            <path
              d={buildPath(pPts.map((p) => ({ k: p.k, val: p.errLog, isZero: p.errZero })))}
              fill="none"
              stroke={COLORS.primaryError}
              strokeWidth="2.4"
            />
          ) : null}
          {(metricView === 'both' || metricView === 'residual') && pPts.length > 0 ? (
            <path
              d={buildPath(pPts.map((p) => ({ k: p.k, val: p.resLog, isZero: p.resZero })))}
              fill="none"
              stroke={COLORS.primaryResidual}
              strokeWidth="1.9"
              strokeDasharray="5 3"
            />
          ) : null}

          {/* Secondary curves */}
          {(metricView === 'both' || metricView === 'abs_error') && sPts.length > 0 ? (
            <path
              d={buildPath(sPts.map((p) => ({ k: p.k, val: p.errLog, isZero: p.errZero })))}
              fill="none"
              stroke={COLORS.secondaryError}
              strokeWidth="2.4"
            />
          ) : null}
          {(metricView === 'both' || metricView === 'residual') && sPts.length > 0 ? (
            <path
              d={buildPath(sPts.map((p) => ({ k: p.k, val: p.resLog, isZero: p.resZero })))}
              fill="none"
              stroke={COLORS.secondaryResidual}
              strokeWidth="1.9"
              strokeDasharray="5 3"
            />
          ) : null}

          {/* Dots */}
          {maxK <= 30 ? (
            <>
              {(metricView === 'both' || metricView === 'abs_error')
                ? pPts
                    .filter((p) => p.errLog !== null || p.errZero)
                    .map((p) => (
                      <circle
                        key={`pe-${p.k}`}
                        cx={xScale(p.k)}
                        cy={yScale(p.errLog ?? minLog)}
                        r="3.2"
                        fill={COLORS.primaryError}
                      />
                    ))
                : null}
              {(metricView === 'both' || metricView === 'residual')
                ? pPts
                    .filter((p) => p.resLog !== null || p.resZero)
                    .map((p) => (
                      <circle
                        key={`pr-${p.k}`}
                        cx={xScale(p.k)}
                        cy={yScale(p.resLog ?? minLog)}
                        r="2.6"
                        fill={COLORS.primaryResidual}
                      />
                    ))
                : null}
              {(metricView === 'both' || metricView === 'abs_error')
                ? sPts
                    .filter((p) => p.errLog !== null || p.errZero)
                    .map((p) => (
                      <circle
                        key={`se-${p.k}`}
                        cx={xScale(p.k)}
                        cy={yScale(p.errLog ?? minLog)}
                        r="3.2"
                        fill={COLORS.secondaryError}
                      />
                    ))
                : null}
              {(metricView === 'both' || metricView === 'residual')
                ? sPts
                    .filter((p) => p.resLog !== null || p.resZero)
                    .map((p) => (
                      <circle
                        key={`sr-${p.k}`}
                        cx={xScale(p.k)}
                        cy={yScale(p.resLog ?? minLog)}
                        r="2.6"
                        fill={COLORS.secondaryResidual}
                      />
                    ))
                : null}
            </>
          ) : null}

          {hoverIter !== null ? (
            <line
              x1={xScale(hoverIter)}
              y1={pad.top}
              x2={xScale(hoverIter)}
              y2={height - pad.bottom}
              stroke="var(--text-secondary)"
              strokeDasharray="4 2"
              strokeWidth="1"
            />
          ) : null}
        </svg>

        {hoverIter !== null ? (
          <div className="mt-2 flex flex-wrap items-center gap-4 text-xs font-mono-num elevated-surface px-3 py-1.5">
            <span className="font-semibold text-[var(--text-primary)]">k = {hoverIter}:</span>
            {findAt(pPts, hoverIter) ? (
              <span className="inline-flex items-center gap-2 text-[var(--text-primary)]">
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: COLORS.primaryError }} />
                {primaryLabel}: paso {formatExp(findAt(pPts, hoverIter)!.errRaw)}
                {findAt(pPts, hoverIter)!.resRaw !== null
                  ? ` · res ${formatExp(findAt(pPts, hoverIter)!.resRaw)}`
                  : ''}
              </span>
            ) : null}
            {hasSecondary && findAt(sPts, hoverIter) ? (
              <span className="inline-flex items-center gap-2 text-[var(--text-primary)]">
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: COLORS.secondaryError }} />
                {secondaryLabel}: paso {formatExp(findAt(sPts, hoverIter)!.errRaw)}
                {findAt(sPts, hoverIter)!.resRaw !== null
                  ? ` · res ${formatExp(findAt(sPts, hoverIter)!.resRaw)}`
                  : ''}
              </span>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
