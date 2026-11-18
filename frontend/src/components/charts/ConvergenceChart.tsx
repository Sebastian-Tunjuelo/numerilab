import { useMemo, useState } from 'react';
import type { IterativeLinearResult } from '../../types/numerical';

interface ConvergenceChartProps {
  jacobi: IterativeLinearResult;
  gaussSeidel: IterativeLinearResult;
  sor: IterativeLinearResult;
  metric: 'abs_error' | 'residual';
}

const COLORS = {
  jacobi: 'var(--chart-jacobi)',
  gaussSeidel: 'var(--chart-gs)',
  sor: 'var(--chart-sor)',
};

function formatExp(val?: number | null): string {
  if (val === null || val === undefined || !Number.isFinite(val)) return '—';
  return val.toExponential(3);
}

export function ConvergenceChart({
  jacobi,
  gaussSeidel,
  sor,
  metric,
}: ConvergenceChartProps) {
  const [hoverIter, setHoverIter] = useState<number | null>(null);

  const chartData = useMemo(() => {
    const extract = (res: IterativeLinearResult) =>
      res.steps.map((s) => {
        const raw = metric === 'abs_error' ? s.abs_error : s.residual;
        return {
          k: s.iteration,
          raw,
          isZero: raw === 0,
          logVal: Number.isFinite(raw) && raw > 0 ? Math.log10(raw) : null,
        };
      });

    const jPts = extract(jacobi);
    const gsPts = extract(gaussSeidel);
    const sorPts = extract(sor);
    const allPts = [...jPts, ...gsPts, ...sorPts];

    const maxK = Math.max(
      1,
      jPts.length ? jPts[jPts.length - 1].k : 1,
      gsPts.length ? gsPts[gsPts.length - 1].k : 1,
      sorPts.length ? sorPts[sorPts.length - 1].k : 1,
    );

    const logValues = allPts.flatMap((p) => p.logVal === null ? [] : [p.logVal]);
    const minLog = logValues.length ? Math.floor(Math.min(...logValues, -8)) : -8;
    const maxLog = logValues.length ? Math.ceil(Math.max(...logValues, 1)) : 1;

    return { jPts, gsPts, sorPts, maxK, minLog, maxLog };
  }, [jacobi, gaussSeidel, sor, metric]);

  const width = 680;
  const height = 270;
  const pad = { top: 22, right: 24, bottom: 38, left: 54 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const { jPts, gsPts, sorPts, maxK, minLog, maxLog } = chartData;
  const logSpan = Math.max(maxLog - minLog, 1);

  const xScale = (k: number) =>
    maxK <= 1
      ? pad.left + plotW / 2
      : pad.left + ((k - 1) / (maxK - 1)) * plotW;
  const yScale = (logVal: number) =>
    pad.top + ((maxLog - logVal) / logSpan) * plotH;

  const buildPath = (pts: Array<{ k: number; isZero: boolean; logVal: number | null }>) => {
    const cmds: string[] = [];
    let drawing = false;
    for (const p of pts) {
      if (p.logVal === null && !p.isZero) {
        drawing = false;
        continue;
      }
      const y = yScale(p.logVal ?? minLog);
      cmds.push(`${drawing ? 'L' : 'M'} ${xScale(p.k).toFixed(2)} ${y.toFixed(2)}`);
      drawing = true;
    }
    return cmds.join(' ');
  };

  const yTicks: number[] = [];
  const stepTick = Math.max(1, Math.ceil(logSpan / 5));
  for (let v = minLog; v <= maxLog; v += stepTick) {
    yTicks.push(v);
  }

  const findAt = (pts: Array<{ k: number; raw: number; isZero: boolean; logVal: number | null }>, k: number) =>
    pts.find((p) => p.k === k);

  const allEmpty = jPts.length === 0 && gsPts.length === 0 && sorPts.length === 0;

  if (allEmpty) {
    return (
      <div className="panel-surface p-4 space-y-1">
        <h4 className="text-sm font-semibold">
          Convergencia comparativa ({metric === 'abs_error' ? 'Error absoluto ||x⁽ᵏ⁾ - x⁽ᵏ⁻¹⁾||∞' : 'Residual ||Ax⁽ᵏ⁾ - b||∞'})
        </h4>
        <p className="text-xs text-[var(--text-secondary)]">
          No hay iteraciones disponibles para graficar (por ejemplo, por diagonal nula en A o fallo previo a la iteración 1).
        </p>
      </div>
    );
  }

  return (
    <div className="panel-surface p-4">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div>
          <h4 className="text-sm font-semibold">
            Convergencia comparativa ({metric === 'abs_error' ? 'Error absoluto ||x⁽ᵏ⁾ - x⁽ᵏ⁻¹⁾||∞' : 'Residual ||Ax⁽ᵏ⁾ - b||∞'})
          </h4>
          <p className="text-xs text-[var(--text-secondary)]">
            Escala logarítmica base 10. Los valores iguales a cero se muestran en el límite inferior.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-4 text-xs">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: COLORS.jacobi }} />
            Jacobi ({jacobi.iterations} it.)
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: COLORS.gaussSeidel }} />
            Gauss-Seidel ({gaussSeidel.iterations} it.)
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: COLORS.sor }} />
            SOR ω={sor.omega.toFixed(2)} ({sor.iterations} it.)
          </span>
        </div>
      </div>

      <div className="relative w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto min-w-[440px]"
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
          {/* Horizontal grid & Y axis labels */}
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

          {/* Axes */}
          <line
            x1={pad.left}
            y1={pad.top}
            x2={pad.left}
            y2={height - pad.bottom}
            stroke="var(--border-strong)"
          />
          <line
            x1={pad.left}
            y1={height - pad.bottom}
            x2={width - pad.right}
            y2={height - pad.bottom}
            stroke="var(--border-strong)"
          />

          {/* X Axis ticks */}
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

          {/* Lines */}
          {jPts.length > 0 ? (
            <path d={buildPath(jPts)} fill="none" stroke={COLORS.jacobi} strokeWidth="2.3" />
          ) : null}
          {gsPts.length > 0 ? (
            <path d={buildPath(gsPts)} fill="none" stroke={COLORS.gaussSeidel} strokeWidth="2.3" />
          ) : null}
          {sorPts.length > 0 ? (
            <path d={buildPath(sorPts)} fill="none" stroke={COLORS.sor} strokeWidth="2.6" />
          ) : null}

          {/* Points when small iteration count */}
          {maxK <= 30
            ? [
                { pts: jPts, color: COLORS.jacobi, key: 'j' },
                { pts: gsPts, color: COLORS.gaussSeidel, key: 'gs' },
                { pts: sorPts, color: COLORS.sor, key: 'sor' },
              ].map(({ pts, color, key }) =>
                pts
                  .filter((p) => p.logVal !== null || p.isZero)
                  .map((p) => (
                    <circle
                      key={`${key}-${p.k}`}
                      cx={xScale(p.k)}
                      cy={yScale(p.logVal ?? minLog)}
                      r="3.2"
                      fill={color}
                    />
                  )),
              )
            : null}

          {/* Hover crosshair */}
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
            <span className="font-semibold text-[var(--text-primary)]">Iteración k = {hoverIter}:</span>
            <span className="inline-flex items-center gap-1.5 font-medium text-[var(--text-primary)]">
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: COLORS.jacobi }} />
              Jacobi: {formatExp(findAt(jPts, hoverIter)?.raw)}
            </span>
            <span className="inline-flex items-center gap-1.5 font-medium text-[var(--text-primary)]">
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: COLORS.gaussSeidel }} />
              GS: {formatExp(findAt(gsPts, hoverIter)?.raw)}
            </span>
            <span className="inline-flex items-center gap-1.5 font-medium text-[var(--text-primary)]">
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: COLORS.sor }} />
              SOR: {formatExp(findAt(sorPts, hoverIter)?.raw)}
            </span>
          </div>
        ) : null}
      </div>
    </div>
  );
}
