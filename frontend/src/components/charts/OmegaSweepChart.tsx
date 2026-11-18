import type { SORSweepResponse } from '../../types/numerical';

interface OmegaSweepChartProps {
  sweep: SORSweepResponse | null;
  loading: boolean;
  failed: boolean;
  isStale?: boolean;
  currentOmega: number;
  onSelectOmega: (omega: number) => void;
}

export function OmegaSweepChart({
  sweep,
  loading,
  failed,
  isStale = false,
  currentOmega,
  onSelectOmega,
}: OmegaSweepChartProps) {
  if (loading) {
    return (
      <div className="panel-surface p-4 text-sm text-[var(--text-secondary)]">
        Analizando barrido de relajación ω ∈ [0.10, 1.95]...
      </div>
    );
  }

  if (!sweep || !sweep.available || sweep.points.length === 0) {
    return (
      <div className="panel-surface p-4 text-sm text-[var(--text-secondary)]">
        {failed
          ? 'No se pudo completar el barrido SOR. Vuelve a resolver el sistema para reintentarlo.'
          : sweep?.reason || 'Barrido SOR no disponible para esta matriz.'}
      </div>
    );
  }

  const maxIterShown = Math.max(...sweep.points.map((p) => p.iterations), 10);
  const width = 680;
  const height = 232;
  const pad = { top: 26, right: 24, bottom: 36, left: 46 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const barW = plotW / sweep.points.length;
  const standardTicks = [0.2, 0.5, 1.0, 1.5, 1.9];

  return (
    <div className="panel-surface p-4">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div>
          <h4 className="text-sm font-semibold">
            Optimizador de sobrerrelajación sucesiva SOR (ω)
          </h4>
          <p className="text-xs text-[var(--text-secondary)]">
            {isStale
              ? 'Barrido desactualizado (corresponde al sistema anterior). Vuelve a resolver para recalcular ω*.'
              : 'Haz clic en cualquier barra para aplicar ese ω. Barras rojas indican divergencia o máximo alcanzado.'}
          </p>
        </div>
        {sweep.optimal_omega !== null ? (
          <button
            type="button"
            disabled={isStale}
            onClick={() => {
              if (!isStale) onSelectOmega(sweep.optimal_omega!);
            }}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-500/15 text-[var(--text-ok)] border border-emerald-500/30 hover:bg-emerald-500/25 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Aplicar óptimo: ω* = {sweep.optimal_omega.toFixed(2)} ({sweep.optimal_iterations} it.
            {sweep.optimal_spectral_radius != null ? ` · ρ=${sweep.optimal_spectral_radius.toFixed(3)}` : ''})
          </button>
        ) : null}
      </div>

      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto min-w-[440px]">
          <line
            x1={pad.left}
            y1={height - pad.bottom}
            x2={width - pad.right}
            y2={height - pad.bottom}
            stroke="var(--border-strong)"
          />
          <line
            x1={pad.left}
            y1={pad.top}
            x2={pad.left}
            y2={height - pad.bottom}
            stroke="var(--border-strong)"
          />

          <text
            x={pad.left - 8}
            y={pad.top + 8}
            textAnchor="end"
            className="fill-[var(--text-secondary)] font-mono-num"
            fontSize="10"
          >
            {maxIterShown}
          </text>
          <text
            x={pad.left - 8}
            y={height - pad.bottom}
            textAnchor="end"
            className="fill-[var(--text-secondary)] font-mono-num"
            fontSize="10"
          >
            0
          </text>

          {sweep.points.map((pt, idx) => {
            const x = pad.left + idx * barW + 1.5;
            const rawH = (pt.iterations / maxIterShown) * plotH;
            const barH = Math.max(rawH, 2);
            const y = height - pad.bottom - barH;
            const isOptimal = sweep.optimal_omega !== null && Math.abs(pt.omega - sweep.optimal_omega) < 0.01;
            const isCurrent = Math.abs(pt.omega - currentOmega) < 0.026;

            let fill = pt.converged ? 'rgba(56, 189, 248, 0.55)' : 'rgba(244, 63, 94, 0.42)';
            if (isOptimal) fill = '#10b981';
            if (isCurrent) fill = '#f59e0b';

            const isStandardTick = standardTicks.some((tick) => Math.abs(pt.omega - tick) < 0.01);
            const barWidth = Math.max(barW - 3, 3);
            const barCenter = x + barWidth / 2;

            return (
              <g
                key={pt.omega}
                role="button"
                tabIndex={isStale ? -1 : 0}
                aria-disabled={isStale || undefined}
                aria-label={`Aplicar omega ${pt.omega.toFixed(2)}, ${pt.converged ? `${pt.iterations} iteraciones` : 'no convergió'}`}
                className={`omega-bar focus:outline-none ${isStale ? 'cursor-not-allowed' : 'cursor-pointer'}`}
                onClick={() => {
                  if (!isStale) onSelectOmega(pt.omega);
                }}
                onKeyDown={(e) => {
                  if (isStale) return;
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onSelectOmega(pt.omega);
                  }
                }}
              >
                <title>
                  {`ω = ${pt.omega.toFixed(2)} | ${
                    pt.converged ? `${pt.iterations} iteraciones` : 'No convergió'
                  }${pt.spectral_radius !== null ? ` | ρ(T_ω) = ${pt.spectral_radius.toFixed(4)}` : ''}`}
                </title>
                <rect
                  x={x}
                  y={y}
                  width={barWidth}
                  height={barH}
                  rx="2"
                  fill={fill}
                />
                {isCurrent || (isOptimal && Math.abs(pt.omega - currentOmega) >= 0.24) ? (
                  <text
                    x={barCenter}
                    y={Math.max(y - 5, 12)}
                    textAnchor="middle"
                    className="font-mono-num font-semibold"
                    fill={isCurrent ? 'var(--text-nl)' : 'var(--text-ok)'}
                    fontSize="9.5"
                  >
                    {isCurrent ? `ω=${pt.omega.toFixed(2)}` : `ω*=${pt.omega.toFixed(2)}`}
                  </text>
                ) : null}
                {isStandardTick ? (
                  <text
                    x={barCenter}
                    y={height - pad.bottom + 14}
                    textAnchor="middle"
                    className="fill-[var(--text-secondary)] font-mono-num"
                    fontSize="9.5"
                  >
                    {pt.omega.toFixed(1)}
                  </text>
                ) : null}
              </g>
            );
          })}

          <text
            x={pad.left + plotW / 2}
            y={height - 4}
            textAnchor="middle"
            className="fill-[var(--text-muted)]"
            fontSize="10.5"
          >
            Parámetro de relajación ω (verde = óptimo ω*, ámbar = ω seleccionado)
          </text>
        </svg>
      </div>
    </div>
  );
}
