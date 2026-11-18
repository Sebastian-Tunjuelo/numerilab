import { useEffect, useMemo, useRef, useState } from 'react';
import { Line, Mafs, MovablePoint, Plot, Point, Theme } from 'mafs';
import { compile } from 'mathjs';
import type { Newton1DResponse, Newton1DStep } from '../../types/numerical';
import { CartesianGrid } from './CartesianGrid';
import { MafsErrorBoundary } from './MafsErrorBoundary';
import {
  balanceAspectRatio,
  clampSpanAroundCenter,
  isPlotCoordValid,
  MAX_ABS_COORD,
  MAX_PLOT_SPAN,
  type PlotViewBox,
} from './plotBounds';

interface Newton1DPlotProps {
  x0: number;
  result: Newton1DResponse | null;
  activeStepIndex: number;
  isStale?: boolean;
  onDragX0: (nextX0: number) => void;
}

type ViewBox = PlotViewBox;

function sanitizeViewBox(
  bounds: Newton1DResponse['plot_bounds'] | undefined,
  safeX0: number,
  steps: Newton1DStep[],
): { viewBox: ViewBox; clamped: boolean } {
  const anchorX = isPlotCoordValid(safeX0) ? safeX0 : 0;
  if (!bounds) {
    return { viewBox: { x: [anchorX - 4, anchorX + 4], y: [-4, 4] }, clamped: false };
  }

  const rawFinite =
    Number.isFinite(bounds.x_min)
    && Number.isFinite(bounds.x_max)
    && Number.isFinite(bounds.y_min)
    && Number.isFinite(bounds.y_max)
    && bounds.x_max > bounds.x_min
    && bounds.y_max > bounds.y_min;

  if (!rawFinite) {
    return { viewBox: { x: [anchorX - 4, anchorX + 4], y: [-4, 4] }, clamped: true };
  }

  let clamped = false;
  const rawXSpan = bounds.x_max - bounds.x_min;
  let xRange: [number, number];

  if (
    rawXSpan > MAX_PLOT_SPAN
    || Math.abs(bounds.x_min) > MAX_ABS_COORD
    || Math.abs(bounds.x_max) > MAX_ABS_COORD
  ) {
    clamped = true;
    const nearX = [
      anchorX,
      ...steps
        .flatMap((s) => [s.x_k_raw, s.x_next_raw])
        .filter((v) => Number.isFinite(v) && Math.abs(v - anchorX) <= MAX_PLOT_SPAN / 2),
    ];
    const minNear = Math.min(...nearX);
    const maxNear = Math.max(...nearX);
    const span = Math.max(maxNear - minNear, 4);
    const pad = Math.min(span * 0.45, MAX_PLOT_SPAN / 4);
    const res = clampSpanAroundCenter(minNear - pad, maxNear + pad);
    xRange = res.range;
  } else {
    const res = clampSpanAroundCenter(bounds.x_min, bounds.x_max);
    xRange = res.range;
    clamped = clamped || res.clamped;
  }

  const rawYSpan = bounds.y_max - bounds.y_min;
  let yRange: [number, number];
  if (
    rawYSpan > MAX_PLOT_SPAN
    || Math.abs(bounds.y_min) > MAX_ABS_COORD
    || Math.abs(bounds.y_max) > MAX_ABS_COORD
  ) {
    clamped = true;
    // Anchor vertical clamping around y = 0 so the X-axis, roots, and draggable x0 remain visible.
    const nearY = [
      0,
      ...steps
        .map((s) => s.fx_k)
        .filter((v) => Number.isFinite(v) && Math.abs(v) <= MAX_PLOT_SPAN / 2),
    ];
    const minNearY = Math.min(...nearY);
    const maxNearY = Math.max(...nearY);
    const ySpanNear = Math.max(maxNearY - minNearY, 4);
    const yPad = Math.min(ySpanNear * 0.45, MAX_PLOT_SPAN / 4);
    const yRes = clampSpanAroundCenter(minNearY - yPad, maxNearY + yPad);
    yRange = yRes.range;
  } else {
    const yRes = clampSpanAroundCenter(bounds.y_min, bounds.y_max);
    yRange = yRes.range;
    clamped = clamped || yRes.clamped;
  }

  // Ensure y = 0 (the X-axis and draggable x0) stays visibly inside yRange.
  if (yRange[0] > -0.5) {
    const shift = yRange[0] + 0.5;
    yRange = [yRange[0] - shift, yRange[1] - shift];
  } else if (yRange[1] < 0.5) {
    const shift = 0.5 - yRange[1];
    yRange = [yRange[0] + shift, yRange[1] + shift];
  }

  const balanced = balanceAspectRatio(xRange, yRange);
  clamped = clamped || balanced.clamped;

  return { viewBox: { x: balanced.x, y: balanced.y }, clamped };
}

export function Newton1DPlot({
  x0,
  result,
  activeStepIndex,
  isStale = false,
  onDragX0,
}: Newton1DPlotProps) {
  const safeX0 = Number.isFinite(x0) ? x0 : 0;
  const [point, setPoint] = useState<[number, number]>([safeX0, 0]);
  const [lockedViewBox, setLockedViewBox] = useState<ViewBox | null>(null);
  const dragTimer = useRef<number | undefined>(undefined);
  const dragUnlockTimer = useRef<number | undefined>(undefined);
  const pointerDownRef = useRef(false);
  const isInteractingRef = useRef(false);
  const hasPendingDragRef = useRef(false);
  const awaitingResultRef = useRef(false);
  const prevResultRef = useRef<Newton1DResponse | null>(result);
  const lockedViewBoxRef = useRef<ViewBox | null>(null);
  lockedViewBoxRef.current = lockedViewBox;

  const bounds = result?.plot_bounds;
  const steps = result?.steps ?? [];
  const { viewBox: computedViewBox, clamped: viewBoxClamped } = useMemo(
    () => sanitizeViewBox(bounds, safeX0, steps),
    [bounds, safeX0, steps],
  );

  useEffect(() => {
    const endDrag = () => {
      pointerDownRef.current = false;
      if (!hasPendingDragRef.current && isInteractingRef.current) {
        isInteractingRef.current = false;
        if (!awaitingResultRef.current) {
          setLockedViewBox(null);
        }
      }
    };
    window.addEventListener('pointerup', endDrag);
    window.addEventListener('pointercancel', endDrag);
    return () => {
      window.removeEventListener('pointerup', endDrag);
      window.removeEventListener('pointercancel', endDrag);
      window.clearTimeout(dragTimer.current);
      window.clearTimeout(dragUnlockTimer.current);
    };
  }, []);

  useEffect(() => {
    if (awaitingResultRef.current && result !== prevResultRef.current) {
      awaitingResultRef.current = false;
      window.clearTimeout(dragUnlockTimer.current);
      if (!isInteractingRef.current && !hasPendingDragRef.current) {
        setLockedViewBox(null);
      }
    }
    prevResultRef.current = result;
  }, [result]);

  useEffect(() => {
    if (!isInteractingRef.current && !hasPendingDragRef.current) {
      setPoint([safeX0, 0]);
    }
  }, [safeX0]);

  const moveInitialPoint = ([x]: [number, number]) => {
    if (!Number.isFinite(x)) return;
    isInteractingRef.current = true;
    hasPendingDragRef.current = true;
    awaitingResultRef.current = false;
    window.clearTimeout(dragUnlockTimer.current);
    if (!lockedViewBoxRef.current) {
      setLockedViewBox(computedViewBox);
    }
    setPoint([x, 0]);
    window.clearTimeout(dragTimer.current);
    dragTimer.current = window.setTimeout(() => {
      hasPendingDragRef.current = false;
      const nextX = Number(x.toFixed(3));
      const didChange = nextX !== safeX0;
      if (didChange) {
        awaitingResultRef.current = true;
        prevResultRef.current = result;
        onDragX0(nextX);
        dragUnlockTimer.current = window.setTimeout(() => {
          awaitingResultRef.current = false;
          if (!isInteractingRef.current && !hasPendingDragRef.current) {
            setLockedViewBox(null);
          }
        }, 1500);
      }
      if (!pointerDownRef.current) {
        isInteractingRef.current = false;
        if (!didChange) {
          setLockedViewBox(null);
        }
      }
    }, 160);
  };

  const expression = result?.expression_normalized;
  const evalFn = useMemo(() => {
    if (!expression) return null;
    try {
      const compiled = compile(expression.replace(/\*\*/g, '^').replace(/\bE\b/g, 'e').replace(/\bAbs\b/g, 'abs'));
      return (x: number) => {
        try {
          const value = compiled.evaluate({ x });
          return typeof value === 'number' && Number.isFinite(value) ? value : Number.NaN;
        } catch {
          return Number.NaN;
        }
      };
    } catch {
      return null;
    }
  }, [expression]);

  const viewBox = lockedViewBox ?? computedViewBox;

  const activeStep =
    result && result.steps.length > 0
      ? result.steps[Math.min(activeStepIndex, result.steps.length - 1)]
      : null;
  const stepInRange = activeStep && [activeStep.x_k_raw, activeStep.fx_k, activeStep.x_next_raw]
    .every((value) => Number.isFinite(value) && Math.abs(value) <= MAX_ABS_COORD);

  return (
    <div className={`panel-surface p-4 transition-opacity ${isStale ? 'opacity-75' : ''}`}>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div>
          <h4 className="text-sm font-semibold">Plano cartesiano interactivo (Mafs)</h4>
          <p className="text-xs text-[var(--text-secondary)]">
            Arrastra el punto naranja <strong className="text-[var(--text-nl)]">x₀</strong> sobre el eje X para recalcular la trayectoria tangente en vivo. Usa la rueda para zoom.
          </p>
          {viewBoxClamped ? (
            <p className="text-xs text-[var(--text-nl)] mt-0.5">
              Rango visual acotado alrededor de x₀ porque algunos iterados divergieron a valores extremos (consulta todos los pasos en la tabla).
            </p>
          ) : null}
          {activeStep && !stepInRange ? (
            <p className="text-xs text-[var(--text-secondary)]">Consulta la tangente de este paso en la tabla. Sus coordenadas superan el rango gráfico.</p>
          ) : null}
        </div>
        <span className="text-xs font-mono-num elevated-surface px-2.5 py-1 text-[var(--text-nl)]">
          x₀ = {point[0].toFixed(3)}
        </span>
      </div>

      <div
        onPointerDownCapture={() => {
          pointerDownRef.current = true;
        }}
      >
        <MafsErrorBoundary resetKey={`${JSON.stringify(viewBox)}:${expression ?? ''}:${safeX0}`}>
          <Mafs height={360} pan zoom={{ min: 0.25, max: 12 }} viewBox={viewBox} preserveAspectRatio={false}>
            <CartesianGrid viewBox={viewBox} />

            {evalFn ? (
              <Plot.OfX y={evalFn} color={Theme.blue} weight={2.5} />
            ) : null}

            {/* Vertical & Tangent Geometry for the active Newton step */}
            {activeStep && stepInRange ? (
              <>
                <Line.Segment
                  point1={[activeStep.x_k_raw, 0]}
                  point2={[activeStep.x_k_raw, activeStep.fx_k]}
                  color="#94a3b8"
                  style="dashed"
                  weight={1.5}
                />
                {Number.isFinite(activeStep.x_next_raw) ? (
                  <>
                    <Line.Segment
                      point1={[activeStep.x_k_raw, activeStep.fx_k]}
                      point2={[activeStep.x_next_raw, 0]}
                      color={Theme.orange}
                      weight={2.2}
                    />
                    <Point x={activeStep.x_next_raw} y={0} color={Theme.green} />
                  </>
                ) : null}
                <Point x={activeStep.x_k_raw} y={activeStep.fx_k} color={Theme.pink} />
              </>
            ) : null}

            {/* Final root marker */}
            {result?.root_raw != null && Number.isFinite(result.root_raw) && Math.abs(result.root_raw) <= MAX_ABS_COORD ? (
              <Point x={result.root_raw} y={0} color={Theme.green} />
            ) : null}

            {Number.isFinite(point[0]) && Math.abs(point[0]) <= MAX_ABS_COORD ? (
              <MovablePoint point={point} onMove={moveInitialPoint} constrain={([x]) => [x, 0]} color={Theme.orange} />
            ) : null}
          </Mafs>
        </MafsErrorBoundary>
      </div>

      <div className="mt-2.5 flex flex-wrap items-center gap-4 text-xs text-[var(--text-secondary)]">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-sky-400" /> Curva f(x)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-amber-500" /> Punto arrastrable x₀ / Recta tangente
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" /> Siguiente xₖ₊₁ / Raíz
        </span>
      </div>
    </div>
  );
}
