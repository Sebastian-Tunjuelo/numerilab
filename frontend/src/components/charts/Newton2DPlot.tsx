import { useMemo, useRef } from 'react';
import { Line, Mafs, Point, Theme } from 'mafs';
import type { Contour2DData, NewtonSystemResponse } from '../../types/numerical';
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

interface Newton2DPlotProps {
  x0: number[];
  result: NewtonSystemResponse | null;
  activeStepIndex: number;
  isStale?: boolean;
  onPickInitial: (point: [number, number]) => void;
}

function sanitize2DViewBox(
  contour: Contour2DData | null | undefined,
  safeX: number,
  safeY: number,
  trajectoryPoints: Array<[number, number]>,
): { viewBox: PlotViewBox; clamped: boolean } {
  const anchorX = isPlotCoordValid(safeX) ? safeX : 0;
  const anchorY = isPlotCoordValid(safeY) ? safeY : 0;
  if (!contour) {
    return {
      viewBox: { x: [anchorX - 4, anchorX + 4], y: [anchorY - 4, anchorY + 4] },
      clamped: false,
    };
  }

  const { bounds } = contour;
  const rawFinite =
    Number.isFinite(bounds.x_min)
    && Number.isFinite(bounds.x_max)
    && Number.isFinite(bounds.y_min)
    && Number.isFinite(bounds.y_max)
    && bounds.x_max > bounds.x_min
    && bounds.y_max > bounds.y_min;

  if (!rawFinite) {
    return {
      viewBox: { x: [anchorX - 4, anchorX + 4], y: [anchorY - 4, anchorY + 4] },
      clamped: true,
    };
  }

  let clamped = false;
  const rawXSpan = bounds.x_max - bounds.x_min;
  const rawYSpan = bounds.y_max - bounds.y_min;

  const clampAxis = (
    minVal: number,
    maxVal: number,
    rawSpan: number,
    anchor: number,
    coordIdx: 0 | 1,
  ): [number, number] => {
    if (
      rawSpan > MAX_PLOT_SPAN
      || Math.abs(minVal) > MAX_ABS_COORD
      || Math.abs(maxVal) > MAX_ABS_COORD
    ) {
      clamped = true;
      const near = [
        anchor,
        ...trajectoryPoints
          .map((pt) => pt[coordIdx])
          .filter((v) => Number.isFinite(v) && Math.abs(v - anchor) <= MAX_PLOT_SPAN / 2),
      ];
      const minNear = Math.min(...near);
      const maxNear = Math.max(...near);
      const span = Math.max(maxNear - minNear, 4);
      const pad = Math.min(span * 0.45, MAX_PLOT_SPAN / 4);
      return clampSpanAroundCenter(minNear - pad, maxNear + pad).range;
    }
    const res = clampSpanAroundCenter(minVal, maxVal);
    clamped = clamped || res.clamped;
    return res.range;
  };

  const xRange = clampAxis(bounds.x_min, bounds.x_max, rawXSpan, anchorX, 0);
  const yRange = clampAxis(bounds.y_min, bounds.y_max, rawYSpan, anchorY, 1);

  const balanced = balanceAspectRatio(xRange, yRange);
  clamped = clamped || balanced.clamped;

  return { viewBox: { x: balanced.x, y: balanced.y }, clamped };
}

export function Newton2DPlot({
  x0,
  result,
  activeStepIndex,
  isStale = false,
  onPickInitial,
}: Newton2DPlotProps) {
  const contour = result?.contour_2d;
  const pointerDownRef = useRef<{ x: number; y: number } | null>(null);

  const safeX = Number.isFinite(x0[0]) ? x0[0] : 0;
  const safeY = Number.isFinite(x0[1]) ? x0[1] : 0;

  const trajectoryPoints = useMemo(() => {
    const pts: Array<[number, number]> = [[safeX, safeY]];
    if (result?.steps) {
      for (const s of result.steps) {
        if (s.x_next_raw.length >= 2) {
          pts.push([s.x_next_raw[0], s.x_next_raw[1]]);
        }
      }
    }
    return pts;
  }, [safeX, safeY, result]);

  const { viewBox, clamped: viewBoxClamped } = useMemo(
    () => sanitize2DViewBox(contour, safeX, safeY, trajectoryPoints),
    [contour, safeX, safeY, trajectoryPoints],
  );

  const inPlot = ([x, y]: [number, number]) => isPlotCoordValid(x) && isPlotCoordValid(y);
  const segInPlot = (seg: [number[], number[]]) =>
    seg[0].length >= 2
    && seg[1].length >= 2
    && isPlotCoordValid(seg[0][0])
    && isPlotCoordValid(seg[0][1])
    && isPlotCoordValid(seg[1][0])
    && isPlotCoordValid(seg[1][1]);

  const activePoint =
    trajectoryPoints[Math.min(activeStepIndex + 1, trajectoryPoints.length - 1)] ??
    trajectoryPoints[0];

  return (
    <div className={`panel-surface p-4 transition-opacity ${isStale ? 'opacity-75' : ''}`}>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div>
          <h4 className="text-sm font-semibold">Curvas de nivel cero y trayectoria 2×2</h4>
          <p className="text-xs text-[var(--text-secondary)]">
            Haz <strong>clic en cualquier punto del plano</strong> para probar otro punto inicial. Las curvas son aproximadas y pueden omitir ceros que la malla no detecta.
          </p>
          {viewBoxClamped || trajectoryPoints.some((point) => !inPlot(point)) ? (
            <p className="text-xs text-[var(--text-nl)] mt-0.5">
              Rango visual acotado porque algunos iterados divergieron a valores extremos (consulta todos los pasos en la tabla).
            </p>
          ) : null}
        </div>
        <span className="text-xs font-mono-num elevated-surface px-2.5 py-1 text-[var(--text-nl)]">
          x⁽⁰⁾ = ({safeX.toFixed(2)}, {safeY.toFixed(2)})
        </span>
      </div>

      <div
        onPointerDownCapture={(e) => {
          pointerDownRef.current = { x: e.clientX, y: e.clientY };
        }}
      >
        <MafsErrorBoundary
          resetKey={`${JSON.stringify(viewBox)}:${safeX}:${safeY}:${result?.iterations ?? 0}`}
          description="Las curvas o iterados divergieron fuera del rango gráfico admitido. Consulta los valores exactos en la tabla de iteraciones o haz clic en otro punto inicial."
        >
          <Mafs
            height={360}
            pan
            zoom={{ min: 0.35, max: 12 }}
            viewBox={viewBox}
            onClick={(pt, event) => {
              const down = pointerDownRef.current;
              pointerDownRef.current = null;
              if (down && Math.hypot(event.clientX - down.x, event.clientY - down.y) > 5) {
                return;
              }
              onPickInitial([Number(pt[0].toFixed(2)), Number(pt[1].toFixed(2))]);
            }}
          >
            <CartesianGrid viewBox={viewBox} />

            {/* Zero-contour f1(x,y) = 0 */}
            {contour?.f1_segments.filter(segInPlot).map((seg, i) => (
              <Line.Segment
                key={`f1-${i}`}
                point1={[seg[0][0], seg[0][1]]}
                point2={[seg[1][0], seg[1][1]]}
                color={Theme.blue}
                weight={2.3}
              />
            ))}

            {/* Zero-contour f2(x,y) = 0 */}
            {contour?.f2_segments.filter(segInPlot).map((seg, i) => (
              <Line.Segment
                key={`f2-${i}`}
                point1={[seg[0][0], seg[0][1]]}
                point2={[seg[1][0], seg[1][1]]}
                color={Theme.orange}
                weight={2.3}
              />
            ))}

            {/* Trajectory segments */}
            {trajectoryPoints.slice(0, -1).map((pt, idx) => inPlot(pt) && inPlot(trajectoryPoints[idx + 1]) && (
              <Line.Segment
                key={`traj-${idx}`}
                point1={pt}
                point2={trajectoryPoints[idx + 1]}
                color={Theme.green}
                style="dashed"
                weight={2}
              />
            ))}

            {trajectoryPoints.map((pt, idx) => inPlot(pt) && (
              <Point
                key={`pt-${idx}`}
                x={pt[0]}
                y={pt[1]}
                color={idx === 0 ? Theme.orange : Theme.green}
              />
            ))}

            {activePoint && inPlot(activePoint) ? (
              <Point x={activePoint[0]} y={activePoint[1]} color={Theme.pink} />
            ) : null}
          </Mafs>
        </MafsErrorBoundary>
      </div>

      <div className="mt-2.5 flex flex-wrap items-center gap-4 text-xs text-[var(--text-secondary)]">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-sky-400" /> f₁(x, y) = 0
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-amber-500" /> f₂(x, y) = 0 / Punto inicial x⁽⁰⁾
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" /> Trayectoria Newton
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-pink-500" /> Iteración activa
        </span>
      </div>
    </div>
  );
}
