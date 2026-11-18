export type PlotViewBox = { x: [number, number]; y: [number, number] };

export const MAX_PLOT_SPAN = 400;
export const MIN_PLOT_SPAN = 0.25;
export const MAX_ABS_COORD = 1e5;
export const MAX_ASPECT_RATIO = 500;

export function isPlotCoordValid(value: number): boolean {
  return Number.isFinite(value) && Math.abs(value) <= MAX_ABS_COORD;
}

export function clampSpanAroundCenter(
  minVal: number,
  maxVal: number,
): { range: [number, number]; clamped: boolean } {
  let clamped = false;
  let min = Number.isFinite(minVal) ? minVal : -4;
  let max = Number.isFinite(maxVal) ? maxVal : 4;
  if (min > max) [min, max] = [max, min];

  if (Math.abs(min) > MAX_ABS_COORD || Math.abs(max) > MAX_ABS_COORD) {
    clamped = true;
    min = Math.max(-MAX_ABS_COORD, Math.min(MAX_ABS_COORD, min));
    max = Math.max(-MAX_ABS_COORD, Math.min(MAX_ABS_COORD, max));
  }

  let span = max - min;
  const center = (min + max) / 2;
  if (!Number.isFinite(span) || span < MIN_PLOT_SPAN) {
    min = center - MIN_PLOT_SPAN / 2;
    max = center + MIN_PLOT_SPAN / 2;
    span = MIN_PLOT_SPAN;
  }
  if (span > MAX_PLOT_SPAN) {
    clamped = true;
    min = center - MAX_PLOT_SPAN / 2;
    max = center + MAX_PLOT_SPAN / 2;
  }
  return { range: [min, max], clamped };
}

export function balanceAspectRatio(
  xRangeInit: [number, number],
  yRangeInit: [number, number],
): { x: [number, number]; y: [number, number]; clamped: boolean } {
  let xRange = xRangeInit;
  let yRange = yRangeInit;
  let clamped = false;

  let xSpan = xRange[1] - xRange[0];
  let ySpan = yRange[1] - yRange[0];
  if (xSpan / ySpan > MAX_ASPECT_RATIO) {
    const targetYSpan = Math.min(MAX_PLOT_SPAN, xSpan / MAX_ASPECT_RATIO);
    const yCenter = (yRange[0] + yRange[1]) / 2;
    yRange = [yCenter - targetYSpan / 2, yCenter + targetYSpan / 2];
    ySpan = targetYSpan;
    clamped = true;
  }
  if (ySpan / xSpan > MAX_ASPECT_RATIO) {
    const targetXSpan = Math.min(MAX_PLOT_SPAN, ySpan / MAX_ASPECT_RATIO);
    const xCenter = (xRange[0] + xRange[1]) / 2;
    xRange = [xCenter - targetXSpan / 2, xCenter + targetXSpan / 2];
    clamped = true;
  }

  return { x: xRange, y: yRange, clamped };
}
