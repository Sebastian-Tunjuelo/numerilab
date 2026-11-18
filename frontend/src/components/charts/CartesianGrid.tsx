import { Coordinates } from 'mafs';

function gridSpacing([min, max]: [number, number]): number {
  const span = Math.abs(max - min);
  if (!Number.isFinite(span) || span <= 0) return 1;
  const target = span / 10;
  const exp = Math.floor(Math.log10(target));
  const base = 10 ** exp;
  const frac = target / base;
  const step = frac <= 1 ? 1 : frac <= 2 ? 2 : frac <= 5 ? 5 : 10;
  return step * base;
}

export function CartesianGrid({ viewBox }: { viewBox: { x: [number, number]; y: [number, number] } }) {
  return (
    <Coordinates.Cartesian
      xAxis={{ lines: gridSpacing(viewBox.x) }}
      yAxis={{ lines: gridSpacing(viewBox.y) }}
    />
  );
}
