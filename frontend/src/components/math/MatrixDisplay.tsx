import { LatexRenderer } from './LatexRenderer';

interface MatrixDisplayProps {
  label?: string;
  latex: string;
  accent?: 'sl' | 'nl' | 'ok' | 'neutral';
}

export function MatrixDisplay({ label, latex, accent = 'neutral' }: MatrixDisplayProps) {
  const borderClass =
    accent === 'sl'
      ? 'border-sky-500/30'
      : accent === 'nl'
        ? 'border-amber-500/30'
        : accent === 'ok'
          ? 'border-emerald-500/30'
          : 'border-[var(--border-subtle)]';

  return (
    <div className={`elevated-surface border ${borderClass} px-3.5 py-2.5 overflow-x-auto`}>
      {label ? (
        <div className="text-xs font-medium text-[var(--text-secondary)] mb-1">
          {label}
        </div>
      ) : null}
      <div className="text-sm">
        <LatexRenderer math={latex} displayMode />
      </div>
    </div>
  );
}
