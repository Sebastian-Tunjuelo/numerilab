import { useMemo } from 'react';
import katex from 'katex';

interface LatexRendererProps {
  math: string;
  displayMode?: boolean;
  className?: string;
}

function escapeHtml(raw: string): string {
  return raw
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

export function LatexRenderer({ math, displayMode = false, className = '' }: LatexRendererProps) {
  const html = useMemo(() => {
    if (!math.trim()) return '';
    try {
      return katex.renderToString(math, {
        displayMode,
        throwOnError: false,
        output: 'htmlAndMathml',
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return `<code>${escapeHtml(math)} (${escapeHtml(msg)})</code>`;
    }
  }, [math, displayMode]);

  return (
    <span
      className={className}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
