import { Component, type ReactNode } from 'react';

interface MafsErrorBoundaryProps {
  resetKey: string;
  description?: string;
  children: ReactNode;
}

interface MafsErrorBoundaryState {
  hasError: boolean;
}

export class MafsErrorBoundary extends Component<MafsErrorBoundaryProps, MafsErrorBoundaryState> {
  state: MafsErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): MafsErrorBoundaryState {
    return { hasError: true };
  }

  componentDidUpdate(prevProps: MafsErrorBoundaryProps) {
    if (this.state.hasError && prevProps.resetKey !== this.props.resetKey) {
      this.setState({ hasError: false });
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          role="alert"
          className="h-[360px] rounded-xl border border-rose-500/30 bg-[var(--plot-bg)] flex flex-col items-center justify-center p-6 text-center gap-3"
        >
          <p className="text-sm font-semibold text-[var(--text-err)]">
            No se pudo renderizar la vista interactiva para este rango numérico.
          </p>
          <p className="text-xs text-[var(--text-secondary)] max-w-md">
            {this.props.description ??
              'Los iterados divergieron fuera del rango gráfico admitido. Consulta los valores exactos en la tabla de iteraciones o prueba otro punto inicial.'}
          </p>
          <button
            type="button"
            onClick={() => this.setState({ hasError: false })}
            className="px-3 py-1.5 rounded-lg text-xs font-medium elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
          >
            Reintentar vista
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
