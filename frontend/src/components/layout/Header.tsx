import { BookOpen, Moon, Sigma, Sun } from 'lucide-react';
import { motion } from 'motion/react';

export type MainMode = 'sl' | 'nl';
export type ThemeMode = 'dark' | 'light';

interface HeaderProps {
  mode: MainMode;
  onChangeMode: (mode: MainMode) => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  onOpenTheory: () => void;
}

export function Header({
  mode,
  onChangeMode,
  theme,
  onToggleTheme,
  onOpenTheory,
}: HeaderProps) {
  return (
    <header className="sticky top-0 z-30 border-b border-[var(--border-subtle)] bg-[var(--bg-canvas)]/90 backdrop-blur-md">
      <div className="max-w-[1440px] mx-auto px-4 sm:px-6 py-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-sky-500/15 border border-sky-500/30 flex items-center justify-center text-[var(--text-sl)]">
            <Sigma className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base sm:text-lg font-bold tracking-tight">NumériLab</h1>
              <span className="text-xs px-2 py-0.5 rounded-md elevated-surface text-[var(--text-secondary)] font-mono-num">
                ING01199
              </span>
            </div>
            <p className="text-xs text-[var(--text-secondary)] hidden sm:block">
              Laboratorio visual de métodos numéricos · Politécnico Colombiano JIC
            </p>
          </div>
        </div>

        {/* Mode Switcher */}
        <nav
          aria-label="Bloques temáticos"
          className="inline-flex p-1 rounded-xl elevated-surface gap-1"
        >
          {(
            [
              { id: 'sl', label: 'S.L. · Sistemas Lineales', sub: 'LU · Jacobi · GS (SOR)' },
              { id: 'nl', label: 'N.L. · No Lineales', sub: 'Newton-Raphson 1D, 2×2 y 3×3' },
            ] as const
          ).map((item) => {
            const active = mode === item.id;
            return (
              <button
                key={item.id}
                type="button"
                aria-pressed={active}
                onClick={() => onChangeMode(item.id)}
                className={`relative px-3.5 py-1.5 rounded-lg text-left transition-colors cursor-pointer ${
                  active
                    ? 'text-[var(--text-primary)]'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                {active ? (
                  <motion.span
                    layoutId="main-mode-pill"
                    transition={{ type: 'spring', stiffness: 420, damping: 32 }}
                    className={`absolute inset-0 rounded-lg border ${
                      item.id === 'sl'
                        ? 'bg-sky-500/15 border-sky-500/40'
                        : 'bg-amber-500/15 border-amber-500/40'
                    }`}
                  />
                ) : null}
                <span className="relative z-10 block text-xs sm:text-sm font-semibold">
                  {item.label}
                </span>
                <span className="relative z-10 hidden md:block text-[11px] text-[var(--text-secondary)]">
                  {item.sub}
                </span>
              </button>
            );
          })}
        </nav>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onOpenTheory}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium elevated-surface hover:border-[var(--border-strong)] transition-colors cursor-pointer"
          >
            <BookOpen className="h-4 w-4 text-[var(--text-sl)]" />
            <span>Guía teórica</span>
          </button>

          <button
            type="button"
            onClick={onToggleTheme}
            aria-label={theme === 'dark' ? 'Cambiar a tema claro' : 'Cambiar a tema oscuro'}
            className="p-2 rounded-lg elevated-surface hover:border-[var(--border-strong)] transition-colors cursor-pointer"
          >
            {theme === 'dark' ? (
              <Sun className="h-4 w-4 text-amber-400" />
            ) : (
              <Moon className="h-4 w-4 text-sky-600" />
            )}
          </button>
        </div>
      </div>
    </header>
  );
}
