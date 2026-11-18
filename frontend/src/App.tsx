import { lazy, Suspense, useEffect, useState } from 'react';
import { Header, type MainMode, type ThemeMode } from './components/layout/Header';
import { LinearSystemsLab } from './components/labs/LinearSystemsLab';
import { TheoryDrawer } from './components/modals/TheoryDrawer';

const NonlinearLab = lazy(() => import('./components/labs/NonlinearLab')
  .then((module) => ({ default: module.NonlinearLab })));

export default function App() {
  const [mode, setMode] = useState<MainMode>('sl');
  const [theme, setTheme] = useState<ThemeMode>(() => {
    const saved = window.localStorage.getItem('numerilab-theme');
    return saved === 'light' ? 'light' : 'dark';
  });
  const [theoryOpen, setTheoryOpen] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    window.localStorage.setItem('numerilab-theme', theme);
  }, [theme]);

  return (
    <div className="min-h-screen flex flex-col">
      <Header
        mode={mode}
        onChangeMode={setMode}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
        onOpenTheory={() => setTheoryOpen(true)}
      />

      <main className="flex-1 max-w-[1440px] w-full mx-auto px-4 sm:px-6 py-6">
        {mode === 'sl' ? <LinearSystemsLab /> : (
          <Suspense fallback={<p role="status">Cargando el laboratorio de Newton-Raphson…</p>}>
            <NonlinearLab />
          </Suspense>
        )}
      </main>

      <footer className="border-t border-[var(--border-subtle)] py-4 px-4 sm:px-6 text-center text-xs text-[var(--text-secondary)]">
        NumériLab · Análisis Numérico (ING01199) · Factorización L.U., Jacobi, Gauss-Seidel (SOR) y Newton-Raphson paso a paso
      </footer>

      <TheoryDrawer open={theoryOpen} onClose={() => setTheoryOpen(false)} />
    </div>
  );
}
