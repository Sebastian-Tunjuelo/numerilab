import { X } from 'lucide-react';
import { motion } from 'motion/react';
import { useEffect, useRef } from 'react';
import { LatexRenderer } from '../math/LatexRenderer';

interface TheoryDrawerProps {
  open: boolean;
  onClose: () => void;
}

export function TheoryDrawer({ open, onClose }: TheoryDrawerProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  return (
    <dialog
      ref={dialogRef}
      aria-label="Guía rápida teórica para sustentación"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
      className="fixed inset-y-0 left-auto right-0 m-0 h-dvh max-h-none w-full max-w-xl overflow-y-auto bg-[var(--bg-panel)] text-[var(--text-primary)] border-l border-[var(--border-subtle)] backdrop:bg-black/55 backdrop:backdrop-blur-xs"
    >
      {open ? (
          <motion.aside
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            transition={{ type: 'spring', stiffness: 360, damping: 34 }}
            className="min-h-full p-6 space-y-5"
          >
            <div className="flex items-start justify-between gap-4 border-b border-[var(--border-subtle)] pb-4">
              <div>
                <h2 className="text-lg font-bold">Guía Teórica — Análisis Numérico</h2>
                <p className="text-xs text-[var(--text-secondary)]">
                  Resumen de fórmulas clave de las Unidades 2 y 3 (ING01199) para consulta durante la exposición.
                </p>
              </div>
              <button
                type="button"
                onClick={onClose}
                aria-label="Cerrar guía teórica"
                className="p-1.5 rounded-lg elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <section className="elevated-surface p-4 space-y-2">
              <h3 className="text-sm font-semibold text-[var(--text-sl)]">1. Factorización L.U. (Método Directo)</h3>
              <p className="text-xs text-[var(--text-secondary)]">
                Descompone la matriz en una triangular inferior <strong className="text-[var(--text-primary)]">L</strong> y una triangular superior <strong className="text-[var(--text-primary)]">U</strong>. Con pivoteo parcial se evita división por cero:
              </p>
              <LatexRenderer
                math={String.raw`\begin{gathered} PA = LU \\ Ly = Pb \;\text{(progresiva)}, \quad Ux = y \;\text{(regresiva)} \end{gathered}`}
                displayMode
              />
              <p className="text-xs text-[var(--text-secondary)]">
                <strong>Doolittle:</strong> Lᵢᵢ = 1 &nbsp;|&nbsp; <strong>Crout:</strong> Uᵢᵢ = 1 &nbsp;|&nbsp; <strong>Costo teórico:</strong>{' '}
                <LatexRenderer math={String.raw`\approx \frac{2}{3}n^3 + 2n^2`} /> FLOPs.
              </p>
            </section>

            <section className="elevated-surface p-4 space-y-2">
              <h3 className="text-sm font-semibold text-[var(--text-sl)]">2. Jacobi, Gauss-Seidel y Sobrerrelajación sucesiva (SOR)</h3>
              <p className="text-xs text-[var(--text-secondary)]">
                Separando la matriz como <LatexRenderer math={String.raw`A = D - L - U`} />, los métodos iterativos se expresan como <LatexRenderer math={String.raw`x^{(k+1)} = Tx^{(k)} + c`} />:
              </p>
              <LatexRenderer math={String.raw`T_J = D^{-1}(L + U), \quad c_J = D^{-1}b`} displayMode />
              <LatexRenderer
                math={String.raw`\begin{gathered} T_\omega = (D - \omega L)^{-1}\left[(1 - \omega)D + \omega U\right] \\ c_\omega = \omega (D - \omega L)^{-1}b \end{gathered}`}
                displayMode
              />
              <p className="text-xs text-[var(--text-secondary)]">
                Si <strong className="text-[var(--text-primary)]">ω = 1</strong> se obtiene Gauss-Seidel estándar. En <strong className="text-[var(--text-primary)]">sobrerrelajación sucesiva (SOR)</strong>, un valor adecuado de <strong className="text-[var(--text-primary)]">1 &lt; ω &lt; 2</strong> puede acelerar la convergencia (costo <LatexRenderer math={String.raw`\mathcal{O}(k \cdot n^2)`} /> FLOPs). El corte de aritmética finita (<strong>redondeo</strong> o <strong>truncamiento</strong> a <em>n</em> decimales, Unidad 1 Sesión 6) se puede activar paso a paso en ambos bloques.
              </p>
            </section>

            <section className="elevated-surface p-4 space-y-2">
              <h3 className="text-sm font-semibold text-[var(--text-ok)]">3. Criterios de Convergencia (S.L.)</h3>
              <ul className="text-xs text-[var(--text-secondary)] space-y-1.5 list-disc pl-4">
                <li>
                  <strong className="text-[var(--text-primary)]">Condición suficiente:</strong> si A es estrictamente diagonalmente dominante (<LatexRenderer math={String.raw`|a_{ii}| > \sum_{j\ne i}|a_{ij}|`} />), Jacobi y Gauss-Seidel convergen para cualquier x⁽⁰⁾.
                </li>
                <li>
                  <strong className="text-[var(--text-primary)]">Convergencia desde cualquier x⁽⁰⁾:</strong> en aritmética exacta ocurre si y solo si <LatexRenderer math={String.raw`\rho(T) = \max_i |\lambda_i(T)| < 1`} />.
                </li>
                <li>
                  <strong className="text-[var(--text-primary)]">Tasa asintótica de convergencia:</strong> <LatexRenderer math={String.raw`R_\infty(T) = -\ln \rho(T)`} />; cuanto menor es <LatexRenderer math={String.raw`\rho(T) < 1`} />, más rápida es la convergencia.
                </li>
              </ul>
            </section>

            <section className="elevated-surface p-4 space-y-2">
              <h3 className="text-sm font-semibold text-[var(--text-nl)]">4. Newton-Raphson (1D y Sistemas No Lineales)</h3>
              <div className="space-y-1">
                <p className="text-[11px] font-medium text-[var(--text-secondary)]">Caso escalar (1D):</p>
                <LatexRenderer math={String.raw`x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}`} displayMode />
                <p className="text-[11px] font-medium text-[var(--text-secondary)]">Sistemas no lineales:</p>
                <LatexRenderer
                  math={String.raw`\begin{gathered} J(x^{(k)}) \Delta x^{(k)} = -F(x^{(k)}) \\ x^{(k+1)} = x^{(k)} + \Delta x^{(k)} \end{gathered}`}
                  displayMode
                />
              </div>
              <p className="text-xs text-[var(--text-secondary)]">
                En cada iteración multivariable se resuelve un sistema lineal con la matriz Jacobiana J(x), usando aquí nuestra propia <strong>Factorización L.U.</strong>
              </p>
              <ul className="text-xs text-[var(--text-secondary)] space-y-1 list-disc pl-4">
                <li>
                  <strong className="text-[var(--text-primary)]">Una condición suficiente (1D):</strong> si f es dos veces continuamente diferenciable en [a, b], f(a)f(b) &lt; 0 y ni f&apos; ni f&apos;&apos; se anulan en el intervalo, el método converge al partir del extremo donde f(x₀)f&apos;&apos;(x₀) &gt; 0.
                </li>
                <li>
                  <strong className="text-[var(--text-primary)]">Orden de convergencia local y empírico:</strong> Newton tiene convergencia cuadrática (<strong className="text-[var(--text-primary)]">p ≈ 2</strong>) cerca de una raíz simple con derivada no nula (o Jacobiano invertible), mientras que en una raíz escalar múltiple la convergencia usual es lineal (<strong className="text-[var(--text-primary)]">p ≈ 1</strong>). A partir de tres pasos consecutivos se estima mediante:
                  <LatexRenderer
                    math={String.raw`p \approx \frac{\ln \left|\Delta x_{k+1} / \Delta x_k\right|}{\ln \left|\Delta x_k / \Delta x_{k-1}\right|}`}
                    displayMode
                  />
                </li>
              </ul>
            </section>
          </motion.aside>
      ) : null}
    </dialog>
  );
}
