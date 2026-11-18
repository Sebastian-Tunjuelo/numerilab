import { useCallback, useEffect, useId, useRef, useState } from 'react';
import {
  AlertTriangle,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  GitBranch,
  Pause,
  Play,
  RotateCcw,
  SkipBack,
  SkipForward,
} from 'lucide-react';
import { solveNewton1D, solveNewtonSystem } from '../../api/client';
import { NEWTON_1D_PRESETS, NEWTON_SYSTEM_PRESETS } from '../../data/presets';
import type {
  Newton1DResponse,
  NewtonStoppingCriterion,
  NewtonSystemResponse,
  RoundingMode,
} from '../../types/numerical';
import { Newton1DPlot } from '../charts/Newton1DPlot';
import { Newton2DPlot } from '../charts/Newton2DPlot';
import { NewtonConvergenceChart } from '../charts/NewtonConvergenceChart';
import { LatexRenderer } from '../math/LatexRenderer';
import { MatrixDisplay } from '../math/MatrixDisplay';

type NLSubMode = '1d' | 'system';

function formatNLStatusBadge(status: string, convergedLabel?: string) {
  if (status === 'converged') {
    return {
      label: convergedLabel ?? 'Convergió',
      cls: 'bg-emerald-500/15 text-[var(--text-ok)] border-emerald-500/30',
      isAlert: false,
    };
  }
  if (status === 'max_iterations' || status === 'stagnated') {
    return {
      label: status === 'max_iterations' ? 'Máx. iteraciones' : 'Estancado',
      cls: 'bg-amber-500/15 text-[var(--text-nl)] border-amber-500/30',
      isAlert: true,
    };
  }
  const errorLabels: Record<string, string> = {
    zero_derivative: "Derivada nula f'(x) = 0",
    singular_jacobian: 'Jacobiano singular',
    parse_error: 'Error de sintaxis',
    domain_error: 'Error de dominio',
    diverged: 'Divergió',
  };
  return {
    label: errorLabels[status] ?? 'No convergió',
    cls: 'bg-rose-500/15 text-[var(--text-err)] border-rose-500/30',
    isAlert: true,
  };
}

function formatScalar(value: number, decimals: number): string {
  if (value === 0) return '0';
  const abs = Math.abs(value);
  if (abs < 1e-3 || abs >= 1e5) return value.toExponential(3);
  return Number(value.toFixed(Math.min(decimals, 6))).toString();
}

function formatRelPct(pct?: number | null): string {
  if (pct === null || pct === undefined || !Number.isFinite(pct)) return '—';
  if (pct === 0) return '0%';
  if (pct < 0.001) return `${pct.toExponential(2)}%`;
  return `${pct.toFixed(4)}%`;
}

function formatRelPctLatex(pct?: number | null): string {
  if (pct === null || pct === undefined || !Number.isFinite(pct)) return '\\text{—}';
  if (pct === 0) return '0\\%';
  if (pct < 0.001) return `${pct.toExponential(2)}\\%`;
  return `${pct.toFixed(4)}\\%`;
}

function parseFiniteInput(raw: string): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const num = Number(trimmed);
  return Number.isFinite(num) ? num : null;
}

function csvCell(value: string | number): string {
  const str = String(value);
  return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
}

function buildNewton1DCsv(res: Newton1DResponse): string {
  if (!res.steps.length) return '';
  const headers = ['Iteracion', 'x_k', 'f(x_k)', "f'(x_k)", "f''(x_k)", 'x_{k+1}', 'Error_Abs', 'e_rel_%'];
  const rows = res.steps.map((s) =>
    [
      s.iteration,
      s.x_k,
      s.fx_k,
      s.dfx_k,
      s.d2fx_k ?? '',
      s.x_next,
      s.abs_error,
      s.rel_error_pct ?? '',
    ].map(csvCell).join(','),
  );
  return [headers.map(csvCell).join(','), ...rows].join('\n');
}

function buildNewton1DLatex(res: Newton1DResponse): string {
  if (!res.steps.length) return '';
  const lines = [
    '\\begin{array}{c|r|r|r|r|r|r|r}',
    "k & x_k & f(x_k) & f'(x_k) & f''(x_k) & x_{k+1} & |\\Delta x| & e_{\\text{rel}}\\,(\\%) \\\\ \\hline",
    ...res.steps.map(
      (s) =>
        `${s.iteration} & ${s.x_k} & ${s.fx_k.toExponential(2)} & ${s.dfx_k.toExponential(2)} & ${
          s.d2fx_k != null ? s.d2fx_k.toExponential(2) : '\\text{—}'
        } & ${s.x_next} & ${s.abs_error.toExponential(2)} & ${formatRelPctLatex(s.rel_error_pct)} \\\\`,
    ),
    '\\end{array}',
  ];
  return lines.join('\n');
}

function buildNewtonSysCsv(res: NewtonSystemResponse, vars: string[]): string {
  if (!res.steps.length) return '';
  const headers = ['Iteracion', ...vars.map((v) => `${v}_{k+1}`), '||dx||_inf', '||F(x_{k+1})||_inf', 'e_rel_%'];
  const rows = res.steps.map((s) =>
    [
      s.iteration,
      ...s.x_next,
      s.abs_error,
      s.residual ?? '',
      s.rel_error_pct ?? '',
    ].map(csvCell).join(','),
  );
  return [headers.map(csvCell).join(','), ...rows].join('\n');
}

function buildNewtonSysLatex(res: NewtonSystemResponse, vars: string[]): string {
  if (!res.steps.length) return '';
  const cols = 'c|' + 'r'.repeat(vars.length) + '|r|r|r';
  const varHeaders = vars.map((v) => `${v}_{k+1}`).join(' & ');
  const lines = [
    `\\begin{array}{${cols}}`,
    `k & ${varHeaders} & \\|\\Delta x\\|_\\infty & \\|F(x_{k+1})\\|_\\infty & e_{\\text{rel}}\\,(\\%) \\\\ \\hline`,
    ...res.steps.map(
      (s) =>
        `${s.iteration} & ${s.x_next.join(' & ')} & ${s.abs_error.toExponential(2)} & ${
          s.residual !== null ? s.residual.toExponential(2) : '\\text{—}'
        } & ${formatRelPctLatex(s.rel_error_pct)} \\\\`,
    ),
    '\\end{array}',
  ];
  return lines.join('\n');
}

export function NonlinearLab() {
  const [subMode, setSubMode] = useState<NLSubMode>('1d');

  // 1D State
  const [expr1D, setExpr1D] = useState(NEWTON_1D_PRESETS[0].expression);
  const [initialField, setX01D] = useState(String(NEWTON_1D_PRESETS[0].x0));
  const [lastValidX01D, setLastValidX01D] = useState(NEWTON_1D_PRESETS[0].x0);
  const parsedX01D = parseFiniteInput(initialField);
  const x01D = parsedX01D ?? lastValidX01D;
  const [criterion, setCriterion] = useState<NewtonStoppingCriterion>('residual_and_step');
  const [tolField, setTolField] = useState('1e-7');
  const tol = Number(tolField);
  const validTol = tolField !== '' && Number.isFinite(tol) && tol > 0
    && (criterion === 'rel_pct' ? tol < 100 : tol <= 1);
  const [maxIterField, setMaxIter] = useState('50');
  const maxIter = Number(maxIterField);
  const validIterations = Number.isInteger(maxIter) && maxIter >= 1 && maxIter <= 200;
  const [decimals, setDecimals] = useState(6);
  const [roundingMode, setRoundingMode] = useState<RoundingMode>('none');
  const [activePreset1D, setActivePreset1D] = useState(NEWTON_1D_PRESETS[0].id);

  // 1D Comparison state
  const [compareEnabled, setCompareEnabled] = useState(false);
  const [compareX0Field, setCompareX0Field] = useState('4');
  const parsedCompareX01D = parseFiniteInput(compareX0Field);
  const compareX01D = parsedCompareX01D ?? Number.NaN;
  const validCompareX0 = !compareEnabled || parsedCompareX01D !== null;
  const [response1DCompare, setResponse1DCompare] = useState<{ key: string; data: Newton1DResponse } | null>(null);

  const [response1D, setResponse1D] = useState<{ key: string; data: Newton1DResponse } | null>(null);
  const [loading1D, setLoading1D] = useState(false);
  const [error1D, setError1D] = useState<{ key: string; message: string } | null>(null);
  const [stepIdx1D, setStepIdx1D] = useState(0);
  const [playing1D, setPlaying1D] = useState(false);
  const [copied1D, setCopied1D] = useState<'csv' | 'latex' | null>(null);
  const [copyError1D, setCopyError1D] = useState(false);

  // System State (2x2 or 3x3)
  const [sysDim, setSysDim] = useState<2 | 3>(2);
  const [sysVars, setSysVars] = useState<string[]>(NEWTON_SYSTEM_PRESETS[0].variables);
  const [sysEqs, setSysEqs] = useState<string[]>(NEWTON_SYSTEM_PRESETS[0].equations);
  const [systemFields, setSysX0] = useState(() => NEWTON_SYSTEM_PRESETS[0].x0.map(String));
  const [lastValidSysX0, setLastValidSysX0] = useState<number[]>(() => [...NEWTON_SYSTEM_PRESETS[0].x0]);
  const parsedSysX0 = systemFields.map(parseFiniteInput);
  const sysX0 = parsedSysX0.map((v, idx) => v ?? lastValidSysX0[idx] ?? 0);
  const [activePresetSys, setActivePresetSys] = useState(NEWTON_SYSTEM_PRESETS[0].id);

  const [responseSys, setResponseSys] = useState<{ dim: 2 | 3; key: string; data: NewtonSystemResponse } | null>(null);
  const [loadingSys, setLoadingSys] = useState(false);
  const [errorSys, setErrorSys] = useState<{ key: string; message: string } | null>(null);
  const [stepIdxSys, setStepIdxSys] = useState(0);
  const [playingSys, setPlayingSys] = useState(false);
  const [copiedSys, setCopiedSys] = useState<'csv' | 'latex' | null>(null);
  const [copyErrorSys, setCopyErrorSys] = useState(false);

  const request1D = useRef<AbortController | null>(null);
  const requestSystem = useRef<AbortController | null>(null);
  const tolInputRef = useRef<HTMLInputElement | null>(null);
  const copyTimer1D = useRef<number | undefined>(undefined);
  const copyTimerSys = useRef<number | undefined>(undefined);
  const systemDrafts = useRef<Partial<Record<2 | 3, {
    equations: string[];
    x0: string[];
    presetId: string;
  }>>>({});

  const table1DCaptionId = useId();
  const tableSysCaptionId = useId();
  const key1D = JSON.stringify([
    expr1D,
    x01D,
    tol,
    maxIter,
    decimals,
    criterion,
    roundingMode,
    compareEnabled,
    compareEnabled ? compareX01D : null,
  ]);
  const keySys = JSON.stringify([sysEqs, sysVars, sysX0, tol, maxIter, decimals, criterion, roundingMode]);
  const valid1D = validIterations && validTol && validCompareX0 && parsedX01D !== null && !!expr1D.trim();
  const completeEquations = sysEqs.every((equation) => !!equation.trim());
  const validSystem = validIterations && validTol && completeEquations
    && parsedSysX0.every((value) => value !== null);
  const res1D = response1D?.data ?? null;
  const is1DStale = !!res1D && (!valid1D || response1D?.key !== key1D);
  const matchedDimSys =
    responseSys
    && responseSys.dim === sysDim
    && (!responseSys.data.variables || responseSys.data.variables.length === sysDim)
      ? responseSys.data
      : null;
  const resSys = matchedDimSys;
  const isSysStale = !!resSys && (!validSystem || responseSys?.key !== keySys);
  const currentError1D = valid1D && error1D?.key === key1D ? error1D.message : null;
  const currentErrorSys = validSystem && errorSys?.key === keySys ? errorSys.message : null;
  const plotRes1D = res1D;
  const plotResSys = resSys;
  const displaySysVars = resSys?.variables && resSys.variables.length === sysDim ? resSys.variables : sysVars;

  const run1D = useCallback(
    async (
      expression: string,
      initialX0: number,
      t: number,
      mIter: number,
      dec: number,
      crit: NewtonStoppingCriterion,
      rMode: RoundingMode = 'none',
      cmpEnabled = false,
      cmpX0?: number | null,
    ) => {
      request1D.current?.abort();
      const controller = new AbortController();
      request1D.current = controller;
      setLoading1D(true);
      setPlaying1D(false);
      setError1D(null);
      const roundFlag = rMode !== 'none';
      const reqKey = JSON.stringify([
        expression,
        initialX0,
        t,
        mIter,
        dec,
        crit,
        rMode,
        cmpEnabled,
        cmpEnabled ? cmpX0 ?? null : null,
      ]);
      try {
        const primaryPromise = solveNewton1D({
          expression,
          x0: initialX0,
          tol: t,
          max_iter: mIter,
          decimals: dec,
          criterion: crit,
          apply_rounding: roundFlag,
          rounding_mode: rMode,
        }, controller.signal);

        if (cmpEnabled && cmpX0 !== null && cmpX0 !== undefined && Number.isFinite(cmpX0)) {
          const secondaryPromise = solveNewton1D({
            expression,
            x0: cmpX0,
            tol: t,
            max_iter: mIter,
            decimals: dec,
            criterion: crit,
            apply_rounding: roundFlag,
            rounding_mode: rMode,
          }, controller.signal);
          const [primaryOutcome, secondaryOutcome] = await Promise.allSettled([
            primaryPromise,
            secondaryPromise,
          ]);
          if (controller.signal.aborted) return;
          if (primaryOutcome.status === 'fulfilled') {
            setResponse1D({ key: reqKey, data: primaryOutcome.value });
            if (secondaryOutcome.status === 'fulfilled') {
              setResponse1DCompare({ key: reqKey, data: secondaryOutcome.value });
            } else {
              setResponse1DCompare(null);
              const secMsg = secondaryOutcome.reason instanceof Error
                ? secondaryOutcome.reason.message
                : 'Error de red o del servidor.';
              setError1D({
                key: reqKey,
                message: `No se pudo calcular la comparación para x₀⁽ᴮ⁾: ${secMsg}`,
              });
            }
          } else {
            throw primaryOutcome.reason;
          }
        } else {
          const data = await primaryPromise;
          if (controller.signal.aborted) return;
          setResponse1D({ key: reqKey, data });
          setResponse1DCompare(null);
        }
        setStepIdx1D(0);
        setPlaying1D(false);
      } catch (err) {
        if (controller.signal.aborted) return;
        setError1D({
          key: reqKey,
          message: err instanceof Error ? err.message : 'Error ejecutando Newton-Raphson 1D.',
        });
      } finally {
        if (!controller.signal.aborted) setLoading1D(false);
      }
    },
    [],
  );

  const runSystem = useCallback(
    async (
      equations: string[],
      variables: string[],
      initialX0: number[],
      t: number,
      mIter: number,
      dec: number,
      crit: NewtonStoppingCriterion,
      rMode: RoundingMode = 'none',
    ) => {
      requestSystem.current?.abort();
      const controller = new AbortController();
      requestSystem.current = controller;
      setLoadingSys(true);
      setStepIdxSys(0);
      setPlayingSys(false);
      setErrorSys(null);
      const dim: 2 | 3 = variables.length === 3 ? 3 : 2;
      const reqKey = JSON.stringify([equations, variables, initialX0, t, mIter, dec, crit, rMode]);
      try {
        const data = await solveNewtonSystem({
          equations,
          variables,
          x0: initialX0,
          tol: t,
          max_iter: mIter,
          decimals: dec,
          criterion: crit,
          apply_rounding: rMode !== 'none',
          rounding_mode: rMode,
        }, controller.signal);
        if (controller.signal.aborted) return;
        setResponseSys({
          dim,
          key: reqKey,
          data,
        });
      } catch (err) {
        if (controller.signal.aborted) return;
        setErrorSys({
          key: reqKey,
          message: err instanceof Error ? err.message : 'Error ejecutando Newton-Raphson para sistemas.',
        });
      } finally {
        if (!controller.signal.aborted) setLoadingSys(false);
      }
    },
    [],
  );

  useEffect(() => {
    run1D(expr1D, x01D, tol, maxIter, decimals, criterion, roundingMode, compareEnabled, compareX01D);
    runSystem(sysEqs, sysVars, sysX0, tol, maxIter, decimals, criterion, roundingMode);
    return () => {
      request1D.current?.abort();
      requestSystem.current?.abort();
      window.clearTimeout(copyTimer1D.current);
      window.clearTimeout(copyTimerSys.current);
    };
    // Initial run only
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const input = tolInputRef.current;
    if (!input) return;
    if (tolField !== '' && !validTol) {
      input.setCustomValidity(
        criterion === 'rel_pct'
          ? 'Ingresa una tolerancia entre 0 y 100 (%).'
          : 'Ingresa una tolerancia positiva menor o igual a 1.',
      );
    } else {
      input.setCustomValidity('');
    }
  }, [tolField, validTol, criterion]);

  // Auto-play timers
  useEffect(() => {
    if (!playing1D || !res1D || res1D.steps.length <= 1) return;
    if (stepIdx1D >= res1D.steps.length - 1) {
      setPlaying1D(false);
      return;
    }
    const timer = window.setTimeout(() => {
      setStepIdx1D((prev) => Math.min(res1D.steps.length - 1, prev + 1));
    }, 950);
    return () => window.clearTimeout(timer);
  }, [playing1D, res1D, stepIdx1D]);

  useEffect(() => {
    if (!playingSys || !resSys || resSys.steps.length <= 1) return;
    if (stepIdxSys >= resSys.steps.length - 1) {
      setPlayingSys(false);
      return;
    }
    const timer = window.setTimeout(() => {
      setStepIdxSys((prev) => Math.min(resSys.steps.length - 1, prev + 1));
    }, 1050);
    return () => window.clearTimeout(timer);
  }, [playingSys, resSys, stepIdxSys]);

  const handleCriterionChange = (next: NewtonStoppingCriterion) => {
    setCriterion(next);
    if (next === 'rel_pct') {
      if (!Number.isFinite(tol) || tol <= 0.0001 || tol >= 100) {
        setTolField('1');
      }
    } else if (!Number.isFinite(tol) || tol <= 0 || tol > 1) {
      setTolField('1e-6');
    }
  };

  const handleCopy1D = async (kind: 'csv' | 'latex') => {
    if (!res1D) return;
    const content = kind === 'csv' ? buildNewton1DCsv(res1D) : buildNewton1DLatex(res1D);
    if (!content) return;
    setCopyError1D(false);
    try {
      await navigator.clipboard.writeText(content);
    } catch {
      setCopied1D(null);
      setCopyError1D(true);
      return;
    }
    window.clearTimeout(copyTimer1D.current);
    setCopied1D(kind);
    copyTimer1D.current = window.setTimeout(() => setCopied1D(null), 1800);
  };

  const handleCopySys = async (kind: 'csv' | 'latex') => {
    if (!resSys) return;
    const content = kind === 'csv'
      ? buildNewtonSysCsv(resSys, displaySysVars)
      : buildNewtonSysLatex(resSys, displaySysVars);
    if (!content) return;
    setCopyErrorSys(false);
    try {
      await navigator.clipboard.writeText(content);
    } catch {
      setCopiedSys(null);
      setCopyErrorSys(true);
      return;
    }
    window.clearTimeout(copyTimerSys.current);
    setCopiedSys(kind);
    copyTimerSys.current = window.setTimeout(() => setCopiedSys(null), 1800);
  };

  const selectPreset1D = (presetId: string) => {
    const p = NEWTON_1D_PRESETS.find((item) => item.id === presetId);
    if (!p) return;
    const nextCriterion: NewtonStoppingCriterion = p.stopping?.criterion ?? 'residual_and_step';
    const nextTolField = p.stopping ? String(p.stopping.tol) : '1e-7';
    const nextTol = p.stopping ? p.stopping.tol : 1e-7;
    const nextMaxIter = validIterations ? maxIter : 50;
    if (!validIterations) setMaxIter('50');

    const currentCmp = parseFiniteInput(compareX0Field);
    let nextCmpX0 = currentCmp ?? Number((p.x0 + 0.5).toFixed(4));
    if (p.compareX0 !== undefined) {
      setCompareX0Field(String(p.compareX0));
      nextCmpX0 = p.compareX0;
    } else if (currentCmp === null) {
      setCompareX0Field(String(nextCmpX0));
    }

    setActivePreset1D(p.id);
    setExpr1D(p.expression);
    setX01D(String(p.x0));
    setLastValidX01D(p.x0);
    setCriterion(nextCriterion);
    setTolField(nextTolField);
    run1D(p.expression, p.x0, nextTol, nextMaxIter, decimals, nextCriterion, roundingMode, compareEnabled, nextCmpX0);
  };

  const selectPresetSys = (presetId: string) => {
    const p = NEWTON_SYSTEM_PRESETS.find((item) => item.id === presetId);
    if (!p) return;
    if (p.variables.length !== sysDim) {
      systemDrafts.current[sysDim] = { equations: sysEqs, x0: systemFields, presetId: activePresetSys };
    }
    const nextTolField = validTol ? tolField : (criterion === 'rel_pct' ? '1' : '1e-7');
    const nextTol = validTol ? tol : (criterion === 'rel_pct' ? 1 : 1e-7);
    const nextMaxIter = validIterations ? maxIter : 50;
    if (!validTol) setTolField(nextTolField);
    if (!validIterations) setMaxIter('50');

    setActivePresetSys(p.id);
    setSysDim(p.variables.length === 3 ? 3 : 2);
    setSysVars([...p.variables]);
    setSysEqs([...p.equations]);
    setSysX0(p.x0.map(String));
    setLastValidSysX0([...p.x0]);
    runSystem(p.equations, p.variables, p.x0, nextTol, nextMaxIter, decimals, criterion, roundingMode);
  };

  const switchSystemDim = (dim: 2 | 3) => {
    if (dim === sysDim) return;
    systemDrafts.current[sysDim] = { equations: sysEqs, x0: systemFields, presetId: activePresetSys };
    const preset = NEWTON_SYSTEM_PRESETS.find((item) => item.variables.length === dim)!;
    const draft = systemDrafts.current[dim];
    const nextVars = [...preset.variables];
    const nextEqs = [...(draft?.equations ?? preset.equations)];
    const nextFields = [...(draft?.x0 ?? preset.x0.map(String))];

    requestSystem.current?.abort();
    setErrorSys(null);
    setPlayingSys(false);
    setStepIdxSys(0);
    setSysDim(dim);
    setSysVars(nextVars);
    setSysEqs(nextEqs);
    setSysX0(nextFields);
    setLastValidSysX0(nextFields.map((value, idx) => parseFiniteInput(value) ?? preset.x0[idx]));
    setActivePresetSys(draft?.presetId ?? preset.id);

    const parsedNextX0 = nextFields.map(parseFiniteInput);
    const nextValid = validIterations && validTol
      && nextEqs.every((eq) => !!eq.trim())
      && parsedNextX0.every((value): value is number => value !== null);
    if (nextValid) {
      runSystem(nextEqs, nextVars, parsedNextX0 as number[], tol, maxIter, decimals, criterion, roundingMode);
    } else {
      setLoadingSys(false);
    }
  };

  const active1DStep = res1D?.steps[stepIdx1D] ?? null;
  const activeSysStep = resSys?.steps[stepIdxSys] ?? null;
  const needsCalculation = subMode === '1d'
    ? !loading1D && (!res1D || is1DStale) && !currentError1D
    : !loadingSys && (!resSys || isSysStale) && !currentErrorSys;
  const badge1D = res1D
    ? formatNLStatusBadge(
        res1D.status,
        res1D.status === 'converged'
          ? res1D.root !== null
            ? `Raíz: ${res1D.root} (${res1D.iterations} it.)`
            : `Convergió (${res1D.iterations} it.)`
          : undefined,
      )
    : null;
  const badgeSys = resSys
    ? formatNLStatusBadge(
        resSys.status,
        resSys.status === 'converged' ? `Convergió (${resSys.iterations} it.)` : undefined,
      )
    : null;

  return (
    <div className="nl-scope space-y-6">
      {/* Sub-mode Toggle & Global NL Controls */}
      <section className="panel-surface p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="inline-flex p-1 rounded-xl elevated-surface gap-1">
          <button
            type="button"
            aria-pressed={subMode === '1d'}
            onClick={() => {
              setPlaying1D(false);
              setPlayingSys(false);
              setSubMode('1d');
            }}
            className={`px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold transition-colors cursor-pointer ${
              subMode === '1d'
                ? 'bg-amber-500 text-slate-950'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            Newton-Raphson 1D · f(x) = 0
          </button>
          <button
            type="button"
            aria-pressed={subMode === 'system'}
            onClick={() => {
              setPlaying1D(false);
              setPlayingSys(false);
              setSubMode('system');
            }}
            className={`px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold transition-colors cursor-pointer ${
              subMode === 'system'
                ? 'bg-amber-500 text-slate-950'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            Sistemas No Lineales · 2×2 y 3×3
          </button>
        </div>

        <div className="flex flex-wrap items-start gap-3 text-xs">
          <label className="inline-flex items-center gap-1.5">
            <span className="text-[var(--text-secondary)]">Criterio:</span>
            <select
              aria-label="Criterio de parada"
              value={criterion}
              onChange={(e) => handleCriterionChange(e.target.value as NewtonStoppingCriterion)}
              className="field-input px-2 py-1"
            >
              <option value="residual_and_step">Residual y paso (estricto)</option>
              <option value="rel_diff">Error relativo |e_rel|</option>
              <option value="rel_pct">Error porcentual e_rel (%)</option>
            </select>
          </label>
          <div className="inline-flex items-start gap-1.5">
            <label htmlFor="nl-tol" className="text-[var(--text-secondary)] pt-1.5">
              {criterion === 'rel_pct' ? 'Tolerancia (%):' : 'Tolerancia:'}
            </label>
            <div>
              <input
                ref={tolInputRef}
                id="nl-tol"
                type="number"
                step="any"
                required
                value={tolField}
                onChange={(e) => setTolField(e.target.value)}
                className="field-input w-24 px-2 py-1 font-mono-num"
              />
              <div className="error-msg">
                {criterion === 'rel_pct' ? 'Entre 0 y 100 (%).' : 'Entre 0 y 1.'}
              </div>
            </div>
          </div>
          <div className="inline-flex items-start gap-1.5">
            <label htmlFor="nl-maxiter" className="text-[var(--text-secondary)] pt-1.5">
              Máx. it.:
            </label>
            <div>
              <input
                id="nl-maxiter"
                type="number"
                step={1}
                min={1}
                max={200}
                required
                value={maxIterField}
                onChange={(e) => setMaxIter(e.target.value)}
                className="field-input w-16 px-2 py-1 font-mono-num"
              />
              <div className="error-msg">Entero entre 1 y 200.</div>
            </div>
          </div>
          <label className="inline-flex items-center gap-1.5">
            <span className="text-[var(--text-secondary)]">Decimales:</span>
            <select
              aria-label="Decimales a mostrar"
              value={decimals}
              onChange={(e) => setDecimals(Number(e.target.value))}
              className="field-input px-2 py-1 font-mono-num"
            >
              <option value={2}>2</option>
              <option value={3}>3</option>
              <option value={4}>4</option>
              <option value={6}>6</option>
              <option value={8}>8</option>
            </select>
          </label>
          <label className="inline-flex items-center gap-1.5">
            <span className="text-[var(--text-secondary)]">Aritmética finita:</span>
            <select
              aria-label="Aritmética finita por paso"
              value={roundingMode}
              onChange={(e) => setRoundingMode(e.target.value as RoundingMode)}
              className="field-input px-2 py-1"
            >
              <option value="none">Sin corte por paso</option>
              <option value="round">Redondeo por paso</option>
              <option value="truncate">Truncamiento por paso</option>
            </select>
          </label>
        </div>
      </section>

      {needsCalculation || !validIterations || !validTol ? (
        <p role="status" className="text-sm text-[var(--text-nl)]">
          {!validIterations ? 'Ingresa un máximo de iteraciones entero entre 1 y 200.'
            : !validTol ? (criterion === 'rel_pct'
                ? 'Ingresa una tolerancia porcentual estrictamente entre 0 y 100 (%).'
                : 'Ingresa una tolerancia positiva menor o igual a 1.')
              : (subMode === '1d' ? valid1D : validSystem)
                ? (subMode === '1d' ? is1DStale : isSysStale)
                  ? 'Mostrando el último resultado calculado (desactualizado). Pulsa calcular/resolver para actualizar.'
                  : 'Calcula con los datos actuales para actualizar el procedimiento.'
                : 'Completa las ecuaciones y los valores iniciales.'}
        </p>
      ) : null}

      {subMode === '1d' ? (
        <div className="space-y-5">
          {/* 1D Presets */}
          <section className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-2.5">
            {NEWTON_1D_PRESETS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                onClick={() => selectPreset1D(preset.id)}
                className={`text-left p-3 rounded-xl border transition-all cursor-pointer ${
                  activePreset1D === preset.id
                    ? 'bg-amber-500/12 border-amber-500/50'
                    : 'panel-surface hover:border-[var(--border-strong)]'
                }`}
              >
                <div className="text-xs font-semibold mb-1">{preset.title}</div>
                <div className="text-xs text-[var(--text-secondary)]">{preset.subtitle}</div>
              </button>
            ))}
          </section>

          {/* 1D Input Form & Symbolic Card */}
          <section className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            <form
              className="lg:col-span-6 panel-surface p-4 space-y-3"
              onSubmit={(e) => {
                e.preventDefault();
                if (valid1D) {
                  run1D(expr1D, x01D, tol, maxIter, decimals, criterion, roundingMode, compareEnabled, compareX01D);
                }
              }}
            >
              <div>
                <label htmlFor="nl-1d-expr" className="block text-sm font-semibold">
                  Ecuación f(x) = 0
                </label>
                <span id="nl-1d-hint" className="block text-xs text-[var(--text-secondary)] mb-1.5">
                  Usa sintaxis matemática estándar: x^3 - 2*x - 5, cos(x) - x, e^(-x) - x, ln(x+2).
                </span>
                <input
                  id="nl-1d-expr"
                  type="text"
                  required
                  maxLength={300}
                  aria-describedby="nl-1d-hint"
                  value={expr1D}
                  onChange={(e) => {
                    setExpr1D(e.target.value);
                    setActivePreset1D('custom');
                  }}
                  className="field-input w-full px-3 py-2 font-mono-num text-sm"
                />
                <div className="error-msg">Ingresa una función matemática válida en x.</div>
              </div>

              <div className="flex flex-wrap items-end gap-3">
                <div className="flex flex-wrap items-start gap-3 flex-1 min-w-[180px]">
                  <div className="flex-1 min-w-[140px]">
                    <label htmlFor="nl-1d-x0" className="block text-xs text-[var(--text-secondary)] mb-1">
                      Valor inicial x₀ (o arrástralo en la gráfica)
                    </label>
                    <input
                      id="nl-1d-x0"
                      type="number"
                      step="any"
                      required
                      value={initialField}
                      onChange={(e) => {
                        const val = e.target.value;
                        setX01D(val);
                        const parsed = parseFiniteInput(val);
                        if (parsed !== null) setLastValidX01D(parsed);
                        setActivePreset1D('custom');
                      }}
                      className="field-input w-full px-3 py-2 font-mono-num text-sm"
                    />
                    <div className="error-msg">Ingresa un número finito para x₀.</div>
                  </div>

                  {compareEnabled ? (
                    <div className="w-36">
                      <label htmlFor="nl-1d-x0-b" className="block text-xs text-[var(--text-sl)] mb-1">
                        Segundo x₀⁽ᴮ⁾
                      </label>
                      <input
                        id="nl-1d-x0-b"
                        type="number"
                        step="any"
                        required
                        value={compareX0Field}
                        onChange={(e) => setCompareX0Field(e.target.value)}
                        className="field-input w-full px-3 py-2 font-mono-num text-sm border-sky-500/40"
                      />
                      <div className="error-msg">Número finito.</div>
                    </div>
                  ) : null}
                </div>

                <button
                  type="submit"
                  disabled={loading1D || !valid1D}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-semibold text-sm cursor-pointer disabled:opacity-60"
                >
                  <Play className="h-4 w-4 fill-current" />
                  {loading1D ? 'Calculando...' : 'Calcular raíz'}
                </button>
              </div>

              <label className="inline-flex items-center gap-2 text-xs text-[var(--text-secondary)] cursor-pointer pt-1">
                <input
                  type="checkbox"
                  checked={compareEnabled}
                  onChange={(e) => {
                    const checked = e.target.checked;
                    setCompareEnabled(checked);
                    if (validIterations && validTol && parsedX01D !== null && expr1D.trim()) {
                      if (!checked || parsedCompareX01D !== null) {
                        run1D(expr1D, x01D, tol, maxIter, decimals, criterion, roundingMode, checked, compareX01D);
                      }
                    }
                  }}
                  className="accent-sky-400"
                />
                <span>Comparar con otro x₀ (superpone curvas de convergencia en escala log₁₀)</span>
              </label>
            </form>

            {/* SymPy Symbolic Derivative Preview */}
            <div className={`lg:col-span-6 panel-surface p-4 space-y-2.5 transition-opacity ${is1DStale ? 'opacity-75' : ''}`}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-sm font-semibold text-[var(--text-nl)]">
                  Derivadas simbólicas automáticas (SymPy)
                </h3>
                {badge1D ? (
                  <span className={`text-xs px-2.5 py-0.5 rounded-full border ${badge1D.cls}`}>
                    {badge1D.label}
                  </span>
                ) : null}
              </div>

              {res1D?.f_latex ? (
                <div className={`grid grid-cols-1 ${res1D.d2f_latex ? 'sm:grid-cols-3' : 'sm:grid-cols-2'} gap-2.5`}>
                  <MatrixDisplay label="Función f(x)" latex={`f(x) = ${res1D.f_latex}`} accent="nl" />
                  <MatrixDisplay label="Derivada f'(x)" latex={`f'(x) = ${res1D.df_latex}`} accent="nl" />
                  {res1D.d2f_latex ? (
                    <MatrixDisplay label="2.ª Derivada f''(x)" latex={`f''(x) = ${res1D.d2f_latex}`} accent="nl" />
                  ) : null}
                </div>
              ) : null}

              {res1D?.fourier_condition ? (
                <div
                  className={`elevated-surface px-3 py-2 text-xs flex flex-wrap items-center justify-between gap-2 border ${
                    res1D.fourier_condition.satisfied
                      ? 'border-emerald-500/30'
                      : 'border-amber-500/30'
                  }`}
                >
                  <span className="text-[var(--text-secondary)]">
                    Criterio de Fourier en x₀ = {res1D.fourier_condition.x0}:{' '}
                    <span className="font-mono-num text-[var(--text-primary)]">
                      f(x₀)·f&apos;&apos;(x₀) ={' '}
                      {res1D.fourier_condition.product !== null ? res1D.fourier_condition.product : '—'}
                    </span>
                  </span>
                  <span
                    className={`font-semibold ${
                      res1D.fourier_condition.satisfied ? 'text-[var(--text-ok)]' : 'text-[var(--text-nl)]'
                    }`}
                  >
                    {res1D.fourier_condition.satisfied
                      ? '> 0 (monotonía inicial favorable)'
                      : '≤ 0 (sin garantía inicial de Fourier)'}
                  </span>
                </div>
              ) : null}

              {res1D?.estimated_order != null ? (
                <div className="elevated-surface px-3 py-2 text-xs flex flex-wrap items-center justify-between gap-2 border border-amber-500/25">
                  <span className="text-[var(--text-secondary)]">Orden empírico de convergencia (últimas 3 iteraciones):</span>
                  <span className="font-mono-num font-semibold text-[var(--text-nl)]">
                    p ≈ {res1D.estimated_order.toFixed(3)}{' '}
                    ({res1D.estimated_order >= 1.65 ? 'Cuadrática' : res1D.estimated_order < 1.35 ? 'Lineal (posible raíz múltiple)' : 'Superlineal'})
                  </span>
                </div>
              ) : null}

              {res1D?.convergence_warnings && res1D.convergence_warnings.length > 0 ? (
                <div className="space-y-1.5">
                  {res1D.convergence_warnings.map((warning, idx) => (
                    <div
                      key={idx}
                      role="status"
                      className="elevated-surface px-3 py-2 text-xs border border-amber-500/35 bg-amber-500/10 text-[var(--text-nl)] flex items-start gap-2"
                    >
                      <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                      <span>{warning}</span>
                    </div>
                  ))}
                </div>
              ) : null}

              {res1D ? (
                badge1D?.isAlert ? (
                  <div
                    role="alert"
                    className={`elevated-surface px-3 py-2 text-xs border flex items-start gap-2 ${
                      res1D.status === 'max_iterations' || res1D.status === 'stagnated'
                        ? 'border-amber-500/35 bg-amber-500/10 text-[var(--text-nl)]'
                        : 'border-rose-500/35 bg-rose-500/10 text-[var(--text-err)]'
                    }`}
                  >
                    <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                    <span>{res1D.message}</span>
                  </div>
                ) : (
                  <p className="text-xs text-[var(--text-secondary)]">{res1D.message}</p>
                )
              ) : null}
            </div>
          </section>

          {currentError1D ? (
            <div role="alert" className="panel-surface p-3 border-rose-500/40 bg-rose-500/10 text-[var(--text-err)] text-sm flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>{currentError1D}</span>
            </div>
          ) : null}

          {/* Interactive Mafs Plot + Step Player + Iteration Table */}
          <section className="grid grid-cols-1 xl:grid-cols-12 gap-5">
            <div className="xl:col-span-7 space-y-3">
              <Newton1DPlot
                x0={x01D}
                result={plotRes1D}
                activeStepIndex={stepIdx1D}
                isStale={is1DStale}
                onDragX0={(nextX0) => {
                  setX01D(String(nextX0));
                  setLastValidX01D(nextX0);
                  setActivePreset1D('custom');
                  if (validIterations && validTol && validCompareX0 && expr1D.trim()) {
                    run1D(expr1D, nextX0, tol, maxIter, decimals, criterion, roundingMode, compareEnabled, compareX01D);
                  }
                }}
              />

              {/* Step Player */}
              {res1D && res1D.steps.length > 0 ? (
                <div className="panel-surface p-3.5 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => {
                        setPlaying1D(false);
                        setStepIdx1D(0);
                      }}
                      aria-label="Primera iteración"
                      className="p-2 rounded-lg elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
                    >
                      <SkipBack className="h-4 w-4" />
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setPlaying1D(false);
                        setStepIdx1D((p) => Math.max(0, p - 1));
                      }}
                      aria-label="Iteración anterior"
                      className="p-2 rounded-lg elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </button>
                    <button
                      type="button"
                      disabled={res1D.steps.length <= 1}
                      onClick={() => {
                        if (playing1D) {
                          setPlaying1D(false);
                          return;
                        }
                        if (stepIdx1D >= res1D.steps.length - 1) setStepIdx1D(0);
                        setPlaying1D(true);
                      }}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500 text-slate-950 font-semibold text-xs cursor-pointer disabled:opacity-50"
                    >
                      {playing1D ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5 fill-current" />}
                      {playing1D ? 'Pausar' : 'Animar tangentes'}
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setPlaying1D(false);
                        setStepIdx1D((p) => Math.min(res1D.steps.length - 1, p + 1));
                      }}
                      aria-label="Siguiente iteración"
                      className="p-2 rounded-lg elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
                    >
                      <ChevronRight className="h-4 w-4" />
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setPlaying1D(false);
                        setStepIdx1D(res1D.steps.length - 1);
                      }}
                      aria-label="Última iteración"
                      className="p-2 rounded-lg elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
                    >
                      <SkipForward className="h-4 w-4" />
                    </button>
                  </div>

                  <span className="text-xs font-mono-num text-[var(--text-secondary)]">
                    Tangente mostrada: paso k = {stepIdx1D + 1} / {res1D.steps.length}
                  </span>
                </div>
              ) : null}

              {res1D ? (
                <NewtonConvergenceChart
                  mode="1d"
                  primaryLabel={`x₀ = ${res1D.fourier_condition?.x0 ?? x01D}`}
                  primarySteps={res1D.steps.map((s) => ({
                    iteration: s.iteration,
                    abs_error: s.abs_error,
                    residual: Math.abs(s.fx_k),
                  }))}
                  primaryOrder={res1D.estimated_order}
                  primaryStatus={res1D.status}
                  primaryRootLabel={res1D.root !== null ? `Raíz: ${res1D.root}` : undefined}
                  secondaryLabel={
                    compareEnabled && response1DCompare?.data
                      ? `x₀⁽ᴮ⁾ = ${response1DCompare.data.fourier_condition?.x0 ?? compareX01D}`
                      : undefined
                  }
                  secondarySteps={
                    compareEnabled && response1DCompare?.data
                      ? response1DCompare.data.steps.map((s) => ({
                          iteration: s.iteration,
                          abs_error: s.abs_error,
                          residual: Math.abs(s.fx_k),
                        }))
                      : undefined
                  }
                  secondaryOrder={compareEnabled ? response1DCompare?.data.estimated_order : undefined}
                  secondaryStatus={compareEnabled ? response1DCompare?.data.status : undefined}
                  secondaryRootLabel={
                    compareEnabled && response1DCompare?.data.root !== null && response1DCompare?.data.root !== undefined
                      ? `Raíz: ${response1DCompare.data.root}`
                      : undefined
                  }
                />
              ) : null}
            </div>

            <div className={`xl:col-span-5 space-y-4 transition-opacity ${is1DStale ? 'opacity-75' : ''}`}>
              {res1D && res1D.steps.length > 0 ? (
                <div className="panel-surface p-4 space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <h4 id={table1DCaptionId} className="text-sm font-semibold">
                      Tabla de iteraciones 1D
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={() => handleCopy1D('csv')}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
                      >
                        {copied1D === 'csv' ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                        Copiar CSV
                      </button>
                      <button
                        type="button"
                        onClick={() => handleCopy1D('latex')}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
                      >
                        {copied1D === 'latex' ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                        Copiar LaTeX
                      </button>
                    </div>
                  </div>
                  {copyError1D ? (
                    <p role="alert" className="text-xs text-[var(--text-err)]">
                      No se pudo copiar la tabla. Comprueba el acceso al portapapeles e inténtalo de nuevo.
                    </p>
                  ) : null}

                  <div
                    tabIndex={0}
                    role="region"
                    aria-label="Tabla de iteraciones 1D"
                    aria-labelledby={table1DCaptionId}
                    className="table-wrapper"
                  >
                    <table className="num-table font-mono-num">
                      <thead>
                        <tr>
                          <th>k</th>
                          <th>xₖ</th>
                          <th>f(xₖ)</th>
                          <th>f&apos;(xₖ)</th>
                          <th>f&apos;&apos;(xₖ)</th>
                          <th>xₖ₊₁</th>
                          <th>|Δx|</th>
                          <th>e_rel (%)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {res1D.steps.map((s, idx) => (
                          <tr
                            key={s.iteration}
                            tabIndex={0}
                            aria-current={idx === stepIdx1D ? 'step' : undefined}
                            data-active={idx === stepIdx1D}
                            onClick={() => {
                              setPlaying1D(false);
                              setStepIdx1D(idx);
                            }}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' || e.key === ' ') {
                                e.preventDefault();
                                setPlaying1D(false);
                                setStepIdx1D(idx);
                              }
                            }}
                          >
                            <th>k={s.iteration}</th>
                            <td data-label="xₖ">{s.x_k}</td>
                            <td data-label="f(xₖ)">{s.fx_k.toExponential(2)}</td>
                            <td data-label="f'(xₖ)">{formatScalar(s.dfx_k, decimals)}</td>
                            <td data-label="f''(xₖ)">
                              {s.d2fx_k !== null && s.d2fx_k !== undefined ? formatScalar(s.d2fx_k, decimals) : '—'}
                            </td>
                            <td data-label="xₖ₊₁">{s.x_next}</td>
                            <td data-label="|Δx|">{s.abs_error.toExponential(2)}</td>
                            <td data-label="e_rel (%)">{formatRelPct(s.rel_error_pct)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {active1DStep ? (
                    <div className="elevated-surface p-3 border border-amber-500/30">
                      <div className="text-xs font-semibold text-[var(--text-nl)] mb-1">
                        Fórmula sustituida (Iteración {active1DStep.iteration})
                      </div>
                      <LatexRenderer math={active1DStep.step_latex} displayMode />
                    </div>
                  ) : null}
                </div>
              ) : res1D ? (
                <div className="panel-surface p-4 space-y-2">
                  <h4 className="text-sm font-semibold">Tabla de iteraciones 1D (0 pasos)</h4>
                  <p className="text-xs text-[var(--text-secondary)]">
                    {res1D.status === 'converged'
                      ? `El valor inicial x₀ = ${res1D.root} ya satisface la tolerancia sin requerir iteraciones.`
                      : res1D.message}
                  </p>
                </div>
              ) : null}
            </div>
          </section>
        </div>
      ) : (
        /* Systems 2x2 / 3x3 View */
        <div className="space-y-5">
          {/* Presets */}
          <section className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
            {NEWTON_SYSTEM_PRESETS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                onClick={() => selectPresetSys(preset.id)}
                className={`text-left p-3 rounded-xl border transition-all cursor-pointer ${
                  activePresetSys === preset.id
                    ? 'bg-amber-500/12 border-amber-500/50'
                    : 'panel-surface hover:border-[var(--border-strong)]'
                }`}
              >
                <div className="text-xs font-semibold mb-1">{preset.title}</div>
                <div className="text-xs text-[var(--text-secondary)]">{preset.subtitle}</div>
              </button>
            ))}
          </section>

          {/* System Inputs + Symbolic Jacobian */}
          <section className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            <form
              className="lg:col-span-6 panel-surface p-4 space-y-3"
              onSubmit={(e) => {
                e.preventDefault();
                if (validSystem) runSystem(sysEqs, sysVars, sysX0, tol, maxIter, decimals, criterion, roundingMode);
              }}
            >
              <div className="flex items-center justify-between gap-2">
                <h3 className="text-sm font-semibold">Sistema no lineal F(x) = 0</h3>
                <div className="inline-flex p-1 rounded-lg elevated-surface gap-1">
                  <button
                    type="button"
                    aria-pressed={sysDim === 2}
                    onClick={() => switchSystemDim(2)}
                    className={`px-2.5 py-1 rounded text-xs font-mono-num cursor-pointer ${
                      sysDim === 2 ? 'bg-amber-500 text-slate-950 font-bold' : 'text-[var(--text-secondary)]'
                    }`}
                  >
                    2×2 (x, y)
                  </button>
                  <button
                    type="button"
                    aria-pressed={sysDim === 3}
                    onClick={() => switchSystemDim(3)}
                    className={`px-2.5 py-1 rounded text-xs font-mono-num cursor-pointer ${
                      sysDim === 3 ? 'bg-amber-500 text-slate-950 font-bold' : 'text-[var(--text-secondary)]'
                    }`}
                  >
                    3×3 (x, y, z)
                  </button>
                </div>
              </div>

              <span id="nl-sys-hint" className="block text-xs text-[var(--text-secondary)]">
                Escribe cada ecuación igualada a cero (ej. x^2 + y^2 - 4 o x^2 + y^2 = 4) usando {sysVars.join(', ')} y funciones (sin/sen, cos, exp, ln, sqrt, pi).
              </span>

              <div className="space-y-2.5">
                {sysEqs.map((eq, idx) => (
                  <div key={idx} className="grid grid-cols-1 sm:grid-cols-12 gap-2 items-start">
                    <div className="sm:col-span-9">
                      <label htmlFor={`sys-eq-${idx}`} className="block text-xs text-[var(--text-secondary)] mb-0.5">
                        f{idx + 1}({sysVars.join(', ')}) = 0
                      </label>
                      <input
                        id={`sys-eq-${idx}`}
                        type="text"
                        required
                        maxLength={300}
                        aria-describedby="nl-sys-hint"
                        value={eq}
                        onChange={(e) => {
                          const val = e.target.value;
                          setSysEqs((prev) => prev.map((item, i) => (i === idx ? val : item)));
                          setActivePresetSys('custom');
                        }}
                        className="field-input w-full px-3 py-1.5 font-mono-num text-sm"
                      />
                      <div className="error-msg">Ingresa una ecuación válida en {sysVars.join(', ')}.</div>
                    </div>
                    <div className="sm:col-span-3">
                      <label htmlFor={`sys-x0-${idx}`} className="block text-xs text-[var(--text-nl)] mb-0.5">
                        {sysVars[idx]}⁽⁰⁾ inicial
                      </label>
                      <input
                        id={`sys-x0-${idx}`}
                        type="number"
                        step="any"
                        required
                        value={systemFields[idx]}
                        onChange={(e) => {
                          const v = e.target.value;
                          setSysX0((prev) => prev.map((item, i) => (i === idx ? v : item)));
                          const parsed = parseFiniteInput(v);
                          if (parsed !== null) {
                            setLastValidSysX0((prev) => prev.map((item, i) => (i === idx ? parsed : item)));
                          }
                          setActivePresetSys('custom');
                        }}
                        className="field-input w-full px-2.5 py-1.5 font-mono-num text-sm border-amber-500/35"
                      />
                      <div className="error-msg">Número finito.</div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="flex flex-wrap gap-2 pt-1">
                <button
                  type="submit"
                  disabled={loadingSys || !validSystem}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-semibold text-sm cursor-pointer disabled:opacity-60"
                >
                  <Play className="h-4 w-4 fill-current" />
                  {loadingSys ? 'Resolviendo sistema...' : 'Resolver sistema F(x) = 0'}
                </button>
                <button
                  type="button"
                  onClick={() => selectPresetSys(sysDim === 2 ? 'circle-exp-2x2' : 'burden-3x3')}
                  className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl elevated-surface text-xs hover:border-[var(--border-strong)] cursor-pointer"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  Restaurar ejemplo
                </button>
              </div>
            </form>

            {/* Symbolic Jacobian Preview */}
            <div className={`lg:col-span-6 panel-surface p-4 space-y-3 transition-opacity ${isSysStale ? 'opacity-75' : ''}`}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-sm font-semibold text-[var(--text-nl)]">
                  Matriz Jacobiana simbólica J(x) · SymPy
                </h3>
                {badgeSys ? (
                  <span className={`text-xs px-2.5 py-0.5 rounded-full border ${badgeSys.cls}`}>
                    {badgeSys.label}
                  </span>
                ) : null}
              </div>

              {resSys?.f_latex_list && resSys.f_latex_list.length > 0 ? (
                <div className="elevated-surface px-3 py-2 space-y-1 overflow-x-auto">
                  <div className="text-xs font-medium text-[var(--text-secondary)]">Sistema interpretado F(x) = 0</div>
                  {resSys.f_latex_list.map((eqTex, idx) => (
                    <div key={idx} className="text-xs">
                      <LatexRenderer math={eqTex} displayMode />
                    </div>
                  ))}
                </div>
              ) : null}

              {resSys?.jacobian_latex ? (
                <MatrixDisplay
                  label="Jacobiano simbólico ∂fᵢ / ∂xⱼ"
                  latex={`J(${displaySysVars.join(',')}) = ${resSys.jacobian_latex}`}
                  accent="nl"
                />
              ) : null}

              {resSys?.solution ? (
                <div className="elevated-surface px-3 py-2 text-xs font-mono-num">
                  <span className="text-[var(--text-secondary)] mr-2">Solución:</span>
                  <span className="text-[var(--text-ok)] font-semibold">
                    ({resSys.solution.join(', ')})
                  </span>
                </div>
              ) : null}

              {resSys?.estimated_order != null ? (
                <div className="elevated-surface px-3 py-2 text-xs flex flex-wrap items-center justify-between gap-2 border border-amber-500/25">
                  <span className="text-[var(--text-secondary)]">Orden empírico de convergencia:</span>
                  <span className="font-mono-num font-semibold text-[var(--text-nl)]">
                    p ≈ {resSys.estimated_order.toFixed(3)}{' '}
                    ({resSys.estimated_order >= 1.65 ? 'Cuadrática' : resSys.estimated_order < 1.35 ? 'Lineal' : 'Superlineal'})
                  </span>
                </div>
              ) : null}

              {resSys ? (
                badgeSys?.isAlert ? (
                  <div
                    role="alert"
                    className={`elevated-surface px-3 py-2 text-xs border flex items-start gap-2 ${
                      resSys.status === 'max_iterations' || resSys.status === 'stagnated'
                        ? 'border-amber-500/35 bg-amber-500/10 text-[var(--text-nl)]'
                        : 'border-rose-500/35 bg-rose-500/10 text-[var(--text-err)]'
                    }`}
                  >
                    <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                    <span>{resSys.message}</span>
                  </div>
                ) : (
                  <p className="text-xs text-[var(--text-secondary)]">{resSys.message}</p>
                )
              ) : null}
            </div>
          </section>

          {currentErrorSys ? (
            <div role="alert" className="panel-surface p-3 border-rose-500/40 bg-rose-500/10 text-[var(--text-err)] text-sm flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>{currentErrorSys}</span>
            </div>
          ) : null}

          {/* 2D Plot + Table + LU Bridge Inspector */}
          <section className="grid grid-cols-1 xl:grid-cols-12 gap-5">
            {sysDim === 2 ? (
              <div className="xl:col-span-6 space-y-3">
                <Newton2DPlot
                  x0={sysX0}
                  result={plotResSys}
                  activeStepIndex={stepIdxSys}
                  isStale={isSysStale}
                  onPickInitial={(pt) => {
                    const nextX0 = [pt[0], pt[1]];
                    setSysX0(nextX0.map(String));
                    setLastValidSysX0(nextX0);
                    setActivePresetSys('custom');
                    if (validIterations && validTol && completeEquations) {
                      runSystem(sysEqs, sysVars, nextX0, tol, maxIter, decimals, criterion, roundingMode);
                    }
                  }}
                />

                {resSys && resSys.steps.length > 0 ? (
                  <div className="panel-surface p-3.5 flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        onClick={() => {
                          setPlayingSys(false);
                          setStepIdxSys(0);
                        }}
                        aria-label="Primera iteración"
                        className="p-2 rounded-lg elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
                      >
                        <SkipBack className="h-4 w-4" />
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setPlayingSys(false);
                          setStepIdxSys((p) => Math.max(0, p - 1));
                        }}
                        aria-label="Iteración anterior"
                        className="p-2 rounded-lg elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
                      >
                        <ChevronLeft className="h-4 w-4" />
                      </button>
                      <button
                        type="button"
                        disabled={resSys.steps.length <= 1}
                        onClick={() => {
                          if (playingSys) {
                            setPlayingSys(false);
                            return;
                          }
                          if (stepIdxSys >= resSys.steps.length - 1) setStepIdxSys(0);
                          setPlayingSys(true);
                        }}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500 text-slate-950 font-semibold text-xs cursor-pointer disabled:opacity-50"
                      >
                        {playingSys ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5 fill-current" />}
                        {playingSys ? 'Pausar' : 'Animar trayectoria'}
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setPlayingSys(false);
                          setStepIdxSys((p) => Math.min(resSys.steps.length - 1, p + 1));
                        }}
                        aria-label="Siguiente iteración"
                        className="p-2 rounded-lg elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
                      >
                        <ChevronRight className="h-4 w-4" />
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setPlayingSys(false);
                          setStepIdxSys(resSys.steps.length - 1);
                        }}
                        aria-label="Última iteración"
                        className="p-2 rounded-lg elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
                      >
                        <SkipForward className="h-4 w-4" />
                      </button>
                    </div>
                    <span className="text-xs font-mono-num text-[var(--text-secondary)]">
                      Paso k = {stepIdxSys + 1} / {resSys.steps.length}
                    </span>
                  </div>
                ) : null}

                {resSys ? (
                  <NewtonConvergenceChart
                    mode="system"
                    primaryLabel={`Sistema ${sysDim}×${sysDim}`}
                    primarySteps={resSys.steps.map((s) => ({
                      iteration: s.iteration,
                      abs_error: s.abs_error,
                      residual: s.residual,
                    }))}
                    primaryOrder={resSys.estimated_order}
                    primaryStatus={resSys.status}
                    primaryRootLabel={resSys.solution ? `Solución: (${resSys.solution.join(', ')})` : undefined}
                  />
                ) : null}
              </div>
            ) : null}

            <div className={`${sysDim === 2 ? 'xl:col-span-6' : 'xl:col-span-12'} space-y-4 transition-opacity ${isSysStale ? 'opacity-75' : ''}`}>
              {sysDim === 3 && resSys ? (
                <NewtonConvergenceChart
                  mode="system"
                  primaryLabel="Sistema 3×3"
                  primarySteps={resSys.steps.map((s) => ({
                    iteration: s.iteration,
                    abs_error: s.abs_error,
                    residual: s.residual,
                  }))}
                  primaryOrder={resSys.estimated_order}
                  primaryStatus={resSys.status}
                  primaryRootLabel={resSys.solution ? `Solución: (${resSys.solution.join(', ')})` : undefined}
                />
              ) : null}
              {resSys && resSys.steps.length > 0 ? (
                <>
                  <div className="panel-surface p-4 space-y-3">
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div>
                        <h4 id={tableSysCaptionId} className="text-sm font-semibold">
                          Tabla de iteraciones del sistema ({resSys.iterations} pasos)
                        </h4>
                        <p className="text-xs text-[var(--text-secondary)]">
                          Haz clic en una fila para inspeccionar cómo el módulo L.U. resolvió J(x⁽ᵏ⁾)Δx = -F(x⁽ᵏ⁾).
                        </p>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <button
                          type="button"
                          onClick={() => handleCopySys('csv')}
                          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
                        >
                          {copiedSys === 'csv' ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                          Copiar CSV
                        </button>
                        <button
                          type="button"
                          onClick={() => handleCopySys('latex')}
                          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium elevated-surface hover:border-[var(--border-strong)] cursor-pointer"
                        >
                          {copiedSys === 'latex' ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                          Copiar LaTeX
                        </button>
                      </div>
                    </div>
                    {copyErrorSys ? (
                      <p role="alert" className="text-xs text-[var(--text-err)]">
                        No se pudo copiar la tabla. Comprueba el acceso al portapapeles e inténtalo de nuevo.
                      </p>
                    ) : null}

                    <div
                      tabIndex={0}
                      role="region"
                      aria-label="Tabla de iteraciones del sistema"
                      aria-labelledby={tableSysCaptionId}
                      className="table-wrapper"
                    >
                      <table className="num-table font-mono-num">
                        <thead>
                          <tr>
                            <th>k</th>
                            {displaySysVars.map((v) => (
                              <th key={v}>{v}ₖ₊₁</th>
                            ))}
                            <th>||Δx||∞</th>
                            <th>||F(xₖ₊₁)||∞</th>
                            <th>e_rel (%)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {resSys.steps.map((s, idx) => (
                            <tr
                              key={s.iteration}
                              tabIndex={0}
                              aria-current={idx === stepIdxSys ? 'step' : undefined}
                              data-active={idx === stepIdxSys}
                              onClick={() => {
                                setPlayingSys(false);
                                setStepIdxSys(idx);
                              }}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter' || e.key === ' ') {
                                  e.preventDefault();
                                  setPlayingSys(false);
                                  setStepIdxSys(idx);
                                }
                              }}
                            >
                              <th>k={s.iteration}</th>
                              {s.x_next.map((val, j) => (
                                <td key={j} data-label={`${displaySysVars[j] ?? `x${j + 1}`}ₖ₊₁`}>
                                  {val}
                                </td>
                              ))}
                              <td data-label="||Δx||∞">{s.abs_error.toExponential(2)}</td>
                              <td data-label="||F||∞">
                                {s.residual !== null ? s.residual.toExponential(2) : '—'}
                              </td>
                              <td data-label="e_rel (%)">{formatRelPct(s.rel_error_pct)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* S.L. <-> N.L. Bridge Inspector */}
                  {activeSysStep ? (
                    <div className="panel-surface p-4 space-y-3 border border-sky-500/30">
                      <div className="flex items-center gap-2">
                        <GitBranch className="h-4 w-4 text-[var(--text-sl)]" />
                        <h4 className="text-sm font-semibold text-[var(--text-sl)]">
                          Puente S.L. ↔ N.L. — Paso lineal con PA = LU (Iteración k = {activeSysStep.iteration})
                        </h4>
                      </div>
                      <p className="text-xs text-[var(--text-secondary)]">
                        En esta iteración se evaluó el Jacobiano y se resolvió <LatexRenderer math="J(x^{(k)})\Delta x^{(k)} = -F(x^{(k)})" /> con nuestra factorización LU con pivoteo parcial:
                      </p>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                        <MatrixDisplay label="Jacobiano evaluado" latex={`J = ${activeSysStep.J_eval_latex}`} accent="nl" />
                        <MatrixDisplay label="Lado derecho del sistema lineal" latex={`-F = ${activeSysStep.neg_F_latex}`} accent="nl" />
                        <MatrixDisplay label="Matriz de permutación P" latex={`P = ${activeSysStep.lu_detail.P_latex}`} accent="sl" />
                        <div className="self-center">
                          <LatexRenderer math="P J = L U" displayMode />
                        </div>
                        <MatrixDisplay label="L del paso" latex={`L = ${activeSysStep.lu_detail.L_latex}`} accent="sl" />
                        <MatrixDisplay label="U del paso" latex={`U = ${activeSysStep.lu_detail.U_latex}`} accent="sl" />
                        <MatrixDisplay label="Vector intermedio y (Ly = −PF)" latex={`y = ${activeSysStep.lu_detail.y_latex}`} />
                        <MatrixDisplay label="Incremento Δx⁽ᵏ⁾ (U Δx = y)" latex={`\\Delta x = ${activeSysStep.lu_detail.dx_latex}`} accent="ok" />
                      </div>
                    </div>
                  ) : null}
                </>
              ) : resSys ? (
                <div className="panel-surface p-4 space-y-2">
                  <h4 className="text-sm font-semibold">Tabla de iteraciones del sistema (0 pasos)</h4>
                  <p className="text-xs text-[var(--text-secondary)]">
                    {resSys.status === 'converged'
                      ? 'El vector inicial x⁽⁰⁾ ya satisface la tolerancia sin requerir iteraciones.'
                      : resSys.message}
                  </p>
                </div>
              ) : null}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
