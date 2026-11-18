import type {
  LinearAnalyzeResponse,
  LUStrategy,
  Newton1DResponse,
  NewtonStoppingCriterion,
  NewtonSystemResponse,
  RoundingMode,
  SORSweepResponse,
  StoppingCriterion,
} from '../types/numerical';

export interface Newton1DRequestPayload {
  expression: string;
  x0: number;
  tol: number;
  max_iter: number;
  decimals: number;
  criterion?: NewtonStoppingCriterion;
  apply_rounding?: boolean;
  rounding_mode?: RoundingMode;
}

export interface NewtonSystemRequestPayload {
  equations: string[];
  variables: string[];
  x0: number[];
  tol: number;
  max_iter: number;
  decimals: number;
  criterion?: NewtonStoppingCriterion;
  apply_rounding?: boolean;
  rounding_mode?: RoundingMode;
  bounds?: { x_min: number; x_max: number; y_min: number; y_max: number };
}

async function postJson<T>(url: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    let detail = `Error del servidor (${response.status})`;
    try {
      const payload = await response.json();
      if (typeof payload?.detail === 'string') {
        detail = payload.detail;
      } else if (Array.isArray(payload?.detail)) {
        const messages = payload.detail
          .map((item: { loc?: unknown[]; msg?: string }) => {
            if (!item || typeof item.msg !== 'string') return null;
            const cleanMsg = item.msg.replace(/^Value error,\s*/i, '');
            const loc = Array.isArray(item.loc)
              ? item.loc.filter((part) => part !== 'body').join('.')
              : '';
            return loc ? `${loc}: ${cleanMsg}` : cleanMsg;
          })
          .filter((msg: string | null): msg is string => Boolean(msg));
        if (messages.length > 0) {
          detail = messages.join(' · ');
        }
      }
    } catch {
      // keep status text
    }
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

export function analyzeLinearSystem(
  payload: {
    A: number[][];
    b: number[];
    x0: number[];
    omega: number;
    tol: number;
    max_iter: number;
    criterion: StoppingCriterion;
    lu_strategy: LUStrategy;
    decimals: number;
    apply_rounding?: boolean;
    rounding_mode?: RoundingMode;
  },
  signal?: AbortSignal,
): Promise<LinearAnalyzeResponse> {
  return postJson<LinearAnalyzeResponse>('/api/linear/analyze', payload, signal);
}

export function sweepSorOmega(
  payload: {
    A: number[][];
    b: number[];
    x0: number[];
    tol: number;
    max_iter: number;
    criterion?: StoppingCriterion;
    decimals?: number;
    apply_rounding?: boolean;
    rounding_mode?: RoundingMode;
  },
  signal?: AbortSignal,
): Promise<SORSweepResponse> {
  return postJson<SORSweepResponse>('/api/linear/sor-sweep', payload, signal);
}

export function solveNewton1D(
  payload: Newton1DRequestPayload,
  signal?: AbortSignal,
): Promise<Newton1DResponse> {
  return postJson<Newton1DResponse>('/api/nonlinear/newton-1d', payload, signal);
}

export function solveNewtonSystem(
  payload: NewtonSystemRequestPayload,
  signal?: AbortSignal,
): Promise<NewtonSystemResponse> {
  return postJson<NewtonSystemResponse>('/api/nonlinear/newton-system', payload, signal);
}
