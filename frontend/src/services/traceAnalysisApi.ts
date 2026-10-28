
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";

export type TraceScenario =
  | "REST"
  | "WALKING"
  | "ELECTRODE_POP";

export interface TraceCompensation {
  decision: string;
  reason: string;
  accepted: boolean;
}

export interface UnifiedWindow {
  scenario: TraceScenario;
  start_seconds: number;
  sample_rate_hz: number;
  sample_count: number;
  duration_seconds: number;
  artifact_label: string;
  quality_flags: string[];
  compensation: TraceCompensation;
  trust_status: string;
  trust_reasons: string[];
  observed_ecg: number[];
  output_ecg: number[];
  simulated_motion?: number[];
  metadata?: Record<string, unknown>;
}

export interface TraceEvent {
  event_type: string;
  timestamp_seconds: number;
  trust_status: string;
  trust_reasons: string[];
  artifact_label?: string;
  quality_flags?: string[];
  requires_review?: boolean;
  clinical_validated?: boolean;
  [key: string]: unknown;
}

export interface TraceEventReport {
  events: TraceEvent[];
  event_count?: number;
  [key: string]: unknown;
}

async function fetchTraceJson<T>(
  path: string,
  signal?: AbortSignal
): Promise<T> {
  const response = await fetch(
    `${API_BASE_URL}${path}`,
    {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
      signal,
    }
  );

  if (!response.ok) {
    throw new Error(
      `TRACE API error: HTTP ${response.status}`
    );
  }

  return (await response.json()) as T;
}

export function getUnifiedWindow(
  scenario: TraceScenario,
  signal?: AbortSignal
): Promise<UnifiedWindow> {
  const params = new URLSearchParams({
    scenario,
    start_seconds: "180",
    seed: "42",
  });

  return fetchTraceJson<UnifiedWindow>(
    `/api/trace/unified-window?${params.toString()}`,
    signal
  );
}

export function getTraceEventAnalysis(
  signal?: AbortSignal
): Promise<TraceEventReport> {
  return fetchTraceJson<TraceEventReport>(
    "/api/trace/event-analysis",
    signal
  );
}
