
/**
 * TRACE — Cardiac Event API Client
 * MEDAITHON'26 | Cardiology PS-001
 *
 * Research-only ECG event evidence.
 * No clinical diagnosis or live patient monitoring.
 */

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  "http://127.0.0.1:8000";

export interface EventQualityContext {
  status: "REVIEW_QUALITY" | "NO_HEURISTIC_WARNING";
  flags: string[];
  method: string;
  clinical_validated: boolean;
  note: string;
}

export interface EventEvidence {
  event_id: string;
  event_type: string;
  timestamp_seconds: number;
  rr_interval_seconds: number;
  baseline_rr_seconds: number;
  ratio_to_baseline: number;
  explanation: string;
  snippet_start_seconds: number;
  snippet_end_seconds: number;
  snippet_sampling_rate: number;
  ecg_samples: number[];
  motion_samples: number[] | null;
  motion_source: string | null;
  quality_context: EventQualityContext;
  requires_review: boolean;
  experimental: boolean;
  clinical_validated: boolean;
}

export interface CardiacEvent {
  event_id: string;
  event_type: string;
  timestamp_seconds: number;
  quality_status: string;
  quality_flags: string[];
  motion_available: boolean;
  evidence: EventEvidence;
  experimental: boolean;
  clinical_validated: boolean;
}

export interface CardiacEventsResponse {
  recording_id: string;
  recording_type: string;
  duration_seconds: number;
  sampling_rate: number;
  peak_count: number;
  estimated_heart_rate_bpm: number | null;
  event_count: number;
  events: CardiacEvent[];
  motion_available: boolean;
  live_monitoring: boolean;
  experimental: boolean;
  clinical_validated: boolean;
  disclaimer: string;
}

async function fetchJson<T>(
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

export async function getCardiacEvents(
  signal?: AbortSignal
): Promise<CardiacEventsResponse> {
  return fetchJson<CardiacEventsResponse>(
    "/api/events",
    signal
  );
}

export async function getCardiacEvent(
  eventId: string,
  signal?: AbortSignal
): Promise<CardiacEvent> {
  return fetchJson<CardiacEvent>(
    `/api/events/${encodeURIComponent(eventId)}`,
    signal
  );
}
