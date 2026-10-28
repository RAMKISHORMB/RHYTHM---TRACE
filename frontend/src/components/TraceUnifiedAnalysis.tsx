
import { useEffect, useState } from "react";
import type { CSSProperties } from "react";

type Scenario = "REST" | "WALKING" | "ELECTRODE_POP";

interface Compensation {
  decision: string;
  reason: string;
  accepted: boolean;
}

interface UnifiedWindow {
  scenario: Scenario;
  sample_rate_hz: number;
  sample_count: number;
  artifact_label: string;
  quality_flags: string[];
  compensation: Compensation;
  trust_status: string;
  trust_reasons: string[];
  observed_ecg: number[];
  output_ecg: number[];
}

interface TraceEvent {
  event_id?: string;
  event_type?: string;
  timestamp_seconds?: number;
  trust_status?: string;
  trust_reasons?: string[];
  artifact_label?: string;
  requires_review?: boolean;
  local_analysis?: {
    artifact_label?: string;
  };
}

interface EventReport {
  events: TraceEvent[];
  duration_seconds?: number;
}

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";

const panel: CSSProperties = {
  background: "#111827",
  border: "1px solid #334155",
  borderRadius: 16,
  padding: 24,
  marginTop: 24,
  color: "#f8fafc",
};

const card: CSSProperties = {
  background: "#020617",
  border: "1px solid #334155",
  borderRadius: 10,
  padding: 16,
};

const muted: CSSProperties = {
  color: "#94a3b8",
  fontSize: 13,
  lineHeight: 1.6,
};

async function fetchJson<T>(
  path: string,
  signal: AbortSignal
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    signal,
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${path}`);
  }

  return (await response.json()) as T;
}

function ECGComparison({
  observed,
  output,
}: {
  observed: number[];
  output: number[];
}) {
  const width = 900;
  const height = 240;
  const padding = 16;

  if (!observed.length || !output.length) {
    return <p style={muted}>No ECG samples available.</p>;
  }

  const all = [...observed, ...output];
  const minimum = Math.min(...all);
  const maximum = Math.max(...all);
  const range = maximum - minimum || 1;

  function makePoints(values: number[]): string {
    return values
      .map((value, index) => {
        const x =
          padding +
          (index / Math.max(values.length - 1, 1)) *
            (width - 2 * padding);

        const y =
          height -
          padding -
          ((value - minimum) / range) *
            (height - 2 * padding);

        return `${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
  }

  return (
    <div style={{ overflowX: "auto" }}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{
          width: "100%",
          minWidth: 480,
          background: "#020617",
          borderRadius: 10,
        }}
        role="img"
        aria-label="Observed and output ECG waveforms"
      >
        <polyline
          points={makePoints(observed)}
          fill="none"
          stroke="#38bdf8"
          strokeWidth={1.5}
        />
        <polyline
          points={makePoints(output)}
          fill="none"
          stroke="#5eead4"
          strokeWidth={1.5}
          opacity={0.85}
        />
      </svg>

      <p style={muted}>
        <span style={{ color: "#38bdf8" }}>
          ● Observed ECG
        </span>
        {"  |  "}
        <span style={{ color: "#5eead4" }}>
          ● Pipeline output
        </span>
      </p>
    </div>
  );
}

export default function TraceUnifiedAnalysis() {
  const [scenario, setScenario] =
    useState<Scenario>("REST");

  const [windowData, setWindowData] =
    useState<UnifiedWindow | null>(null);

  const [eventReport, setEventReport] =
    useState<EventReport | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [eventError, setEventError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    setLoading(true);
    setError("");
    setWindowData(null);

    const params = new URLSearchParams({
      scenario,
      start_seconds: "180",
      seed: "42",
    });

    fetchJson<UnifiedWindow>(
      `/api/trace/unified-window?${params}`,
      controller.signal
    )
      .then((data) => {
        if (!controller.signal.aborted) {
          setWindowData(data);
        }
      })
      .catch((err: unknown) => {
        if (!controller.signal.aborted) {
          setError(
            err instanceof Error ? err.message : String(err)
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      });

    return () => controller.abort();
  }, [scenario]);

  useEffect(() => {
    const controller = new AbortController();

    fetchJson<EventReport>(
      "/api/trace/event-analysis",
      controller.signal
    )
      .then((data) => {
        if (!controller.signal.aborted) {
          setEventReport(data);
        }
      })
      .catch((err: unknown) => {
        if (!controller.signal.aborted) {
          setEventError(
            err instanceof Error ? err.message : String(err)
          );
        }
      });

    return () => controller.abort();
  }, []);

  return (
    <section style={panel}>
      <h2>TRACE Unified ECG Analysis</h2>

      <p style={muted}>
        Experimental ECG analysis with simulated motion
        scenarios and event-local trust routing.
      </p>

      <label
        htmlFor="trace-scenario"
        style={{ display: "block", marginBottom: 8 }}
      >
        Select scenario
      </label>

      <select
        id="trace-scenario"
        value={scenario}
        onChange={(event) =>
          setScenario(event.target.value as Scenario)
        }
        style={{
          background: "#020617",
          color: "#f8fafc",
          border: "1px solid #475569",
          borderRadius: 8,
          padding: 12,
          marginBottom: 20,
        }}
      >
        <option value="REST">REST</option>
        <option value="WALKING">WALKING</option>
        <option value="ELECTRODE_POP">
          ELECTRODE_POP
        </option>
      </select>

      {loading && <p>Loading ECG analysis...</p>}

      {error && (
        <p role="alert" style={{ color: "#fca5a5" }}>
          {error}
        </p>
      )}

      {windowData && (
        <>
          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "repeat(auto-fit, minmax(180px, 1fr))",
              gap: 12,
            }}
          >
            <div style={card}>
              <p style={muted}>Artefact prediction</p>
              <strong>{windowData.artifact_label}</strong>
            </div>

            <div style={card}>
              <p style={muted}>Trust status</p>
              <strong>{windowData.trust_status}</strong>
            </div>

            <div style={card}>
              <p style={muted}>Compensation</p>
              <strong>
                {windowData.compensation?.decision ??
                  "NOT_AVAILABLE"}
              </strong>
            </div>

            <div style={card}>
              <p style={muted}>Quality flags</p>
              <strong>
                {windowData.quality_flags.length
                  ? windowData.quality_flags.join(", ")
                  : "None detected"}
              </strong>
            </div>
          </div>

          <h3 style={{ marginTop: 24 }}>
            ECG Waveform Comparison
          </h3>

          <ECGComparison
            observed={windowData.observed_ecg}
            output={windowData.output_ecg}
          />

          <p style={muted}>
            <strong>Compensation reason:</strong>{" "}
            {windowData.compensation?.reason ??
              "NOT_AVAILABLE"}
          </p>

          <p style={muted}>
            <strong>Correction accepted:</strong>{" "}
            {windowData.compensation?.accepted === true
              ? "Yes"
              : "No"}
          </p>

          <p style={muted}>
            <strong>Trust reasons:</strong>{" "}
            {windowData.trust_reasons.join(", ") ||
              "None supplied"}
          </p>

          <p style={muted}>
            Motion is simulated. The artefact classifier
            and quality heuristics are experimental.
            A clean prediction does not establish
            interference-free ECG.
          </p>
        </>
      )}

      <h3 style={{ marginTop: 28 }}>
        Event-Local Trust Evidence
      </h3>

      {eventError && (
        <p role="alert" style={{ color: "#fca5a5" }}>
          Event analysis unavailable: {eventError}
        </p>
      )}

      {!eventReport && !eventError && (
        <p style={muted}>Loading event evidence...</p>
      )}

      {eventReport && (
        <>
          <p style={muted}>
            {eventReport.events.length} candidates
            {typeof eventReport.duration_seconds === "number"
              ? ` | ${eventReport.duration_seconds}s recording`
              : " | Recorded ECG"}
          </p>

          {eventReport.events.map((event, index) => {
            const time =
              typeof event.timestamp_seconds === "number"
                ? event.timestamp_seconds
                : null;

            const reasons = event.trust_reasons ?? [];

            return (
              <div
                key={event.event_id ?? `event-${index}`}
                style={{
                  ...card,
                  marginBottom: 12,
                }}
              >
                <strong>
                  Event {index + 1}:{" "}
                  {event.event_type ?? "EVENT_CANDIDATE"}
                </strong>

                <p style={muted}>
                  Time:{" "}
                  {time !== null
                    ? `${time.toFixed(3)}s`
                    : "Not available"}
                </p>

                <p style={muted}>
                  Trust:{" "}
                  {event.trust_status ?? "NOT_AVAILABLE"}
                </p>

                <p style={muted}>
                  Artefact prediction:{" "}
                  {event.local_analysis?.artifact_label ??
                    event.artifact_label ??
                    "Not available"}
                </p>

                <p style={muted}>
                  Evidence:{" "}
                  {reasons.length
                    ? reasons.join(", ")
                    : "No trust reasons supplied"}
                </p>

                <p style={muted}>
                  Requires review:{" "}
                  {event.requires_review === false
                    ? "No"
                    : "Yes"}
                </p>
              </div>
            );
          })}
        </>
      )}

      <p style={muted}>
        Recorded MIT-BIH ECG is not live monitoring.
        Physical motion measurements are unavailable
        for event-local analysis. All event candidates
        require research review; this system is not
        clinically validated for diagnosis or treatment.
      </p>
    </section>
  );
}
