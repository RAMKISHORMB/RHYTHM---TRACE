
import { useEffect, useState } from "react";
import {
  getCardiacEvents,
  type CardiacEvent,
  type CardiacEventsResponse,
} from "../services/eventApi";

const panelStyle: React.CSSProperties = {
  background: "#0f172a",
  border: "1px solid #334155",
  borderRadius: 14,
  padding: 20,
  marginTop: 24,
};

const innerStyle: React.CSSProperties = {
  background: "#1e293b",
  border: "1px solid #334155",
  borderRadius: 12,
  padding: 16,
};

function formatSeconds(value: number): string {
  return `${value.toFixed(3)} s`;
}

function EventWaveform({
  event,
}: {
  event: CardiacEvent;
}) {
  const evidence = event.evidence;
  const samples = evidence.ecg_samples;
  const fs = evidence.snippet_sampling_rate;

  const width = 900;
  const height = 240;
  const padding = 28;

  if (samples.length < 2 || fs <= 0) {
    return <p>ECG waveform unavailable.</p>;
  }

  const min = Math.min(...samples);
  const max = Math.max(...samples);
  const range = Math.max(max - min, 1e-6);

  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;

  const xAt = (index: number) =>
    padding + (index / (samples.length - 1)) * plotWidth;

  const yAt = (value: number) =>
    padding +
    (1 - (value - min) / range) * plotHeight;

  const points = samples
    .map((value, index) => `${xAt(index)},${yAt(value)}`)
    .join(" ");

  const eventOffset =
    (evidence.timestamp_seconds -
      evidence.snippet_start_seconds) * fs;

  const eventX = xAt(
    Math.max(0, Math.min(samples.length - 1, eventOffset))
  );

  const gridLines = Array.from(
    { length: 9 },
    (_, index) => padding + (index * plotWidth) / 8
  );

  return (
    <div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Recorded ECG around event at ${formatSeconds(
          event.timestamp_seconds
        )}`}
        style={{
          width: "100%",
          background: "#020617",
          borderRadius: 10,
          border: "1px solid #334155",
        }}
      >
        {gridLines.map((x, index) => (
          <line
            key={`vertical-${index}`}
            x1={x}
            x2={x}
            y1={padding}
            y2={height - padding}
            stroke="#1e293b"
            strokeWidth={1}
          />
        ))}

        {Array.from({ length: 5 }, (_, index) => {
          const y =
            padding + (index * plotHeight) / 4;

          return (
            <line
              key={`horizontal-${index}`}
              x1={padding}
              x2={width - padding}
              y1={y}
              y2={y}
              stroke="#1e293b"
              strokeWidth={1}
            />
          );
        })}

        <polyline
          points={points}
          fill="none"
          stroke="#38bdf8"
          strokeWidth={1.8}
          strokeLinejoin="round"
        />

        <line
          x1={eventX}
          x2={eventX}
          y1={padding}
          y2={height - padding}
          stroke="#fbbf24"
          strokeWidth={2}
          strokeDasharray="6 5"
        />

        <text
          x={Math.min(eventX + 8, width - 145)}
          y={22}
          fill="#fbbf24"
          fontSize={13}
        >
          Event candidate
        </text>

        <text
          x={padding}
          y={height - 7}
          fill="#94a3b8"
          fontSize={12}
        >
          {formatSeconds(evidence.snippet_start_seconds)}
        </text>

        <text
          x={width - padding}
          y={height - 7}
          textAnchor="end"
          fill="#94a3b8"
          fontSize={12}
        >
          {formatSeconds(evidence.snippet_end_seconds)}
        </text>
      </svg>

      <p style={{ fontSize: 12, color: "#94a3b8" }}>
        Recorded ECG · {fs} Hz · {samples.length} samples ·
        Automatically scaled digital values
      </p>
    </div>
  );
}

export default function CardiacEventReview() {
  const [data, setData] =
    useState<CardiacEventsResponse | null>(null);

  const [selectedId, setSelectedId] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    getCardiacEvents(controller.signal)
      .then((response) => {
        if (controller.signal.aborted) return;

        setData(response);
        setSelectedId(response.events[0]?.event_id ?? "");
        setError("");
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;

        setError(
          reason instanceof Error
            ? reason.message
            : "Unable to load cardiac-event evidence."
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      });

    return () => controller.abort();
  }, []);

  const selectedEvent =
    data?.events.find((event) => event.event_id === selectedId) ??
    null;

  return (
    <section style={panelStyle}>
      <h2>Cardiac Event Review</h2>

      <p style={{ color: "#94a3b8" }}>
        Transient RR-timing candidates detected from
        recorded ECG. Select an event to inspect its
        evidence.
      </p>

      {loading && <p>Loading cardiac-event evidence...</p>}

      {error && (
        <p role="alert" style={{ color: "#f87171" }}>
          {error}
        </p>
      )}

      {data && (
        <>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 12,
              marginBottom: 20,
            }}
          >
            <div style={innerStyle}>
              <small>Recording</small>
              <h3>{data.recording_id}</h3>
            </div>

            <div style={innerStyle}>
              <small>Event Candidates</small>
              <h3>{data.event_count}</h3>
            </div>

            <div style={innerStyle}>
              <small>Recording Duration</small>
              <h3>{data.duration_seconds.toFixed(0)} s</h3>
            </div>

            <div style={innerStyle}>
              <small>Motion Context</small>
              <h3>
                {data.motion_available
                  ? "Available"
                  : "Not available"}
              </h3>
            </div>
          </div>

          {data.events.length === 0 ? (
            <p>No event candidates detected in this recording.</p>
          ) : (
            <>
              <div style={{ marginBottom: 20 }}>
                <label htmlFor="trace-event-select">
                  <strong>Select an event</strong>
                </label>

                <select
                  id="trace-event-select"
                  value={selectedId}
                  onChange={(e) =>
                    setSelectedId(e.target.value)
                  }
                  style={{
                    display: "block",
                    marginTop: 10,
                    width: "100%",
                    maxWidth: 480,
                    padding: 12,
                    background: "#020617",
                    color: "#e2e8f0",
                    border: "1px solid #475569",
                    borderRadius: 8,
                  }}
                >
                  {data.events.map((event) => (
                    <option
                      key={event.event_id}
                      value={event.event_id}
                    >
                      {event.event_type} —{" "}
                      {formatSeconds(event.timestamp_seconds)}
                    </option>
                  ))}
                </select>
              </div>

              {selectedEvent && (
                <div style={innerStyle}>
                  <h3>Recorded ECG Evidence</h3>

                  <EventWaveform event={selectedEvent} />

                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns:
                        "repeat(auto-fit, minmax(180px, 1fr))",
                      gap: 12,
                      marginTop: 20,
                    }}
                  >
                    <div>
                      <small>Observed RR Interval</small>
                      <h3>
                        {(
                          selectedEvent.evidence
                            .rr_interval_seconds * 1000
                        ).toFixed(0)}{" "}
                        ms
                      </h3>
                    </div>

                    <div>
                      <small>Baseline RR Interval</small>
                      <h3>
                        {(
                          selectedEvent.evidence
                            .baseline_rr_seconds * 1000
                        ).toFixed(0)}{" "}
                        ms
                      </h3>
                    </div>

                    <div>
                      <small>RR / Baseline Ratio</small>
                      <h3>
                        {selectedEvent.evidence
                          .ratio_to_baseline.toFixed(3)}
                      </h3>
                    </div>

                    <div>
                      <small>ECG Quality</small>
                      <h3>{selectedEvent.quality_status}</h3>
                    </div>
                  </div>

                  <h4>Why TRACE flagged this event</h4>

                  <p>
                    {selectedEvent.evidence.explanation}
                  </p>

                  <h4>Signal Quality Evidence</h4>

                  <p>
                    {selectedEvent.quality_flags.length > 0
                      ? selectedEvent.quality_flags.join(", ")
                      : "No warning from current experimental quality heuristics."}
                  </p>

                  <p style={{ color: "#94a3b8" }}>
                    Motion data:{" "}
                    {selectedEvent.motion_available
                      ? "Available"
                      : "Not available for this ECG recording"}
                  </p>

                  <p style={{ color: "#fbbf24" }}>
                    {selectedEvent.evidence.requires_review
                      ? "Review required — candidate event, not a diagnosis."
                      : "Research evidence only."}
                  </p>
                </div>
              )}
            </>
          )}

          <p
            style={{
              marginTop: 20,
              fontSize: 12,
              color: "#fbbf24",
            }}
          >
            {data.disclaimer}
          </p>
        </>
      )}
    </section>
  );
}
