
import { useEffect, useRef, useState } from "react";

import ArtifactPredictionPanel, {
  type ArtifactPrediction,
} from "./ArtifactPredictionPanel";

import DualECGChart, {
  type BeatMarker,
} from "./DualECGChart";

// ============================================================
// TRACE — AI ARTEFACT DEMONSTRATION WITH BEAT-LEVEL TRUST
// Recorded ECG + Synthetic Artefacts + Simulated IMU
// ============================================================

type ConnectionStatus =
  | "connecting"
  | "connected"
  | "completed"
  | "disconnected"
  | "error";

type ArtifactECGFrame = {
  type: "artifact_ecg_frame";
  window_index: number;
  sample_offset: number;
  fs: number;
  ecg_reference: number[];
  ecg_corrupted: number[];
  imu_simulated_values: number[];
  ecg_units: string;
  ecg_replayed: boolean;
  artifact_injected: boolean;
  imu_simulated: boolean;
};

type AuditEntry = {
  event_id: string;
  timestamp_utc: string;
  window_index: number;
  artifact_class: string;
  trust_level: "HIGH" | "REVIEW" | "UNRELIABLE";
  action: "PRESERVED" | "CANDIDATE_FILTERED" | "ABSTAINED";
  explanation: string;
  experimental: boolean;
  clinical_validated: boolean;
};

type ArtifactAuditMessage = {
  type: "artifact_audit";
  window_index: number;
  audit: AuditEntry;
};

type DemoComplete = {
  type: "artifact_demo_complete";
  windows_streamed: number;
};

type StreamError = {
  type: "error";
  message: string;
};

type StreamMessage =
  | ArtifactPrediction
  | ArtifactECGFrame
  | ArtifactAuditMessage
  | DemoComplete
  | StreamError;

const WS_URL = "ws://127.0.0.1:8000/ws/artifact-demo";

const WINDOW_SECONDS = 4;
const EXPECTED_WINDOWS = 5;

export default function ArtifactDemo() {
  const [status, setStatus] =
    useState<ConnectionStatus>("disconnected");

  const [prediction, setPrediction] =
    useState<ArtifactPrediction | null>(null);

  const [auditEntries, setAuditEntries] = useState<AuditEntry[]>([]);

  const [beatMarkers, setBeatMarkers] =
    useState<BeatMarker[]>([]);

  const [referenceSamples, setReferenceSamples] =
    useState<number[]>([]);

  const [corruptedSamples, setCorruptedSamples] =
    useState<number[]>([]);

  const [samplingRate, setSamplingRate] = useState(250);

  const [activeWindow, setActiveWindow] =
    useState<number | null>(null);

  const [frameCount, setFrameCount] = useState(0);
  const [sampleCount, setSampleCount] = useState(0);
  const [completedWindows, setCompletedWindows] = useState(0);
  const [errorMessage, setErrorMessage] = useState("");

  const socketRef = useRef<WebSocket | null>(null);
  const windowRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, []);

  function startDemo() {
    socketRef.current?.close();
    windowRef.current = null;

    setStatus("connecting");
    setPrediction(null);
    setAuditEntries([]);
    setBeatMarkers([]);

    setReferenceSamples([]);
    setCorruptedSamples([]);

    setActiveWindow(null);
    setSamplingRate(250);

    setFrameCount(0);
    setSampleCount(0);
    setCompletedWindows(0);
    setErrorMessage("");

    const socket = new WebSocket(WS_URL);
    socketRef.current = socket;

    socket.onopen = () => {
      if (socketRef.current !== socket) return;
      setStatus("connected");
    };

    socket.onmessage = (event: MessageEvent<string>) => {
      if (socketRef.current !== socket) return;

      let data: StreamMessage;

      try {
        data = JSON.parse(event.data) as StreamMessage;
      } catch {
        setErrorMessage("Received invalid WebSocket JSON.");
        setStatus("error");
        socket.close();
        return;
      }

      switch (data.type) {
        case "artifact_prediction": {
          // Each prediction belongs to a new ECG window.
          windowRef.current = data.window_index;

          // Reset the previous window's waveform and markers.
          setReferenceSamples([]);
          setCorruptedSamples([]);
          setBeatMarkers([]);

          setActiveWindow(data.window_index);
          setPrediction(data);

          // These positions are MIT-BIH reference annotations.
          // Their trust labels are experimental assessments.
          const incomingBeats = data.beats ?? [];

          setBeatMarkers(incomingBeats);

          break;
        }

        case "artifact_audit": {
          if (data.window_index !== data.audit.window_index) {
            setErrorMessage("Audit event window mismatch.");
            setStatus("error");
            socket.close();
            break;
          }

          setAuditEntries((previous) => {
            if (previous.some((entry) => entry.event_id === data.audit.event_id)) {
              return previous;
            }
            return [...previous, data.audit];
          });
          break;
        }

        case "artifact_ecg_frame": {
          // Ignore stale or out-of-order windows.
          if (data.window_index !== windowRef.current) {
            break;
          }

          if (
            data.ecg_reference.length !==
            data.ecg_corrupted.length
          ) {
            setErrorMessage(
              "ECG channel lengths do not match."
            );
            setStatus("error");
            socket.close();
            break;
          }

          if (
            !Number.isFinite(data.fs) ||
            data.fs <= 0
          ) {
            setErrorMessage("Invalid ECG sampling rate.");
            setStatus("error");
            socket.close();
            break;
          }

          setSamplingRate(data.fs);

          const maxSamples =
            data.fs * WINDOW_SECONDS;

          setReferenceSamples((previous) =>
            [...previous, ...data.ecg_reference].slice(
              -maxSamples
            )
          );

          setCorruptedSamples((previous) =>
            [...previous, ...data.ecg_corrupted].slice(
              -maxSamples
            )
          );

          setFrameCount((previous) => previous + 1);

          setSampleCount(
            (previous) =>
              previous + data.ecg_corrupted.length
          );

          break;
        }

        case "artifact_demo_complete": {
          setCompletedWindows(data.windows_streamed);
          setStatus("completed");
          break;
        }

        case "error": {
          setErrorMessage(data.message);
          setStatus("error");
          socket.close();
          break;
        }
      }
    };

    socket.onerror = () => {
      if (socketRef.current !== socket) return;

      setErrorMessage(
        "Unable to connect to the TRACE AI WebSocket."
      );

      setStatus("error");
    };

    socket.onclose = () => {
      if (socketRef.current !== socket) return;

      socketRef.current = null;

      setStatus((previous) =>
        previous === "completed" || previous === "error"
          ? previous
          : "disconnected"
      );
    };
  }

  function stopDemo() {
    socketRef.current?.close();
    socketRef.current = null;
    windowRef.current = null;

    setStatus("disconnected");
  }

  const cardStyle = {
    background: "#0f172a",
    border: "1px solid #334155",
    borderRadius: 14,
    padding: 20,
    marginTop: 20,
  };

  const statStyle = {
    background: "#1e293b",
    border: "1px solid #334155",
    borderRadius: 12,
    padding: 16,
  };

  return (
    <section style={cardStyle}>
      <h2>TRACE — AI Artefact Demonstration</h2>

      <p style={{ color: "#94a3b8" }}>
        Recorded MIT-BIH ECG · Injected synthetic artefacts ·
        Simulated IMU
      </p>

      {/* Controls */}

      <div
        style={{
          display: "flex",
          gap: 12,
          flexWrap: "wrap",
          marginBottom: 20,
        }}
      >
        <button
          onClick={startDemo}
          disabled={
            status === "connecting" ||
            status === "connected"
          }
        >
          Start AI Demo
        </button>

        <button
          onClick={stopDemo}
          disabled={
            status !== "connecting" &&
            status !== "connected"
          }
        >
          Stop Demo
        </button>
      </div>

      {/* Connection statistics */}

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fit, minmax(170px, 1fr))",
          gap: 12,
          marginBottom: 20,
        }}
      >
        <div style={statStyle}>
          <small>Connection</small>
          <h3>{status.toUpperCase()}</h3>
        </div>

        <div style={statStyle}>
          <small>ECG Frames</small>
          <h3>{frameCount}</h3>
        </div>

        <div style={statStyle}>
          <small>ECG Samples</small>
          <h3>{sampleCount.toLocaleString()}</h3>
        </div>

        <div style={statStyle}>
          <small>Completed Windows</small>
          <h3>{completedWindows}</h3>
        </div>
      </div>

      {/* Dual ECG visualization */}

      <div style={statStyle}>
        <h3>Original vs Artefact-Corrupted ECG</h3>

        <p style={{ color: "#94a3b8" }}>
          Synchronized four-second window ·
          250 Hz · Uncalibrated digital values
        </p>

        <p>
          <strong>Current window:</strong>{" "}
          {activeWindow === null
            ? "Waiting for stream"
            : `${activeWindow + 1} of ${EXPECTED_WINDOWS}`}
        </p>

        <DualECGChart
          referenceSamples={referenceSamples}
          corruptedSamples={corruptedSamples}
          fs={samplingRate}
          beats={beatMarkers}
        />

        <div
          style={{
            display: "flex",
            gap: 20,
            flexWrap: "wrap",
            fontSize: 13,
            marginTop: 12,
          }}
        >
          <span style={{ color: "#10b981" }}>
            ● Original recorded ECG
          </span>

          <span style={{ color: "#f59e0b" }}>
            ● Artefact-corrupted ECG
          </span>
        </div>

        {/* Beat trust summary */}

        {prediction && (
          <div
            style={{
              marginTop: 18,
              padding: 14,
              borderRadius: 10,
              background: "#0f172a",
              border: "1px solid #334155",
            }}
          >
            <h4 style={{ marginTop: 0 }}>
              Beat-Level Trust Analysis
            </h4>

            <p style={{ color: "#94a3b8" }}>
              {beatMarkers.length} MIT-BIH reference beats
              aligned to the current four-second window.
            </p>

            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: 12,
              }}
            >
              {beatMarkers.map((beat) => {
                const color =
                  beat.trust_level === "HIGH"
                    ? "#10b981"
                    : beat.trust_level === "REVIEW"
                      ? "#f59e0b"
                      : "#ef4444";

                return (
                  <div
                    key={beat.relative_sample}
                    style={{
                      background: "#1e293b",
                      border: `1px solid ${color}`,
                      borderRadius: 10,
                      padding: "10px 14px",
                      minWidth: 125,
                    }}
                  >
                    <div
                      style={{
                        color: "#94a3b8",
                        fontSize: 12,
                      }}
                    >
                      Beat @{" "}
                      {(
                        beat.relative_sample /
                        samplingRate
                      ).toFixed(2)}
                      s
                    </div>

                    <strong
                      style={{
                        color,
                        display: "block",
                        marginTop: 5,
                      }}
                    >
                      {beat.trust_level}
                    </strong>

                    <small>
                      Reference: {beat.symbol}
                    </small>
                  </div>
                );
              })}
            </div>

            <p
              style={{
                color: "#94a3b8",
                fontSize: 12,
                marginTop: 14,
                marginBottom: 0,
              }}
            >
              Beat positions and symbols are dataset
              annotations. Trust levels are experimental
              engineering assessments, not clinical diagnoses.
            </p>
          </div>
        )}
      </div>

      {/* Decision audit timeline: generated during this replay only */}
      <div style={cardStyle}>
        <h3 style={{ marginTop: 0 }}>TRACE Decision Timeline</h3>
        <p style={{ color: "#94a3b8", fontSize: 13 }}>
          {auditEntries.length} decision events received · Generated during demo replay
          · Not a persistent or tamper-evident clinical audit log
        </p>
        {auditEntries.length === 0 ? (
          <p style={{ color: "#94a3b8" }}>
            Start the AI demo to view decisions as they arrive.
          </p>
        ) : (
          <div style={{ display: "grid", gap: 12 }}>
            {auditEntries.map((entry) => {
              const color = entry.trust_level === "HIGH"
                ? "#10b981"
                : entry.trust_level === "REVIEW"
                  ? "#f59e0b"
                  : "#ef4444";
              return (
                <div key={entry.event_id} style={{
                  background: "#1e293b",
                  border: `1px solid ${color}`,
                  borderRadius: 10,
                  padding: 14,
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                    <strong>Window {entry.window_index + 1} · {entry.artifact_class.replace(/_/g, " ")}</strong>
                    <span style={{ color, fontWeight: 700 }}>
                      {entry.trust_level} · {entry.action.replace(/_/g, " ")}
                    </span>
                  </div>
                  <p style={{ marginBottom: 8, color: "#cbd5e1" }}>
                    {entry.explanation}
                  </p>
                  <small style={{ color: "#94a3b8" }}>
                    Processed: {new Date(entry.timestamp_utc).toLocaleString()}
                    {" · "}Experimental: {entry.experimental ? "Yes" : "No"}
                    {" · "}Clinically validated: {entry.clinical_validated ? "Yes" : "No"}
                  </small>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Existing AI prediction panel */}

      <ArtifactPredictionPanel prediction={prediction} />

      {errorMessage && (
        <p role="alert" style={{ color: "#f87171" }}>
          {errorMessage}
        </p>
      )}

      <p
        style={{
          fontSize: 12,
          color: "#94a3b8",
          marginTop: 20,
        }}
      >
        Research-only demonstration. ECG is replayed from
        a dataset. Artefacts and IMU are simulated.
        AI probabilities are not calibrated clinical confidence.
        No live patient monitoring or medical decision-making.
      </p>
    </section>
  );
}
