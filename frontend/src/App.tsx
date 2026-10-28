
import { useEffect, useRef, useState } from "react";

import ECGChart from "./components/ECGChart";
import ArtifactDemo from "./components/ArtifactDemo";
import CardiacEventReview from "./components/CardiacEventReview";
import EventReviewActions from "./components/EventReviewActions";
import TraceUnifiedAnalysis from "./components/TraceUnifiedAnalysis";

type ConnectionStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  | "completed"
  | "error";

interface ECGFrame {
  type: "ecg_frame";
  t0_sample: number;
  fs: number;
  ecg_raw: number[];
  record_id?: string;
  ecg_units?: string;
  beats?: Array<{
    sample: number;
    symbol: string;
    source: string;
  }>;
}

interface ReplayComplete {
  type: "replay_complete";
  record_id?: string;
  samples_streamed?: number;
}

interface StreamError {
  type: "error";
  message: string;
}

type StreamMessage =
  | ECGFrame
  | ReplayComplete
  | StreamError;

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ??
  "http://127.0.0.1:8000";

const WS_BASE = API_BASE.replace(/^http/, "ws");

const MAX_ECG_SAMPLES = 1500;

export default function App() {
  const socketRef = useRef<WebSocket | null>(null);

  const [status, setStatus] =
    useState<ConnectionStatus>("disconnected");

  const [samples, setSamples] = useState<number[]>([]);
  const [samplingRate, setSamplingRate] = useState(250);
  const [recordId, setRecordId] = useState("MITBIH-100");
  const [beatSymbol, setBeatSymbol] = useState("—");
  const [samplesReceived, setSamplesReceived] = useState(0);
  const [error, setError] = useState("");

  function stopReplay() {
    const socket = socketRef.current;
    socketRef.current = null;

    if (socket) {
      socket.close();
    }

    setStatus("disconnected");
  }

  function startReplay() {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }

    setSamples([]);
    setSamplesReceived(0);
    setBeatSymbol("—");
    setError("");
    setStatus("connecting");

    const socket = new WebSocket(`${WS_BASE}/ws/ecg`);
    socketRef.current = socket;

    socket.onopen = () => {
      if (socketRef.current === socket) {
        setStatus("connected");
      }
    };

    socket.onmessage = (event: MessageEvent<string>) => {
      if (socketRef.current !== socket) {
        return;
      }

      try {
        const message = JSON.parse(
          event.data
        ) as StreamMessage;

        if (message.type === "ecg_frame") {
          const incoming = message.ecg_raw ?? [];

          setSamples((previous) =>
            [...previous, ...incoming].slice(
              -MAX_ECG_SAMPLES
            )
          );

          setSamplesReceived(
            (previous) => previous + incoming.length
          );

          setSamplingRate(message.fs);

          if (message.record_id) {
            setRecordId(message.record_id);
          }

          const beats = message.beats ?? [];

          if (beats.length > 0) {
            setBeatSymbol(
              beats[beats.length - 1].symbol
            );
          }
        }

        if (message.type === "replay_complete") {
          setStatus("completed");
          socket.close();
        }

        if (message.type === "error") {
          setError(message.message);
          setStatus("error");
          socket.close();
        }
      } catch {
        setError("Unable to process ECG stream data.");
        setStatus("error");
        socket.close();
      }
    };

    socket.onerror = () => {
      if (socketRef.current === socket) {
        setError("ECG WebSocket connection failed.");
        setStatus("error");
      }
    };

    socket.onclose = () => {
      if (socketRef.current === socket) {
        socketRef.current = null;

        setStatus((previous) =>
          previous === "connected" ||
          previous === "connecting"
            ? "disconnected"
            : previous
        );
      }
    };
  }

  useEffect(() => {
    return () => {
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, []);

  const containerStyle: React.CSSProperties = {
    minHeight: "100vh",
    background: "#020617",
    color: "#f8fafc",
    padding: "32px 20px",
    fontFamily:
      "Inter, system-ui, -apple-system, sans-serif",
  };

  const contentStyle: React.CSSProperties = {
    maxWidth: 1200,
    margin: "0 auto",
  };

  const panelStyle: React.CSSProperties = {
    background: "#111827",
    border: "1px solid #334155",
    borderRadius: 16,
    padding: 24,
    marginTop: 24,
  };

  const buttonStyle: React.CSSProperties = {
    background: "#0f766e",
    color: "#ffffff",
    border: "none",
    borderRadius: 8,
    padding: "11px 18px",
    cursor: "pointer",
    fontWeight: 600,
  };

  const isStreaming =
    status === "connecting" ||
    status === "connected";

  return (
    <main style={containerStyle}>
      <div style={contentStyle}>
        <header style={{ marginBottom: 28 }}>
          <h1
            style={{
              color: "#5eead4",
              fontSize: "clamp(32px, 5vw, 52px)",
              margin: 0,
              fontWeight: 800,
              letterSpacing: 1,
            }}
          >
            RHYTHM
          </h1>
        </header>

        <section style={panelStyle}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              flexWrap: "wrap",
              gap: 12,
            }}
          >
            <div>
              <h2 style={{ margin: 0 }}>
                Recorded ECG Replay
              </h2>

              <p
                style={{
                  color: "#94a3b8",
                  marginBottom: 0,
                }}
              >
                MIT-BIH research ECG · 250 Hz ·
                not live patient monitoring
              </p>
            </div>

            <span
              style={{
                color:
                  status === "connected"
                    ? "#5eead4"
                    : "#cbd5e1",
                fontWeight: 600,
              }}
            >
              {status.toUpperCase()}
            </span>
          </div>

          <div
            style={{
              display: "flex",
              gap: 12,
              flexWrap: "wrap",
              marginTop: 20,
              marginBottom: 20,
            }}
          >
            <button
              type="button"
              onClick={startReplay}
              disabled={isStreaming}
              style={{
                ...buttonStyle,
                opacity: isStreaming ? 0.5 : 1,
              }}
            >
              Start ECG Replay
            </button>

            <button
              type="button"
              onClick={stopReplay}
              disabled={!isStreaming}
              style={{
                ...buttonStyle,
                background: "#334155",
                opacity: !isStreaming ? 0.5 : 1,
              }}
            >
              Stop Replay
            </button>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "repeat(auto-fit, minmax(170px, 1fr))",
              gap: 12,
              marginBottom: 20,
            }}
          >
            {[
              ["Recording", recordId],
              ["Sampling Rate", `${samplingRate} Hz`],
              [
                "Samples Received",
                samplesReceived.toLocaleString(),
              ],
              ["Latest Reference Beat", beatSymbol],
            ].map(([label, value]) => (
              <div
                key={label}
                style={{
                  background: "#020617",
                  border: "1px solid #334155",
                  borderRadius: 10,
                  padding: 14,
                }}
              >
                <div
                  style={{
                    color: "#94a3b8",
                    fontSize: 12,
                    marginBottom: 8,
                  }}
                >
                  {label}
                </div>

                <strong>{value}</strong>
              </div>
            ))}
          </div>

          <ECGChart
            samples={samples}
            fs={samplingRate}
          />

          {error && (
            <p
              role="alert"
              style={{
                color: "#fca5a5",
                marginTop: 16,
              }}
            >
              {error}
            </p>
          )}

          <p
            style={{
              color: "#94a3b8",
              fontSize: 12,
              marginBottom: 0,
            }}
          >
            Reference beat symbols originate from
            dataset annotations and are not model
            predictions. Motion sensor data is not
            available for this recording.
          </p>
        </section>
        <ArtifactDemo />

<TraceUnifiedAnalysis />

<CardiacEventReview />

<EventReviewActions />

        <footer
          style={{
            color: "#94a3b8",
            fontSize: 12,
            textAlign: "center",
            marginTop: 36,
            paddingBottom: 20,
            lineHeight: 1.7,
          }}
        >
          Research prototype only. Not clinically
          validated. Not for diagnosis, treatment,
          or patient monitoring.
        </footer>
      </div>
    </main>
  );
}
