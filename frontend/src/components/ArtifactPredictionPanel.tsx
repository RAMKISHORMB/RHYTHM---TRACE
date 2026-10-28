
import type { CSSProperties } from "react";
import type { BeatMarker } from "./DualECGChart";

// ============================================================
// TRACE — AI ARTEFACT PREDICTION PANEL
// Window-Level Trust + Beat-Level Metadata
// Research Prototype — Not for Clinical Use
// ============================================================

export type TrustLevel =
  | "HIGH"
  | "REVIEW"
  | "UNRELIABLE";

export type TrustResult = {
  level: TrustLevel;
  score: number;
  predicted_class: string;
  model_probability: number;
  flatline_fraction: number;
  reasons: string[];
  policy_version: string;
  experimental: boolean;
  clinical_validated: boolean;
};

export type ArtifactPrediction = {
  type: "artifact_prediction";

  window_index: number;

  true_artifact: string;
  predicted_artifact: string;

  model_probability: number;
  class_probabilities: Record<string, number>;

  trust?: TrustResult;

  // Stage 5.10 — Beat-level trust metadata
  beats?: BeatMarker[];
  beat_count?: number;
  window_start_sample?: number;

  experimental: boolean;
  imu_simulated: boolean;
  ecg_replayed: boolean;
  artifact_injected: boolean;
  clinical_validated: boolean;
};

type ArtifactPredictionPanelProps = {
  prediction: ArtifactPrediction | null;
};

const CLASS_LABELS: Record<string, string> = {
  clean: "CLEAN",
  baseline_wander: "BASELINE WANDER",
  walking: "WALKING",
  electrode_pop: "ELECTRODE POP",
  lead_dropout: "LEAD DROPOUT",
};

const TRUST_COLORS: Record<TrustLevel, string> = {
  HIGH: "#10b981",
  REVIEW: "#f59e0b",
  UNRELIABLE: "#ef4444",
};

const TRUST_DESCRIPTIONS: Record<TrustLevel, string> = {
  HIGH:
    "The current signal window meets the experimental " +
    "HIGH trust policy criteria. This does not establish " +
    "clinical ECG reliability.",

  REVIEW:
    "The current signal window may contain interference " +
    "or uncertain classification and requires review.",

  UNRELIABLE:
    "The current signal window does not meet the " +
    "experimental quality criteria for reliable analysis.",
};

const panelStyle: CSSProperties = {
  background: "#0f172a",
  border: "1px solid #334155",
  borderRadius: 14,
  padding: 20,
  marginTop: 20,
};

const statStyle: CSSProperties = {
  background: "#1e293b",
  border: "1px solid #334155",
  borderRadius: 12,
  padding: 16,
};

function formatClassName(value: string): string {
  return (
    CLASS_LABELS[value] ??
    value.replace(/_/g, " ").toUpperCase()
  );
}

function formatPercentage(value: number): string {
  if (!Number.isFinite(value)) {
    return "N/A";
  }

  return `${(value * 100).toFixed(2)}%`;
}

export default function ArtifactPredictionPanel({
  prediction,
}: ArtifactPredictionPanelProps) {
  if (!prediction) {
    return (
      <section style={panelStyle}>
        <h2>AI Artefact Classification</h2>

        <p style={{ color: "#94a3b8" }}>
          Waiting for an experimental artefact prediction.
        </p>
      </section>
    );
  }

  const {
    window_index,
    true_artifact,
    predicted_artifact,
    model_probability,
    class_probabilities,
    trust,
  } = prediction;

  const isMatch = true_artifact === predicted_artifact;

  const trustColor = trust
    ? TRUST_COLORS[trust.level]
    : "#94a3b8";

  const classEntries = Object.entries(
    class_probabilities
  ).sort(([classA], [classB]) => {
    const order = Object.keys(CLASS_LABELS);

    return (
      order.indexOf(classA) -
      order.indexOf(classB)
    );
  });

  return (
    <section style={panelStyle}>
      <h2>AI Artefact Classification</h2>

      <p style={{ color: "#94a3b8" }}>
        Window {window_index + 1}
      </p>

      <h2
        style={{
          color: "#38bdf8",
          marginBottom: 8,
        }}
      >
        {formatClassName(predicted_artifact)}
      </h2>

      <p>
        Model probability:{" "}
        <strong>
          {formatPercentage(model_probability)}
        </strong>
      </p>

      <p>
        Injected ground truth:{" "}
        <strong>
          {formatClassName(true_artifact)}
        </strong>
      </p>

      <p>
        Prediction match:{" "}
        <strong
          style={{
            color: isMatch
              ? "#10b981"
              : "#ef4444",
          }}
        >
          {isMatch ? "MATCH" : "MISMATCH"}
        </strong>
      </p>

      {/* Class probabilities */}

      <div style={{ marginTop: 24 }}>
        <h3>Class Probabilities</h3>

        {classEntries.map(([className, probability]) => {
          const safeProbability = Number.isFinite(
            probability
          )
            ? Math.min(1, Math.max(0, probability))
            : 0;

          return (
            <div
              key={className}
              style={{ marginBottom: 16 }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  gap: 12,
                  marginBottom: 6,
                  fontSize: 13,
                }}
              >
                <span>
                  {formatClassName(className)}
                </span>

                <strong>
                  {formatPercentage(probability)}
                </strong>
              </div>

              <div
                style={{
                  height: 9,
                  background: "#334155",
                  borderRadius: 6,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${safeProbability * 100}%`,
                    height: "100%",
                    background:
                      className === predicted_artifact
                        ? "#38bdf8"
                        : "#64748b",
                    borderRadius: 6,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Window-level trust */}

      <div style={{ marginTop: 28 }}>
        <h3>TRACE Signal Trust</h3>

        {trust ? (
          <>
            <div
              style={{
                ...statStyle,
                borderColor: trustColor,
              }}
            >
              <small style={{ color: "#94a3b8" }}>
                Signal trust level
              </small>

              <h2
                style={{
                  color: trustColor,
                  marginTop: 8,
                  marginBottom: 8,
                }}
              >
                {trust.level}
              </h2>

              <p
                style={{
                  color: "#cbd5e1",
                  marginBottom: 0,
                }}
              >
                {TRUST_DESCRIPTIONS[trust.level]}
              </p>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "repeat(auto-fit, minmax(150px, 1fr))",
                gap: 12,
                marginTop: 14,
              }}
            >
              <div style={statStyle}>
                <small style={{ color: "#94a3b8" }}>
                  Policy score
                </small>

                <h3>
                  {trust.score.toFixed(1)} / 1.0
                </h3>
              </div>

              <div style={statStyle}>
                <small style={{ color: "#94a3b8" }}>
                  Flatline fraction
                </small>

                <h3>
                  {formatPercentage(
                    trust.flatline_fraction
                  )}
                </h3>
              </div>

              <div style={statStyle}>
                <small style={{ color: "#94a3b8" }}>
                  Policy version
                </small>

                <p
                  style={{
                    overflowWrap: "anywhere",
                    marginBottom: 0,
                  }}
                >
                  {trust.policy_version}
                </p>
              </div>
            </div>

            <div style={{ marginTop: 22 }}>
              <h4>Decision Explanation</h4>

              {trust.reasons.length > 0 ? (
                <ul
                  style={{
                    color: "#cbd5e1",
                    paddingLeft: 22,
                    lineHeight: 1.7,
                  }}
                >
                  {trust.reasons.map(
                    (reason, index) => (
                      <li key={index}>
                        {reason}
                      </li>
                    )
                  )}
                </ul>
              ) : (
                <p style={{ color: "#94a3b8" }}>
                  No additional explanation available.
                </p>
              )}
            </div>
          </>
        ) : (
          <p style={{ color: "#94a3b8" }}>
            Window-level trust information unavailable.
          </p>
        )}
      </div>

      {/* Beat-level metadata summary */}

      {prediction.beats && (
        <div
          style={{
            ...statStyle,
            marginTop: 22,
          }}
        >
          <h4 style={{ marginTop: 0 }}>
            Beat-Level Trust Integration
          </h4>

          <p style={{ marginBottom: 0 }}>
            <strong>
              {prediction.beats.length}
            </strong>{" "}
            reference beats aligned to this window.
            Individual trust markers are displayed
            on the ECG chart.
          </p>
        </div>
      )}

      {/* Research provenance */}

      <p
        style={{
          color: "#94a3b8",
          fontSize: 12,
          lineHeight: 1.7,
          marginTop: 24,
        }}
      >
        Recorded ECG replay · Injected synthetic artefact ·
        Simulated IMU
      </p>

      <p
        style={{
          color: "#94a3b8",
          fontSize: 12,
          lineHeight: 1.7,
        }}
      >
        The trust score is a categorical engineering value,
        not a calibrated probability of ECG reliability.
        HIGH does not imply clinical safety.
      </p>

      <p
        style={{
          color: "#94a3b8",
          fontSize: 12,
          lineHeight: 1.7,
        }}
      >
        Experimental research output only. Model
        probabilities are not calibrated clinical
        confidence. Dataset beat annotations are reference
        labels, not AI-detected arrhythmias.
        Not for patient care.
      </p>
    </section>
  );
}
