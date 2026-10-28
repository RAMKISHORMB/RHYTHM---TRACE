
import { useCallback, useEffect, useState } from "react";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ??
  "http://127.0.0.1:8000";

type ReviewAction = "REVIEWED" | "NEEDS_FOLLOWUP";

interface AuditEntry {
  id: number;
  event_id: string;
  action: string;
  reviewer: string;
  note: string;
  timestamp_utc: string;
}

interface AuditResponse {
  event_id: string;
  review_count: number;
  current_review_status: string;
  history: AuditEntry[];
}

interface EventSummary {
  event_id: string;
  event_type: string;
  timestamp_seconds: number;
}

interface EventsResponse {
  events: EventSummary[];
}

export default function EventReviewActions() {
  const [events, setEvents] = useState<EventSummary[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [reviewer, setReviewer] = useState("Demo Reviewer");
  const [note, setNote] = useState("");
  const [action, setAction] =
    useState<ReviewAction>("REVIEWED");
  const [audit, setAudit] = useState<AuditResponse | null>(
    null
  );
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    async function loadEvents() {
      try {
        const response = await fetch(
          `${API_BASE}/api/events`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error("Could not load cardiac events.");
        }

        const data: EventsResponse = await response.json();

        setEvents(data.events);
        setSelectedId(data.events[0]?.event_id ?? "");
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(
            err instanceof Error
              ? err.message
              : "Failed to load events."
          );
        }
      }
    }

    void loadEvents();

    return () => controller.abort();
  }, []);

  const loadAudit = useCallback(
    async (eventId: string, signal?: AbortSignal) => {
      const response = await fetch(
        `${API_BASE}/api/event-audit/${encodeURIComponent(
          eventId
        )}`,
        { signal }
      );

      if (!response.ok) {
        throw new Error("Could not load audit history.");
      }

      const data: AuditResponse = await response.json();
      setAudit(data);
    },
    []
  );

  useEffect(() => {
    if (!selectedId) {
      setAudit(null);
      return;
    }

    const controller = new AbortController();
    setAudit(null);
    setError("");

    void loadAudit(selectedId, controller.signal).catch(
      (err) => {
        if (!controller.signal.aborted) {
          setError(
            err instanceof Error
              ? err.message
              : "Audit history unavailable."
          );
        }
      }
    );

    return () => controller.abort();
  }, [selectedId, loadAudit]);

  function downloadEvidence(format: "json" | "csv") {
    if (!selectedId) return;

    const url =
      `${API_BASE}/api/events/` +
      `${encodeURIComponent(selectedId)}` +
      `/export/${format}`;

    // Opens the export endpoint, which supplies
    // Content-Disposition: attachment.
    window.open(url, "_blank", "noopener,noreferrer");
  }

  async function submitReview() {
    if (!selectedId || busy) return;

    if (!reviewer.trim()) {
      setError("Enter a reviewer label.");
      return;
    }

    setBusy(true);
    setError("");
    setMessage("");

    try {
      const response = await fetch(
        `${API_BASE}/api/event-audit/review`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            event_id: selectedId,
            action,
            reviewer: reviewer.trim(),
            note: note.trim(),
          }),
        }
      );

      if (!response.ok) {
        const details = await response.text();
        throw new Error(
          `Review failed (${response.status}): ${details}`
        );
      }

      await loadAudit(selectedId);

      setMessage("Review action saved to the audit log.");
      setNote("");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not save review."
      );
    } finally {
      setBusy(false);
    }
  }

  const panelStyle: React.CSSProperties = {
    background: "#111827",
    border: "1px solid #334155",
    borderRadius: 16,
    padding: 24,
    color: "#f8fafc",
    marginTop: 24,
  };

  const fieldStyle: React.CSSProperties = {
    background: "#020617",
    color: "#f8fafc",
    border: "1px solid #475569",
    borderRadius: 8,
    padding: "10px 12px",
    width: "100%",
    boxSizing: "border-box",
  };

  const buttonStyle: React.CSSProperties = {
    background: "#0f766e",
    color: "#fff",
    border: 0,
    borderRadius: 8,
    padding: "11px 16px",
    cursor: "pointer",
  };

  return (
    <section style={panelStyle}>
      <h2 style={{ marginTop: 0 }}>
        Event Evidence &amp; Review Audit
      </h2>

      <p style={{ color: "#94a3b8" }}>
        Export recorded ECG evidence and log simulated
        reviewer decisions. Research demonstration only.
      </p>

      <label
        htmlFor="trace-event-select"
        style={{ display: "block", marginBottom: 8 }}
      >
        Select cardiac-event candidate
      </label>

      <select
        id="trace-event-select"
        value={selectedId}
        onChange={(e) => {
          setSelectedId(e.target.value);
          setMessage("");
        }}
        style={fieldStyle}
      >
        {events.map((event) => (
          <option key={event.event_id} value={event.event_id}>
            {event.event_type} —{" "}
            {event.timestamp_seconds.toFixed(3)} s
          </option>
        ))}
      </select>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 12,
          marginTop: 16,
        }}
      >
        <button
          type="button"
          onClick={() => downloadEvidence("json")}
          disabled={!selectedId}
          style={buttonStyle}
        >
          Download JSON
        </button>

        <button
          type="button"
          onClick={() => downloadEvidence("csv")}
          disabled={!selectedId}
          style={buttonStyle}
        >
          Download ECG CSV
        </button>
      </div>

      <hr
        style={{
          border: 0,
          borderTop: "1px solid #334155",
          margin: "24px 0",
        }}
      />

      <h3>Record Review Action</h3>

      <div
        style={{
          display: "grid",
          gap: 14,
          maxWidth: 600,
        }}
      >
        <label>
          Reviewer label
          <input
            value={reviewer}
            onChange={(e) => setReviewer(e.target.value)}
            maxLength={80}
            style={{ ...fieldStyle, marginTop: 6 }}
          />
        </label>

        <label>
          Review action
          <select
            value={action}
            onChange={(e) =>
              setAction(e.target.value as ReviewAction)
            }
            style={{ ...fieldStyle, marginTop: 6 }}
          >
            <option value="REVIEWED">REVIEWED</option>
            <option value="NEEDS_FOLLOWUP">
              NEEDS_FOLLOWUP
            </option>
          </select>
        </label>

        <label>
          Review note
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            maxLength={1000}
            rows={3}
            style={{
              ...fieldStyle,
              marginTop: 6,
              resize: "vertical",
            }}
            placeholder="Research demonstration note..."
          />
        </label>

        <button
          type="button"
          onClick={() => void submitReview()}
          disabled={!selectedId || busy}
          style={{
            ...buttonStyle,
            opacity: busy ? 0.6 : 1,
          }}
        >
          {busy ? "Saving..." : "Save Review Action"}
        </button>
      </div>

      {message && (
        <p role="status" style={{ color: "#5eead4" }}>
          {message}
        </p>
      )}

      {error && (
        <p role="alert" style={{ color: "#fca5a5" }}>
          {error}
        </p>
      )}

      <h3 style={{ marginTop: 28 }}>Audit History</h3>

      <p style={{ color: "#cbd5e1" }}>
        Current status:{" "}
        <strong>
          {audit?.current_review_status ?? "Loading..."}
        </strong>
        {" · "}
        Reviews: {audit?.review_count ?? 0}
      </p>

      {audit?.history.length === 0 && (
        <p style={{ color: "#94a3b8" }}>
          No review actions recorded for this event.
        </p>
      )}

      <div style={{ display: "grid", gap: 10 }}>
        {audit?.history.map((entry) => (
          <article
            key={entry.id}
            style={{
              border: "1px solid #334155",
              borderRadius: 10,
              padding: 14,
              background: "#020617",
            }}
          >
            <strong>{entry.action}</strong>
            <p style={{ margin: "8px 0" }}>
              Reviewer: {entry.reviewer}
            </p>
            <p style={{ margin: "8px 0" }}>
              {entry.note || "No review note."}
            </p>
            <small style={{ color: "#94a3b8" }}>
              {new Date(entry.timestamp_utc).toLocaleString()}
              {" · "}Audit #{entry.id}
            </small>
          </article>
        ))}
      </div>

      <p
        style={{
          marginTop: 24,
          fontSize: 12,
          color: "#94a3b8",
        }}
      >
        Experimental research system. Reviewer labels are
        not authenticated. Review actions do not change
        ECG evidence or establish a medical diagnosis.
      </p>
    </section>
  );
}
