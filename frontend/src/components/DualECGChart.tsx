
import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

export type BeatMarker = {
  absolute_sample: number;
  relative_sample: number;
  symbol: string;
  trust_level: "HIGH" | "REVIEW" | "UNRELIABLE";
  trust_score: number;
  annotation_source: string;
};

type DualECGChartProps = {
  referenceSamples: number[];
  corruptedSamples: number[];
  fs: number;
  beats?: BeatMarker[];
};

const WINDOW_SECONDS = 4;

export default function DualECGChart({
  referenceSamples,
  corruptedSamples,
  fs,
  beats = [],
}: DualECGChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<uPlot | null>(null);

  useEffect(() => {
    const container = containerRef.current;

    if (!container) return;

    const options: uPlot.Options = {
      width: Math.max(container.clientWidth, 320),
      height: 340,

      scales: {
        x: { time: false },
        y: { auto: true },
      },

      series: [
        {},
        {
          label: "Original ECG",
          stroke: "#10b981",
          width: 2,
          points: { show: false },
        },
        {
          label: "Corrupted ECG",
          stroke: "#f59e0b",
          width: 2,
          points: { show: false },
        },
      ],

      axes: [
        {
          label: "Time (seconds)",
          stroke: "#94a3b8",
          grid: {
            stroke: "rgba(148,163,184,0.15)",
          },
        },
        {
          label: "Amplitude (digital units)",
          stroke: "#94a3b8",
          grid: {
            stroke: "rgba(148,163,184,0.15)",
          },
        },
      ],

      legend: { show: true },
    };

    const chart = new uPlot(
      options,
      [[], [], []],
      container
    );

    chartRef.current = chart;

    const resizeObserver = new ResizeObserver(() => {
      chart.setSize({
        width: Math.max(container.clientWidth, 320),
        height: 340,
      });
    });

    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      chartRef.current = null;
      chart.destroy();
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;

    if (!chart || fs <= 0) return;

    const count = Math.min(
      referenceSamples.length,
      corruptedSamples.length,
      fs * WINDOW_SECONDS
    );

    if (count === 0) {
      chart.setData([[], [], []]);
      return;
    }

    const reference = referenceSamples.slice(-count);
    const corrupted = corruptedSamples.slice(-count);

    const time = Array.from(
      { length: count },
      (_, index) => index / fs
    );

    chart.setData([
      time,
      reference,
      corrupted,
    ]);
  }, [referenceSamples, corruptedSamples, fs]);

  return (
    <div>
      <div
        ref={containerRef}
        style={{
          width: "100%",
          minHeight: 340,
        }}
      />

      <p
        style={{
          color: "#94a3b8",
          fontSize: 12,
          marginTop: 12,
        }}
      >
        MIT-BIH reference beats received: {beats.length}
        {" "}· Beat markers temporarily disabled for
        graph verification.
      </p>
    </div>
  );
}
