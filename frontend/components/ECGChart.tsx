
import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

interface ECGChartProps {
  samples: number[];
  fs: number;
}

const WINDOW_SECONDS = 10;

export default function ECGChart({
  samples,
  fs,
}: ECGChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<uPlot | null>(null);

  // Create chart once.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const options: uPlot.Options = {
      width: Math.max(container.clientWidth, 320),
      height: 340,

      scales: {
        x: { time: false },
        y: {
          auto: false,
          range: [-0.2, 1.3],
        },
      },

      series: [
        {},
        {
          label: "ECG",
          stroke: "#10b981",
          width: 2,
          points: { show: false },
        },
      ],

      axes: [
        {
          label: "Time (seconds)",
          stroke: "#94a3b8",
          grid: {
            stroke: "rgba(148, 163, 184, 0.15)",
          },
        },
        {
          label: "Amplitude (a.u.)",
          stroke: "#94a3b8",
          grid: {
            stroke: "rgba(148, 163, 184, 0.15)",
          },
        },
      ],

      legend: {
        show: false,
      },
    };

    const chart = new uPlot(
      options,
      [[], []],
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

  // Update the chart whenever samples arrive.
  useEffect(() => {
    const chart = chartRef.current;

    if (!chart || samples.length === 0 || fs <= 0) {
      return;
    }

    const maxSamples = fs * WINDOW_SECONDS;
    const visible = samples.slice(-maxSamples);

    const time = visible.map(
      (_, index) => index / fs
    );

    chart.setData([time, visible]);
  }, [samples, fs]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        minHeight: "340px",
      }}
    />
  );
}
