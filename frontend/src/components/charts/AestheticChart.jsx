import { useEffect, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Bar, Line, Scatter, Pie, Doughnut, Radar, PolarArea, Bubble } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Backend chart_type → react-chartjs-2 component (unknown types fall back to Bar).
const CHARTS = {
  bar: Bar,
  histogram: Bar,
  line: Line,
  scatter: Scatter,
  pie: Pie,
  doughnut: Doughnut,
  radar: Radar,
  polararea: PolarArea,
  bubble: Bubble,
};

const PALETTES = {
  dark: [
    'rgba(234, 179, 8, 1)',   // Gold
    'rgba(74, 222, 128, 1)',  // Mint Green
    'rgba(96, 165, 250, 1)',  // Light Blue
    'rgba(248, 113, 113, 1)', // Soft Red
    'rgba(192, 132, 252, 1)', // Purple
    'rgba(251, 146, 60, 1)'   // Orange
  ],
  light: [
    'rgba(245, 158, 11, 1)',  // Vibrant Amber
    'rgba(16, 185, 129, 1)',  // Vibrant Emerald
    'rgba(59, 130, 246, 1)',  // Vibrant Blue
    'rgba(239, 68, 68, 1)',   // Vibrant Red
    'rgba(168, 85, 247, 1)',  // Vibrant Purple
    'rgba(249, 115, 22, 1)'   // Vibrant Orange
  ]
};

const LINE_LIKE = ['line', 'radar'];                                  // one filled series colour
const SOLID_SHAPES = ['pie', 'doughnut', 'bar', 'histogram', 'polararea']; // soft or no borders

export default function AestheticChart({ data, height = "400px" }) {
  const [isDark, setIsDark] = useState(
    document.documentElement.dataset.theme === 'dark' ||
    localStorage.getItem("app-theme") === 'dark'
  );

  // Reactively update theme when user clicks the toggle without reloading
  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.dataset.theme === 'dark');
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);

  if (!data || !data.chart_type) return null;

  const { labels, datasets } = data;
  const type = data.chart_type.toLowerCase();
  const ChartComponent = CHARTS[type] || Bar;
  const isLineLike = LINE_LIKE.includes(type);

  const textColor = isDark ? 'rgba(255, 255, 255, 0.7)' : 'rgba(17, 20, 29, 0.8)';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(17, 20, 29, 0.08)';
  const palette = isDark ? PALETTES.dark : PALETTES.light;
  const fills = palette.map(c => c.replace('1)', isDark ? '0.7)' : '0.85)'));
  const shapeBorder = isDark ? 'transparent' : 'rgba(255,255,255,0.5)';

  const formattedDatasets = datasets.map((ds) => ({
    ...ds,
    backgroundColor: isLineLike ? fills[0] : fills,
    borderColor: isLineLike ? palette[0] : SOLID_SHAPES.includes(type) ? shapeBorder : palette,
    borderWidth: type === 'bar' || type === 'histogram' ? 0 : 2,
    pointBackgroundColor: palette,
    pointBorderColor: isDark ? 'transparent' : 'rgba(255,255,255,0.2)',
    pointRadius: 4,
    pointHoverRadius: 6,
    fill: isLineLike,
    tension: 0.4
  }));

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    color: textColor,
    plugins: {
      legend: {
        position: 'top',
        labels: { color: textColor, usePointStyle: true, boxWidth: 8 }
      },
      tooltip: {
        backgroundColor: isDark ? 'rgba(0,0,0,0.8)' : 'rgba(255,255,255,0.9)',
        titleColor: isDark ? '#fff' : '#000',
        bodyColor: isDark ? '#ccc' : '#333',
        titleFont: { size: 14, family: 'Manrope' },
        bodyFont: { size: 13, family: 'JetBrains Mono' },
        padding: 12,
        cornerRadius: 8,
        displayColors: false,
        borderColor: gridColor,
        borderWidth: 1
      }
    },
    scales: chartScales(type, textColor, gridColor, isDark)
  };

  return (
    <div className="w-full relative" style={{ height }}>
      <ChartComponent data={{ labels: labels || [], datasets: formattedDatasets }} options={options} />
    </div>
  );
}

/** Axis config per chart family: none for pie, a radial axis for radar/polar, linear x for point charts. */
function chartScales(type, textColor, gridColor, isDark) {
  if (type === 'pie' || type === 'doughnut') {
    return { x: { display: false }, y: { display: false } };
  }
  if (type === 'radar' || type === 'polararea') {
    // More visible grid lines specifically for radial charts
    const radialGrid = isDark ? 'rgba(255, 255, 255, 0.25)' : 'rgba(17, 20, 29, 0.25)';
    return {
      x: { display: false },
      y: { display: false },
      r: {
        ticks: { color: textColor, backdropColor: 'transparent', z: 10 },
        grid: { color: radialGrid },
        angleLines: { color: radialGrid },
        pointLabels: { color: textColor, font: { family: 'JetBrains Mono', size: 12, weight: 'bold' } }
      }
    };
  }
  if (type === 'scatter' || type === 'bubble') {
    return {
      x: { type: 'linear', position: 'bottom', ticks: { color: textColor }, grid: { color: gridColor } },
      y: { ticks: { color: textColor }, grid: { color: gridColor } }
    };
  }
  return {
    x: { ticks: { color: textColor }, grid: { color: gridColor, drawBorder: false } },
    y: { ticks: { color: textColor }, grid: { color: gridColor, drawBorder: false } }
  };
}
