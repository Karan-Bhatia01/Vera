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
ChartJS.defaults.font.family = "'Atkinson Hyperlegible Next', system-ui, sans-serif";

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

// Vera's mascot colours (violet first: charts live on the EDA stage), then navy and coral.
// Light-theme shades are deeper so bars keep contrast against the paper background.
const PALETTES = {
  dark: [
    'rgba(176, 114, 230, 1)', // violet
    'rgba(47, 143, 230, 1)',  // sky
    'rgba(245, 196, 0, 1)',   // sun
    'rgba(155, 209, 46, 1)',  // lime
    'rgba(243, 240, 232, 1)', // chalk
    'rgba(255, 138, 128, 1)'  // coral
  ],
  light: [
    'rgba(140, 62, 200, 1)',  // violet
    'rgba(31, 117, 204, 1)',  // sky
    'rgba(214, 168, 0, 1)',   // sun
    'rgba(110, 160, 24, 1)',  // lime
    'rgba(30, 36, 82, 1)',    // felt navy
    'rgba(226, 92, 80, 1)'    // coral
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

  const textColor = isDark ? 'rgba(243, 240, 232, 0.8)' : 'rgba(30, 36, 82, 0.85)';
  const gridColor = isDark ? 'rgba(243, 240, 232, 0.08)' : 'rgba(30, 36, 82, 0.08)';
  const palette = isDark ? PALETTES.dark : PALETTES.light;
  const fills = palette.map(c => c.replace('1)', isDark ? '0.7)' : '0.85)'));
  const shapeBorder = isDark ? 'transparent' : 'rgba(255,255,255,0.5)';

  const formattedDatasets = datasets.map((ds) => ({
    ...ds,
    backgroundColor: isLineLike ? fills[0] : fills,
    borderColor: isLineLike ? palette[0] : SOLID_SHAPES.includes(type) ? shapeBorder : palette,
    borderWidth: type === 'bar' || type === 'histogram' ? 0 : 2,
    pointBackgroundColor: palette,
    pointBorderColor: isDark ? 'transparent' : 'rgba(255,255,255,0.6)',
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
        backgroundColor: isDark ? 'rgba(29, 27, 25, 0.96)' : 'rgba(255, 255, 255, 0.97)',
        titleColor: isDark ? '#f3f0e8' : '#1e2452',
        bodyColor: isDark ? '#b5aea2' : '#565d86',
        titleFont: { size: 14, family: 'Atkinson Hyperlegible Next', weight: 'bold' },
        bodyFont: { size: 13, family: 'Atkinson Hyperlegible Mono' },
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
    const radialGrid = isDark ? 'rgba(243, 240, 232, 0.25)' : 'rgba(30, 36, 82, 0.25)';
    return {
      x: { display: false },
      y: { display: false },
      r: {
        ticks: { color: textColor, backdropColor: 'transparent', z: 10 },
        grid: { color: radialGrid },
        angleLines: { color: radialGrid },
        pointLabels: { color: textColor, font: { family: 'Atkinson Hyperlegible Next', size: 12, weight: 'bold' } }
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
    // Skip crowded labels and keep the angle shallow so narrow screens stay readable.
    x: { ticks: { color: textColor, autoSkip: true, autoSkipPadding: 8, maxRotation: 45 }, grid: { color: gridColor, drawBorder: false } },
    y: { ticks: { color: textColor }, grid: { color: gridColor, drawBorder: false } }
  };
}
