import React from 'react';
import type { FishInstance } from '../types/annotations';
import { getSpeciesLabel, getSpeciesColor } from '../lib/speciesColors';

interface StatsPanelProps {
  instances: FishInstance[];
  selectedFish: FishInstance | null;
  selectedVisible: boolean;
  totalUniqueTracks: number;
}

/** Right-sidebar dashboard panel with insights + selected fish detail. */
const StatsPanel: React.FC<StatsPanelProps> = ({
  instances,
  selectedFish,
  selectedVisible,
  totalUniqueTracks,
}) => {
  const totalCount = instances.length;

  const speciesBreakdown = instances.reduce<Record<string, number>>((acc, inst) => {
    acc[inst.speciesId] = (acc[inst.speciesId] || 0) + 1;
    return acc;
  }, {});

  const avgLength =
    totalCount > 0 ? instances.reduce((s, i) => s + i.metrics.length_cm, 0) / totalCount : 0;
  const avgWellness =
    totalCount > 0 ? instances.reduce((s, i) => s + i.metrics.wellness_index, 0) / totalCount : 0;
  const avgConfidence =
    totalCount > 0 ? instances.reduce((s, i) => s + i.metrics.confidence, 0) / totalCount : 0;

  // Health status label
  const healthStatus = avgWellness > 0.75 ? 'Excellent' : avgWellness > 0.5 ? 'Good' : avgWellness > 0.3 ? 'Fair' : totalCount === 0 ? '--' : 'Needs Attention';
  const healthColor = avgWellness > 0.75 ? '#34d399' : avgWellness > 0.5 ? '#fbbf24' : '#f87171';

  return (
    <div className="stats-panel">
      {/* --- Population Health Overview --- */}
      <div className="panel-section health-overview">
        <h3 className="panel-heading">Population Health</h3>
        <div className="health-gauge-row">
          <div className="health-gauge">
            <svg viewBox="0 0 120 70" className="health-arc">
              <path d="M10,65 A50,50 0 0,1 110,65" fill="none" stroke="#1e293b" strokeWidth="10" strokeLinecap="round" />
              <path
                d="M10,65 A50,50 0 0,1 110,65"
                fill="none"
                stroke={healthColor}
                strokeWidth="10"
                strokeLinecap="round"
                strokeDasharray={`${avgWellness * 157} 157`}
              />
            </svg>
            <div className="health-gauge-text">
              <span className="health-gauge-value" style={{ color: healthColor }}>
                {totalCount > 0 ? `${(avgWellness * 100).toFixed(0)}%` : '--'}
              </span>
              <span className="health-gauge-label">{healthStatus}</span>
            </div>
          </div>
        </div>

        {/* Quick stats below the gauge */}
        <div className="quick-stats">
          <div className="quick-stat">
            <span className="qs-value">{totalCount}</span>
            <span className="qs-label">In Frame</span>
          </div>
          <div className="quick-stat">
            <span className="qs-value">{totalUniqueTracks}</span>
            <span className="qs-label">Total Tracked</span>
          </div>
          <div className="quick-stat">
            <span className="qs-value">{avgLength > 0 ? avgLength.toFixed(1) : '--'}</span>
            <span className="qs-label">Avg cm</span>
          </div>
          <div className="quick-stat">
            <span className="qs-value">{avgConfidence > 0 ? `${(avgConfidence * 100).toFixed(0)}%` : '--'}</span>
            <span className="qs-label">Detection</span>
          </div>
        </div>
      </div>

      {/* --- Species Breakdown --- */}
      <div className="panel-section">
        <h3 className="panel-heading">Species Distribution</h3>
        {totalCount > 0 ? (
          <div className="species-bars">
            {Object.entries(speciesBreakdown).map(([speciesId, count]) => (
              <div key={speciesId} className="species-bar-row">
                <div className="species-bar-label">
                  <span className="species-dot" style={{ backgroundColor: getSpeciesColor(speciesId) }} />
                  <span>{getSpeciesLabel(speciesId)}</span>
                </div>
                <div className="species-bar-track">
                  <div
                    className="species-bar-fill"
                    style={{
                      width: `${(count / totalCount) * 100}%`,
                      backgroundColor: getSpeciesColor(speciesId),
                    }}
                  />
                </div>
                <span className="species-bar-count">{count}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="panel-empty">No fish in view</p>
        )}
      </div>

      {/* --- Selected Fish Card --- */}
      <div className="panel-section">
        <h3 className="panel-heading">Selected Fish</h3>
        {selectedFish ? (
          <div className="selected-fish-card">
            {!selectedVisible && (
              <div className="selected-fish-warning">Not visible in current frame</div>
            )}
            <div className="selected-fish-header-row">
              <div
                className="selected-fish-species-badge"
                style={{ backgroundColor: getSpeciesColor(selectedFish.speciesId) }}
              >
                {getSpeciesLabel(selectedFish.speciesId)}
              </div>
              <span className="selected-fish-id">Track #{selectedFish.trackId}</span>
            </div>
            <div className="selected-fish-metrics">
              <MetricRow label="Body Length" value={`${selectedFish.metrics.length_cm.toFixed(1)} cm`} />
              <MetricRow label="Est. Weight" value={`${selectedFish.metrics.weight_g_est.toFixed(0)} g`} />
              <MetricGauge label="Wellness" value={selectedFish.metrics.wellness_index} />
              <MetricGauge label="Satiety" value={selectedFish.metrics.satiety_index} />
              <MetricGauge label="Confidence" value={selectedFish.metrics.confidence} />
            </div>
          </div>
        ) : (
          <p className="panel-empty">Click a fish to inspect</p>
        )}
      </div>
    </div>
  );
};

function MetricRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="metric-row">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}

function MetricGauge({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100);
  const hue = value > 0.7 ? 142 : value > 0.4 ? 45 : 0; // green / yellow / red
  const sat = 70;
  const light = value > 0.7 ? 45 : value > 0.4 ? 50 : 45;
  const color = `hsl(${hue},${sat}%,${light}%)`;
  return (
    <div className="metric-gauge-row">
      <span className="metric-label">{label}</span>
      <div className="metric-gauge-track">
        <div className="metric-gauge-fill" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="metric-gauge-pct" style={{ color }}>{pct}%</span>
    </div>
  );
}

export default StatsPanel;
