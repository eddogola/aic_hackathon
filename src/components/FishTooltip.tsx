import React from 'react';
import type { FishInstance } from '../types/annotations';
import { getSpeciesLabel, getSpeciesColor } from '../lib/speciesColors';

interface FishTooltipProps {
  fish: FishInstance;
  /** Position in canvas/page pixel coords */
  x: number;
  y: number;
  /** Bounding rect of the overlay container for clamp calculations */
  containerRect: DOMRect | null;
}

/**
 * Small tooltip anchored near a fish's centroid.
 * Clamps to stay within the container bounds.
 */
const FishTooltip: React.FC<FishTooltipProps> = ({ fish, x, y, containerRect }) => {
  const tooltipWidth = 200;
  const tooltipHeight = 130;
  const offset = 12;

  // Default position: offset to the right and below the centroid
  let left = x + offset;
  let top = y + offset;

  // Clamp so tooltip doesn't overflow the container
  if (containerRect) {
    if (left + tooltipWidth > containerRect.width) {
      left = x - tooltipWidth - offset;
    }
    if (top + tooltipHeight > containerRect.height) {
      top = y - tooltipHeight - offset;
    }
    if (left < 0) left = offset;
    if (top < 0) top = offset;
  }

  const color = getSpeciesColor(fish.speciesId);

  return (
    <div
      className="fish-tooltip"
      style={{
        position: 'absolute',
        left: `${left}px`,
        top: `${top}px`,
        width: `${tooltipWidth}px`,
        pointerEvents: 'none',
        zIndex: 20,
      }}
    >
      <div className="fish-tooltip-header" style={{ borderLeftColor: color }}>
        <span className="fish-tooltip-species">{getSpeciesLabel(fish.speciesId)}</span>
        <span className="fish-tooltip-id">ID: {fish.trackId}</span>
      </div>
      <div className="fish-tooltip-body">
        <div className="fish-tooltip-row">
          <span>Length</span>
          <span>{fish.metrics.length_cm.toFixed(1)} cm</span>
        </div>
        <div className="fish-tooltip-row">
          <span>Weight</span>
          <span>{fish.metrics.weight_g_est.toFixed(0)} g</span>
        </div>
        <div className="fish-tooltip-row">
          <span>Confidence</span>
          <span>{(fish.metrics.confidence * 100).toFixed(0)}%</span>
        </div>
      </div>
    </div>
  );
};

export default FishTooltip;
