import React, { useState, useRef, useCallback, useEffect } from 'react';
import type { FishInstance, AnnotationFrame, AnnotationChunk } from '../types/annotations';
import { getChunkIndex, loadChunk } from '../lib/chunkLoader';
import { SPECIES_COLOR_MAP } from '../lib/speciesColors';
import VideoWithOverlay from './VideoWithOverlay';
import StatsPanel from './StatsPanel';
import FishTable from './FishTable';
import TimelineScrubber from './TimelineScrubber';

const VIDEO_SRC = '/demo/video.mp4';
/** Max age (seconds) to carry forward annotations when current frame is empty */
const PERSIST_WINDOW = 1.0;

/**
 * Main page layout. Manages all shared state:
 *  - current time, duration, playback controls
 *  - chunk loading as time advances
 *  - hovered / selected fish
 *  - species filters, confidence threshold
 */
const SessionViewer: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);

  // --- Playback state ---
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);

  // --- Video native dimensions ---
  const [, setVideoDims] = useState<{ w: number; h: number }>({ w: 720, h: 1280 });

  // --- Annotation state ---
  const [currentChunk, setCurrentChunk] = useState<AnnotationChunk | null>(null);
  const [currentFrame, setCurrentFrame] = useState<AnnotationFrame | null>(null);
  const lastChunkIndexRef = useRef<number>(-1);
  /** Keep the last frame that had actual detections so overlays persist */
  const lastNonEmptyFrameRef = useRef<AnnotationFrame | null>(null);

  // --- Interaction state ---
  const [hoveredTrackId, setHoveredTrackId] = useState<number | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);

  // --- Filters ---
  const [confidenceThreshold, setConfidenceThreshold] = useState(0);
  const allSpecies = Object.keys(SPECIES_COLOR_MAP);
  const [speciesFilter, setSpeciesFilter] = useState<Set<string>>(new Set(allSpecies));

  // --- Cumulative tracking data for the session ---
  const allTracksRef = useRef<Map<number, FishInstance>>(new Map());
  const [totalUniqueTracks, setTotalUniqueTracks] = useState(0);

  // --- Load chunk when time changes ---
  const loadChunkForTime = useCallback(async (t: number) => {
    const idx = getChunkIndex(t);
    if (idx !== lastChunkIndexRef.current) {
      lastChunkIndexRef.current = idx;
      const chunk = await loadChunk(idx);
      setCurrentChunk(chunk);
      // Index all tracks from this chunk for cumulative stats
      if (chunk) {
        for (const frame of chunk.frames) {
          for (const inst of frame.instances) {
            allTracksRef.current.set(inst.trackId, inst);
          }
        }
        setTotalUniqueTracks(allTracksRef.current.size);
      }
      // Also preload next chunk
      loadChunk(idx + 1);
    }
  }, []);

  // --- Find the closest frame in the current chunk for time t ---
  const findFrame = useCallback(
    (t: number): AnnotationFrame | null => {
      if (!currentChunk || currentChunk.frames.length === 0) return null;
      let best: AnnotationFrame | null = null;
      let bestDelta = Infinity;
      for (const frame of currentChunk.frames) {
        const delta = Math.abs(frame.t - t);
        if (delta < bestDelta) {
          bestDelta = delta;
          best = frame;
        }
      }
      return best;
    },
    [currentChunk]
  );

  // --- Handle time updates from the video element ---
  const handleTimeUpdate = useCallback(
    (t: number) => {
      setCurrentTime(t);
      loadChunkForTime(t);
      const frame = findFrame(t);

      // Carry forward: if frame is empty, persist last non-empty frame
      if (frame && frame.instances.length > 0) {
        lastNonEmptyFrameRef.current = frame;
        setCurrentFrame(frame);
      } else if (lastNonEmptyFrameRef.current) {
        const age = Math.abs(t - lastNonEmptyFrameRef.current.t);
        setCurrentFrame(age < PERSIST_WINDOW ? lastNonEmptyFrameRef.current : frame);
      } else {
        setCurrentFrame(frame);
      }
    },
    [loadChunkForTime, findFrame]
  );

  // --- Timeline seek ---
  const handleSeek = useCallback(
    (t: number) => {
      const v = videoRef.current;
      if (v) {
        v.currentTime = t;
        setCurrentTime(t);
        loadChunkForTime(t);
      }
    },
    [loadChunkForTime]
  );

  // --- Play / Pause ---
  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      v.play();
      setIsPlaying(true);
    } else {
      v.pause();
      setIsPlaying(false);
    }
  }, []);

  // --- Playback speed ---
  const handleSpeedChange = useCallback((rate: number) => {
    setPlaybackRate(rate);
    const v = videoRef.current;
    if (v) v.playbackRate = rate;
  }, []);

  // --- Species filter toggle ---
  const toggleSpecies = useCallback((speciesId: string) => {
    setSpeciesFilter((prev) => {
      const next = new Set(prev);
      if (next.has(speciesId)) {
        next.delete(speciesId);
      } else {
        next.add(speciesId);
      }
      return next;
    });
  }, []);

  // --- Fish selection from table ---
  const handleTableSelect = useCallback((trackId: number) => {
    setSelectedTrackId((prev) => (prev === trackId ? null : trackId));
  }, []);

  // --- Resolve selected fish instance (may not be in current frame) ---
  const selectedFish: FishInstance | null = (() => {
    if (selectedTrackId == null) return null;
    // Try to find in current frame first
    if (currentFrame) {
      const found = currentFrame.instances.find((i) => i.trackId === selectedTrackId);
      if (found) return found;
    }
    // Not in current frame — return null (StatsPanel handles "not visible" message)
    return null;
  })();

  const selectedVisible = selectedFish != null;

  // The instances displayed in panels come from current frame, filtered
  const visibleInstances: FishInstance[] = currentFrame
    ? currentFrame.instances.filter(
        (i) =>
          i.metrics.confidence >= confidenceThreshold && speciesFilter.has(i.speciesId)
      )
    : [];

  // Load initial chunk on mount
  useEffect(() => {
    loadChunkForTime(0);
  }, [loadChunkForTime]);

  // If selectedTrackId is set but the fish isn't in the current frame, we still keep
  // the selection. We need to provide the last known instance for the StatsPanel card.
  // We'll search all loaded frames across chunks.
  const selectedFishForPanel: FishInstance | null = (() => {
    if (selectedTrackId == null) return null;
    if (selectedFish) return selectedFish;
    // Search current chunk frames for last known appearance
    if (currentChunk) {
      for (let i = currentChunk.frames.length - 1; i >= 0; i--) {
        const found = currentChunk.frames[i].instances.find(
          (inst) => inst.trackId === selectedTrackId
        );
        if (found) return found;
      }
    }
    return null;
  })();

  // --- Aggregate health from all currently-visible instances ---
  const avgWellness = visibleInstances.length > 0
    ? visibleInstances.reduce((s, i) => s + i.metrics.wellness_index, 0) / visibleInstances.length
    : 0;
  const speciesSet = new Set(visibleInstances.map(i => i.speciesId));

  return (
    <div className="session-viewer">
      {/* Header */}
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-icon">AV</div>
          <div className="brand-text">
            <h1 className="app-title">AquaVision Monitor</h1>
            <span className="app-subtitle">Real-time Fish Health Intelligence</span>
          </div>
        </div>

        {/* Live metric badges */}
        <div className="header-metrics">
          <div className="header-metric">
            <span className="hm-value">{visibleInstances.length}</span>
            <span className="hm-label">Fish Detected</span>
          </div>
          <div className="header-metric">
            <span className="hm-value">{totalUniqueTracks}</span>
            <span className="hm-label">Tracked Total</span>
          </div>
          <div className="header-metric">
            <span className="hm-value" style={{ color: avgWellness > 0.7 ? '#34d399' : avgWellness > 0.4 ? '#fbbf24' : '#f87171' }}>
              {avgWellness > 0 ? `${(avgWellness * 100).toFixed(0)}%` : '--'}
            </span>
            <span className="hm-label">Avg Health</span>
          </div>
          <div className="header-metric">
            <span className="hm-value">{speciesSet.size}</span>
            <span className="hm-label">Species</span>
          </div>
        </div>

        <div className="header-controls">
          {/* Species filters */}
          <div className="species-filters">
            {allSpecies.map((sid) => (
              <label key={sid} className="species-checkbox">
                <input
                  type="checkbox"
                  checked={speciesFilter.has(sid)}
                  onChange={() => toggleSpecies(sid)}
                />
                <span
                  className="species-dot"
                  style={{
                    backgroundColor: SPECIES_COLOR_MAP[sid].color,
                  }}
                />
                {SPECIES_COLOR_MAP[sid].label}
              </label>
            ))}
          </div>

          {/* Confidence threshold */}
          <div className="confidence-control">
            <label>
              Min Confidence {(confidenceThreshold * 100).toFixed(0)}%
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
              />
            </label>
          </div>
        </div>
      </header>

      {/* Main content */}
      <div className="main-content">
        {/* Center: video + controls */}
        <div className="center-column">
          <VideoWithOverlay
            videoSrc={VIDEO_SRC}
            currentFrame={currentFrame}
            hoveredTrackId={hoveredTrackId}
            selectedTrackId={selectedTrackId}
            onHoverChange={setHoveredTrackId}
            onSelectChange={setSelectedTrackId}
            onTimeUpdate={handleTimeUpdate}
            onDurationChange={setDuration}
            onMetadataLoaded={(w, h) => setVideoDims({ w, h })}
            videoRef={videoRef}
            confidenceThreshold={confidenceThreshold}
            speciesFilter={speciesFilter}
          />

          {/* Playback controls */}
          <div className="playback-controls">
            <button className="play-btn" onClick={togglePlay}>
              {isPlaying ? '⏸ Pause' : '▶ Play'}
            </button>
            <div className="speed-selector">
              {[0.5, 1, 1.5].map((rate) => (
                <button
                  key={rate}
                  className={`speed-btn ${playbackRate === rate ? 'speed-active' : ''}`}
                  onClick={() => handleSpeedChange(rate)}
                >
                  {rate}x
                </button>
              ))}
            </div>
          </div>

          {/* Timeline */}
          <TimelineScrubber
            currentTime={currentTime}
            duration={duration}
            onSeek={handleSeek}
          />
        </div>

        {/* Right sidebar */}
        <div className="right-sidebar">
          <StatsPanel
            instances={visibleInstances}
            selectedFish={selectedFishForPanel}
            selectedVisible={selectedVisible}
            totalUniqueTracks={totalUniqueTracks}
          />
          <FishTable
            instances={visibleInstances}
            selectedTrackId={selectedTrackId}
            onSelectTrack={handleTableSelect}
          />
        </div>
      </div>
    </div>
  );
};

export default SessionViewer;
