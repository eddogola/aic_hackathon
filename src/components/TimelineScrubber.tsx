import React from 'react';

interface TimelineScrubberProps {
  currentTime: number;
  duration: number;
  onSeek: (time: number) => void;
}

/** Timeline scrubber synced to video currentTime. */
const TimelineScrubber: React.FC<TimelineScrubberProps> = ({
  currentTime,
  duration,
  onSeek,
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onSeek(parseFloat(e.target.value));
  };

  const formatTime = (sec: number): string => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 10);
    return `${m}:${s.toString().padStart(2, '0')}.${ms}`;
  };

  // Progress percentage for the filled track
  const pct = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="timeline-scrubber">
      <span className="timeline-time">{formatTime(currentTime)}</span>
      <div className="timeline-track-wrapper">
        <input
          type="range"
          className="timeline-range"
          min={0}
          max={duration || 1}
          step={0.01}
          value={currentTime}
          onChange={handleChange}
          style={{
            background: `linear-gradient(to right, #0d9488 0%, #0d9488 ${pct}%, #374151 ${pct}%, #374151 100%)`,
          }}
        />
        {/* Stub event markers — could be populated from metadata */}
        <div className="timeline-markers">
          <div className="timeline-marker" style={{ left: '25%' }} title="Feeding event" />
          <div className="timeline-marker" style={{ left: '60%' }} title="Activity spike" />
        </div>
      </div>
      <span className="timeline-time">{formatTime(duration)}</span>
    </div>
  );
};

export default TimelineScrubber;
