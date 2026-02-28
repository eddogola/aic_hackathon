import React, { useRef, useEffect, useCallback, useState } from 'react';
import type { FishInstance, AnnotationFrame } from '../types/annotations';
import { encodeTrackId, hitTestAtPoint } from '../lib/hitTest';
import { drawFilledPolygon, drawStrokedPolygon, tracePath, toCanvasCoords } from '../lib/geometry';
import { getSpeciesColor, hexToRgba } from '../lib/speciesColors';
import FishTooltip from './FishTooltip';

/**
 * Compute the actual rendered rectangle of a video within its container
 * when using object-fit: contain. The video may have letterbox bars.
 */
function getVideoDisplayRect(video: HTMLVideoElement): { x: number; y: number; w: number; h: number } {
  const containerW = video.clientWidth || 1;
  const containerH = video.clientHeight || 1;
  const videoW = video.videoWidth || 960;
  const videoH = video.videoHeight || 540;

  const containerAspect = containerW / containerH;
  const videoAspect = videoW / videoH;

  let renderW: number, renderH: number, offsetX: number, offsetY: number;

  if (videoAspect > containerAspect) {
    // Video is wider than container — pillarbox (bars top/bottom)
    renderW = containerW;
    renderH = containerW / videoAspect;
    offsetX = 0;
    offsetY = (containerH - renderH) / 2;
  } else {
    // Video is taller than container — letterbox (bars left/right)
    renderH = containerH;
    renderW = containerH * videoAspect;
    offsetX = (containerW - renderW) / 2;
    offsetY = 0;
  }

  return { x: offsetX, y: offsetY, w: renderW, h: renderH };
}

interface VideoWithOverlayProps {
  videoSrc: string;
  currentFrame: AnnotationFrame | null;
  hoveredTrackId: number | null;
  selectedTrackId: number | null;
  onHoverChange: (trackId: number | null) => void;
  onSelectChange: (trackId: number | null) => void;
  onTimeUpdate: (time: number) => void;
  onDurationChange: (dur: number) => void;
  onMetadataLoaded: (w: number, h: number) => void;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  confidenceThreshold: number;
  speciesFilter: Set<string>;
}

/**
 * Video element with two overlay canvases:
 *  - visibleCanvas: pretty overlays (outlines + fills)
 *  - hitCanvas: offscreen, each fish mask painted with unique RGB-encoded trackId
 */
const VideoWithOverlay: React.FC<VideoWithOverlayProps> = ({
  videoSrc,
  currentFrame,
  hoveredTrackId,
  selectedTrackId,
  onHoverChange,
  onSelectChange,
  onTimeUpdate,
  onDurationChange,
  onMetadataLoaded,
  videoRef,
  confidenceThreshold,
  speciesFilter,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const visibleCanvasRef = useRef<HTMLCanvasElement>(null);
  const hitCanvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const [containerRect, setContainerRect] = useState<DOMRect | null>(null);

  // Native video dimensions (set once on loadedmetadata)
  const videoDimsRef = useRef<{ w: number; h: number }>({ w: 960, h: 540 });

  /** Filter instances by confidence threshold and species filter */
  const getVisibleInstances = useCallback((): FishInstance[] => {
    if (!currentFrame) return [];
    return currentFrame.instances.filter(
      (inst) =>
        inst.metrics.confidence >= confidenceThreshold &&
        speciesFilter.has(inst.speciesId)
    );
  }, [currentFrame, confidenceThreshold, speciesFilter]);

  // The actual rendered video area within the container (accounts for object-fit: contain)
  const videoRectRef = useRef<{ x: number; y: number; w: number; h: number }>({ x: 0, y: 0, w: 960, h: 540 });

  /** Sync canvas dimensions to the video element's actual rendered area */
  const syncCanvasSize = useCallback(() => {
    const video = videoRef.current;
    const visibleCanvas = visibleCanvasRef.current;
    const hitCanvas = hitCanvasRef.current;
    if (!video || !visibleCanvas || !hitCanvas) return;

    const containerW = video.clientWidth;
    const containerH = video.clientHeight;

    // Canvases fill the full container (same as the video element)
    if (visibleCanvas.width !== containerW || visibleCanvas.height !== containerH) {
      visibleCanvas.width = containerW;
      visibleCanvas.height = containerH;
      hitCanvas.width = containerW;
      hitCanvas.height = containerH;
    }

    // Compute the actual rendered video rect (for coordinate mapping)
    videoRectRef.current = getVideoDisplayRect(video);
    setContainerRect(new DOMRect(0, 0, containerW, containerH));
  }, [videoRef]);

  /**
   * Compute scale factors from video-pixel coords to canvas-pixel coords,
   * accounting for the object-fit: contain offset.
   * The returned sx/sy map video pixels into the rendered video area,
   * and ox/oy provide the offset of that area within the canvas.
   */
  const getScaleAndOffset = useCallback(() => {
    const { w: vw, h: vh } = videoDimsRef.current;
    const vr = videoRectRef.current;
    const sx = vr.w / vw;
    const sy = vr.h / vh;
    return { sx, sy, ox: vr.x, oy: vr.y };
  }, []);

  /** Draw all overlays on the visible canvas */
  const drawVisible = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      const canvas = ctx.canvas;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const instances = getVisibleInstances();
      if (instances.length === 0) return;

      const { sx, sy, ox, oy } = getScaleAndOffset();

      // Translate context so all drawing is offset to the actual video area
      ctx.save();
      ctx.translate(ox, oy);

      for (const inst of instances) {
        const color = getSpeciesColor(inst.speciesId);
        const isHovered = inst.trackId === hoveredTrackId;
        const isSelected = inst.trackId === selectedTrackId;

        // Draw outline for every instance
        drawStrokedPolygon(ctx, inst.polygon, hexToRgba(color, 0.7), isSelected ? 3 : 1.5, sx, sy);

        // Fill hovered instance
        if (isHovered && !isSelected) {
          drawFilledPolygon(ctx, inst.polygon, hexToRgba(color, 0.35), sx, sy);
        }

        // Fill selected instance (thicker outline + stronger fill)
        if (isSelected) {
          drawFilledPolygon(ctx, inst.polygon, hexToRgba(color, 0.5), sx, sy);
          drawStrokedPolygon(ctx, inst.polygon, hexToRgba(color, 1), 3, sx, sy);

          // Draw ID label near centroid
          const [cx, cy] = toCanvasCoords(inst.centroid, sx, sy);
          ctx.font = 'bold 13px Inter, system-ui, sans-serif';
          ctx.fillStyle = '#fff';
          ctx.strokeStyle = 'rgba(0,0,0,0.6)';
          ctx.lineWidth = 3;
          const label = `ID: ${inst.trackId}`;
          ctx.strokeText(label, cx - 20, cy - 10);
          ctx.fillText(label, cx - 20, cy - 10);
        }
      }

      ctx.restore();
    },
    [getVisibleInstances, getScaleAndOffset, hoveredTrackId, selectedTrackId]
  );

  /** Draw the hit-test canvas: each fish polygon filled with its encoded RGB color */
  const drawHit = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      const canvas = ctx.canvas;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const instances = getVisibleInstances();
      if (instances.length === 0) return;

      const { sx, sy, ox, oy } = getScaleAndOffset();

      // Disable anti-aliasing on the hit canvas for accurate pixel reads
      ctx.imageSmoothingEnabled = false;

      ctx.save();
      ctx.translate(ox, oy);

      for (const inst of instances) {
        const [r, g, b] = encodeTrackId(inst.trackId);
        const fillColor = `rgb(${r},${g},${b})`;
        tracePath(ctx, inst.polygon, sx, sy);
        ctx.fillStyle = fillColor;
        ctx.fill();
      }

      ctx.restore();
    },
    [getVisibleInstances, getScaleAndOffset]
  );

  /** Main render loop — redraws both canvases */
  const renderLoop = useCallback(() => {
    const visCtx = visibleCanvasRef.current?.getContext('2d');
    const hitCtx = hitCanvasRef.current?.getContext('2d');
    if (visCtx && hitCtx) {
      syncCanvasSize();
      drawVisible(visCtx);
      drawHit(hitCtx);
    }
    animFrameRef.current = requestAnimationFrame(renderLoop);
  }, [syncCanvasSize, drawVisible, drawHit]);

  // Start/stop the render loop
  useEffect(() => {
    animFrameRef.current = requestAnimationFrame(renderLoop);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [renderLoop]);

  // Handle window resize
  useEffect(() => {
    const onResize = () => syncCanvasSize();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [syncCanvasSize]);

  /**
   * Get canvas-space coordinates from a mouse event.
   * Uses the visible canvas's bounding rect (hit canvas is display:none).
   */
  const getCanvasMouseCoords = useCallback((e: React.MouseEvent): { x: number; y: number } | null => {
    const visibleCanvas = visibleCanvasRef.current;
    if (!visibleCanvas) return null;
    const rect = visibleCanvas.getBoundingClientRect();
    // Scale from CSS pixels to canvas pixels (handles DPR-like differences)
    const scaleX = visibleCanvas.width / rect.width;
    const scaleY = visibleCanvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    return { x, y };
  }, []);

  /** Mouse-move handler: read hit canvas to determine hovered fish */
  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const coords = getCanvasMouseCoords(e);
      if (!coords) return;
      const hitCtx = hitCanvasRef.current?.getContext('2d');
      if (!hitCtx) return;
      const trackId = hitTestAtPoint(hitCtx, coords.x, coords.y);
      onHoverChange(trackId);
    },
    [getCanvasMouseCoords, onHoverChange]
  );

  /** Click handler: select fish or clear selection on background click */
  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      const coords = getCanvasMouseCoords(e);
      if (!coords) return;
      const hitCtx = hitCanvasRef.current?.getContext('2d');
      if (!hitCtx) return;
      const trackId = hitTestAtPoint(hitCtx, coords.x, coords.y);
      // Click on fish -> select; click on background -> deselect
      onSelectChange(trackId);
    },
    [getCanvasMouseCoords, onSelectChange]
  );

  const handleMouseLeave = useCallback(() => {
    onHoverChange(null);
  }, [onHoverChange]);

  /** Get the hovered fish instance for tooltip rendering */
  const tooltipFish = (() => {
    const id = hoveredTrackId ?? selectedTrackId;
    if (id == null || !currentFrame) return null;
    return currentFrame.instances.find((i) => i.trackId === id) ?? null;
  })();

  /** Compute tooltip position from the fish centroid (in canvas coords with offset) */
  const tooltipPos = (() => {
    if (!tooltipFish) return { x: 0, y: 0 };
    const vr = videoRectRef.current;
    const { w: vw, h: vh } = videoDimsRef.current;
    const sx = vr.w / vw;
    const sy = vr.h / vh;
    const [cx, cy] = toCanvasCoords(tooltipFish.centroid, sx, sy);
    return { x: cx + vr.x, y: cy + vr.y };
  })();

  return (
    <div className="video-overlay-container" ref={containerRef}>
      <video
        ref={videoRef}
        src={videoSrc}
        className="video-element"
        onLoadedMetadata={() => {
          const v = videoRef.current;
          if (v) {
            videoDimsRef.current = { w: v.videoWidth, h: v.videoHeight };
            onMetadataLoaded(v.videoWidth, v.videoHeight);
            onDurationChange(v.duration);
            syncCanvasSize();
          }
        }}
        onTimeUpdate={() => {
          const v = videoRef.current;
          if (v) onTimeUpdate(v.currentTime);
        }}
        muted
        playsInline
      />
      {/* Visible overlay canvas */}
      <canvas
        ref={visibleCanvasRef}
        className="overlay-canvas"
        onMouseMove={handleMouseMove}
        onClick={handleClick}
        onMouseLeave={handleMouseLeave}
      />
      {/* Hit-test canvas (hidden) */}
      <canvas ref={hitCanvasRef} className="hit-canvas" />

      {/* Tooltip */}
      {tooltipFish && (
        <FishTooltip
          fish={tooltipFish}
          x={tooltipPos.x}
          y={tooltipPos.y}
          containerRect={containerRect}
        />
      )}
    </div>
  );
};

export default VideoWithOverlay;
