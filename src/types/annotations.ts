/** Metrics for an individual fish instance in a single frame. */
export interface FishMetrics {
  length_cm: number;
  weight_g_est: number;
  /** 0–1 scale where 1 = perfectly healthy */
  wellness_index: number;
  /** 0–1 scale where 1 = fully satiated */
  satiety_index: number;
  /** Detection confidence 0–1 */
  confidence: number;
}

/** A single fish instance within one frame. */
export interface FishInstance {
  /** Stable integer ID that persists across frames for the same fish */
  trackId: number;
  /** Species identifier, e.g. "tilapia" | "catfish" */
  speciesId: string;
  /** Bounding box in video-pixel coords: [x, y, width, height] */
  bbox: [number, number, number, number];
  /** Contour polygon in video-pixel coords: [[x1,y1], [x2,y2], ...] */
  polygon: [number, number][];
  /** Center point in video-pixel coords: [cx, cy] */
  centroid: [number, number];
  metrics: FishMetrics;
}

/** A single frame of annotation data. */
export interface AnnotationFrame {
  /** Time in seconds relative to video start */
  t: number;
  instances: FishInstance[];
}

/** A chunk of precomputed annotations covering a time range. */
export interface AnnotationChunk {
  chunkIndex: number;
  chunkStartSec: number;
  chunkEndSec: number;
  fps: number;
  frames: AnnotationFrame[];
}

/** Species → display color mapping entry */
export interface SpeciesColorEntry {
  speciesId: string;
  color: string; // CSS color string, e.g. "rgba(0,128,128,1)"
  label: string; // Human-readable name
}
