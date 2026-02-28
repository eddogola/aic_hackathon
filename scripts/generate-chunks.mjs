/**
 * Generate realistic annotation chunk JSON files for the demo video.
 * Video: 720x1280 portrait, ~12.82 seconds.
 *
 * Run: node scripts/generate-chunks.mjs
 */

import { writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(__dirname, '..', 'public', 'demo', 'chunks');

const VIDEO_W = 720;
const VIDEO_H = 1280;
const DURATION = 12.82;
const CHUNK_DUR = 2; // seconds per chunk
const FPS = 5; // annotation frames per second
const NUM_CHUNKS = Math.ceil(DURATION / CHUNK_DUR);

// --- Fish track definitions ---
// Each track defines a fish that exists for a time range, with start/end positions and size.
const tracks = [
  {
    trackId: 1, speciesId: 'tilapia',
    tStart: 0, tEnd: 8,
    // Swims right across upper-mid area
    startCx: 80, startCy: 350, endCx: 650, endCy: 420,
    bodyW: 120, bodyH: 55,
    metrics: { length_cm: 22.4, weight_g_est: 310, wellness_index: 0.87, satiety_index: 0.65, confidence: 0.94 },
  },
  {
    trackId: 2, speciesId: 'catfish',
    tStart: 0, tEnd: 10,
    // Swims left across lower area
    startCx: 620, startCy: 880, endCx: 100, endCy: 820,
    bodyW: 150, bodyH: 65,
    metrics: { length_cm: 28.1, weight_g_est: 520, wellness_index: 0.72, satiety_index: 0.80, confidence: 0.91 },
  },
  {
    trackId: 3, speciesId: 'tilapia',
    tStart: 0, tEnd: 12.82,
    // Moves slowly down-right in upper area
    startCx: 300, startCy: 200, endCx: 500, endCy: 380,
    bodyW: 95, bodyH: 42,
    metrics: { length_cm: 18.6, weight_g_est: 210, wellness_index: 0.91, satiety_index: 0.55, confidence: 0.88 },
  },
  {
    trackId: 4, speciesId: 'catfish',
    tStart: 0, tEnd: 6,
    // Mostly stationary in lower-left
    startCx: 180, startCy: 1020, endCx: 220, endCy: 1000,
    bodyW: 135, bodyH: 58,
    metrics: { length_cm: 25.3, weight_g_est: 440, wellness_index: 0.68, satiety_index: 0.72, confidence: 0.86 },
  },
  {
    trackId: 5, speciesId: 'tilapia',
    tStart: 1.5, tEnd: 9,
    // Swims right across middle area
    startCx: 50, startCy: 580, endCx: 670, endCy: 550,
    bodyW: 100, bodyH: 45,
    metrics: { length_cm: 16.2, weight_g_est: 175, wellness_index: 0.95, satiety_index: 0.48, confidence: 0.82 },
  },
  {
    trackId: 6, speciesId: 'catfish',
    tStart: 3, tEnd: 12.82,
    // Enters from right, swims left through mid-lower
    startCx: 680, startCy: 700, endCx: 120, endCy: 750,
    bodyW: 160, bodyH: 70,
    metrics: { length_cm: 30.5, weight_g_est: 580, wellness_index: 0.76, satiety_index: 0.77, confidence: 0.93 },
  },
  {
    trackId: 7, speciesId: 'tilapia',
    tStart: 5, tEnd: 12.82,
    // Swims up-right from lower area
    startCx: 150, startCy: 1100, endCx: 550, endCy: 650,
    bodyW: 110, bodyH: 48,
    metrics: { length_cm: 20.1, weight_g_est: 260, wellness_index: 0.89, satiety_index: 0.58, confidence: 0.90 },
  },
  {
    trackId: 8, speciesId: 'tilapia',
    tStart: 8, tEnd: 12.82,
    // Small fish, enters top-left, swims right
    startCx: 60, startCy: 150, endCx: 600, endCy: 250,
    bodyW: 80, bodyH: 35,
    metrics: { length_cm: 14.5, weight_g_est: 140, wellness_index: 0.97, satiety_index: 0.40, confidence: 0.78 },
  },
  {
    trackId: 9, speciesId: 'catfish',
    tStart: 7, tEnd: 12.82,
    // Slow mover in the middle
    startCx: 400, startCy: 500, endCx: 320, endCy: 600,
    bodyW: 140, bodyH: 60,
    metrics: { length_cm: 26.8, weight_g_est: 470, wellness_index: 0.71, satiety_index: 0.83, confidence: 0.87 },
  },
];

/**
 * Generate a fish-shaped polygon given a center point and body dimensions.
 * Returns 10 points forming a fish-like contour.
 */
function fishPolygon(cx, cy, w, h) {
  const hw = w / 2;
  const hh = h / 2;
  return [
    [Math.round(cx + hw),      Math.round(cy)],           // nose
    [Math.round(cx + hw * 0.55), Math.round(cy - hh)],     // upper front
    [Math.round(cx - hw * 0.15), Math.round(cy - hh * 0.9)], // upper mid
    [Math.round(cx - hw * 0.6),  Math.round(cy - hh * 0.7)], // upper back
    [Math.round(cx - hw * 0.9),  Math.round(cy - hh * 0.5)], // tail top
    [Math.round(cx - hw),       Math.round(cy)],           // tail tip
    [Math.round(cx - hw * 0.9),  Math.round(cy + hh * 0.5)], // tail bottom
    [Math.round(cx - hw * 0.6),  Math.round(cy + hh * 0.7)], // lower back
    [Math.round(cx - hw * 0.15), Math.round(cy + hh * 0.9)], // lower mid
    [Math.round(cx + hw * 0.55), Math.round(cy + hh)],     // lower front
  ];
}

/** Linearly interpolate between two values. */
function lerp(a, b, t) {
  return a + (b - a) * t;
}

/** Add small random jitter for realism. */
function jitter(val, amount = 3) {
  return Math.round(val + (Math.random() - 0.5) * amount * 2);
}

/** Slightly vary a metric value for realism. */
function varyMetric(base, variance = 0.02) {
  return +(base + (Math.random() - 0.5) * variance * 2).toFixed(2);
}

// --- Generate chunks ---
mkdirSync(OUT_DIR, { recursive: true });

for (let ci = 0; ci < NUM_CHUNKS; ci++) {
  const chunkStart = ci * CHUNK_DUR;
  const chunkEnd = Math.min((ci + 1) * CHUNK_DUR, DURATION);
  const numFrames = Math.round((chunkEnd - chunkStart) * FPS);

  const frames = [];

  for (let fi = 0; fi < numFrames; fi++) {
    const t = +(chunkStart + fi / FPS).toFixed(2);
    if (t > DURATION) break;

    const instances = [];

    for (const track of tracks) {
      // Skip if this track isn't active at time t
      if (t < track.tStart || t > track.tEnd) continue;

      // Progress through this track's lifetime (0 to 1)
      const progress = (t - track.tStart) / (track.tEnd - track.tStart);

      // Interpolate position with jitter
      const cx = jitter(lerp(track.startCx, track.endCx, progress), 4);
      const cy = jitter(lerp(track.startCy, track.endCy, progress), 4);

      // Clamp to video bounds with padding
      const pad = Math.max(track.bodyW, track.bodyH) / 2 + 5;
      const clampedCx = Math.max(pad, Math.min(VIDEO_W - pad, cx));
      const clampedCy = Math.max(pad, Math.min(VIDEO_H - pad, cy));

      const poly = fishPolygon(clampedCx, clampedCy, track.bodyW, track.bodyH);
      const bboxX = Math.min(...poly.map(p => p[0]));
      const bboxY = Math.min(...poly.map(p => p[1]));
      const bboxW = Math.max(...poly.map(p => p[0])) - bboxX;
      const bboxH = Math.max(...poly.map(p => p[1])) - bboxY;

      instances.push({
        trackId: track.trackId,
        speciesId: track.speciesId,
        bbox: [bboxX, bboxY, bboxW, bboxH],
        polygon: poly,
        centroid: [Math.round(clampedCx), Math.round(clampedCy)],
        metrics: {
          length_cm: varyMetric(track.metrics.length_cm, 0.3),
          weight_g_est: Math.round(varyMetric(track.metrics.weight_g_est, 5)),
          wellness_index: +Math.max(0, Math.min(1, varyMetric(track.metrics.wellness_index, 0.03))).toFixed(2),
          satiety_index: +Math.max(0, Math.min(1, varyMetric(track.metrics.satiety_index, 0.03))).toFixed(2),
          confidence: +Math.max(0, Math.min(1, varyMetric(track.metrics.confidence, 0.03))).toFixed(2),
        },
      });
    }

    frames.push({ t, instances });
  }

  const chunk = {
    chunkIndex: ci,
    chunkStartSec: chunkStart,
    chunkEndSec: +chunkEnd.toFixed(2),
    fps: FPS,
    frames,
  };

  const padded = String(ci).padStart(3, '0');
  const filePath = join(OUT_DIR, `chunk_${padded}.json`);
  writeFileSync(filePath, JSON.stringify(chunk, null, 2));
  console.log(`Wrote ${filePath} — ${frames.length} frames, ${frames.reduce((s, f) => s + f.instances.length, 0)} total instances`);
}

console.log(`\nDone! Generated ${NUM_CHUNKS} chunks for ${VIDEO_W}x${VIDEO_H} video (${DURATION}s).`);
