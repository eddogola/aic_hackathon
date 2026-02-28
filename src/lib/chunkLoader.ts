import type { AnnotationChunk } from '../types/annotations';

/**
 * Chunk loader with in-memory caching.
 * Fetches annotation chunks from /public/demo/chunks/ as playback time advances.
 */

const CHUNK_DURATION = 2; // seconds per chunk
const CHUNK_BASE_PATH = '/demo/chunks';

/** In-memory cache: chunkIndex -> AnnotationChunk */
const chunkCache = new Map<number, AnnotationChunk>();

/** Determine which chunk index covers a given time in seconds. */
export function getChunkIndex(timeSec: number): number {
  return Math.max(0, Math.floor(timeSec / CHUNK_DURATION));
}

/** Build the URL for a chunk JSON file by index. */
function chunkUrl(index: number): string {
  const padded = String(index).padStart(3, '0');
  return `${CHUNK_BASE_PATH}/chunk_${padded}.json`;
}

/**
 * Load a chunk by index. Returns from cache if available,
 * otherwise fetches from the public directory.
 */
export async function loadChunk(index: number): Promise<AnnotationChunk | null> {
  if (chunkCache.has(index)) {
    return chunkCache.get(index)!;
  }
  try {
    const resp = await fetch(chunkUrl(index));
    if (!resp.ok) return null;
    const data: AnnotationChunk = await resp.json();
    chunkCache.set(index, data);
    return data;
  } catch {
    return null;
  }
}

/**
 * Preload the current chunk and the next one for seamless playback.
 */
export async function preloadChunks(timeSec: number): Promise<void> {
  const idx = getChunkIndex(timeSec);
  await Promise.all([loadChunk(idx), loadChunk(idx + 1)]);
}

/** Clear all cached chunks (useful on video change). */
export function clearChunkCache(): void {
  chunkCache.clear();
}
