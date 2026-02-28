/**
 * Hit-testing utilities for instance segmentation.
 *
 * Each fish trackId is encoded as a unique RGB color on the hidden hit canvas.
 * Reading a single pixel under the mouse decodes back to a trackId.
 * This avoids expensive point-in-polygon math on every mouse move.
 */

/** Encode a trackId (integer) into an RGB triplet. Supports up to 16,777,215 IDs. */
export function encodeTrackId(id: number): [number, number, number] {
  const r = id & 0xff;
  const g = (id >> 8) & 0xff;
  const b = (id >> 16) & 0xff;
  return [r, g, b];
}

/** Decode an RGB triplet back to a trackId. Returns 0 if the pixel is background (black). */
export function decodeTrackId(r: number, g: number, b: number): number | null {
  const id = r + (g << 8) + (b << 16);
  // 0 means background (no fish)
  return id === 0 ? null : id;
}

/**
 * Read a single pixel from a canvas at (x, y) and decode it to a trackId.
 * Returns null if the pixel is background or out of bounds.
 */
export function hitTestAtPoint(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number
): number | null {
  if (x < 0 || y < 0 || x >= ctx.canvas.width || y >= ctx.canvas.height) {
    return null;
  }
  const pixel = ctx.getImageData(Math.floor(x), Math.floor(y), 1, 1).data;
  return decodeTrackId(pixel[0], pixel[1], pixel[2]);
}
