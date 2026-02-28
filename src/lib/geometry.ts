/**
 * Geometry helpers for drawing polygons and transforming coordinates
 * between the original video pixel space and the displayed canvas space.
 */

/**
 * Compute the scale factors to transform from video-pixel coords
 * to the canvas (displayed) pixel coords.
 *
 * @param videoWidth  - native video width in pixels
 * @param videoHeight - native video height in pixels
 * @param canvasWidth - displayed canvas width in pixels
 * @param canvasHeight - displayed canvas height in pixels
 */
export function getTransformScale(
  videoWidth: number,
  videoHeight: number,
  canvasWidth: number,
  canvasHeight: number
): { sx: number; sy: number } {
  return {
    sx: canvasWidth / videoWidth,
    sy: canvasHeight / videoHeight,
  };
}

/** Transform a point from video coords to canvas coords. */
export function toCanvasCoords(
  point: [number, number],
  sx: number,
  sy: number
): [number, number] {
  return [point[0] * sx, point[1] * sy];
}

/**
 * Draw a polygon path on a 2D context.
 * Does NOT call fill() or stroke() — caller decides rendering.
 */
export function tracePath(
  ctx: CanvasRenderingContext2D,
  polygon: [number, number][],
  sx: number,
  sy: number
): void {
  if (polygon.length === 0) return;
  ctx.beginPath();
  const [startX, startY] = toCanvasCoords(polygon[0], sx, sy);
  ctx.moveTo(startX, startY);
  for (let i = 1; i < polygon.length; i++) {
    const [px, py] = toCanvasCoords(polygon[i], sx, sy);
    ctx.lineTo(px, py);
  }
  ctx.closePath();
}

/**
 * Draw a filled polygon with the given color.
 * Used for the hit canvas (solid color, no alpha blending).
 */
export function drawFilledPolygon(
  ctx: CanvasRenderingContext2D,
  polygon: [number, number][],
  fillStyle: string,
  sx: number,
  sy: number
): void {
  tracePath(ctx, polygon, sx, sy);
  ctx.fillStyle = fillStyle;
  ctx.fill();
}

/**
 * Draw a stroked polygon outline.
 */
export function drawStrokedPolygon(
  ctx: CanvasRenderingContext2D,
  polygon: [number, number][],
  strokeStyle: string,
  lineWidth: number,
  sx: number,
  sy: number
): void {
  tracePath(ctx, polygon, sx, sy);
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
}
