/**
 * Species color map.
 * Maps speciesId to a display color used for overlays and UI badges.
 */

export const SPECIES_COLOR_MAP: Record<string, { color: string; label: string }> = {
  tilapia: { color: '#0d9488', label: 'Tilapia' },   // teal
  catfish: { color: '#7c3aed', label: 'Catfish' },    // purple
};

/** Get a CSS color string for a species, with fallback for unknown species. */
export function getSpeciesColor(speciesId: string): string {
  return SPECIES_COLOR_MAP[speciesId]?.color ?? '#6b7280'; // gray fallback
}

/** Get human-readable label for a species. */
export function getSpeciesLabel(speciesId: string): string {
  return SPECIES_COLOR_MAP[speciesId]?.label ?? speciesId;
}

/** Convert hex color to rgba string with given alpha. */
export function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
