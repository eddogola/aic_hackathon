# Aquaculture Vision Demo

Interactive instance-segmentation video demo for fish monitoring. Hover or click a fish to see its colored mask and metrics. Built with React + TypeScript + Vite. No backend required.

## Quick Start

```bash
npm install
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173).

## Adding Your Video

Place your video file at:

```
public/demo/video.mp4
```

The sample annotation coordinates assume a 960x540 video. The overlay scales to any resolution.

## Project Structure

```
src/
  components/
    SessionViewer.tsx    - Page layout, shared state, controls
    VideoWithOverlay.tsx - Video + dual-canvas overlay + render loop
    StatsPanel.tsx       - Selected fish card + aggregate stats
    FishTable.tsx        - Sortable fish track table
    FishTooltip.tsx      - Hover/selection tooltip
    TimelineScrubber.tsx - Timeline range input synced to video
  lib/
    chunkLoader.ts       - Fetch + cache annotation chunks
    hitTest.ts           - RGB encode/decode trackId for pixel hit-testing
    geometry.ts          - Polygon drawing + coordinate transforms
    speciesColors.ts     - Species color map
  types/
    annotations.ts       - TypeScript types for the chunk JSON schema
public/demo/
  video.mp4              - Your video file (placeholder)
  chunks/
    chunk_000.json       - Annotations for t=[0,2) seconds
    chunk_001.json       - Annotations for t=[2,4) seconds
```

## Key Features

- Dual-canvas overlay: visible canvas for rendering + hidden hit canvas for pixel-accurate fish detection
- Chunk-based annotation loading with in-memory caching
- object-fit:contain alignment so overlays match the actual video area
- Species filters, confidence threshold, sortable fish table
- Play/Pause, speed control (0.5x/1x/1.5x), timeline scrubber
