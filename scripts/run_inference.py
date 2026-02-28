"""
Run fish instance segmentation on the demo video, then output annotation
chunk JSON files that the React app consumes.

Default backend is DINOv2-guided segmentation (better underwater object
separation than generic COCO YOLO in many scenes). If unavailable or weak,
the script falls back to YOLOv8-seg, then to OpenCV contour detection.

Usage:
    python3 scripts/run_inference.py

Optional env vars:
    AQUA_MODEL=dinov2|yolo|auto   (default: dinov2)
    DINOV2_MODEL_ID=<hf_model_id> (default: facebook/dinov2-base)

Output:
    public/demo/chunks/chunk_000.json, chunk_001.json, ...
"""

import json, os, sys, math
from pathlib import Path
import numpy as np
import cv2

# ── paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = ROOT / "public" / "demo" / "video.mp4"
CHUNKS_DIR = ROOT / "public" / "demo" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_DURATION = 2  # seconds per chunk
SAMPLE_FPS = 5      # how many frames per second we annotate
CONF_THRESH = 0.25  # YOLO confidence threshold
MIN_CONTOUR_AREA = 800  # minimum contour area for fallback detection
MODEL_BACKEND = os.getenv("AQUA_MODEL", "dinov2").strip().lower()
DINOV2_MODEL_ID = os.getenv("DINOV2_MODEL_ID", "facebook/dinov2-base")

# Species assignment heuristic: classify by area
# (larger blobs → catfish, smaller → tilapia)
AREA_THRESHOLD = 5000  # pixels²


def _sample_frames(video_path, sample_fps):
    """Return (fps, sampled_frames) where sampled_frames = [(t, frame_bgr), ...]."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / sample_fps))

    frames_data = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            t = frame_idx / fps
            frames_data.append((t, frame.copy()))
        frame_idx += 1

    cap.release()
    return fps, frames_data


def _mask_to_detections(mask, frame):
    """Convert a binary mask into detection dicts expected by downstream tracker."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / (min(w, h) + 1)
        if aspect > 8 or (w < 20 and h < 20):
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        polygon = approx.reshape(-1, 2).tolist()
        if len(polygon) > 20:
            step = max(1, len(polygon) // 20)
            polygon = polygon[::step]
        if len(polygon) < 4:
            polygon = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        polygon = [[int(p[0]), int(p[1])] for p in polygon]

        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else y + h // 2

        species = "catfish" if area > AREA_THRESHOLD else "tilapia"
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        conf = min(0.98, max(0.5, solidity * 0.8 + 0.3))

        dets.append({
            "bbox": [x, y, w, h],
            "polygon": polygon,
            "centroid": [cx, cy],
            "speciesId": species,
            "confidence": round(conf, 3),
            "area": area,
        })

    return dets

# ── simple IoU tracker ─────────────────────────────────────────────
class SimpleTracker:
    """Assign stable track IDs across frames using IoU matching."""
    def __init__(self, iou_thresh=0.25, max_lost=8):
        self.next_id = 1
        self.tracks = {}  # track_id -> last_bbox
        self.lost_count = {}  # track_id -> frames since last seen
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost

    def update(self, detections):
        """
        detections: list of dicts with 'bbox' key [x,y,w,h].
        Returns list of (track_id, detection) pairs.
        """
        if not self.tracks:
            # First frame: assign new IDs to all
            results = []
            for det in detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = det['bbox']
                self.lost_count[tid] = 0
                results.append((tid, det))
            return results

        # Compute IoU matrix between existing tracks and new detections
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))

        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(self.tracks[tid], det['bbox'])

        # Greedy matching
        matched_tracks = set()
        matched_dets = set()
        results = []

        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < self.iou_thresh:
                break
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            tid = track_ids[i]
            self.tracks[tid] = detections[j]['bbox']
            self.lost_count[tid] = 0
            matched_tracks.add(i)
            matched_dets.add(j)
            results.append((tid, detections[j]))
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0

        # New tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = det['bbox']
                self.lost_count[tid] = 0
                results.append((tid, det))

        # Increment lost count for unmatched tracks, remove old ones
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.lost_count[tid] = self.lost_count.get(tid, 0) + 1
                if self.lost_count[tid] > self.max_lost:
                    del self.tracks[tid]
                    del self.lost_count[tid]

        return results

    @staticmethod
    def _iou(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0


# ── detection methods ──────────────────────────────────────────────

def try_yolo(video_path):
    """Attempt YOLOv8-seg inference. Returns list of per-frame detections or None."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n-seg.pt")
        print("Running YOLOv8-seg inference...")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        frame_interval = max(1, int(fps / SAMPLE_FPS))

        all_frame_dets = []
        frame_idx = 0
        useful_dets = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                t = frame_idx / fps
                results = model(frame, conf=CONF_THRESH, verbose=False)[0]

                dets = []
                if results.masks is not None and len(results.masks) > 0:
                    for i in range(len(results.boxes)):
                        box = results.boxes[i]
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        # COCO class names - accept any detection
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        w, h = x2 - x1, y2 - y1

                        # Get mask polygon
                        mask = results.masks[i]
                        # masks.xy gives polygon points in image coords
                        if hasattr(mask, 'xy') and len(mask.xy) > 0:
                            polygon = mask.xy[0].tolist()
                        else:
                            # Fallback: use bbox as polygon
                            polygon = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]

                        # Simplify polygon to max 20 points
                        if len(polygon) > 20:
                            step = max(1, len(polygon) // 20)
                            polygon = polygon[::step]

                        polygon = [[int(round(float(p[0]))), int(round(float(p[1])))] for p in polygon]

                        area = float(w * h)
                        species = "catfish" if area > AREA_THRESHOLD else "tilapia"

                        centroid = [int(round(float((x1+x2)/2))), int(round(float((y1+y2)/2)))]
                        bbox = [int(round(float(x1))), int(round(float(y1))), int(round(float(w))), int(round(float(h)))]

                        dets.append({
                            "bbox": bbox,
                            "polygon": polygon,
                            "centroid": centroid,
                            "speciesId": species,
                            "confidence": round(conf, 3),
                            "area": area,
                        })
                        useful_dets += 1

                all_frame_dets.append({"t": round(t, 3), "dets": dets})

            frame_idx += 1

        cap.release()
        print(f"YOLO found {useful_dets} detections across {len(all_frame_dets)} frames")

        # If YOLO found very few detections, return None to trigger fallback
        if useful_dets < len(all_frame_dets) * 0.3:
            print("Too few YOLO detections — falling back to contour detection")
            return None

        return all_frame_dets

    except Exception as e:
        print(f"YOLO failed: {e}")
        return None


def try_dinov2(video_path):
    """
    Attempt DINOv2-guided segmentation.
    Uses DINO patch saliency + motion/color cues to generate fish masks.
    Returns list of per-frame detections or None.
    """
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoImageProcessor, AutoModel
    except Exception as e:
        print(f"DINOv2 dependencies unavailable: {e}")
        return None

    try:
        print(f"Running DINOv2-guided segmentation ({DINOV2_MODEL_ID})...")
        processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
        model = AutoModel.from_pretrained(DINOV2_MODEL_ID)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        _, frames_data = _sample_frames(video_path, SAMPLE_FPS)
        if not frames_data:
            print("No sampled frames for DINOv2")
            return None

        sample_for_bg = [f for _, f in frames_data[::max(1, len(frames_data) // 20)]]
        bg = np.median(np.stack(sample_for_bg), axis=0).astype(np.uint8)

        all_frame_dets = []
        total_dets = 0

        for t, frame in frames_data:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                inputs = processor(images=rgb, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)

                tokens = outputs.last_hidden_state[0]  # [1 + num_patches, dim]
                if tokens.shape[0] <= 1:
                    all_frame_dets.append({"t": round(t, 3), "dets": []})
                    continue

                cls_token = tokens[0:1]
                patch_tokens = tokens[1:]
                grid = int(math.sqrt(patch_tokens.shape[0]))
                if grid * grid != patch_tokens.shape[0]:
                    all_frame_dets.append({"t": round(t, 3), "dets": []})
                    continue

                saliency = F.cosine_similarity(
                    patch_tokens,
                    cls_token.expand_as(patch_tokens),
                    dim=1,
                )
                saliency = saliency.reshape(grid, grid).detach().cpu().numpy()

            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
            saliency_u8 = (saliency * 255).astype(np.uint8)
            saliency_u8 = cv2.resize(
                saliency_u8,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

            # Dynamic cues for aquatic scenes
            diff = cv2.absdiff(frame, bg)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, motion_mask = cv2.threshold(gray_diff, 22, 255, cv2.THRESH_BINARY)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv, np.array([0, 20, 50]), np.array([180, 255, 255]))

            _, dino_mask = cv2.threshold(saliency_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            combined = cv2.bitwise_or(dino_mask, motion_mask)
            combined = cv2.bitwise_or(combined, color_mask)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)

            dets = _mask_to_detections(combined, frame)
            total_dets += len(dets)
            all_frame_dets.append({"t": round(t, 3), "dets": dets})

        print(f"DINOv2 produced {total_dets} detections across {len(all_frame_dets)} frames")
        if total_dets < len(all_frame_dets) * 0.4:
            print("DINOv2 detections too sparse — trying fallback backend")
            return None

        return all_frame_dets

    except Exception as e:
        print(f"DINOv2 failed: {e}")
        return None


def contour_detect(video_path):
    """
    Fallback: use background subtraction + contour detection to find
    fish-like objects in the underwater video.
    """
    print("Running contour-based fish detection...")

    _, frames_data = _sample_frames(video_path, SAMPLE_FPS)

    if not frames_data:
        print("No frames read!")
        return []

    # Compute median background from sampled frames
    sample_for_bg = [f for _, f in frames_data[::max(1, len(frames_data)//20)]]
    bg = np.median(np.stack(sample_for_bg), axis=0).astype(np.uint8)

    all_frame_dets = []

    for t, frame in frames_data:
        # Background subtraction
        diff = cv2.absdiff(frame, bg)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Also use color segmentation: fish tend to be brighter/different
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Broad mask for non-background colors
        lower = np.array([0, 20, 50])
        upper = np.array([180, 255, 255])
        color_mask = cv2.inRange(hsv, lower, upper)

        # Combine: strong diff OR strong color difference
        _, thresh_diff = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(thresh_diff, color_mask)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

        # Edge-enhanced approach: use adaptive threshold on grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        edges = cv2.Canny(blurred, 30, 100)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        # Final mask: combine all approaches
        final_mask = cv2.bitwise_or(combined, edges_dilated)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        dets = _mask_to_detections(final_mask, frame)

        all_frame_dets.append({"t": round(t, 3), "dets": dets})

    total_dets = sum(len(f["dets"]) for f in all_frame_dets)
    print(f"Contour detection found {total_dets} detections across {len(all_frame_dets)} frames")
    return all_frame_dets


# ── build chunk JSONs ──────────────────────────────────────────────

def build_chunks(all_frame_dets):
    """Group tracked detections into time-chunked annotation files."""

    tracker = SimpleTracker(iou_thresh=0.2, max_lost=10)
    # Track species per track_id for consistency
    track_species = {}
    # Smooth biologic size estimates per track to avoid noisy frame-to-frame jumps
    track_length_cm = {}

    # Process all frames through tracker
    tracked_frames = []
    for frame_data in all_frame_dets:
        t = frame_data["t"]
        dets = frame_data["dets"]
        matched = tracker.update(dets)

        instances = []
        for tid, det in matched:
            # Lock species per track ID (first detection wins)
            if tid not in track_species:
                track_species[tid] = det["speciesId"]

            area = float(det["area"])
            _, _, bw, bh = det["bbox"]
            body_scale_px = math.sqrt(max(1.0, bw * bh))

            # Derive a more realistic length from bbox size with species offset
            # (catfish tend to be larger than tilapia in this demo)
            species_offset = 1.8 if track_species[tid] == "catfish" else -0.8
            raw_length = 7.5 + body_scale_px * 0.14 + species_offset
            raw_length = min(42.0, max(10.0, raw_length))

            # Exponential smoothing per track to reduce flicker
            prev_length = track_length_cm.get(tid)
            length_val = raw_length if prev_length is None else (0.75 * prev_length + 0.25 * raw_length)
            track_length_cm[tid] = length_val
            length_cm = round(length_val, 1)

            # Weight from length via rough allometric relation (kept simple)
            weight_g = round(max(30.0, 0.53 * (length_cm ** 2.22)))
            wellness = round(min(1.0, 0.6 + (det["confidence"] * 0.35)), 2)
            satiety_raw = 0.45 + min(0.35, area / 90000) + min(0.2, det["confidence"] * 0.25)
            satiety = round(min(1.0, max(0.3, satiety_raw)), 2)

            instances.append({
                "trackId": tid,
                "speciesId": track_species[tid],
                "bbox": det["bbox"],
                "polygon": det["polygon"],
                "centroid": det["centroid"],
                "metrics": {
                    "length_cm": length_cm,
                    "weight_g_est": weight_g,
                    "wellness_index": wellness,
                    "satiety_index": satiety,
                    "confidence": det["confidence"],
                },
            })

        tracked_frames.append({"t": t, "instances": instances})

    # Split into chunks
    if not tracked_frames:
        print("No frames to write!")
        return

    max_t = max(f["t"] for f in tracked_frames)
    num_chunks = math.ceil((max_t + 0.01) / CHUNK_DURATION)

    for ci in range(num_chunks):
        chunk_start = ci * CHUNK_DURATION
        chunk_end = min((ci + 1) * CHUNK_DURATION, max_t + 0.01)

        frames_in_chunk = [
            f for f in tracked_frames
            if chunk_start <= f["t"] < chunk_end
        ]

        chunk = {
            "chunkIndex": ci,
            "chunkStartSec": chunk_start,
            "chunkEndSec": round(chunk_end, 2),
            "fps": SAMPLE_FPS,
            "frames": frames_in_chunk,
        }

        padded = str(ci).zfill(3)
        path = CHUNKS_DIR / f"chunk_{padded}.json"
        with open(path, "w") as f:
            json.dump(chunk, f, indent=2)

        total_inst = sum(len(fr["instances"]) for fr in frames_in_chunk)
        print(f"  {path.name}: {len(frames_in_chunk)} frames, {total_inst} instances")

    # Clean up old chunk files beyond what we generated
    for old in CHUNKS_DIR.glob("chunk_*.json"):
        idx = int(old.stem.split("_")[1])
        if idx >= num_chunks:
            old.unlink()
            print(f"  Removed old {old.name}")

    print(f"\nDone! {num_chunks} chunks, {len(track_species)} unique tracks")
    print(f"Track species: { {k: v for k, v in sorted(track_species.items())} }")


# ── main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not VIDEO_PATH.exists():
        print(f"Video not found at {VIDEO_PATH}")
        sys.exit(1)

    print(f"Video: {VIDEO_PATH}")
    print(f"Resolution check...")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"  {w}x{h} @ {fps:.1f} fps, {total} frames, {total/fps:.2f}s")

    # Backend order controlled by AQUA_MODEL:
    # - dinov2 (default): DINOv2 -> YOLO -> contours
    # - yolo: YOLO -> DINOv2 -> contours
    # - auto: DINOv2 -> YOLO -> contours
    if MODEL_BACKEND == "yolo":
        backends = [try_yolo, try_dinov2, contour_detect]
    else:
        backends = [try_dinov2, try_yolo, contour_detect]

    frame_dets = None
    for backend in backends:
        frame_dets = backend(VIDEO_PATH)
        if frame_dets is not None and len(frame_dets) > 0:
            print(f"Using backend: {backend.__name__}")
            break

    if not frame_dets:
        print("No detections at all — exiting")
        sys.exit(1)

    print(f"\nBuilding chunk JSON files...")
    build_chunks(frame_dets)
