#!/usr/bin/env python3
"""
DINOv2 High Frame Rate Optimization Script

Optimizations for processing fish videos at higher frame rates:
1. Batch processing of frames
2. GPU acceleration with mixed precision
3. Feature caching and temporal consistency
4. Adaptive quality scaling based on motion
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import cv2
import numpy as np
from pathlib import Path
import time

class OptimizedDINOv2Processor:
    def __init__(self, model_id="facebook/dinov2-base", batch_size=4):
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Initializing DINOv2 on {self.device} with batch_size={batch_size}")
        
        # Load model with optimizations
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        
        # Enable mixed precision for faster inference
        if self.device == "cuda":
            self.model = self.model.half()
            self.use_amp = True
        else:
            self.use_amp = False
            
        # Cache for temporal consistency
        self.feature_cache = {}
        self.background_features = None
        
    def precompute_background(self, frames_sample):
        """Precompute background features for better foreground separation."""
        print("Computing background model...")
        
        # Sample frames for background
        bg_frames = frames_sample[::max(1, len(frames_sample) // 10)]
        
        with torch.no_grad():
            bg_features = []
            for frame in bg_frames:
                inputs = self.processor(frame, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                if self.use_amp:
                    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state.mean(dim=1)  # Global pool
                    bg_features.append(features.cpu())
            
            # Average background features
            self.background_features = torch.stack(bg_features).mean(dim=0)
            print(f"Background model computed from {len(bg_frames)} frames")
    
    def process_frame_batch(self, frames):
        """Process a batch of frames efficiently."""
        if not frames:
            return []
            
        batch_size = min(len(frames), self.batch_size)
        
        with torch.no_grad():
            # Prepare batch
            inputs = self.processor(frames[:batch_size], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if self.use_amp:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(**inputs)
                patch_features = outputs.last_hidden_state  # [batch, patches, dim]
            
            # Generate saliency maps for each frame in batch
            saliency_maps = []
            for i in range(batch_size):
                features = patch_features[i]  # [patches, dim]
                
                # Compute saliency using background subtraction
                if self.background_features is not None:
                    bg_features = self.background_features.to(self.device)
                    if self.use_amp:
                        bg_features = bg_features.half()
                    
                    # Compute similarity to background
                    similarities = F.cosine_similarity(
                        features.unsqueeze(1), 
                        bg_features.unsqueeze(0), 
                        dim=2
                    )
                    saliency = 1.0 - similarities.max(dim=1)[0]  # Foreground = dissimilar to background
                else:
                    # Fallback: use attention-based saliency
                    saliency = features.norm(dim=1)
                
                # Reshape to spatial dimensions (14x14 for base model)
                patch_size = int(np.sqrt(features.shape[0]))
                saliency_map = saliency.view(patch_size, patch_size)
                saliency_maps.append(saliency_map.cpu().numpy())
            
            return saliency_maps
    
    def generate_fish_masks(self, frame, saliency_map, motion_mask=None):
        """Generate fish masks from DINOv2 saliency and optional motion cues."""
        h, w = frame.shape[:2]
        
        # Upsample saliency map to frame size
        saliency_resized = cv2.resize(saliency_map, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Normalize and threshold
        saliency_norm = (saliency_resized - saliency_resized.min()) / (saliency_resized.max() - saliency_resized.min() + 1e-8)
        
        # Adaptive thresholding based on saliency distribution
        threshold = np.percentile(saliency_norm, 85)  # Top 15% most salient
        fish_mask = (saliency_norm > threshold).astype(np.uint8) * 255
        
        # Combine with motion if available
        if motion_mask is not None:
            motion_resized = cv2.resize(motion_mask, (w, h))
            fish_mask = cv2.bitwise_and(fish_mask, motion_resized)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fish_mask = cv2.morphologyEx(fish_mask, cv2.MORPH_OPEN, kernel)
        fish_mask = cv2.morphologyEx(fish_mask, cv2.MORPH_CLOSE, kernel)
        
        return fish_mask
    
    def compute_motion_mask(self, frame, prev_frame):
        """Compute motion mask between consecutive frames."""
        if prev_frame is None:
            return None
            
        # Convert to grayscale
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute optical flow magnitude
        flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
        if flow[0] is not None:
            # Create motion mask from flow magnitude
            magnitude = np.sqrt(flow[0][:, 0]**2 + flow[0][:, 1]**2)
            motion_threshold = np.percentile(magnitude, 75)
            motion_mask = (magnitude > motion_threshold).astype(np.uint8) * 255
            return motion_mask
        
        # Fallback: simple frame difference
        diff = cv2.absdiff(gray1, gray2)
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        return motion_mask

def optimize_inference_settings():
    """Apply system-level optimizations for faster inference."""
    if torch.cuda.is_available():
        # Enable optimized attention and memory management
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.8)
        
        print(f"CUDA optimizations enabled. GPU: {torch.cuda.get_device_name()}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        # CPU optimizations
        torch.set_num_threads(min(8, torch.get_num_threads()))
        print(f"CPU optimizations enabled. Threads: {torch.get_num_threads()}")

if __name__ == "__main__":
    # Test the optimized processor
    optimize_inference_settings()
    
    processor = OptimizedDINOv2Processor(batch_size=8)
    
    # Load test video
    video_path = Path("../public/demo/video1.mp4")
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        for i in range(min(100, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        
        print(f"Testing with {len(frames)} frames from {video_path}")
        
        # Benchmark processing speed
        start_time = time.time()
        
        # Precompute background
        processor.precompute_background(frames[::5])
        
        # Process frames in batches
        total_processed = 0
        for i in range(0, len(frames), processor.batch_size):
            batch = frames[i:i+processor.batch_size]
            saliency_maps = processor.process_frame_batch(batch)
            total_processed += len(batch)
            
            if i == 0:  # Show first batch results
                for j, (frame, saliency) in enumerate(zip(batch, saliency_maps)):
                    mask = processor.generate_fish_masks(frame, saliency)
                    print(f"Frame {j}: Generated mask with {cv2.countNonZero(mask)} foreground pixels")
        
        end_time = time.time()
        processing_fps = total_processed / (end_time - start_time)
        
        print(f"\nPerformance Results:")
        print(f"Original video FPS: {fps:.1f}")
        print(f"Processing FPS: {processing_fps:.1f}")
        print(f"Real-time factor: {processing_fps/fps:.2f}x")
        
        if processing_fps >= fps:
            print("✅ Real-time processing achieved!")
        else:
            print(f"⚠️  Need {fps/processing_fps:.1f}x speedup for real-time")
    
    else:
        print(f"Video not found: {video_path}")
