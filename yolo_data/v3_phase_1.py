#!/usr/bin/env python3
"""
v3 Phase 1: Video Inference with CPU-parallel frame reading
- One video at a time (sequential)
- Multiple CPU reader threads (parallel frame reading)
- GPU inference continuously fed with frames (no stalling)
"""

import cv2
import torch
import numpy as np
import threading
import queue
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime
try:
    import decord
    decord.gpu([0])  # Use GPU for decoding
    HAS_DECORD = True
except (ImportError, Exception):
    HAS_DECORD = False

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = Path("/datadrive/polyp/yolo_output/runs/detect/2026_04_15(1)/weights/best.pt")
VIDEO_DIR = Path("videos/")
OUT_ROOT = Path("/datadrive/polyp/data/pre_v3")

# Inference settings
CONF_THRESH = 0.5
BATCH_SIZE = 256  # Smaller batch to reduce memory

# Frame queue (prefetch buffer)
FRAME_QUEUE_SIZE = 512  # ~8 seconds @ 60fps, much less memory

# Review settings
KEYFRAME = 3000
REVIEW_EVERY_before = 500
REVIEW_EVERY_after = 100

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Initialization
# ============================================================================

print(f"Model      : {MODEL_PATH}")
print(f"Video dir  : {VIDEO_DIR}")
print(f"Output root: {OUT_ROOT.resolve()}")
print(f"CPU cores  : {os.cpu_count()}")
print(f"Device     : {DEVICE}")
print(f"Keyframe   : {KEYFRAME}")
print()

model = YOLO(str(MODEL_PATH))
model.to(DEVICE)
print(f"✓ Model loaded: {MODEL_PATH.name}")
print(f"  Classes: {model.names}")

if HAS_DECORD:
    print(f"✓ Video reader: decord (GPU hardware-accelerated)")
else:
    print(f"⚠️  Video reader: OpenCV (CPU decoding)")
print()

# ============================================================================
# Helper Functions
# ============================================================================

def get_scope_bbox(frame: np.ndarray):
    """Detect scope bounding box from frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    k25 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k25)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k15)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return cv2.boundingRect(max(cnts, key=cv2.contourArea))

def crop_scope(frame: np.ndarray, bbox) -> np.ndarray:
    """Crop frame to scope bounding box."""
    if bbox is None:
        return frame
    x, y, w, h = bbox
    return frame[y:y+h, x:x+w]

def patient_id_from_path(p: Path) -> str:
    """Extract patient ID from video filename."""
    return p.stem.split("_")[0]

def boxes_to_yolo(r) -> str:
    """Convert detection to YOLO format."""
    confs = r.boxes.conf.cpu().numpy()
    best = int(confs.argmax())
    cls = int(r.boxes.cls[best].cpu().numpy())
    cx, cy, w, h = r.boxes.xywhn[best].cpu().numpy()
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

# ============================================================================
# Video Frame Reader
# ============================================================================

class VideoReader:
    """Efficient video reader with GPU hardware-accelerated decoding (decord) or fallback to OpenCV."""

    def __init__(self, video_path: Path, queue_size: int = 512):
        self.video_path = video_path
        self.queue_size = queue_size
        self.frame_q = queue.Queue(maxsize=queue_size)
        self.read_err = None
        self.total_frames = 0
        self.fps = 30.0
        self.use_decord = HAS_DECORD

        # Get metadata
        if HAS_DECORD:
            try:
                vr = decord.VideoReader(str(video_path))
                self.total_frames = len(vr)
                self.fps = vr.get_avg_fps()
                self.use_decord = True
            except Exception as e:
                print(f"  ⚠️  decord failed, falling back to OpenCV: {e}")
                self.use_decord = False

        if not self.use_decord:
            cap = cv2.VideoCapture(str(video_path))
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

        self.reader_thread = None

    def start(self):
        """Start continuous reader thread."""
        self.reader_thread = threading.Thread(target=self._continuous_reader, daemon=True)
        self.reader_thread.start()

    def _continuous_reader(self):
        """Single thread that continuously reads frames sequentially."""
        try:
            if self.use_decord:
                self._read_with_decord()
            else:
                self._read_with_opencv()
        except Exception as e:
            self.read_err = e
        finally:
            self.frame_q.put(None)  # Signal end

    def _read_with_decord(self):
        """Read frames using decord (GPU hardware decoding)."""
        vr = decord.VideoReader(str(self.video_path))
        for fid in range(len(vr)):
            frame = vr[fid].asnumpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # decord returns RGB, convert to BGR
            self.frame_q.put((fid, frame))

    def _read_with_opencv(self):
        """Read frames using OpenCV (CPU decoding)."""
        cap = cv2.VideoCapture(str(self.video_path))
        fid = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self.frame_q.put((fid, frame))
            fid += 1
        cap.release()

    def get_frame(self, timeout=10):
        """Get next frame from queue."""
        try:
            return self.frame_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def join(self):
        """Wait for reader thread."""
        if self.reader_thread:
            self.reader_thread.join()

# ============================================================================
# Video Inference Pipeline
# ============================================================================

def infer_video(video_path: Path) -> dict:
    """Process single video with CPU-parallel frame reading."""
    pid = patient_id_from_path(video_path)

    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing: {video_path.name}")
    print(f"  Patient ID: {pid}")

    # Initialize reader
    reader = VideoReader(video_path, queue_size=FRAME_QUEUE_SIZE)
    total_frames = reader.total_frames
    fps = reader.fps
    print(f"  Frames: {total_frames} @ {fps:.1f} fps")
    print(f"  Config: Queue: {FRAME_QUEUE_SIZE} | Batch: {BATCH_SIZE}")

    # Detect scope bounding box from first few frames
    cap_temp = cv2.VideoCapture(str(video_path))
    scope_bbox = None
    for _ in range(min(60, total_frames)):
        ok, fr = cap_temp.read()
        if ok:
            scope_bbox = get_scope_bbox(fr)
            if scope_bbox:
                break
    cap_temp.release()

    if scope_bbox:
        bx, by, bw, bh = scope_bbox
        print(f"  Scope: {bw}×{bh} (x={bx}, y={by})")
    else:
        print(f"  WARNING: scope bbox not found — using full frame")

    # Create output directory
    tmp_dir = OUT_ROOT / pid / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Start reading threads
    reader.start()

    # Batch inference
    def flush_batch(batch):
        """Infer batch of frames."""
        imgs = [item[1] for item in batch]
        res = model.predict(source=imgs, conf=CONF_THRESH, imgsz=640, device=DEVICE, verbose=False, stream=True)
        hits = []
        for (fid, orig), r in zip(batch, res):
            if r.boxes is not None and len(r.boxes) > 0:
                hit = (fid, orig, r.plot(), boxes_to_yolo(r))
                hits.append(hit)
        return hits

    def save_detection(fid, orig, annotated, label_txt):
        """Save detection results."""
        stem = f"{pid}_{fid:06d}"
        cv2.imwrite(str(tmp_dir / f"{stem}.jpg"), orig)
        cv2.imwrite(str(tmp_dir / f"{stem}_vis.jpg"), annotated)
        (tmp_dir / f"{stem}.txt").write_text(label_txt)

    # Saver thread pool
    save_pool = ThreadPoolExecutor(max_workers=4)

    # Main inference loop
    infer_buf = []
    pre_buf = []
    post_buf = []
    frame_id = 0
    total_detected = 0
    crossed = False
    saved_count = 0
    LOG_EVERY = max(1000, BATCH_SIZE * 2)

    # Queue monitoring
    import time
    start_time = time.time()
    last_log_time = start_time
    last_log_frames = 0

    pbar = tqdm(total=total_frames, desc=f"  {pid} inference", unit="frame", leave=False)

    while True:
        item = reader.get_frame()
        if item is None:
            break

        fid, frame = item
        if frame is None:
            continue

        # Crop to scope
        cropped = crop_scope(frame, scope_bbox)
        infer_buf.append((fid, cropped))
        frame_id += 1
        pbar.update(1)

        # Queue monitoring
        if frame_id % LOG_EVERY == 0:
            q_size = reader.frame_q.qsize()
            current_time = time.time()
            elapsed = current_time - last_log_time
            fps = (frame_id - last_log_frames) / elapsed if elapsed > 0 else 0
            pct = frame_id / total_frames * 100

            queue_pct = (q_size / FRAME_QUEUE_SIZE) * 100 if FRAME_QUEUE_SIZE > 0 else 0
            queue_status = "⚠️ EMPTY" if q_size < 500 else ("✓ OK" if queue_pct < 80 else "⚠️ FULL")

            # print(f"    [{pid}] {frame_id:>6}/{total_frames} ({pct:5.1f}%) | "
            #       f"Queue: {q_size:>5}/{FRAME_QUEUE_SIZE} ({queue_pct:3.0f}%) {queue_status} | "
            #       f"FPS: {fps:6.1f}")

            last_log_time = current_time
            last_log_frames = frame_id

        # Batch inference
        if len(infer_buf) >= BATCH_SIZE:
            hits = flush_batch(infer_buf)
            infer_buf = []
            total_detected += len(hits)

            for fid, orig, annotated, label_txt in hits:
                if fid < KEYFRAME:
                    pre_buf.append((fid, orig, annotated, label_txt))
                    # Save every REVIEW_EVERY_before
                    while len(pre_buf) >= REVIEW_EVERY_before:
                        f, o, a, l = pre_buf[REVIEW_EVERY_before - 1]
                        save_pool.submit(save_detection, f, o, a, l)
                        saved_count += 1
                        pre_buf = pre_buf[REVIEW_EVERY_before:]
                else:
                    if not crossed:
                        crossed = True
                        if pre_buf:
                            f, o, a, l = pre_buf[-1]
                            save_pool.submit(save_detection, f, o, a, l)
                            saved_count += 1
                            pre_buf = []
                    post_buf.append((fid, orig, annotated, label_txt))
                    while len(post_buf) >= REVIEW_EVERY_after:
                        f, o, a, l = post_buf[REVIEW_EVERY_after - 1]
                        save_pool.submit(save_detection, f, o, a, l)
                        saved_count += 1
                        post_buf = post_buf[REVIEW_EVERY_after:]

    # Final batch
    if infer_buf:
        hits = flush_batch(infer_buf)
        total_detected += len(hits)
        for fid, orig, annotated, label_txt in hits:
            if fid < KEYFRAME:
                pre_buf.append((fid, orig, annotated, label_txt))
            else:
                if not crossed:
                    crossed = True
                    if pre_buf:
                        f, o, a, l = pre_buf[-1]
                        save_pool.submit(save_detection, f, o, a, l)
                        saved_count += 1
                        pre_buf = []
                post_buf.append((fid, orig, annotated, label_txt))

    # Save remaining
    for buf in (pre_buf, post_buf):
        if buf:
            f, o, a, l = buf[-1]
            save_pool.submit(save_detection, f, o, a, l)
            saved_count += 1

    # Wait for saves
    save_pool.shutdown(wait=True)
    reader.join()

    pbar.close()

    if reader.read_err:
        raise RuntimeError(f"[{pid}] Reader error: {reader.read_err}")

    # Final stats
    total_time = time.time() - start_time
    avg_fps = frame_id / total_time if total_time > 0 else 0

    print(f"  ✓ Done: {frame_id}/{total_frames} frames | {total_detected} detected | {saved_count} saved")
    print(f"         Total time: {total_time:.1f}s | Avg FPS: {avg_fps:.1f}")

    return {
        "pid": pid,
        "frames": frame_id,
        "detected": total_detected,
        "saved": saved_count
    }

# ============================================================================
# Main: Process Videos
# ============================================================================

def main():
    # Find videos
    video_exts = ("*.MOV", "*.mov", "*.mp4", "*.MP4", "*.avi", "*.AVI")
    video_files = []
    for ext in video_exts:
        video_files.extend(VIDEO_DIR.glob(ext))
    video_files = sorted(set(video_files))

    if not video_files:
        print(f"❌ No videos found in {VIDEO_DIR}")
        return

    print(f"Found {len(video_files)} video(s)")

    # Filter: skip already processed
    todo = [vp for vp in video_files
            if not (OUT_ROOT / patient_id_from_path(vp) / "tmp").exists()]
    skip = [patient_id_from_path(vp) for vp in video_files if vp not in todo]

    if skip:
        print(f"  → Skipping (already done): {', '.join(sorted(skip))}")

    if not todo:
        print("✓ All videos already processed!")
        return

    print(f"  → To process: {len(todo)} video(s)\n")

    # Process videos sequentially (one at a time)
    summary = []
    for i, video_path in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}]", end=" ")
        try:
            result = infer_video(video_path)
            summary.append(result)
        except Exception as e:
            print(f"❌ ERROR in {video_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'Patient':<12} {'Frames':>10} {'Detected':>10} {'Saved':>10}")
    print("-" * 44)
    for s in sorted(summary, key=lambda x: x["pid"]):
        print(f"{s['pid']:<12} {s['frames']:>10} {s['detected']:>10} {s['saved']:>10}")
    print("-" * 44)
    total_frames = sum(s['frames'] for s in summary)
    total_detected = sum(s['detected'] for s in summary)
    total_saved = sum(s['saved'] for s in summary)
    print(f"{'TOTAL':<12} {total_frames:>10} {total_detected:>10} {total_saved:>10}")

    print(f"\n✓ Done! Now run v3_phase_2.py to label the frames.")

if __name__ == "__main__":
    main()
