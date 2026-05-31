"""Extracts uniformly-sampled frames from video clips and saves augmented copies."""
import os
import random
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from utils.config import (
    ASL_FRAMES_DIR, ASL_VIDEOS_DIR,
    FRAME_SIZE, MAX_FRAMES, SHORT_SIDE_SIZE,
)
from utils.logger import get_logger

_log = get_logger("preparation_manager")


class PreparationManager:
    """For every mp4 clip, writes 4 augmentation sub-directories of JPEG frames:
    original / flipped / brightened / contrasted.
    """

    _SPLITS = ("train", "val", "test")

    def __init__(
        self,
        videos_dir:        str   = ASL_VIDEOS_DIR,
        frames_dir:        str   = ASL_FRAMES_DIR,
        num_classes:       int   = 100,
        max_frames:        int   = MAX_FRAMES,
        log_callback:      Optional[Callable[[str], None]]        = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        self.videos_dir        = videos_dir
        self.frames_dir        = frames_dir
        self.num_classes       = num_classes
        self.max_frames        = max_frames
        self.log_callback      = log_callback
        self.progress_callback = progress_callback
        self._stop             = False

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        _log.info(msg)
        if self.log_callback:
            self.log_callback(msg)

    def _progress(self, value: float, msg: str = ""):
        if self.progress_callback:
            self.progress_callback(value, msg)

    # ── frame utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _uniform_indices(total: int, n: int) -> List[int]:
        """Return n evenly-spaced frame indices within [0, total)."""
        if total <= n:
            return list(range(total))
        step = total / n
        return [int(i * step) for i in range(n)]

    @staticmethod
    def _resize_short_side(frame: np.ndarray, size: int) -> np.ndarray:
        h, w = frame.shape[:2]
        if h <= w:
            new_h, new_w = size, int(w * size / h)
        else:
            new_h, new_w = int(h * size / w), size
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _center_crop(frame: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
        h, w = frame.shape[:2]
        y = max((h - crop_h) // 2, 0)
        x = max((w - crop_w) // 2, 0)
        return frame[y: y + crop_h, x: x + crop_w]

    @staticmethod
    def _augment(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (flipped, brightened, contrasted) augmentations."""
        flipped     = cv2.flip(frame, 1)
        factor      = random.uniform(0.5, 1.5)
        brightened  = np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        mean        = float(frame.mean())
        contrasted  = np.clip(
            (frame.astype(np.float32) - mean) * factor + mean, 0, 255
        ).astype(np.uint8)
        return flipped, brightened, contrasted

    # ── per-video extraction ──────────────────────────────────────────────────

    def _extract_video(self, video_path: str, out_dir: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"Cannot open: {video_path}")
            return

        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = set(self._uniform_indices(total, self.max_frames))

        raw_frames: List[np.ndarray] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in indices:
                frame = self._resize_short_side(frame, SHORT_SIDE_SIZE)
                frame = self._center_crop(frame, FRAME_SIZE[0], FRAME_SIZE[1])
                raw_frames.append(frame)
            idx += 1
        cap.release()

        if not raw_frames:
            return

        aug_map = {
            "original":   raw_frames,
            "flipped":    [],
            "brightened": [],
            "contrasted": [],
        }
        for f in raw_frames:
            fl, br, co = self._augment(f)
            aug_map["flipped"].append(fl)
            aug_map["brightened"].append(br)
            aug_map["contrasted"].append(co)

        for aug_name, frames in aug_map.items():
            aug_path = os.path.join(out_dir, aug_name)
            os.makedirs(aug_path, exist_ok=True)
            for i, frame in enumerate(frames):
                cv2.imwrite(os.path.join(aug_path, f"frame_{i:04d}.jpg"), frame)

    # ── public API ────────────────────────────────────────────────────────────

    def run(self):
        self._stop = False
        for split in self._SPLITS:
            if self._stop:
                break
            split_dir = os.path.join(self.videos_dir, split)
            if not os.path.isdir(split_dir):
                self._log(f"Split directory not found: {split_dir}")
                continue

            # Collect all mp4 paths
            video_files: List[Tuple[str, str]] = []
            for class_dir in os.listdir(split_dir):
                class_path = os.path.join(split_dir, class_dir)
                if not os.path.isdir(class_path):
                    continue
                try:
                    label = int(class_dir.split("_")[1])
                except (IndexError, ValueError):
                    label = 0
                if label >= self.num_classes:
                    continue
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(".mp4"):
                        video_files.append((class_dir, fname))

            total = len(video_files)
            self._log(f"[{split}] {total} videos to process")

            for i, (class_dir, fname) in enumerate(video_files):
                if self._stop:
                    break
                video_path = os.path.join(split_dir, class_dir, fname)
                stem       = os.path.splitext(fname)[0]
                out_dir    = os.path.join(self.frames_dir, split, class_dir, stem)
                os.makedirs(out_dir, exist_ok=True)

                # Skip if already extracted
                if not os.path.exists(os.path.join(out_dir, "original")):
                    self._extract_video(video_path, out_dir)

                self._progress((i + 1) / max(total, 1), f"{split}  {i + 1} / {total}  clips")

        self._log("Frame extraction complete.")

    def stop(self):
        self._stop = True
