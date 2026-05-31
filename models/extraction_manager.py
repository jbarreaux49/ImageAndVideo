"""Downloads and organises ASL video clips from YouTube using MS-ASL metadata."""
import json
import os
import subprocess
from typing import Callable, Dict, List, Optional

import yt_dlp

from utils.config import (
    ASL_VIDEOS_DIR, ERROR_LOG_PATH,
    NUM_CLASSES, TRAIN_JSON, VAL_JSON, TEST_JSON,
)
from utils.logger import get_logger

_log = get_logger("extraction_manager", log_file=ERROR_LOG_PATH)


class _SilentLogger:
    """Redirects yt-dlp debug/warning noise to /dev/null; keeps errors."""
    def debug(self, msg):   pass
    def warning(self, msg): pass
    def error(self, msg):   _log.error(msg)


class ExtractionManager:
    """Groups clips by source URL, downloads each video once, then trims clips."""

    _SPLITS: Dict[str, str] = {
        "train": TRAIN_JSON,
        "val":   VAL_JSON,
        "test":  TEST_JSON,
    }

    def __init__(
        self,
        output_dir:        str   = ASL_VIDEOS_DIR,
        num_classes:       int   = NUM_CLASSES,
        log_callback:      Optional[Callable[[str], None]]         = None,
        progress_callback: Optional[Callable[[float, str], None]]  = None,
    ):
        self.output_dir        = output_dir
        self.num_classes       = num_classes
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

    # ── download helpers ──────────────────────────────────────────────────────

    def _download(self, url: str, out_path: str) -> bool:
        """Download a YouTube video to `out_path` (without extension); return success."""
        opts = {
            "outtmpl":      out_path,
            "format":       "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "quiet":        True,
            "no_warnings":  True,
            "logger":       _SilentLogger(),
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            # yt-dlp may append .mp4 automatically
            return os.path.exists(out_path) or os.path.exists(out_path + ".mp4")
        except yt_dlp.utils.DownloadError as exc:
            self._log(f"Download failed [{url}]: {exc}")
            return False

    def _trim(self, src: str, dst: str, start: float, end: float) -> bool:
        """Trim `src` to [start, end] seconds and write to `dst` via ffmpeg."""
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-i", src,
            "-t",  str(end - start),
            "-c",  "copy",
            "-loglevel", "error",
            dst,
        ]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0 and os.path.exists(dst)

    # ── per-split processing ──────────────────────────────────────────────────

    def _process_split(self, split: str, samples: List[Dict]):
        src_dir = os.path.join(self.output_dir, "source_videos")
        os.makedirs(src_dir, exist_ok=True)

        # Group clips by URL so each video is downloaded once
        by_url: Dict[str, List[Dict]] = {}
        for s in samples:
            if s.get("label", 9999) < self.num_classes:
                by_url.setdefault(s["url"], []).append(s)

        total = len(by_url)
        for i, (url, clips) in enumerate(by_url.items()):
            if self._stop:
                self._log("Extraction stopped by user.")
                return

            video_id = url.split("v=")[-1].split("&")[0]
            # Accept the file with or without .mp4 suffix
            src_mp4  = os.path.join(src_dir, f"{video_id}.mp4")
            src_base = os.path.join(src_dir, video_id)

            done = i + 1
            if not os.path.exists(src_mp4):
                self._progress(i / total, f"{split}  {done}/{total}  downloading…")
                if not self._download(url, src_base):
                    # Skipped counts as processed
                    self._progress(done / total, f"{split}  {done}/{total}  skipped")
                    continue
                if os.path.exists(src_base) and not src_base.endswith(".mp4"):
                    os.rename(src_base, src_mp4)

            actual_src = src_mp4 if os.path.exists(src_mp4) else src_base
            for j, clip in enumerate(clips):
                class_dir = os.path.join(self.output_dir, split, f"class_{clip['label']}")
                os.makedirs(class_dir, exist_ok=True)
                out_clip = os.path.join(class_dir, f"{video_id}_{j}.mp4")
                if not os.path.exists(out_clip):
                    self._trim(actual_src, out_clip, clip["start_time"], clip["end_time"])

            self._progress(done / total, f"{split}  {done} / {total}  videos")

    # ── public API ────────────────────────────────────────────────────────────

    def run(self):
        self._stop = False
        for split, json_path in self._SPLITS.items():
            if self._stop:
                break
            self._log(f"── Split: {split} ──")
            with open(json_path, "r") as f:
                samples = json.load(f)
            self._process_split(split, samples)
        self._log("Extraction complete.")

    def stop(self):
        self._stop = True
