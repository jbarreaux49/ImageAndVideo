"""Tab 1 — Data Pipeline: extraction (left) and preparation (right)."""
import tkinter as tk

from views.base_view import (
    C_BG, C_SURFACE, C_BORDER, C_ACCENT, C_ACCENT2, C_DANGER, C_DIM, C_TEXT,
    FieldRow, SpinRow, SectionHeader,
    btn, label, separator,
)
from utils.config import ASL_VIDEOS_DIR, ASL_FRAMES_DIR, NUM_CLASSES, MAX_FRAMES
from tkinter import ttk


class DataView(tk.Frame):

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", C_BG)
        super().__init__(parent, **kwargs)

        self.ext_start_command  = None
        self.ext_stop_command   = None
        self.prep_start_command = None
        self.prep_stop_command  = None

        self._build_ui()

    def _build_ui(self):
        bg = self["bg"]

        inner = tk.Frame(self, bg=C_SURFACE)
        inner.pack(fill="both", expand=True, padx=14, pady=14)

        self._build_extraction(inner)
        separator(inner, bg=C_BORDER).pack(fill="x", padx=14, pady=8)
        self._build_preparation(inner)

    # ── extraction ────────────────────────────────────────────────────────────

    def _build_extraction(self, parent):
        bg = parent["bg"]
        pad = {"padx": 14, "pady": 3}

        SectionHeader(parent, "① Extract Videos", color=C_ACCENT, bg=bg).pack(
            fill="x", padx=14, pady=(14, 6))
        label(parent, "Downloads ASL clips from YouTube then trims with ffmpeg.",
              dim=True, size=8, bg=bg).pack(anchor="w", padx=14, pady=(0, 10))

        self.ext_output_dir  = FieldRow(parent, "Output dir:", ASL_VIDEOS_DIR, browse="dir", bg=bg)
        self.ext_num_classes = SpinRow(parent, "Num classes:", NUM_CLASSES, lo=1, hi=1000, bg=bg)
        self.ext_output_dir.pack(anchor="w", **pad)
        self.ext_num_classes.pack(anchor="w", **pad)

        # cookies.txt file (Netscape format — export with "Get cookies.txt LOCALLY" extension)
        self.ext_cookies_file = FieldRow(parent, "cookies.txt:", "", browse="file", bg=bg)
        self.ext_cookies_file.pack(anchor="w", **pad)

        # Fallback: read cookies directly from a browser (may fail if browser is open)
        cookie_row = tk.Frame(parent, bg=bg)
        cookie_row.pack(anchor="w", **pad)
        label(cookie_row, "Or browser:", bg=bg, size=9).pack(side="left", padx=(0, 6))
        self.ext_cookies_browser = tk.StringVar(value="none")
        ttk.Combobox(
            cookie_row,
            textvariable=self.ext_cookies_browser,
            values=["none", "chrome", "firefox", "edge", "brave", "opera", "chromium"],
            state="readonly", width=12,
        ).pack(side="left")

        separator(parent, bg=C_BORDER).pack(fill="x", padx=14, pady=10)

        row = tk.Frame(parent, bg=bg)
        row.pack(anchor="w", padx=14)
        self.ext_start_btn = btn(row, "▶  Start", command=self._ext_start, color=C_ACCENT, width=12)
        self.ext_start_btn.pack(side="left", padx=(0, 6))
        self.ext_stop_btn  = btn(row, "■  Stop",  command=self._ext_stop,  color=C_DANGER, disabled=True, width=8)
        self.ext_stop_btn.pack(side="left")

        # Progress bar
        self._ext_bar = ttk.Progressbar(parent, mode="determinate", maximum=100)
        self._ext_bar.pack(fill="x", padx=14, pady=(12, 4))

        # Counter label
        self.ext_status = label(parent, "—", dim=True, size=9, bg=bg, anchor="w")
        self.ext_status.pack(fill="x", padx=14)

    # ── preparation ───────────────────────────────────────────────────────────

    def _build_preparation(self, parent):
        bg = parent["bg"]
        pad = {"padx": 14, "pady": 3}

        SectionHeader(parent, "② Prepare Frames", color=C_ACCENT2, bg=bg).pack(
            fill="x", padx=14, pady=(14, 6))
        label(parent, "Samples frames from each clip and saves 4 augmentation variants.",
              dim=True, size=8, bg=bg).pack(anchor="w", padx=14, pady=(0, 10))

        self.prep_videos_dir  = FieldRow(parent, "Videos dir:", ASL_VIDEOS_DIR, browse="dir", bg=bg)
        self.prep_frames_dir  = FieldRow(parent, "Frames dir:", ASL_FRAMES_DIR, browse="dir", bg=bg)
        self.prep_num_classes = SpinRow(parent, "Num classes:", NUM_CLASSES, lo=1, hi=1000, bg=bg)
        self.prep_max_frames  = SpinRow(parent, "Max frames:", MAX_FRAMES, lo=8, hi=128, bg=bg)
        self.prep_videos_dir.pack(anchor="w", **pad)
        self.prep_frames_dir.pack(anchor="w", **pad)
        self.prep_num_classes.pack(anchor="w", **pad)
        self.prep_max_frames.pack(anchor="w", **pad)

        separator(parent, bg=C_BORDER).pack(fill="x", padx=14, pady=10)

        row = tk.Frame(parent, bg=bg)
        row.pack(anchor="w", padx=14)
        self.prep_start_btn = btn(row, "▶  Start", command=self._prep_start, color=C_ACCENT2, width=12)
        self.prep_start_btn.pack(side="left", padx=(0, 6))
        self.prep_stop_btn  = btn(row, "■  Stop",  command=self._prep_stop,  color=C_DANGER, disabled=True, width=8)
        self.prep_stop_btn.pack(side="left")

        self._prep_bar = ttk.Progressbar(parent, mode="determinate", maximum=100)
        self._prep_bar.pack(fill="x", padx=14, pady=(12, 4))

        self.prep_status = label(parent, "—", dim=True, size=9, bg=bg, anchor="w")
        self.prep_status.pack(fill="x", padx=14)

    # ── event forwarders ──────────────────────────────────────────────────────

    def _ext_start(self):
        if self.ext_start_command:
            self.ext_start_command()

    def _ext_stop(self):
        if self.ext_stop_command:
            self.ext_stop_btn.configure(state="disabled", text="⏳ …")
            self.ext_stop_command()

    def _prep_start(self):
        if self.prep_start_command:
            self.prep_start_command()

    def _prep_stop(self):
        if self.prep_stop_command:
            self.prep_stop_btn.configure(state="disabled", text="⏳ …")
            self.prep_stop_command()

    # ── state helpers ─────────────────────────────────────────────────────────

    def set_ext_running(self, running: bool):
        self.ext_start_btn.configure(state="disabled" if running else "normal")
        self.ext_stop_btn.configure(state="normal" if running else "disabled", text="■  Stop")
        if not running:
            self._ext_bar["value"] = 0

    def set_prep_running(self, running: bool):
        self.prep_start_btn.configure(state="disabled" if running else "normal")
        self.prep_stop_btn.configure(state="normal" if running else "disabled", text="■  Stop")
        if not running:
            self._prep_bar["value"] = 0

    def update_ext_progress(self, fraction: float, status: str):
        self._ext_bar["value"] = fraction * 100
        self.ext_status.configure(text=status)

    def update_prep_progress(self, fraction: float, status: str):
        self._prep_bar["value"] = fraction * 100
        self.prep_status.configure(text=status)
