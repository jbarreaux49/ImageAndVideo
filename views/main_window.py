"""Single scrollable page — pure white theme."""
import tkinter as tk
from tkinter import ttk

from views.data_view     import DataView
from views.training_view import TrainingView
from views.analysis_view import AnalysisView
from views.base_view     import C_BG, C_SURFACE, C_BORDER, C_DIM, C_ACCENT, C_TEXT


class MainWindow(tk.Tk):
    """Root window: fixed header + vertically scrollable content area."""

    def __init__(self):
        print("[window] building UI...", flush=True)
        super().__init__()
        self.title("ASL Sign Language Recognition")
        self.geometry("1100x860")
        self.minsize(920, 620)
        self.configure(bg=C_BG)

        self._apply_theme()
        self._build_header()

        print("[window] creating scroll area...", flush=True)
        self._build_scroll_area()
        print("[window] UI ready.", flush=True)

    # ── theme ─────────────────────────────────────────────────────────────────

    def _apply_theme(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TProgressbar",
                    troughcolor=C_BORDER, background=C_ACCENT, thickness=3)
        s.configure("TScrollbar",
                    background=C_SURFACE, troughcolor=C_BG,
                    arrowcolor=C_DIM, borderwidth=0)
        s.configure("Treeview",
                    background=C_BG, foreground=C_TEXT,
                    fieldbackground=C_BG, rowheight=22, font=("Segoe UI", 8))
        s.configure("Treeview.Heading",
                    background=C_SURFACE, foreground=C_DIM,
                    font=("Segoe UI", 8, "bold"), relief="flat")
        s.map("Treeview", background=[("selected", "#dbeafe")],
              foreground=[("selected", C_ACCENT)])

    # ── header ────────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self, bg=C_BG)
        hdr.pack(fill="x")
        # Bottom border
        tk.Frame(hdr, bg=C_BORDER, height=1).pack(fill="x", side="bottom")

        inner = tk.Frame(hdr, bg=C_BG)
        inner.pack(fill="x", padx=20, pady=12)

        tk.Label(inner, text="ASL Recognition",
                 bg=C_BG, fg=C_TEXT,
                 font=("Segoe UI", 14, "bold")).pack(side="left")
        tk.Label(inner, text="R3D-18  ·  MS-ASL  ·  PyTorch",
                 bg=C_BG, fg=C_DIM,
                 font=("Segoe UI", 8)).pack(side="right", pady=5)

    # ── scrollable area ───────────────────────────────────────────────────────

    def _build_scroll_area(self):
        container = tk.Frame(self, bg=C_BG)
        container.pack(fill="both", expand=True)

        self._cv = tk.Canvas(container, bg=C_BG, highlightthickness=0)
        sb = ttk.Scrollbar(container, orient="vertical", command=self._cv.yview)
        self._cv.configure(yscrollcommand=sb.set)

        sb.pack(side="right", fill="y")
        self._cv.pack(side="left", fill="both", expand=True)

        self._inner = tk.Frame(self._cv, bg=C_BG)
        self._win_id = self._cv.create_window((0, 0), window=self._inner, anchor="nw")

        self._cv.bind("<Configure>",
                      lambda e: self._cv.itemconfig(self._win_id, width=e.width))
        self._inner.bind("<Configure>",
                         lambda e: self._cv.configure(scrollregion=self._cv.bbox("all")))
        self._main_scroll = lambda: self._cv.bind_all(
            "<MouseWheel>", lambda e: self._cv.yview_scroll(-1*(e.delta//120), "units"))
        self._main_scroll()

        self._build_sections(self._inner)

    # ── sections ──────────────────────────────────────────────────────────────

    def _build_sections(self, parent):
        P = dict(fill="x", padx=24)

        # ① Extract & Prepare
        print("[window] building data section...", flush=True)
        self._section_label(parent, "Extract & Prepare").pack(**P, pady=(22, 8))
        data_wrap = tk.Frame(parent, bg=C_BG)
        data_wrap.pack(**P, pady=(0, 4))
        self.data_view = DataView(data_wrap, bg=C_BG)
        self.data_view.pack(fill="both", expand=True)

        self._hr(parent).pack(**P, pady=18)

        # ② Train & Search
        print("[window] building training section...", flush=True)
        self._section_label(parent, "Train & Search").pack(**P, pady=(0, 8))
        train_wrap = tk.Frame(parent, bg=C_BG)
        train_wrap.pack(**P, pady=(0, 4))
        self.training_view = TrainingView(train_wrap, bg=C_BG)
        self.training_view.pack(fill="both", expand=True)
        self.training_view.set_page_scroll_fn(self._main_scroll)

        self._hr(parent).pack(**P, pady=18)

        # ③ Analysis
        print("[window] building analysis section...", flush=True)
        self._section_label(parent, "Analysis").pack(**P, pady=(0, 8))
        analysis_wrap = tk.Frame(parent, bg=C_BG)
        analysis_wrap.pack(**P, pady=(0, 28))
        self.analysis_view = AnalysisView(analysis_wrap, bg=C_BG)
        self.analysis_view.pack(fill="both", expand=True)

    @staticmethod
    def _section_label(parent, text: str) -> tk.Label:
        return tk.Label(parent, text=text,
                        bg=parent["bg"], fg=C_TEXT,
                        font=("Segoe UI", 12, "bold"), anchor="w")

    @staticmethod
    def _hr(parent) -> tk.Frame:
        return tk.Frame(parent, bg=C_BORDER, height=1)
