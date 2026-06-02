"""Tab 2 — Train & Search: one view, mode toggle switches left panel + right panel."""
import tkinter as tk
from tkinter import ttk
from typing import Dict, List

import matplotlib.ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from views.base_view import (
    C_BG, C_SURFACE, C_SURFACE2, C_INPUT, C_BORDER, C_TEXT, C_DIM,
    C_ACCENT, C_ACCENT2, C_DANGER, C_SUCCESS,
    LogPanel, FieldRow, SpinRow, FloatRow, ComboRow, ProgressBar,
    btn, label, separator,
)
from utils.config import (
    ASL_FRAMES_DIR, BATCH_SIZE, DROPOUT_RATE,
    LEARNING_RATE, MAX_FRAMES, MODEL_SAVE_PATH,
    NUM_CLASSES, NUM_EPOCHS,
)

_PLT_BG    = "#f8fafc"
_PLT_SURF  = "#ffffff"
_COL_TRAIN = "#2563eb"
_COL_VAL   = "#d97706"


class TrainingView(tk.Frame):
    """Single view: a mode toggle (Train / Search) drives which form and
    which right-side panel is visible."""

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", C_BG)
        super().__init__(parent, **kwargs)

        self.train_start_command  = None
        self.train_stop_command   = None
        self.search_start_command = None
        self.search_stop_command  = None
        self._page_scroll_fn      = None

        self._build_ui()

    def set_page_scroll_fn(self, fn):
        self._page_scroll_fn = fn

    # ── top-level layout ──────────────────────────────────────────────────────

    def _build_ui(self):
        bg = self["bg"]

        body = tk.Frame(self, bg=bg)
        body.pack(fill="both", expand=True, padx=10, pady=10)

        # ── Left panel (scrollable) ───────────────────────────────────────────
        left_outer = tk.Frame(body, bg=bg, width=330)
        left_outer.pack(side="left", fill="y")
        left_outer.pack_propagate(False)

        left_cv = tk.Canvas(left_outer, bg=bg, highlightthickness=0, width=310)
        left_sb = ttk.Scrollbar(left_outer, orient="vertical", command=left_cv.yview)
        left_cv.configure(yscrollcommand=left_sb.set)
        left_sb.pack(side="right", fill="y")
        left_cv.pack(side="left", fill="both", expand=True)

        left = tk.Frame(left_cv, bg=bg)
        left_win = left_cv.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", lambda e: left_cv.configure(
            scrollregion=left_cv.bbox("all")))
        left_cv.bind("<Configure>", lambda e: left_cv.itemconfig(
            left_win, width=e.width))
        def _enable_left(e=None):
            left_cv.bind_all("<MouseWheel>",
                             lambda ev: left_cv.yview_scroll(-1*(ev.delta//120), "units"))

        def _restore_page(e=None):
            if self._page_scroll_fn:
                self._page_scroll_fn()

        left_cv.bind("<Enter>", _enable_left)
        left_cv.bind("<Leave>", _restore_page)

        # Mode toggle at the top of the left panel
        self._build_mode_toggle(left)
        separator(left, bg=C_BORDER).pack(fill="x", pady=6)

        # Two forms stacked; only one is shown at a time
        self._train_form  = tk.Frame(left, bg=bg)
        self._search_form = tk.Frame(left, bg=bg)
        self._build_train_form(self._train_form)
        self._build_search_form(self._search_form)

        # ── Divider ───────────────────────────────────────────────────────────
        tk.Frame(body, bg=C_BORDER, width=1).pack(side="left", fill="y", padx=8)

        # ── Right panel ───────────────────────────────────────────────────────
        right = tk.Frame(body, bg=bg)
        right.pack(side="left", fill="both", expand=True)

        self._chart_panel  = tk.Frame(right, bg=bg)
        self._result_panel = tk.Frame(right, bg=bg)
        self._build_chart_panel(self._chart_panel)
        self._build_result_panel(self._result_panel)

        for panel in (right, self._chart_panel, self._result_panel):
            panel.bind("<Enter>", lambda e: self._page_scroll_fn() if self._page_scroll_fn else None)

        # Start in Train mode
        self._set_mode("train")

    # ── Mode toggle ───────────────────────────────────────────────────────────

    def _build_mode_toggle(self, parent):
        bg = parent["bg"]
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", padx=10, pady=(10, 0))

        label(row, "Mode", dim=True, size=8, bg=bg).pack(side="left", padx=(0, 10))

        self._mode_train_btn  = self._toggle_btn(row, "Train",  lambda: self._set_mode("train"))
        self._mode_search_btn = self._toggle_btn(row, "Search", lambda: self._set_mode("search"))
        self._mode_train_btn.pack( side="left")
        self._mode_search_btn.pack(side="left")

    @staticmethod
    def _toggle_btn(parent, text: str, command) -> tk.Button:
        return tk.Button(parent, text=text, command=command,
                         bg=C_SURFACE2, fg=C_DIM, relief="flat",
                         font=("Segoe UI", 9), padx=14, pady=4,
                         cursor="hand2", activebackground=C_SURFACE2)

    def _set_mode(self, mode: str):
        self._mode = mode

        # Toggle button styles
        active_bg, active_fg   = C_ACCENT,   "#ffffff"
        inactive_bg, inactive_fg = C_SURFACE2, C_DIM

        if mode == "train":
            self._mode_train_btn.configure( bg=active_bg,   fg=active_fg)
            self._mode_search_btn.configure(bg=inactive_bg, fg=inactive_fg)
            self._search_form.pack_forget()
            self._train_form.pack(fill="both", expand=True)
            self._result_panel.pack_forget()
            self._chart_panel.pack(fill="both", expand=True)
        else:
            self._mode_train_btn.configure( bg=inactive_bg, fg=inactive_fg)
            self._mode_search_btn.configure(bg=active_bg,   fg=active_fg)
            self._train_form.pack_forget()
            self._search_form.pack(fill="both", expand=True)
            self._chart_panel.pack_forget()
            self._result_panel.pack(fill="both", expand=True)

    # ── Train form ────────────────────────────────────────────────────────────

    def _build_train_form(self, parent):
        bg = parent["bg"]
        pad = {"anchor": "w", "padx": 10, "pady": 2}

        self.tr_frames_dir = FieldRow(parent, "Frames dir:", ASL_FRAMES_DIR, browse="dir", bg=bg)
        self.tr_model_path = FieldRow(parent, "Model path:", MODEL_SAVE_PATH, browse="save", bg=bg)
        self.tr_frames_dir.pack(**pad)
        self.tr_model_path.pack(**pad)

        separator(parent, bg=C_BORDER).pack(fill="x", padx=10, pady=6)

        self.tr_num_classes    = SpinRow(parent, "Num classes:",     NUM_CLASSES,  1, 1000, bg=bg)
        self.tr_batch_size     = SpinRow(parent, "Batch size:",       BATCH_SIZE,   1, 64,   bg=bg)
        self.tr_max_frames     = SpinRow(parent, "Max frames:",       64,           4, 128,  bg=bg)
        self.tr_steps_epoch    = SpinRow(parent, "Steps/epoch:",      0,            0, 9999, bg=bg)
        self.tr_num_epochs     = SpinRow(parent, "Epochs:",           NUM_EPOCHS,   1, 500,  bg=bg)
        self.tr_lr             = FloatRow(parent, "Learning rate:",   LEARNING_RATE, bg=bg)
        self.tr_dropout        = FloatRow(parent, "Dropout rate:",    DROPOUT_RATE,  bg=bg)
        self.tr_optimizer      = ComboRow(parent, "Optimizer:",
                                          ["Adam", "AdamW", "SGD", "RMSprop"],
                                          default="Adam", bg=bg)
        self.tr_loss_fn        = ComboRow(parent, "Loss function:",
                                          ["CrossEntropy (weighted)", "CrossEntropy",
                                           "LabelSmoothing", "Focal Loss"],
                                          default="CrossEntropy (weighted)", bg=bg)
        self.tr_regularization = ComboRow(parent, "Regularization:",
                                          ["None", "L2", "L1", "L1+L2"],
                                          default="L2", bg=bg)
        self.tr_reg_strength   = FloatRow(parent, "Reg. strength:", 1e-4, bg=bg)
        for w in (self.tr_num_classes, self.tr_batch_size, self.tr_max_frames,
                  self.tr_steps_epoch, self.tr_num_epochs, self.tr_lr, self.tr_dropout,
                  self.tr_optimizer, self.tr_loss_fn, self.tr_regularization,
                  self.tr_reg_strength):
            w.pack(**pad)

        separator(parent, bg=C_BORDER).pack(fill="x", padx=10, pady=6)

        row = tk.Frame(parent, bg=bg)
        row.pack(anchor="w", padx=10, pady=2)
        self.tr_start_btn = btn(row, "▶  Start Training", command=self._tr_start,
                                 color=C_ACCENT, width=16)
        self.tr_start_btn.pack(side="left", padx=(0, 6))
        self.tr_stop_btn = btn(row, "■  Stop", command=self._tr_stop,
                                color=C_DANGER, disabled=True, width=8)
        self.tr_stop_btn.pack(side="left")

        self.tr_progress = ProgressBar(parent, bg=bg)
        self.tr_progress.pack(fill="x", padx=10, pady=4)

        separator(parent, bg=C_BORDER).pack(fill="x", padx=10, pady=4)
        label(parent, "Log", dim=True, size=8, bg=bg).pack(anchor="w", padx=10)
        self.tr_log = LogPanel(parent, height=9, bg=bg)
        self.tr_log.pack(fill="both", expand=True, padx=10, pady=(2, 10))

    # ── Search form ───────────────────────────────────────────────────────────

    def _build_search_form(self, parent):
        bg = parent["bg"]
        pad = {"anchor": "w", "padx": 10, "pady": 2}

        self.sr_frames_dir  = FieldRow(parent, "Frames dir:",   ASL_FRAMES_DIR,   browse="dir", bg=bg)
        self.sr_save_dir    = FieldRow(parent, "Results dir:",  "search_results", browse="dir", bg=bg)
        self.sr_num_classes = SpinRow(parent, "Num classes:",   NUM_CLASSES, 1, 1000, bg=bg)
        self.sr_max_frames  = SpinRow(parent, "Max frames:",    64,          8, 128,  bg=bg)
        self.sr_epochs      = SpinRow(parent, "Epochs/trial:",  3,           1, 50,   bg=bg)
        self.sr_steps_epoch = SpinRow(parent, "Steps/epoch:",   0,           0, 9999, bg=bg)
        self.sr_n_trials    = SpinRow(parent, "Max trials:",    6,           1, 100,  bg=bg)
        self.sr_optimizer      = ComboRow(parent, "Optimizer:",
                                          ["Adam", "AdamW", "SGD", "RMSprop"],
                                          default="Adam", bg=bg)
        self.sr_loss_fn        = ComboRow(parent, "Loss function:",
                                          ["CrossEntropy (weighted)", "CrossEntropy",
                                           "LabelSmoothing", "Focal Loss"],
                                          default="CrossEntropy (weighted)", bg=bg)
        self.sr_regularization = ComboRow(parent, "Regularization:",
                                          ["None", "L2", "L1", "L1+L2"],
                                          default="L2", bg=bg)
        self.sr_reg_strength   = FloatRow(parent, "Reg. strength:", 1e-4, bg=bg)
        for w in (self.sr_frames_dir, self.sr_save_dir, self.sr_num_classes,
                  self.sr_max_frames, self.sr_epochs, self.sr_steps_epoch,
                  self.sr_n_trials, self.sr_optimizer, self.sr_loss_fn,
                  self.sr_regularization, self.sr_reg_strength):
            w.pack(**pad)

        separator(parent, bg=C_BORDER).pack(fill="x", padx=10, pady=6)

        # Mode radio
        mode_row = tk.Frame(parent, bg=bg)
        mode_row.pack(anchor="w", padx=10, pady=2)
        label(mode_row, "Strategy:", dim=True, size=8, bg=bg).pack(side="left", padx=(0, 8))
        self.sr_mode = tk.StringVar(value="random")
        for val, txt in (("random", "Random"), ("grid", "Grid")):
            tk.Radiobutton(mode_row, text=txt, variable=self.sr_mode, value=val,
                           bg=bg, fg=C_TEXT, selectcolor=C_SURFACE2,
                           activebackground=bg, font=("Segoe UI", 9)
                           ).pack(side="left", padx=(0, 10))

        separator(parent, bg=C_BORDER).pack(fill="x", padx=10, pady=6)

        label(parent, "Parameter values  (comma-separated)", dim=True, size=8, bg=bg
              ).pack(anchor="w", padx=10, pady=(0, 4))
        self.sr_lr_vals      = FieldRow(parent, "LR values:",      "1e-4,5e-5,1e-5", bg=bg)
        self.sr_dropout_vals = FieldRow(parent, "Dropout values:", "0.3,0.5,0.7",    bg=bg)
        self.sr_bs_vals      = FieldRow(parent, "Batch sizes:",    "4,8",             bg=bg)
        self.sr_lr_vals.pack(**pad)
        self.sr_dropout_vals.pack(**pad)
        self.sr_bs_vals.pack(**pad)

        separator(parent, bg=C_BORDER).pack(fill="x", padx=10, pady=6)

        row = tk.Frame(parent, bg=bg)
        row.pack(anchor="w", padx=10, pady=2)
        self.sr_start_btn = btn(row, "▶  Run Search", command=self._sr_start,
                                 color=C_ACCENT2, width=14)
        self.sr_start_btn.pack(side="left", padx=(0, 6))
        self.sr_stop_btn = btn(row, "■  Stop", command=self._sr_stop,
                                color=C_DANGER, disabled=True, width=8)
        self.sr_stop_btn.pack(side="left")

        self.sr_progress = ProgressBar(parent, bg=bg)
        self.sr_progress.pack(fill="x", padx=10, pady=4)

        label(parent, "Log", dim=True, size=8, bg=bg).pack(anchor="w", padx=10)
        self.sr_log = LogPanel(parent, height=8, bg=bg)
        self.sr_log.pack(fill="both", expand=True, padx=10, pady=(2, 10))

    # ── Right panel: training charts ──────────────────────────────────────────

    def _build_chart_panel(self, parent):
        bg = parent["bg"]
        label(parent, "Live metrics", dim=True, size=8, bg=bg).pack(
            anchor="w", padx=10, pady=(10, 4))

        self._tr_fig  = Figure(figsize=(5, 5.2), dpi=80, facecolor=_PLT_SURF)
        self._ax_loss = self._tr_fig.add_subplot(211)
        self._ax_acc  = self._tr_fig.add_subplot(212)
        for ax in (self._ax_loss, self._ax_acc):
            ax.set_facecolor(_PLT_BG)
            ax.tick_params(colors="#64748b", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#e2e8f0")
        self._ax_loss.set_title("Loss",     color="#475569", fontsize=8, pad=4)
        self._ax_acc.set_title( "Accuracy", color="#475569", fontsize=8, pad=4)
        self._tr_fig.tight_layout(pad=2.5)

        self._tr_canvas = FigureCanvasTkAgg(self._tr_fig, master=parent)
        self._tr_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 10))

    # ── Right panel: search results ───────────────────────────────────────────

    def _build_result_panel(self, parent):
        bg = parent["bg"]
        label(parent, "Search results", dim=True, size=8, bg=bg).pack(
            anchor="w", padx=10, pady=(10, 4))

        cols   = ("trial", "lr", "dropout", "batch", "best_val_acc", "epochs")
        hdrs   = ("Trial", "LR", "Dropout", "Batch", "Best Val Acc", "Epochs")
        widths = (50, 85, 70, 55, 100, 60)
        self.sr_tree = ttk.Treeview(parent, columns=cols, show="headings", height=8)
        for c, h, w in zip(cols, hdrs, widths):
            self.sr_tree.heading(c, text=h)
            self.sr_tree.column(c, width=w, anchor="center")
        self.sr_tree.pack(fill="x", padx=10, pady=(0, 6))

        self._sr_fig = Figure(figsize=(5, 3.2), dpi=80, facecolor=_PLT_SURF)
        self._sr_ax  = self._sr_fig.add_subplot(111)
        self._sr_ax.set_facecolor(_PLT_BG)
        self._sr_ax.tick_params(colors="#64748b", labelsize=7)
        for sp in self._sr_ax.spines.values():
            sp.set_edgecolor("#e2e8f0")
        self._sr_ax.set_title("Trial comparison", color="#475569", fontsize=8)
        self._sr_fig.tight_layout(pad=2.5)

        self._sr_canvas = FigureCanvasTkAgg(self._sr_fig, master=parent)
        self._sr_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 10))

    # ── event forwarders ──────────────────────────────────────────────────────

    def _tr_start(self):
        if self.train_start_command:
            self.train_start_command()

    def _tr_stop(self):
        if self.train_stop_command:
            self.tr_stop_btn.configure(state="disabled", text="⏳ …")
            self.tr_log.append("Stop requested — finishing current batch…")
            self.train_stop_command()

    def _sr_start(self):
        if self.search_start_command:
            self.search_start_command()

    def _sr_stop(self):
        if self.search_stop_command:
            self.sr_stop_btn.configure(state="disabled", text="⏳ …")
            self.sr_log.append("Stop requested…")
            self.search_stop_command()

    # ── state helpers ─────────────────────────────────────────────────────────

    def set_train_running(self, running: bool):
        self.tr_start_btn.configure(state="disabled" if running else "normal")
        self.tr_stop_btn.configure(state="normal" if running else "disabled", text="■  Stop")

    def set_search_running(self, running: bool):
        self.sr_start_btn.configure(state="disabled" if running else "normal")
        self.sr_stop_btn.configure(state="normal" if running else "disabled", text="■  Stop")

    def update_charts(self, history: List[Dict]):
        if not history:
            return
        epochs     = [r["epoch"]      for r in history]
        train_loss = [r["train_loss"] for r in history]
        val_loss   = [r["val_loss"]   for r in history]
        train_acc  = [r["train_acc"]  for r in history]
        val_acc    = [r["val_acc"]    for r in history]

        for ax in (self._ax_loss, self._ax_acc):
            ax.clear()
            ax.set_facecolor(_PLT_BG)
            ax.tick_params(colors="#64748b", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#e2e8f0")

        self._ax_loss.plot(epochs, train_loss, color=_COL_TRAIN, linewidth=1.5, label="train")
        self._ax_loss.plot(epochs, val_loss,   color=_COL_VAL,   linewidth=1.5, label="val")
        self._ax_loss.set_title("Loss", color="#475569", fontsize=8, pad=4)
        self._ax_loss.set_xlabel("Epoch", color="#64748b", fontsize=7)
        self._ax_loss.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        self._ax_loss.legend(fontsize=7, facecolor=_PLT_SURF, labelcolor="#475569")

        self._ax_acc.plot(epochs, train_acc, color=_COL_TRAIN, linewidth=1.5, label="train")
        self._ax_acc.plot(epochs, val_acc,   color=_COL_VAL,   linewidth=1.5, label="val")
        self._ax_acc.set_title("Accuracy", color="#475569", fontsize=8, pad=4)
        self._ax_acc.set_xlabel("Epoch", color="#64748b", fontsize=7)
        self._ax_acc.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        self._ax_acc.legend(fontsize=7, facecolor=_PLT_SURF, labelcolor="#475569")
        self._ax_acc.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0%}"))

        self._tr_fig.tight_layout(pad=2.5)
        self._tr_canvas.draw()   # force immediate redraw, not deferred

    def update_search_results(self, results: List[Dict]):
        for row in self.sr_tree.get_children():
            self.sr_tree.delete(row)
        for r in results:
            self.sr_tree.insert("", "end", values=(
                r.get("trial", ""),
                f"{r.get('learning_rate', 0):.2e}",
                f"{r.get('dropout_rate', 0):.2f}",
                r.get("batch_size", ""),
                f"{r.get('best_val_acc', 0):.2%}",
                r.get("epochs_run", ""),
            ))

        self._sr_ax.clear()
        self._sr_ax.set_facecolor(_PLT_BG)
        self._sr_ax.tick_params(colors="#666666", labelsize=7)
        trials = [r.get("trial", i+1) for i, r in enumerate(results)]
        accs   = [r.get("best_val_acc", 0) for r in results]
        colors = [C_SUCCESS if a == max(accs) else C_ACCENT for a in accs]
        self._sr_ax.bar(trials, accs, color=colors, alpha=.8, width=.6)
        self._sr_ax.set_title("Trial comparison", color="#aaaaaa", fontsize=8)
        self._sr_ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
        self._sr_fig.tight_layout(pad=2.5)
        self._sr_canvas.draw_idle()
