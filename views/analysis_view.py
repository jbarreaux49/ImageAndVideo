"""Tab 3 — Analysis: rich model inspection and visualisation tools."""
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from views.base_view import (
    C_BG, C_SURFACE, C_SURFACE2, C_INPUT, C_BORDER,
    C_TEXT, C_DIM, C_ACCENT, C_ACCENT2, C_DANGER, C_SUCCESS, C_WARN,
    FieldRow, SpinRow, ProgressBar, SectionHeader,
    btn, label, separator,
)
from utils.config import MODEL_SAVE_PATH, NUM_CLASSES, MAX_FRAMES, ASL_FRAMES_DIR

_PLT_BG   = "#f8fafc"
_PLT_SURF = "#ffffff"

_TOOLS = [
    ("history",     "📈  History",         C_ACCENT,  "Load and visualise a training history CSV."),
    ("weights",     "🔬  Layer Weights",    C_ACCENT2, "Show first-layer conv filter visualisation."),
    ("gradflow",    "⚡  Gradient Flow",    C_WARN,    "Per-layer gradient norms — detects vanishing/exploding gradients. No data needed."),
    ("tsne",        "🗺  t-SNE / PCA",     C_SUCCESS, "Project embeddings into 2D by class."),
    ("weightstats", "📊  Weight Stats",     C_DANGER,  "Per-layer weight norm, mean and std. No data needed."),
]


class AnalysisView(tk.Frame):
    """Sidebar nav + large matplotlib canvas for all inspection tools."""

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", C_BG)
        super().__init__(parent, **kwargs)

        # Commands wired by InspectorController
        self.load_history_cmd    = None
        self.run_weights_cmd     = None
        self.run_gradflow_cmd    = None
        self.run_tsne_cmd        = None
        self.run_weightstats_cmd = None

        self._active_tool = tk.StringVar(value="history")
        self._nav_btns: Dict[str, tk.Button] = {}
        self._ctrl_panels: Dict[str, tk.Frame] = {}

        self._build_ui()

    # ── top-level layout ──────────────────────────────────────────────────────

    def _build_ui(self):
        bg = self["bg"]

        # Model loader bar
        top = tk.Frame(self, bg=C_SURFACE)
        top.pack(fill="x", padx=12, pady=(12, 0))
        self._build_model_bar(top)

        separator(self, bg=C_BORDER).pack(fill="x", padx=12, pady=6)

        # Main area: sidebar | canvas
        body = tk.Frame(self, bg=bg)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        sidebar = tk.Frame(body, bg=C_SURFACE, width=200)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        tk.Frame(body, bg=C_BORDER, width=1).pack(side="left", fill="y", padx=8)

        canvas_area = tk.Frame(body, bg=bg)
        canvas_area.pack(side="left", fill="both", expand=True)
        self._build_canvas(canvas_area)

        # Select default tool after canvas_title exists
        self._select_tool("history")

    # ── model loader bar ──────────────────────────────────────────────────────

    def _build_model_bar(self, parent):
        bg = parent["bg"]
        pad = {"side": "left", "padx": 6, "pady": 8}

        label(parent, "Model:", dim=True, bg=bg).pack(**pad)
        self.model_path_var = tk.StringVar(value=MODEL_SAVE_PATH)
        tk.Entry(parent, textvariable=self.model_path_var, width=44,
                 bg=C_INPUT, fg=C_TEXT, insertbackground=C_TEXT, relief="flat").pack(**pad)
        btn(parent, "Browse", command=self._browse_model, color=C_SURFACE2,
            fg=C_TEXT, width=8).pack(**pad)

        self.load_model_btn = btn(parent, "Load Model", command=self._on_load_model,
                                   color=C_ACCENT, width=12)
        self.load_model_btn.pack(**pad)

        self.model_status = label(parent, "No model loaded.", dim=True, bg=bg)
        self.model_status.pack(**pad)

    # ── sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self, parent):
        bg = parent["bg"]

        label(parent, "Tools", dim=True, size=8, bg=bg).pack(
            anchor="w", padx=12, pady=(12, 4))

        # Navigation buttons
        for tool_id, title, color, _ in _TOOLS:
            b = tk.Button(
                parent, text=title, anchor="w",
                bg=bg, fg=C_DIM, relief="flat",
                font=("Segoe UI", 9), padx=12, pady=5,
                activebackground=C_SURFACE2, cursor="hand2",
                command=lambda t=tool_id: self._select_tool(t),
            )
            b.pack(fill="x")
            self._nav_btns[tool_id] = b

        separator(parent, bg=C_BORDER).pack(fill="x", padx=8, pady=10)

        # Context-specific control panels
        ctrl_area = tk.Frame(parent, bg=bg)
        ctrl_area.pack(fill="both", expand=True)

        for tool_id, _, color, desc in _TOOLS:
            panel = tk.Frame(ctrl_area, bg=bg)
            self._ctrl_panels[tool_id] = panel
            self._build_ctrl_panel(panel, tool_id, color, desc)

        self.progress = ProgressBar(parent, bg=bg)
        self.progress.pack(fill="x", padx=10, pady=6)

    def _build_ctrl_panel(self, panel: tk.Frame, tool_id: str, color: str, desc: str):
        bg = panel["bg"]
        label(panel, desc, dim=True, size=8, bg=bg, wraplength=175, justify="left"
              ).pack(anchor="w", padx=8, pady=(0, 8))

        if tool_id == "history":
            label(panel, "Auto-detected from model path.\nOr browse to override.", dim=True, size=8, bg=bg
                  ).pack(anchor="w", padx=8, pady=(0, 4))
            self.hist_csv = FieldRow(panel, "History CSV:", "", browse="file", bg=bg)
            self.hist_csv.pack(anchor="w", padx=6, pady=2)
            btn(panel, "Load & Plot", command=self._on_load_history,
                color=color, width=14).pack(anchor="w", padx=6, pady=6)

        elif tool_id == "weights":
            label(panel, "Requires loaded model.", dim=True, size=8, bg=bg
                  ).pack(anchor="w", padx=8, pady=4)
            btn(panel, "Visualise Filters", command=self._on_run_weights,
                color=color, width=14).pack(anchor="w", padx=6, pady=6)

        elif tool_id == "gradflow":
            label(panel, "Requires loaded model.\nUses a random dummy input.", dim=True, size=8, bg=bg
                  ).pack(anchor="w", padx=8, pady=4)
            btn(panel, "Run Gradient Flow", command=self._on_run_gradflow,
                color=color, width=14).pack(anchor="w", padx=6, pady=6)

        elif tool_id == "tsne":
            self.tsne_frames_dir = FieldRow(panel, "Frames dir:", ASL_FRAMES_DIR, browse="dir", bg=bg)
            self.tsne_frames_dir.pack(anchor="w", padx=6, pady=2)
            self.tsne_samples = SpinRow(panel, "Max samples:", 300, lo=50, hi=2000, bg=bg)
            self.tsne_samples.pack(anchor="w", padx=6, pady=2)
            split_row = tk.Frame(panel, bg=bg)
            split_row.pack(anchor="w", padx=6, pady=2)
            label(split_row, "Split:", dim=True, size=8, bg=bg).pack(side="left", padx=(0, 6))
            self.tsne_split = tk.StringVar(value="val")
            for val, txt in (("train", "Train"), ("val", "Val"), ("test", "Test")):
                tk.Radiobutton(split_row, text=txt, variable=self.tsne_split, value=val,
                               bg=bg, fg=C_TEXT, selectcolor=C_SURFACE2,
                               activebackground=bg, font=("Segoe UI", 9)
                               ).pack(side="left", padx=(0, 6))
            self.tsne_mode = tk.StringVar(value="tsne")
            mode_row = tk.Frame(panel, bg=bg)
            mode_row.pack(anchor="w", padx=6, pady=2)
            label(mode_row, "Method:", dim=True, size=8, bg=bg).pack(side="left", padx=(0, 6))
            for val, txt in (("tsne", "t-SNE"), ("pca", "PCA")):
                tk.Radiobutton(mode_row, text=txt, variable=self.tsne_mode, value=val,
                               bg=bg, fg=C_TEXT, selectcolor=C_SURFACE2,
                               activebackground=bg, font=("Segoe UI", 9)
                               ).pack(side="left", padx=(0, 10))
            btn(panel, "Run Projection", command=self._on_run_tsne,
                color=color, width=14).pack(anchor="w", padx=6, pady=6)

        elif tool_id == "weightstats":
            label(panel, "Requires loaded model.\nOne histogram per layer.", dim=True, size=8, bg=bg
                  ).pack(anchor="w", padx=8, pady=4)
            self.ws_start = SpinRow(panel, "Start layer:", 0,  lo=0, hi=200, bg=bg)
            self.ws_count = SpinRow(panel, "Num layers:",  8,  lo=1, hi=20,  bg=bg)
            self.ws_start.pack(anchor="w", padx=6, pady=2)
            self.ws_count.pack(anchor="w", padx=6, pady=2)
            btn(panel, "Show Histograms", command=self._on_run_weightstats,
                color=color, width=14).pack(anchor="w", padx=6, pady=6)

    # ── canvas area ───────────────────────────────────────────────────────────

    def _build_canvas(self, parent):
        bg = parent["bg"]
        self._canvas_parent = parent
        self.canvas_title = label(parent, "Select a tool →", dim=True, bg=bg)
        self.canvas_title.pack(anchor="w", padx=4, pady=(4, 2))

        self._fig = Figure(figsize=(8, 5.5), dpi=90, facecolor=_PLT_SURF)
        self._ax  = self._fig.add_subplot(111)
        self._ax.set_facecolor(_PLT_BG)
        self._ax.axis("off")
        self._fig.tight_layout(pad=1.5)

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── tool selection ────────────────────────────────────────────────────────

    def _select_tool(self, tool_id: str):
        self._active_tool.set(tool_id)
        for tid, b in self._nav_btns.items():
            _, title, color, _ = next(t for t in _TOOLS if t[0] == tid)
            if tid == tool_id:
                b.configure(bg=color, fg="#ffffff")
            else:
                b.configure(bg=self["bg"], fg=C_DIM)

        for tid, panel in self._ctrl_panels.items():
            panel.pack_forget()
        self._ctrl_panels[tool_id].pack(fill="both", expand=True, padx=4, pady=4)

        _, title, _, _ = next(t for t in _TOOLS if t[0] == tool_id)
        self.canvas_title.configure(text=title)

    # ── event forwarders ──────────────────────────────────────────────────────

    def _browse_model(self):
        p = filedialog.askopenfilename(filetypes=[("Model weights", "*.pth"), ("All", "*.*")])
        if p:
            self.model_path_var.set(p)

    def _on_load_model(self):
        self.load_model_btn.configure(state="disabled", text="Loading…")
        self.after(10, self._fire_load_model)

    def _fire_load_model(self):
        if self.load_history_cmd:
            # Delegate actual loading to controller; it will call set_model_status()
            pass
        self.load_model_btn.configure(state="normal", text="Load Model")

    def _on_load_history(self):
        path = self.hist_csv.get()
        if path and self.load_history_cmd:
            self.load_history_cmd(path)

    def _on_run_weights(self):
        if self.run_weights_cmd:
            self.run_weights_cmd()

    def _on_run_gradflow(self):
        if self.run_gradflow_cmd:
            self.run_gradflow_cmd()

    def _on_run_tsne(self):
        if self.run_tsne_cmd:
            self.run_tsne_cmd(
                self.tsne_frames_dir.get(),
                self.tsne_split.get(),
                self.tsne_samples.get(),
                self.tsne_mode.get(),
            )

    def _on_run_weightstats(self):
        if self.run_weightstats_cmd:
            self.run_weightstats_cmd(self.ws_start.get(), self.ws_count.get())

    # ── state helpers (called by InspectorController) ─────────────────────────

    def set_model_status(self, msg: str, ok: bool = True):
        self.model_status.configure(text=msg, fg=C_SUCCESS if ok else C_DANGER)
        self.load_model_btn.configure(state="normal", text="Load Model")

    def update_progress(self, frac: float, msg: str = ""):
        self.progress.update(frac, msg)

    # ── plot helpers ──────────────────────────────────────────────────────────

    def _reset_fig(self, nrows: int = 1, ncols: int = 1, figsize=(8, 5.5)):
        # Destroy old canvas widget so it doesn't linger below the new one
        self._canvas.get_tk_widget().destroy()

        self._fig = Figure(figsize=figsize, dpi=90, facecolor=_PLT_SURF)
        axes = self._fig.subplots(nrows, ncols)

        self._canvas = FigureCanvasTkAgg(self._fig, master=self._canvas_parent)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        return axes

    def _redraw(self):
        self._fig.tight_layout(pad=1.5)
        self._canvas.draw()

    # ── renderers ─────────────────────────────────────────────────────────────

    def plot_history(self, history: List[Dict]):
        import matplotlib.ticker
        ax_loss, ax_acc = self._reset_fig(2, 1)
        for ax in (ax_loss, ax_acc):
            ax.set_facecolor(_PLT_BG)
            ax.tick_params(colors="#64748b", labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("#e2e8f0")

        epochs     = [r["epoch"]      for r in history]
        train_loss = [r["train_loss"] for r in history]
        val_loss   = [r["val_loss"]   for r in history]
        train_acc  = [r["train_acc"]  for r in history]
        val_acc    = [r["val_acc"]    for r in history]

        ax_loss.plot(epochs, train_loss, color=C_ACCENT,  linewidth=1.5, label="train")
        ax_loss.plot(epochs, val_loss,   color=C_WARN,    linewidth=1.5, label="val")
        ax_loss.set_title("Loss",     color="#475569", fontsize=9)
        ax_loss.legend(fontsize=8, facecolor=_PLT_SURF, labelcolor="#475569")

        ax_acc.plot(epochs, train_acc, color=C_ACCENT,  linewidth=1.5, label="train")
        ax_acc.plot(epochs, val_acc,   color=C_WARN,    linewidth=1.5, label="val")
        ax_acc.set_title("Accuracy", color="#475569", fontsize=9)
        ax_acc.legend(fontsize=8, facecolor=_PLT_SURF, labelcolor="#475569")
        ax_acc.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0%}"))

        best_acc = max(val_acc)
        best_ep  = epochs[val_acc.index(best_acc)]
        self.canvas_title.configure(
            text=f"Training History  ·  best val acc {best_acc:.2%} (epoch {best_ep})"
        )
        self._redraw()

    def plot_weights(self, weights: np.ndarray):
        """Display first-layer conv filters as an RGB grid.

        weights: (N, 3, H, W)
        """
        n = weights.shape[0]
        cols = 8
        rows = math.ceil(n / cols)
        axes = self._reset_fig(rows, cols, figsize=(8, rows * 1.1))
        if rows == 1:
            axes = [axes]
        for i in range(rows):
            for j in range(cols):
                ax = axes[i][j]
                ax.axis("off")
                idx = i * cols + j
                if idx >= n:
                    continue
                f = weights[idx]                    # (3, H, W)
                f = f.transpose(1, 2, 0)            # (H, W, 3)
                lo, hi = f.min(), f.max()
                f = (f - lo) / (hi - lo + 1e-8)
                ax.imshow(f)

        self.canvas_title.configure(
            text=f"Layer Weights — {n} filters ({weights.shape[2]}×{weights.shape[3]} spatial, RGB channels)"
        )
        self._redraw()

    def plot_gradient_flow(self, grad_data: dict):
        """Bar chart of per-parameter gradient norms (vanishing gradient view)."""
        names  = list(grad_data.keys())
        values = [grad_data[n] for n in names]

        # Shorten names for readability
        short = [n.replace("features.", "").replace(".weight", ".w")
                  .replace(".bias", ".b") for n in names]

        ax = self._reset_fig(1, 1, figsize=(8, 5.5))
        ax.set_facecolor(_PLT_BG)
        ax.tick_params(colors="#64748b", labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#e2e8f0")

        colors = [C_DANGER if v < 1e-6 else C_WARN if v < 1e-4 else C_ACCENT
                  for v in values]
        ax.bar(range(len(values)), values, color=colors, width=0.7)
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, rotation=90, fontsize=5)
        ax.set_ylabel("Mean |grad|", color="#64748b", fontsize=8)
        ax.set_yscale("log")
        ax.set_title("Gradient Flow  (red = vanishing < 1e-6,  orange = weak < 1e-4,  blue = OK)",
                     color="#475569", fontsize=8)

        self.canvas_title.configure(text=f"Gradient Flow — {len(names)} parameters")
        self._redraw()

    def plot_weight_histograms(self, layers: list, total: int):
        """Grid of weight histograms — one per layer, like the reference image."""
        n = len(layers)
        if n == 0:
            return
        cols   = min(n, 5)
        rows   = math.ceil(n / cols)
        fw     = max(8, cols * 1.7)
        fh     = max(3, rows * 2.2)
        axes   = self._reset_fig(rows, cols, figsize=(fw, fh))

        # Normalise to always be a 2-D list
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [list(axes)]
        elif cols == 1:
            axes = [[a] for a in axes]

        for k, info in enumerate(layers):
            r, c = divmod(k, cols)
            ax   = axes[r][c]
            w    = info["weights"]

            ax.set_facecolor(_PLT_BG)
            ax.tick_params(colors="#64748b", labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor("#e2e8f0")

            ax.hist(w, bins=60, color=C_ACCENT2, alpha=0.85, edgecolor="none")
            ax.axvline(0, color=C_DIM, linewidth=0.6, linestyle="--")

            short = info["name"].replace("features.", "").replace("classifier", "cls")
            ax.set_title(
                f"{short}\n{info['layer_type']}  act:{info['activation']}",
                fontsize=6, color="#475569", pad=2,
            )
            ax.set_xlabel(f"{info['mean']:.3f} ± {info['std']:.3f}", fontsize=6, color=C_DIM)
            ax.set_ylabel("")
            ax.set_yticks([])

        # Hide unused axes
        for k in range(n, rows * cols):
            r, c = divmod(k, cols)
            axes[r][c].axis("off")

        start = layers[0]["idx"]
        end   = layers[-1]["idx"]
        self.canvas_title.configure(
            text=f"Weight Histograms — layers {start}–{end}  (total {total})"
        )
        self._redraw()

    def plot_tsne(self, coords: np.ndarray, labels: np.ndarray,
                  class_names: Optional[List[str]] = None, mode: str = "t-SNE"):
        ax = self._reset_fig(1, 1)
        ax.set_facecolor(_PLT_BG)
        ax.tick_params(colors="#64748b", labelsize=7)

        unique = np.unique(labels)
        cmap   = __import__("matplotlib.cm", fromlist=["tab20"]).tab20
        n_col  = len(cmap.colors)
        for cls in unique:
            mask = labels == cls
            c    = cmap(int(cls) % n_col)
            name = class_names[cls] if class_names and cls < len(class_names) else str(cls)
            ax.scatter(coords[mask, 0], coords[mask, 1], color=c, s=8, alpha=.7, label=name)

        ax.set_title(f"{mode} projection  ·  {len(unique)} classes", color="#475569", fontsize=9)
        if len(unique) <= 20:
            ax.legend(fontsize=6, facecolor=_PLT_SURF, labelcolor=C_DIM,
                      loc="best", markerscale=2)
        self.canvas_title.configure(text=f"{mode} / PCA — {len(coords)} samples")
        self._redraw()

    def plot_gradcam(self, heatmap: np.ndarray, pred_class: int,
                     confidence: float, frame_rgb: Optional[np.ndarray] = None):
        if frame_rgb is not None:
            ax1, ax2 = self._reset_fig(1, 2)
            ax1.imshow(frame_rgb)
            ax1.set_title("Sample frame", color="#475569", fontsize=8)
            ax1.axis("off")
        else:
            ax2 = self._reset_fig(1, 1)

        im = ax2.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        self._fig.colorbar(im, ax=ax2, fraction=.04, pad=.02)
        ax2.set_title(f"Grad-CAM  class {pred_class}  ({confidence:.1%})",
                      color="#475569", fontsize=8)
        ax2.axis("off")

        self.canvas_title.configure(
            text=f"Grad-CAM  ·  predicted class {pred_class}  ({confidence:.1%} confidence)"
        )
        self._redraw()
