"""Shared palette, style helpers, and reusable widgets — light theme."""
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Optional

# ── Light palette ─────────────────────────────────────────────────────────────
C_BG      = "#ffffff"   # pure white
C_SURFACE = "#f8fafc"   # off-white panels
C_SURFACE2= "#f1f5f9"   # slightly darker surface
C_INPUT   = "#f1f5f9"   # input field background
C_BORDER  = "#e2e8f0"   # subtle border
C_TEXT    = "#0f172a"   # near-black text
C_DIM     = "#94a3b8"   # muted / placeholder
C_ACCENT  = "#2563eb"   # blue — primary action
C_ACCENT2 = "#7c3aed"   # purple — secondary
C_SUCCESS = "#059669"   # green
C_DANGER  = "#dc2626"   # red
C_WARN    = "#d97706"   # amber


def separator(parent, bg: str = C_BORDER, height: int = 1, **kwargs) -> tk.Frame:
    return tk.Frame(parent, bg=bg, height=height, **kwargs)


def label(parent, text: str, dim: bool = False, bold: bool = False,
          size: int = 9, bg: str = None, **kwargs) -> tk.Label:
    bg  = bg or parent.cget("bg")
    fg  = C_DIM if dim else C_TEXT
    font = ("Segoe UI", size, "bold" if bold else "normal")
    return tk.Label(parent, text=text, bg=bg, fg=fg, font=font, **kwargs)


def btn(parent, text: str, command=None,
        color: str = C_ACCENT, disabled: bool = False,
        width: int = None, **kwargs) -> tk.Button:
    kw = dict(
        text=text, command=command,
        bg=color, fg="#ffffff",
        relief="flat", font=("Segoe UI", 9),
        activebackground=_darken(color),
        state="disabled" if disabled else "normal",
        cursor="hand2", padx=10, pady=4,
    )
    if width:
        kw["width"] = width
    kw.update(kwargs)
    return tk.Button(parent, **kw)


def _darken(hex_color: str) -> str:
    r = max(int(hex_color[1:3], 16) - 25, 0)
    g = max(int(hex_color[3:5], 16) - 25, 0)
    b = max(int(hex_color[5:7], 16) - 25, 0)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Reusable compound widgets ─────────────────────────────────────────────────

class LogPanel(tk.Frame):
    """Scrollable, append-only log text area."""

    def __init__(self, parent, height: int = 10, **kwargs):
        kwargs.setdefault("bg", C_BG)
        super().__init__(parent, **kwargs)
        self._text = tk.Text(
            self, height=height, state="disabled", wrap="word",
            bg=C_SURFACE, fg=C_TEXT, font=("Consolas", 8),
            relief="flat", borderwidth=0, padx=8, pady=6,
            selectbackground=C_ACCENT, insertbackground=C_TEXT,
        )
        sb = ttk.Scrollbar(self, command=self._text.yview)
        self._text.configure(yscrollcommand=sb.set)
        self._text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def append(self, message: str):
        self._text.configure(state="normal")
        self._text.insert("end", message + "\n")
        self._text.see("end")
        self._text.configure(state="disabled")

    def clear(self):
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")


class FieldRow(tk.Frame):
    """Label + Entry + optional browse button."""

    def __init__(self, parent, text: str, default: str = "",
                 browse: Optional[str] = None, **kwargs):
        kwargs.setdefault("bg", parent.cget("bg"))
        super().__init__(parent, **kwargs)
        self._var = tk.StringVar(value=default)

        label(self, text, dim=True, width=18, anchor="w").pack(side="left")
        tk.Entry(
            self, textvariable=self._var, width=36,
            bg=C_INPUT, fg=C_TEXT, insertbackground=C_TEXT,
            relief="flat", highlightthickness=1,
            highlightbackground=C_BORDER, highlightcolor=C_ACCENT,
        ).pack(side="left", padx=(4, 2))
        if browse:
            btn(self, "Browse", lambda: self._browse(browse),
                color=C_SURFACE2, fg=C_TEXT, width=7).pack(side="left")

    def _browse(self, mode: str):
        if mode == "dir":
            p = filedialog.askdirectory()
        elif mode == "file":
            p = filedialog.askopenfilename()
        elif mode == "save":
            p = filedialog.asksaveasfilename(defaultextension=".pth")
        else:
            p = ""
        if p:
            self._var.set(p)

    def get(self) -> str:  return self._var.get()
    def set(self, v: str): self._var.set(v)


class SpinRow(tk.Frame):
    """Label + Spinbox."""

    def __init__(self, parent, text: str, default: int,
                 lo: int = 1, hi: int = 1000, **kwargs):
        kwargs.setdefault("bg", parent.cget("bg"))
        super().__init__(parent, **kwargs)
        label(self, text, dim=True, width=18, anchor="w").pack(side="left")
        self.var = tk.IntVar(value=default)
        tk.Spinbox(
            self, from_=lo, to=hi, textvariable=self.var, width=7,
            bg=C_INPUT, fg=C_TEXT, buttonbackground=C_SURFACE2,
            relief="flat", highlightthickness=1,
            highlightbackground=C_BORDER, highlightcolor=C_ACCENT,
        ).pack(side="left", padx=4)

    def get(self) -> int: return self.var.get()


class FloatRow(tk.Frame):
    """Label + Entry for float values."""

    def __init__(self, parent, text: str, default: float, **kwargs):
        kwargs.setdefault("bg", parent.cget("bg"))
        super().__init__(parent, **kwargs)
        label(self, text, dim=True, width=18, anchor="w").pack(side="left")
        self.var = tk.StringVar(value=str(default))
        tk.Entry(
            self, textvariable=self.var, width=10,
            bg=C_INPUT, fg=C_TEXT, insertbackground=C_TEXT, relief="flat",
        ).pack(side="left", padx=4)

    def get(self) -> str: return self.var.get()


class ComboRow(tk.Frame):
    """Label + Combobox (read-only dropdown)."""

    def __init__(self, parent, text: str, choices: list, default: str = "", **kwargs):
        kwargs.setdefault("bg", parent.cget("bg"))
        super().__init__(parent, **kwargs)
        label(self, text, dim=True, width=18, anchor="w").pack(side="left")
        self.var = tk.StringVar(value=default or (choices[0] if choices else ""))
        cb = ttk.Combobox(self, textvariable=self.var, values=choices,
                          state="readonly", width=14, font=("Segoe UI", 9))
        cb.pack(side="left", padx=4)

    def get(self) -> str: return self.var.get()


class ProgressBar(tk.Frame):
    """Slim progress bar with status label."""

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", parent.cget("bg"))
        super().__init__(parent, **kwargs)
        self._lbl = label(self, "", dim=True, size=8, anchor="w")
        self._lbl.pack(fill="x")
        self._bar = ttk.Progressbar(self, mode="determinate", maximum=100)
        self._bar.pack(fill="x", pady=(1, 0))

    def update(self, frac: float, msg: str = ""):
        self._bar["value"] = frac * 100
        if msg:
            self._lbl.configure(text=msg)

    def reset(self):
        self._bar["value"] = 0
        self._lbl.configure(text="")


class SectionHeader(tk.Frame):
    """Small coloured left bar + bold title."""

    def __init__(self, parent, text: str, color: str = C_ACCENT, **kwargs):
        kwargs.setdefault("bg", parent.cget("bg"))
        super().__init__(parent, **kwargs)
        tk.Frame(self, bg=color, width=3).pack(side="left", fill="y", padx=(0, 8))
        label(self, text, bold=True, bg=self.cget("bg")).pack(side="left", pady=4)
