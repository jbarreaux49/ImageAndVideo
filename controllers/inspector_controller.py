"""Controller for the Analysis tab: loads model and drives all inspection tools."""
import csv
import queue
import threading
from tkinter import messagebox
from typing import List, Optional

import numpy as np

from models.model_inspector import ModelInspector
from views.analysis_view import AnalysisView


class InspectorController:
    def __init__(self, view: AnalysisView):
        self.view         = view
        self._inspector:  Optional[ModelInspector] = None
        self._queue       = queue.Queue()
        self._model_path  = ""
        self._num_classes = 100

        # Wire view commands
        view.load_history_cmd    = self._load_history
        view.run_weights_cmd     = self._run_weights
        view.run_gradflow_cmd    = self._run_gradflow
        view.run_tsne_cmd        = self._run_tsne
        view.run_weightstats_cmd = self._run_weightstats

        # Wire model loader button via monkey-patch
        view.load_model_btn.configure(command=self._load_model)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        path = self.view.model_path_var.get()
        if not path:
            messagebox.showwarning("No path", "Enter or browse to a model .pth file.")
            return
        self.view.load_model_btn.configure(state="disabled", text="Loading…")
        threading.Thread(
            target=self._do_load_model, args=(path,), daemon=True
        ).start()
        self.view.after(200, self._poll_load)

    def _do_load_model(self, path: str):
        try:
            self._inspector   = ModelInspector(path)
            self._model_path  = path
            self._num_classes = self._inspector.num_classes
            # Auto-detect history CSV next to the model file
            import os
            history_csv = os.path.splitext(path)[0] + "_history.csv"
            self._queue.put(("model_ok",
                             f"Loaded: {os.path.basename(path)}",
                             history_csv if os.path.exists(history_csv) else "",
                             self._inspector.num_classes))
        except Exception as exc:
            self._queue.put(("model_err", str(exc)))

    def _poll_load(self):
        try:
            item = self._queue.get_nowait()
            if item[0] == "model_ok":
                msg, history_csv, num_classes = item[1], item[2], item[3]
                self.view.set_model_status(f"{msg}  ({num_classes} classes)", ok=True)
                if history_csv:
                    self.view.hist_csv.set(history_csv)
            elif item[0] == "model_err":
                self.view.set_model_status(f"Error: {item[1]}", ok=False)
            return
        except queue.Empty:
            pass
        self.view.after(200, self._poll_load)

    # ── Training History ──────────────────────────────────────────────────────

    def _load_history(self, path: str):
        try:
            history = []
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    history.append({k: float(v) for k, v in row.items()})
            if not history:
                messagebox.showwarning("Empty", "CSV file contains no data.")
                return
            self.view.plot_history(history)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not load history:\n{exc}")

    # ── Layer Weights ─────────────────────────────────────────────────────────

    def _run_weights(self):
        if not self._check_model():
            return
        weights = self._inspector.stem_weights()   # (N, 3, H, W)
        self.view.plot_weights(weights)

    # ── Gradient Flow ─────────────────────────────────────────────────────────

    def _run_gradflow(self):
        if not self._check_model():
            return
        self.view.update_progress(0.2, "Computing gradients…")

        def work():
            try:
                data = self._inspector.gradient_flow()
                self._queue.put(("gradflow", data))
            except Exception as exc:
                self._queue.put(("err", str(exc)))

        threading.Thread(target=work, daemon=True).start()
        self.view.after(100, lambda: self._poll_generic("gradflow"))

    # ── t-SNE / PCA ───────────────────────────────────────────────────────────

    def _run_tsne(self, frames_dir: str, split: str, max_samples: int, mode: str):
        if not self._check_model():
            return
        self.view.update_progress(0.1, "Building index…")

        def work():
            try:
                import os, tempfile
                split_dir = os.path.join(frames_dir, split)
                if not os.path.isdir(split_dir):
                    self._queue.put(("err", f"Split directory not found: {split_dir}"))
                    return

                tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                                  delete=False, newline="")
                n_seqs = 0
                for class_dir in sorted(os.listdir(split_dir)):
                    class_path = os.path.join(split_dir, class_dir)
                    if not os.path.isdir(class_path):
                        continue
                    try:
                        label = int(class_dir.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    if label >= self._num_classes:
                        continue
                    for seq in os.listdir(class_path):
                        seq_path = os.path.join(class_path, seq)
                        if os.path.isdir(seq_path):
                            tmp.write(f"{seq_path},{label}\n")
                            n_seqs += 1
                tmp.close()

                self.view.after(0, lambda: self.view.update_progress(
                    0.3, f"Extracting embeddings for {min(n_seqs, max_samples)} samples… (CPU — please wait)"))

                embs, labels = self._inspector.get_embeddings(tmp.name, max_samples)
                os.unlink(tmp.name)
                self._queue.put(("emb_raw", embs, labels, mode))
            except Exception as exc:
                self._queue.put(("err", str(exc)))

        threading.Thread(target=work, daemon=True).start()
        self.view.after(100, lambda: self._poll_generic("emb_raw"))

    def _reduce_embeddings(self, embs: np.ndarray, mode: str) -> np.ndarray:
        if mode == "pca":
            from sklearn.decomposition import PCA
            return PCA(n_components=2).fit_transform(embs)
        else:
            try:
                from sklearn.manifold import TSNE
                perp = min(30, max(5, len(embs) // 10))
                return TSNE(n_components=2, perplexity=perp,
                            random_state=42, n_iter=500).fit_transform(embs)
            except ImportError:
                messagebox.showerror("Missing", "Install scikit-learn: pip install scikit-learn")
                return None

    # ── Weight Histograms ─────────────────────────────────────────────────────

    def _run_weightstats(self, start: int, count: int):
        if not self._check_model():
            return
        self.view.update_progress(0.2, "Extracting weights…")

        def work():
            try:
                layers, total = self._inspector.multi_weight_histograms(start, count)
                self._queue.put(("weightstats", layers, total))
            except Exception as exc:
                self._queue.put(("err", str(exc)))

        threading.Thread(target=work, daemon=True).start()
        self.view.after(100, lambda: self._poll_generic("weightstats"))

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll_generic(self, expected_tag: str):
        try:
            item = self._queue.get_nowait()
            tag  = item[0]

            if tag == "err":
                messagebox.showerror("Error", item[1])
                self.view.update_progress(0, "")
                return

            if tag == "gradflow":
                self.view.plot_gradient_flow(item[1])
                self.view.update_progress(1.0, "Done")

            elif tag == "emb_raw":
                embs, labels, mode = item[1], item[2], item[3]
                self.view.update_progress(0.6, "Reducing dimensions…")
                coords = self._reduce_embeddings(embs, mode)
                if coords is not None:
                    self.view.plot_tsne(coords, labels, mode=mode.upper())
                    self.view.update_progress(1.0, "Done")

            elif tag == "weightstats":
                _, layers, total = item
                self.view.plot_weight_histograms(layers, total)
                self.view.update_progress(1.0, "Done")

            return
        except queue.Empty:
            pass
        self.view.after(150, lambda: self._poll_generic(expected_tag))

    # ── Guard ─────────────────────────────────────────────────────────────────

    def _check_model(self) -> bool:
        if self._inspector is None:
            messagebox.showwarning("No model", "Load a model first.")
            return False
        return True
