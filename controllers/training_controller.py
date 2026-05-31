"""Training + search controller — uses multiprocessing.Process so Tkinter never freezes."""
import multiprocessing as mp
from typing import Dict, List

from models.workers import training_worker, search_worker
from models.hyper_search_manager import HyperSearchManager
from views.training_view import TrainingView


class TrainingController:
    def __init__(self, view: TrainingView):
        self.view = view
        self._tr_proc: mp.Process = None
        self._sr_proc: mp.Process = None
        self._tr_q: mp.Queue = None
        self._sr_q: mp.Queue = None

        view.train_start_command  = self._start_training
        view.train_stop_command   = self._stop_training
        view.search_start_command = self._start_search
        view.search_stop_command  = self._stop_search

    # ── Training ──────────────────────────────────────────────────────────────

    def _start_training(self):
        try:
            lr           = float(self.view.tr_lr.get())
            dropout      = float(self.view.tr_dropout.get())
            reg_strength = float(self.view.tr_reg_strength.get())
        except ValueError:
            self.view.tr_log.append("Invalid value: learning rate, dropout and reg. strength must be floats.")
            return

        config = dict(
            frames_dir       = self.view.tr_frames_dir.get(),
            model_save_path  = self.view.tr_model_path.get(),
            num_classes      = self.view.tr_num_classes.get(),
            batch_size       = self.view.tr_batch_size.get(),
            max_frames       = self.view.tr_max_frames.get(),
            steps_per_epoch  = self.view.tr_steps_epoch.get(),
            num_epochs       = self.view.tr_num_epochs.get(),
            learning_rate    = lr,
            dropout_rate     = dropout,
            optimizer_name   = self.view.tr_optimizer.get(),
            loss_fn_name     = self.view.tr_loss_fn.get(),
            regularization   = self.view.tr_regularization.get(),
            reg_strength     = reg_strength,
        )

        self._tr_q    = mp.Queue()
        self._tr_proc = mp.Process(
            target=training_worker, args=(self._tr_q, config), daemon=True
        )
        self._tr_proc.start()

        self.view.tr_log.clear()
        self.view.tr_progress.reset()
        self.view.set_train_running(True)
        self.view.after(200, self._poll_tr)

    def _stop_training(self):
        if self._tr_proc and self._tr_proc.is_alive():
            self._tr_proc.terminate()
            self._tr_proc.join(timeout=2)
        self._tr_proc = None
        self.view.set_train_running(False)

    def _poll_tr(self):
        if self._tr_q is None:
            return
        try:
            while True:
                item = self._tr_q.get_nowait()
                tag  = item[0]
                if tag == "log":
                    self.view.tr_log.append(item[1])
                elif tag == "progress":
                    self.view.tr_progress.update(item[1], item[2])
                elif tag == "epoch":
                    self.view.update_charts(item[1])
                elif tag == "error":
                    self.view.tr_log.append(f"ERROR: {item[1]}")
                elif tag == "done":
                    self.view.set_train_running(False)
                    return
        except Exception:
            pass
        self.view.after(200, self._poll_tr)

    # ── Search ────────────────────────────────────────────────────────────────

    def _start_search(self):
        mode     = self.view.sr_mode.get()
        n_trials = self.view.sr_n_trials.get()

        param_space = self._parse_param_space()
        if not param_space:
            self.view.sr_log.append("Define at least one parameter range.")
            return

        combos = (HyperSearchManager.grid_combinations(param_space)
                  if mode == "grid"
                  else HyperSearchManager.random_combinations(param_space, n_trials))

        try:
            sr_reg_strength = float(self.view.sr_reg_strength.get())
        except ValueError:
            sr_reg_strength = 1e-4

        config = dict(
            frames_dir        = self.view.sr_frames_dir.get(),
            model_save_dir    = self.view.sr_save_dir.get(),
            num_classes       = self.view.sr_num_classes.get(),
            max_frames        = self.view.sr_max_frames.get(),
            epochs_per_trial  = self.view.sr_epochs.get(),
            steps_per_epoch   = self.view.sr_steps_epoch.get(),
            optimizer_name    = self.view.sr_optimizer.get(),
            loss_fn_name      = self.view.sr_loss_fn.get(),
            regularization    = self.view.sr_regularization.get(),
            reg_strength      = sr_reg_strength,
        )

        self._sr_q    = mp.Queue()
        self._sr_proc = mp.Process(
            target=search_worker, args=(self._sr_q, config, combos), daemon=True
        )
        self._sr_proc.start()

        self.view.sr_log.clear()
        self.view.sr_progress.reset()
        self.view.set_search_running(True)
        self.view.sr_log.append(f"Starting {mode} search — {len(combos)} trial(s)…")
        self.view.after(200, self._poll_sr)

    def _parse_param_space(self) -> Dict:
        space = {}
        for attr, key, cast in [
            ("sr_lr_vals",      "learning_rate", float),
            ("sr_dropout_vals", "dropout_rate",  float),
            ("sr_bs_vals",      "batch_size",    int),
        ]:
            try:
                vals = [cast(v.strip())
                        for v in getattr(self.view, attr).get().split(",") if v.strip()]
                if vals:
                    space[key] = vals
            except ValueError:
                pass
        return space

    def _stop_search(self):
        if self._sr_proc and self._sr_proc.is_alive():
            self._sr_proc.terminate()
            self._sr_proc.join(timeout=2)
        self._sr_proc = None
        self.view.set_search_running(False)

    def _poll_sr(self):
        if self._sr_q is None:
            return
        try:
            while True:
                item = self._sr_q.get_nowait()
                tag  = item[0]
                if tag == "log":
                    self.view.sr_log.append(item[1])
                elif tag == "progress":
                    self.view.sr_progress.update(item[1], item[2])
                elif tag == "results":
                    self.view.update_search_results(item[1])
                elif tag == "error":
                    self.view.sr_log.append(f"ERROR: {item[1]}")
                elif tag == "done":
                    self.view.set_search_running(False)
                    return
        except Exception:
            pass
        self.view.after(200, self._poll_sr)
