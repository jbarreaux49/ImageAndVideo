"""Grid and random hyperparameter search wrapping TrainingManager."""
import os
import random
from itertools import product
from typing import Callable, Dict, List, Optional

from models.training_manager import TrainingManager
from utils.config import ASL_FRAMES_DIR, NUM_CLASSES, MAX_FRAMES
from utils.logger import get_logger

_log = get_logger("hyper_search")


class HyperSearchManager:
    """Runs multiple short TrainingManager runs and collects comparison results."""

    def __init__(
        self,
        frames_dir:         str   = ASL_FRAMES_DIR,
        model_save_dir:     str   = "search_results",
        num_classes:        int   = NUM_CLASSES,
        max_frames:         int   = MAX_FRAMES,
        epochs_per_trial:   int   = 5,
        steps_per_epoch:    int   = 0,
        optimizer_name:     str   = "Adam",
        loss_fn_name:       str   = "CrossEntropy (weighted)",
        regularization:     str   = "L2",
        reg_strength:       float = 1e-4,
        log_callback:       Optional[Callable[[str], None]]              = None,
        progress_callback:  Optional[Callable[[float, str], None]]       = None,
        result_callback:    Optional[Callable[[List[Dict]], None]]        = None,
    ):
        self.frames_dir       = frames_dir
        self.model_save_dir   = model_save_dir
        self.num_classes      = num_classes
        self.max_frames       = max_frames
        self.epochs_per_trial = epochs_per_trial
        self.steps_per_epoch  = steps_per_epoch
        self.optimizer_name   = optimizer_name
        self.loss_fn_name     = loss_fn_name
        self.regularization   = regularization
        self.reg_strength     = reg_strength
        self.log_callback     = log_callback
        self.progress_callback = progress_callback
        self.result_callback  = result_callback
        self.results: List[Dict] = []
        self._stop = False
        os.makedirs(model_save_dir, exist_ok=True)

    # ── combination generators ────────────────────────────────────────────────

    @staticmethod
    def grid_combinations(param_grid: Dict[str, List]) -> List[Dict]:
        """Cartesian product of all parameter lists."""
        keys = list(param_grid.keys())
        return [dict(zip(keys, combo)) for combo in product(*param_grid.values())]

    @staticmethod
    def random_combinations(param_space: Dict[str, List], n_trials: int) -> List[Dict]:
        """Sample `n_trials` random combinations (uniform over each list)."""
        combos = []
        for _ in range(n_trials):
            combos.append({k: random.choice(v) for k, v in param_space.items()})
        return combos

    # ── helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        _log.info(msg)
        if self.log_callback:
            self.log_callback(msg)

    def _progress(self, v: float, msg: str = ""):
        if self.progress_callback:
            self.progress_callback(v, msg)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self, combinations: List[Dict]) -> List[Dict]:
        """Train one model per combination; return results sorted by best val accuracy."""
        self._stop   = False
        self.results = []
        total = len(combinations)

        for i, params in enumerate(combinations):
            if self._stop:
                self._log("Search stopped by user.")
                break

            trial_id   = i + 1
            slug       = "_".join(f"{k[:2]}={v}" for k, v in params.items())
            model_path = os.path.join(self.model_save_dir, f"trial_{trial_id:02d}_{slug}.pth")

            self._log(f"\n── Trial {trial_id}/{total}  {params}")
            self._progress(i / total, f"Trial {trial_id}/{total}")

            mgr = TrainingManager(
                frames_dir      = self.frames_dir,
                model_save_path = model_path,
                num_classes     = self.num_classes,
                max_frames      = self.max_frames,
                num_epochs      = self.epochs_per_trial,
                steps_per_epoch = self.steps_per_epoch,
                learning_rate   = float(params.get("learning_rate", 5e-5)),
                batch_size      = int(params.get("batch_size", 4)),
                dropout_rate    = float(params.get("dropout_rate", 0.5)),
                optimizer_name  = self.optimizer_name,
                loss_fn_name    = self.loss_fn_name,
                regularization  = self.regularization,
                reg_strength    = self.reg_strength,
                log_callback    = self.log_callback,
            )

            try:
                history       = mgr.run()
                best_val_acc  = max((r["val_acc"] for r in history), default=0.0)
                final_val_loss= history[-1]["val_loss"] if history else 0.0
                result = {
                    **params,
                    "trial":          trial_id,
                    "best_val_acc":   best_val_acc,
                    "final_val_loss": final_val_loss,
                    "epochs_run":     len(history),
                    "model_path":     model_path,
                }
                self.results.append(result)
                self._log(f"Trial {trial_id} best val acc: {best_val_acc:.2%}")
                if self.result_callback:
                    self.result_callback(list(self.results))
            except Exception as exc:
                self._log(f"Trial {trial_id} failed: {exc}")

        self.results.sort(key=lambda r: r["best_val_acc"], reverse=True)
        if self.results:
            best = self.results[0]
            self._log(f"\nBest trial: {best}")
        self._progress(1.0, "Search complete.")
        return self.results

    def stop(self):
        self._stop = True
