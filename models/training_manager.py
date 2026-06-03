"""Full training pipeline: data loading, training loop, evaluation, early stopping."""
import csv
import os
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.sign_dataset import SignDataset
from models.sign_model import SignModel
from utils.config import (
    BATCH_SIZE, DROPOUT_RATE, EARLY_STOPPING_PATIENCE,
    LEARNING_RATE, LR_DECAY_EPOCH, MAX_FRAMES, NUM_CLASSES, NUM_EPOCHS,
)
from utils.logger import get_logger

_log = get_logger("training_manager")


class _FocalLoss(nn.Module):
    """Focal Loss — down-weights easy examples to focus on hard ones."""

    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt   = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class EarlyStopping:
    """Triggers after `patience` epochs with no improvement in validation loss.

    `delta` sets the minimum improvement required to reset the patience counter,
    preventing near-flat plateaus from being mistaken for real progress.
    """

    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE, delta: float = 0.001):
        self.patience   = patience
        self.delta      = delta
        self.best_loss  = float("inf")
        self.counter    = 0

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class TrainingManager:
    """Orchestrates dataset building, model training, and evaluation.

    Heavy work is meant to run inside a background thread.
    UI feedback is delivered through optional callbacks so this class has
    zero dependency on tkinter.
    """

    def __init__(
        self,
        frames_dir:        str,
        model_save_path:   str,
        num_classes:       int   = NUM_CLASSES,
        batch_size:        int   = BATCH_SIZE,
        max_frames:        int   = MAX_FRAMES,
        num_epochs:        int   = NUM_EPOCHS,
        steps_per_epoch:   int   = 0,           # 0 = use all batches
        learning_rate:     float = LEARNING_RATE,
        dropout_rate:      float = DROPOUT_RATE,
        optimizer_name:    str   = "SGD",
        loss_fn_name:      str   = "CrossEntropy (weighted)",
        regularization:    str   = "L2",
        reg_strength:      float = 1e-4,
        lr_decay_epoch:    int   = LR_DECAY_EPOCH,
        log_callback:      Optional[Callable[[str], None]]         = None,
        progress_callback: Optional[Callable[[float, str], None]]  = None,
        epoch_callback:    Optional[Callable[[List[Dict]], None]]   = None,
    ):
        self.frames_dir       = frames_dir
        self.model_save_path  = model_save_path
        self.num_classes      = num_classes
        self.batch_size       = batch_size
        self.max_frames       = max_frames
        self.num_epochs       = num_epochs
        self.steps_per_epoch  = steps_per_epoch
        self.learning_rate    = learning_rate
        self.dropout_rate     = dropout_rate
        self.optimizer_name   = optimizer_name
        self.loss_fn_name     = loss_fn_name
        self.regularization   = regularization
        self.reg_strength     = reg_strength
        self.lr_decay_epoch   = lr_decay_epoch
        self.log_callback     = log_callback
        self.progress_callback = progress_callback
        self.epoch_callback   = epoch_callback

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model:  Optional[SignModel] = None
        self.history: List[Dict]         = []
        self._stop   = False

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        _log.info(msg)
        if self.log_callback:
            self.log_callback(msg)

    def _progress(self, value: float, msg: str = ""):
        if self.progress_callback:
            self.progress_callback(value, msg)

    # ── data helpers ──────────────────────────────────────────────────────────

    def _build_index(self, split: str) -> str:
        """Scan the frames directory for a split and write a CSV index."""
        split_dir  = os.path.join(self.frames_dir, split)
        index_path = os.path.join(self.frames_dir, f"{split}_index.csv")

        with open(index_path, "w", newline="") as f:
            for class_dir in sorted(os.listdir(split_dir)):
                class_path = os.path.join(split_dir, class_dir)
                if not os.path.isdir(class_path):
                    continue
                try:
                    label = int(class_dir.split("_")[1])
                except (IndexError, ValueError):
                    continue
                if label >= self.num_classes:
                    continue
                for seq in os.listdir(class_path):
                    seq_path = os.path.join(class_path, seq)
                    if os.path.isdir(seq_path):
                        f.write(f"{seq_path},{label}\n")

        return index_path

    def _make_sampler(self, dataset: SignDataset) -> WeightedRandomSampler:
        counts  = Counter(s.label for s in dataset.samples)
        weights = [1.0 / counts[s.label] for s in dataset.samples]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def _build_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        self._log("Building dataset indices…")
        train_idx = self._build_index("train")
        val_idx   = self._build_index("val")
        test_idx  = self._build_index("test")

        train_ds = SignDataset(train_idx, self.max_frames, training=True)
        val_ds   = SignDataset(val_idx,   self.max_frames, training=False)
        test_ds  = SignDataset(test_idx,  self.max_frames, training=False)

        self._log(
            f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
        )

        pin = self.device.type == "cuda"
        kw  = dict(num_workers=0, pin_memory=pin)
        return (
            DataLoader(train_ds, batch_size=self.batch_size, sampler=self._make_sampler(train_ds), **kw),
            DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False, **kw),
            DataLoader(test_ds,  batch_size=self.batch_size, shuffle=False, **kw),
        )

    # ── training / evaluation ─────────────────────────────────────────────────

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        scaler: torch.cuda.amp.GradScaler,
    ) -> Tuple[float, float]:
        self.model.train()
        total_loss = correct = total = 0
        n_batches  = len(loader)
        limit      = self.steps_per_epoch if self.steps_per_epoch > 0 else n_batches

        # Last known val metrics (from previous epoch, or 0 if first)
        last_val_loss = self.history[-1]["val_loss"] if self.history else 0.0
        last_val_acc  = self.history[-1]["val_acc"]  if self.history else 0.0

        for batch_idx, (inputs, labels) in enumerate(loader):
            if self._stop or batch_idx >= limit:
                break

            print(f"  [ep{epoch}/{self.num_epochs} batch {batch_idx+1}/{limit}] forward...",
                  end="", flush=True)

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                out  = self.model(inputs)
                loss = criterion(out, labels)
                if self.regularization in ("L1", "L1+L2"):
                    l1 = sum(p.abs().sum() for p in self.model.parameters())
                    loss = loss + self.reg_strength * l1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(labels)
            correct    += out.argmax(1).eq(labels).sum().item()
            total      += len(labels)

            print(f" loss={loss.item():.4f}", flush=True)

            frac = ((epoch - 1) + (batch_idx + 1) / limit) / self.num_epochs
            self._progress(frac, f"Epoch {epoch}/{self.num_epochs}  batch {batch_idx+1}/{limit}")

        return (total_loss / total, correct / total) if total else (0.0, 0.0)

    def _eval_epoch(self, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        self.model.eval()
        total_loss = correct = total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                out  = self.model(inputs)
                loss = criterion(out, labels)
                total_loss += loss.item() * len(labels)
                correct    += out.argmax(1).eq(labels).sum().item()
                total      += len(labels)

        return (total_loss / total, correct / total) if total else (0.0, 0.0)

    # ── loss / optimizer builders ─────────────────────────────────────────────

    def _build_loss(self, samples) -> nn.Module:
        counts = Counter(s.label for s in samples)
        w = torch.zeros(self.num_classes, device=self.device)
        for c, n in counts.items():
            w[c] = 1.0 / n

        name = self.loss_fn_name
        if name == "CrossEntropy (weighted)":
            return nn.CrossEntropyLoss(weight=w)
        elif name == "CrossEntropy":
            return nn.CrossEntropyLoss()
        elif name == "LabelSmoothing":
            return nn.CrossEntropyLoss(label_smoothing=0.1)
        elif name == "Focal Loss":
            return _FocalLoss(gamma=2.0, weight=w)
        return nn.CrossEntropyLoss(weight=w)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        p    = self.model.parameters()
        lr   = self.learning_rate
        wd   = self.reg_strength if self.regularization in ("L2", "L1+L2") else 0.0
        name = self.optimizer_name
        if name == "AdamW":
            return torch.optim.AdamW(p, lr=lr, weight_decay=wd)
        elif name == "SGD":
            return torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=wd)
        elif name == "RMSprop":
            return torch.optim.RMSprop(p, lr=lr, momentum=0.9, weight_decay=wd)
        return torch.optim.AdamW(p, lr=lr, weight_decay=wd)

    # ── public API ────────────────────────────────────────────────────────────

    def run(self) -> List[Dict]:
        """Execute the full training pipeline and return the history list."""
        import os
        self._stop   = False
        self.history = []

        print(f"[train pid={os.getpid()}] starting...", flush=True)
        torch.set_num_threads(1)

        train_loader, val_loader, test_loader = self._build_loaders()

        print(f"[train pid={os.getpid()}] loading R3D-18 weights (may take ~30s)...", flush=True)
        self.model = SignModel(self.num_classes, self.dropout_rate).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()   # wait for model weights to land on GPU
        print(f"[train pid={os.getpid()}] model ready.", flush=True)

        criterion = self._build_loss(train_loader.dataset.samples)
        optimizer = self._build_optimizer()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[self.lr_decay_epoch], gamma=0.1
        )
        scaler    = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        stopper   = EarlyStopping()

        self._log(
            f"Device: {self.device} | classes: {self.num_classes} | epochs: {self.num_epochs}"
        )

        best_val_loss = float("inf")

        for epoch in range(1, self.num_epochs + 1):
            if self._stop:
                self._log("Training stopped by user.")
                break

            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer, epoch, scaler)
            print(f"  [ep{epoch}/{self.num_epochs}] running validation...", flush=True)
            val_loss, val_acc     = self._eval_epoch(val_loader, criterion)
            scheduler.step()

            record = dict(
                epoch=epoch,
                train_loss=train_loss, train_acc=train_acc,
                val_loss=val_loss,     val_acc=val_acc,
            )
            self.history.append(record)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_save_path)
                self._log(f"Best model saved (val loss={val_loss:.4f})")

            self._save_history()

            msg = (
                f"Epoch {epoch}/{self.num_epochs}  "
                f"train loss={train_loss:.4f} acc={train_acc:.2%}  "
                f"val loss={val_loss:.4f} acc={val_acc:.2%}"
            )
            self._log(msg)
            self._progress(epoch / self.num_epochs, msg)

            # Final update with val metrics
            if self.epoch_callback:
                self.epoch_callback(list(self.history))

            if stopper(val_loss):
                self._log(f"Early stopping at epoch {epoch}.")
                break

        self._log("Evaluating on test set…")
        test_loss, test_acc = self._eval_epoch(test_loader, criterion)
        self._log(f"Test — loss={test_loss:.4f}  acc={test_acc:.2%}")

        self._log(f"Best model saved at {self.model_save_path}")
        self._save_history()
        self._progress(1.0, f"Done. Test accuracy: {test_acc:.2%}")
        return self.history

    def stop(self):
        self._stop = True

    def _save_history(self):
        if not self.history:
            return
        path = os.path.splitext(self.model_save_path)[0] + "_history.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)
        self._log(f"History saved → {path}")
