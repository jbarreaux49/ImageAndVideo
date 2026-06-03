"""Dataset that loads pre-extracted sign language frame sequences."""
import os
import random
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.io import read_image

from utils.config import MAX_FRAMES, MEAN, STD


@dataclass
class SignSample:
    path: str
    label: int


class SignDataset(Dataset):
    """Reads index CSV files produced by TrainingManager._build_index().

    Each sample directory must contain sub-dirs:
        original/  flipped/  brightened/  contrasted/
    One augmentation is chosen at random for every __getitem__ call.
    """

    _AUG_DIRS = ("original", "flipped", "brightened", "contrasted")

    def __init__(self, index_file: str, max_frames: int = MAX_FRAMES, training: bool = True):
        self.max_frames = max_frames
        self.training   = training
        self.samples: List[SignSample] = self._parse_index(index_file)
        self._transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    # ── index parsing ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_index(path: str) -> List[SignSample]:
        samples = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                seq_path, label = line.rsplit(",", 1)
                samples.append(SignSample(path=seq_path, label=int(label)))
        return samples

    # ── frame loading ─────────────────────────────────────────────────────────

    def _load_frames(self, seq_dir: str) -> torch.Tensor:
        aug_subdir = os.path.join(seq_dir, random.choice(self._AUG_DIRS))
        if not os.path.isdir(aug_subdir):
            aug_subdir = seq_dir

        files = sorted(
            f for f in os.listdir(aug_subdir)
            if f.lower().endswith(".jpg") and os.path.getsize(os.path.join(aug_subdir, f)) > 0
        )
        if not files:
            return torch.zeros(3, self.max_frames, 224, 224)

        raw: List[torch.Tensor] = []
        for fname in files:
            img = read_image(os.path.join(aug_subdir, fname))
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            raw.append(img)

        t = len(raw)

        # ── temporal sampling: random consecutive window (paper §4) ───────────
        if t >= self.max_frames:
            start = random.randint(0, t - self.max_frames) if self.training else 0
            raw = raw[start: start + self.max_frames]
        else:
            # Repeat last frame to reach max_frames (paper §4, not zero-padding)
            raw = raw + [raw[-1]] * (self.max_frames - t)

        # ── spatial jitter: ±10% scale/translation, consistent per clip ───────
        # Applied only during training (paper §4)
        if self.training:
            h, w = raw[0].shape[1], raw[0].shape[2]
            tx = random.uniform(-0.1, 0.1) * w
            ty = random.uniform(-0.1, 0.1) * h
            scale = random.uniform(0.9, 1.1)
            raw = [
                TF.affine(img, angle=0.0, translate=[tx, ty], scale=scale, shear=0.0,
                          interpolation=TF.InterpolationMode.BILINEAR)
                for img in raw
            ]

        frames = [self._transform(img) for img in raw]

        # stack → (T, C, H, W) then permute → (C, T, H, W)
        return torch.stack(frames).permute(1, 0, 2, 3)

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return self._load_frames(sample.path), sample.label
