"""Dataset that loads pre-extracted sign language frame sequences."""
import os
import random
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
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

    def __init__(self, index_file: str, max_frames: int = MAX_FRAMES):
        self.max_frames = max_frames
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

        files = sorted(f for f in os.listdir(aug_subdir) if f.lower().endswith(".jpg"))
        if not files:
            return torch.zeros(3, self.max_frames, 224, 224)

        frames = []
        for fname in files:
            img = read_image(os.path.join(aug_subdir, fname))
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            frames.append(self._transform(img))

        # stack → (T, C, H, W) then permute → (C, T, H, W)
        tensor = torch.stack(frames).permute(1, 0, 2, 3)

        t = tensor.shape[1]
        if t < self.max_frames:
            pad = torch.zeros(3, self.max_frames - t, *tensor.shape[2:])
            tensor = torch.cat([tensor, pad], dim=1)
        else:
            tensor = tensor[:, :self.max_frames]

        return tensor

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return self._load_frames(sample.path), sample.label
