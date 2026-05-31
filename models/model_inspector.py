"""Visual inspection tools for a trained SignModel.

Provides: first-layer weight grids, layer activation maps,
feature embeddings for t-SNE/PCA, and Grad-CAM heatmaps.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.sign_model import SignModel


class ModelInspector:
    """Loads a saved SignModel and exposes inspection utilities.

    All methods return plain numpy arrays so the view layer has zero
    PyTorch dependency.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        state       = torch.load(model_path, map_location=self.device, weights_only=True)
        self.num_classes = state["classifier.weight"].shape[0]
        self.model  = SignModel(self.num_classes).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self._hooks: List = []
        self._cache: Dict = {}

    def clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._cache.clear()

    # ── Layer weight visualisation ────────────────────────────────────────────

    def stem_weights(self) -> np.ndarray:
        """Return the first Conv3d weights shaped (N, 3, H, W).

        Each of the N filters is collapsed over the temporal dimension
        so the result is a grid of RGB spatial filters.
        """
        for m in self.model.features[0].modules():
            if isinstance(m, nn.Conv3d):
                # shape: (out_ch, in_ch=3, T, H, W)
                w = m.weight.detach().cpu().numpy()
                # mean over T → (out_ch, 3, H, W)
                return w.mean(axis=2)
        return np.zeros((64, 3, 7, 7))

    # ── Gradient flow ─────────────────────────────────────────────────────────

    def gradient_flow(self) -> dict:
        """Dummy forward+backward to measure per-parameter gradient norms.

        Returns {param_name: grad_abs_mean} for all parameters that receive
        a gradient. Low values in early layers indicate vanishing gradients.
        """
        self.model.train()
        x      = torch.randn(1, 3, 16, 224, 224).to(self.device)
        target = torch.zeros(1, dtype=torch.long).to(self.device)
        out    = self.model(x)
        loss   = torch.nn.functional.cross_entropy(out, target)
        loss.backward()

        result = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                result[name] = float(param.grad.abs().mean().item())

        self.model.zero_grad()
        self.model.eval()
        return result

    # ── Embeddings (t-SNE / PCA) ──────────────────────────────────────────────

    def get_embeddings(
        self, index_file: str, max_samples: int = 300, max_frames: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 512-d feature vectors for dataset samples.

        Returns (embeddings (N, 512), labels (N,)).
        """
        from models.sign_dataset import SignDataset
        from torch.utils.data import DataLoader

        ds = SignDataset(index_file, max_frames)
        if len(ds) > max_samples:
            idx = np.random.choice(len(ds), max_samples, replace=False)
            ds.samples = [ds.samples[i] for i in idx]

        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

        embeddings, labels = [], []
        cache = {}

        def hook(m, inp, out):
            cache["f"] = out.detach().cpu()

        h = self.model.features.register_forward_hook(hook)

        with torch.no_grad():
            for videos, lbls in loader:
                videos = videos.to(self.device)
                self.model(videos)
                feat = cache["f"].flatten(start_dim=1)  # (B, 512)
                embeddings.append(feat.numpy())
                labels.extend(lbls.tolist())

        h.remove()
        return np.vstack(embeddings), np.array(labels)

    # ── Weight histograms ─────────────────────────────────────────────────────

    def _flat_modules(self) -> list:
        """Flat ordered list of (name, module) for all modules in the model."""
        return list(self.model.named_modules())

    def _find_activation(self, module_name: str) -> str:
        """Look at the next sibling module after `module_name` for an activation."""
        _ACT = {
            "ReLU": "ReLU", "LeakyReLU": "LeakyReLU", "GELU": "GELU",
            "Sigmoid": "Sigmoid", "Tanh": "Tanh", "SiLU": "SiLU",
            "ELU": "ELU", "PReLU": "PReLU",
        }
        flat = self._flat_modules()
        for i, (name, _) in enumerate(flat):
            if name == module_name:
                for _, nxt in flat[i + 1: i + 4]:
                    t = type(nxt).__name__
                    if t in _ACT:
                        return _ACT[t]
                return "—"
        return "—"

    def weight_layers_info(self) -> list:
        """Return ordered list of weight-bearing layers with metadata."""
        _SKIP = {"BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm"}
        result = []
        for name, module in self._flat_modules():
            if not hasattr(module, "weight") or module.weight is None:
                continue
            layer_type = type(module).__name__
            if layer_type in _SKIP:
                continue
            result.append({
                "idx":        len(result),
                "name":       name,
                "layer_type": layer_type,
                "activation": self._find_activation(name),
                "n_params":   module.weight.numel(),
            })
        return result

    def multi_weight_histograms(self, start: int, count: int) -> tuple:
        """Return list of histogram dicts for `count` layers starting at `start`.

        Each dict: weights (np array), name, layer_type, activation, mean, std.
        Also returns total number of weight layers.
        """
        layers = self.weight_layers_info()
        total  = len(layers)
        selected = layers[start: start + count]
        result = []
        param_map = {n: p for n, p in self.model.named_parameters()}
        for info in selected:
            key = info["name"] + ".weight"
            if key not in param_map:
                continue
            w = param_map[key].data.float().cpu().numpy().flatten()
            result.append({**info, "weights": w, "mean": float(w.mean()), "std": float(w.std())})
        return result, total

    # ── Grad-CAM ──────────────────────────────────────────────────────────────

    def gradcam(
        self, video_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """Compute Grad-CAM on layer4 of the R3D-18 backbone.

        Returns (heatmap H×W normalised 0-1, predicted class index, confidence).
        """
        target_layer = self.model.features[4]  # layer4

        acts_store = {}
        grad_store = {}

        fh = target_layer.register_forward_hook(
            lambda m, i, o: acts_store.update({"a": o})
        )
        bh = target_layer.register_full_backward_hook(
            lambda m, gi, go: grad_store.update({"g": go[0]})
        )

        x      = video_tensor.unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs  = torch.softmax(logits, dim=1)[0]

        if target_class is None:
            target_class = probs.argmax().item()
        confidence = probs[target_class].item()

        self.model.zero_grad()
        logits[0, target_class].backward()

        fh.remove()
        bh.remove()

        grads = grad_store["g"].detach().cpu().numpy()[0]   # (C, T, H, W)
        acts  = acts_store["a"].detach().cpu().numpy()[0]

        alpha = grads.mean(axis=(1, 2, 3))                  # (C,)
        cam   = np.tensordot(alpha, acts, axes=([0], [0]))  # (T, H, W)
        cam   = np.maximum(cam, 0).mean(axis=0)             # (H, W)

        lo, hi = cam.min(), cam.max()
        if hi > lo:
            cam = (cam - lo) / (hi - lo)

        return cam, int(target_class), float(confidence)

    # ── Convenience: load a single video tensor ───────────────────────────────

    @staticmethod
    def load_video_tensor(video_path: str, max_frames: int = 64) -> torch.Tensor:
        """Load a video file and return a (C, T, H, W) tensor ready for inference."""
        import cv2
        from torchvision import transforms
        from utils.config import MEAN, STD, SHORT_SIDE_SIZE, FRAME_SIZE

        cap    = cv2.VideoCapture(video_path)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step   = max(total // max_frames, 1)
        wanted = set(range(0, total, step)[:max_frames])

        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

        frames = []
        idx    = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in wanted:
                h, w = frame.shape[:2]
                scale = SHORT_SIDE_SIZE / min(h, w)
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                ch, cw = FRAME_SIZE
                y = (frame.shape[0] - ch) // 2
                x = (frame.shape[1] - cw) // 2
                frame = frame[y:y+ch, x:x+cw]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(tf(frame))
            idx += 1
        cap.release()

        if not frames:
            return torch.zeros(3, max_frames, 224, 224)

        tensor = torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)
        t = tensor.shape[1]
        if t < max_frames:
            pad = torch.zeros(3, max_frames - t, *tensor.shape[2:])
            tensor = torch.cat([tensor, pad], dim=1)
        return tensor[:, :max_frames]
