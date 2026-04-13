"""LPIPS — Learned Perceptual Image Patch Similarity. JAX/Flax NNX port."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .lpips_utils import get_ckpt_path


class ScalingLayer(nnx.Module):
    """Normalize input for VGG."""
    def __init__(self):
        self.shift = jnp.array([-.030, -.088, -.188], dtype=jnp.float32).reshape(1, 3, 1, 1)
        self.scale = jnp.array([.458, .448, .450], dtype=jnp.float32).reshape(1, 3, 1, 1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return (x - self.shift) / self.scale


class NetLinLayer(nnx.Module):
    """Linear 1x1 conv layer for LPIPS."""
    def __init__(self, chn_in: int, chn_out: int = 1, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        # Implemented as (B, C, H, W) → 1x1 conv → (B, 1, H, W)
        self.conv = nnx.Conv(chn_in, chn_out, kernel_size=(1, 1), strides=(1, 1),
                             padding='VALID', use_bias=False, dtype=dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, C, H, W) → (B, H, W, C) for NNX Conv → back
        x = x.transpose(0, 2, 3, 1)
        x = self.conv(x)
        return x.transpose(0, 3, 1, 2)


class VGG16Slice(nnx.Module):
    """A slice of VGG16 features (sequential Conv+ReLU+Pool blocks)."""
    def __init__(self, layer_configs: list, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.layers = []
        for cfg in layer_configs:
            if cfg == 'M':
                self.layers.append(('pool', None))
            else:
                in_c, out_c = cfg
                conv = nnx.Conv(in_c, out_c, kernel_size=(3, 3), padding='SAME',
                                use_bias=True, dtype=dtype, rngs=rngs)
                self.layers.append(('conv', conv))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, H, W, C)
        for kind, layer in self.layers:
            if kind == 'pool':
                x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
            else:
                x = jax.nn.relu(layer(x))
        return x


class VGG16FeatureExtractor(nnx.Module):
    """VGG16 feature extractor for LPIPS — 5 slices."""
    # VGG16 layer configs: [(in_channels, out_channels), 'M', ...]
    SLICE_CONFIGS = [
        [(3, 64), (64, 64)],                                    # slice1: relu1_2
        ['M', (64, 128), (128, 128)],                           # slice2: relu2_2
        ['M', (128, 256), (256, 256), (256, 256)],               # slice3: relu3_3
        ['M', (256, 512), (512, 512), (512, 512)],               # slice4: relu4_3
        ['M', (512, 512), (512, 512), (512, 512)],               # slice5: relu5_3
    ]

    def __init__(self, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.slices = [
            VGG16Slice(cfg, rngs=rngs, dtype=dtype)
            for cfg in self.SLICE_CONFIGS
        ]

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        # x: (B, C, H, W) → (B, H, W, C) for NNX Conv
        h = x.transpose(0, 2, 3, 1)
        outputs = []
        for s in self.slices:
            h = s(h)
            outputs.append(h.transpose(0, 3, 1, 2))  # back to (B, C, H, W)
        return tuple(outputs)


def _normalize(x: jnp.ndarray, eps: float = 1e-10) -> jnp.ndarray:
    norm = jnp.sqrt(jnp.sum(x ** 2, axis=1, keepdims=True))
    return x / (norm + eps)


def _spatial_average(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(x, axis=(2, 3), keepdims=True)


class LPIPS(nnx.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS) with VGG16."""

    def __init__(self, *, rngs: nnx.Rngs = nnx.Rngs(0), dtype: jnp.dtype = jnp.float32):
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = VGG16FeatureExtractor(rngs=rngs, dtype=dtype)

        self.lins = [
            NetLinLayer(c, chn_out=1, rngs=rngs, dtype=dtype)
            for c in self.chns
        ]

        self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """Load pretrained VGG LPIPS weights.

        VGG16 features from torchvision, linear layers from LPIPS checkpoint.
        """
        import torch
        import torchvision.models as tv_models

        # --- Load VGG16 features from torchvision ---
        vgg16 = tv_models.vgg16(weights='IMAGENET1K_V1')
        vgg_state = vgg16.features.state_dict()

        # Map VGG features to our slice structure
        # features.0, features.2 → slice0 (conv layers only)
        # features.5, features.7 → slice1
        # features.10, features.12, features.14 → slice2
        # features.17, features.19, features.21 → slice3
        # features.24, features.26, features.28 → slice4
        vgg_feat_indices = [
            ['0', '2'],
            ['5', '7'],
            ['10', '12', '14'],
            ['17', '19', '21'],
            ['24', '26', '28'],
        ]

        for slice_idx, s in enumerate(self.net.slices):
            conv_count = 0
            for kind, layer in s.layers:
                if kind == 'conv':
                    feat_idx = vgg_feat_indices[slice_idx][conv_count]
                    w = vgg_state[f"{feat_idx}.weight"].detach().cpu().numpy()
                    b = vgg_state[f"{feat_idx}.bias"].detach().cpu().numpy()
                    # PyTorch Conv2d: (out, in, kH, kW) → JAX: (kH, kW, in, out)
                    layer.kernel.value = jnp.array(w.transpose(2, 3, 1, 0))
                    layer.bias.value = jnp.array(b)
                    conv_count += 1

        # --- Load LPIPS linear layers ---
        ckpt = get_ckpt_path("vgg_lpips")
        lpips_state = torch.load(ckpt, map_location="cpu")

        for i, lin in enumerate(self.lins):
            w = lpips_state[f"lin{i}.model.1.weight"].detach().cpu().numpy()
            # PyTorch Conv2d: (out, in, kH, kW) → JAX: (kH, kW, in, out)
            lin.conv.kernel.value = jnp.array(w.transpose(2, 3, 1, 0))

        print(f"[LPIPS] Loaded VGG16 from torchvision + LPIPS linears from {ckpt}")

    def __call__(self, input: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """Compute LPIPS distance.

        Args:
            input, target: Images (B, C, H, W) in [0, 1] or [-1, 1].

        Returns:
            Scalar loss.
        """
        in_scaled = self.scaling_layer(input)
        tgt_scaled = self.scaling_layer(target)

        feats_in = self.net(in_scaled)
        feats_tgt = self.net(tgt_scaled)

        value = jnp.zeros(())
        for i, (f_in, f_tgt) in enumerate(zip(feats_in, feats_tgt)):
            f_in = _normalize(f_in)
            f_tgt = _normalize(f_tgt)
            diff = (f_in - f_tgt) ** 2
            value = value + jnp.mean(_spatial_average(self.lins[i](diff)))

        return value
