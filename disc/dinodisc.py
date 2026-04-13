"""DINO-based discriminator — JAX/Flax NNX port of dinodisc.py."""

from __future__ import annotations

import math
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .utils import RandomWindowCrop


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class MLPNoDrop(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int = 0,
                 *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        out_features = out_features or in_features
        self.fc1 = nnx.Linear(in_features, hidden_features, dtype=dtype, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_features, out_features, dtype=dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.fc2(jax.nn.gelu(self.fc1(x), approximate=True))


class SelfAttentionNoDrop(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nnx.Linear(embed_dim, embed_dim * 3, use_bias=True, dtype=dtype, rngs=rngs)
        self.proj = nnx.Linear(embed_dim, embed_dim, use_bias=True, dtype=dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = jnp.split(qkv.transpose(2, 0, 3, 1, 4), 3, axis=0)
        q, k, v = q[0], k[0], v[0]  # each (B, H, L, D)

        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        attn = jax.nn.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, C)
        return self.proj(out)


class SABlockNoDrop(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, norm_eps: float,
                 *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.norm1 = nnx.LayerNorm(embed_dim, epsilon=norm_eps, dtype=dtype, rngs=rngs)
        self.attn = SelfAttentionNoDrop(embed_dim, num_heads, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.LayerNorm(embed_dim, epsilon=norm_eps, dtype=dtype, rngs=rngs)
        self.mlp = MLPNoDrop(embed_dim, round(embed_dim * mlp_ratio), rngs=rngs, dtype=dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# BatchNormLocal — virtual batch norm
# ---------------------------------------------------------------------------
class BatchNormLocal(nnx.Module):
    def __init__(self, num_features: int, eps: float = 1e-6, virtual_bs: int = 1,
                 *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.weight = nnx.Param(jnp.ones(num_features, dtype=dtype))
        self.bias = nnx.Param(jnp.zeros(num_features, dtype=dtype))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, C, L) — 1D conv format
        shape = x.shape
        B = shape[0]
        G = int(math.ceil(B / self.virtual_bs))
        x = x.reshape(G, -1, x.shape[-2], x.shape[-1])
        mean = jnp.mean(x, axis=(1, 3), keepdims=True)
        var = jnp.var(x, axis=(1, 3), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        x = x * self.weight.value[None, :, None] + self.bias.value[None, :, None]
        return x.reshape(shape)


# ---------------------------------------------------------------------------
# Conv1D block for discriminator heads
# ---------------------------------------------------------------------------
class Conv1DBlock(nnx.Module):
    """Conv1D + BatchNormLocal + LeakyReLU."""
    def __init__(self, channels: int, kernel_size: int, norm_eps: float,
                 *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        # NNX Conv for 1D: use Conv with kernel_size on last dim
        self.conv = nnx.Conv(channels, channels, kernel_size=(kernel_size,),
                             padding='CIRCULAR', dtype=dtype, rngs=rngs)
        self.norm = BatchNormLocal(channels, eps=norm_eps, rngs=rngs, dtype=dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, C, L) → need (B, L, C) for NNX Conv then back
        x_t = x.transpose(0, 2, 1)  # (B, L, C)
        x_t = self.conv(x_t)
        x_t = x_t.transpose(0, 2, 1)  # (B, C, L)
        x_t = self.norm(x_t)
        return jax.nn.leaky_relu(x_t, negative_slope=0.2)


class ResidualBlock(nnx.Module):
    def __init__(self, channels: int, kernel_size: int, norm_eps: float,
                 *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.fn = Conv1DBlock(channels, kernel_size, norm_eps, rngs=rngs, dtype=dtype)
        self.ratio = 1.0 / math.sqrt(2.0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.fn(x) + x) * self.ratio


class DiscHead(nnx.Module):
    """Single discriminator head: Conv1d(1x1) → ResBlock → Conv1d(1x1→1)."""
    def __init__(self, channels: int, kernel_size: int, norm_eps: float,
                 *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.block1 = Conv1DBlock(channels, kernel_size=1, norm_eps=norm_eps, rngs=rngs, dtype=dtype)
        self.res = ResidualBlock(channels, kernel_size, norm_eps, rngs=rngs, dtype=dtype)
        # Final 1x1 conv to scalar
        self.final_conv = nnx.Conv(channels, 1, kernel_size=(1,), padding='VALID', dtype=dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, C, L)
        x = self.block1(x)
        x = self.res(x)
        x = x.transpose(0, 2, 1)  # (B, L, C)
        x = self.final_conv(x)  # (B, L, 1)
        return x.transpose(0, 2, 1)  # (B, 1, L)


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------
class PatchEmbed(nnx.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int,
                 *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nnx.Conv(in_chans, embed_dim,
                             kernel_size=(patch_size, patch_size),
                             strides=(patch_size, patch_size),
                             padding='VALID', dtype=dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, H, W, C) → Conv → (B, H', W', embed) → flatten
        x = self.proj(x)
        B, H, W, C = x.shape
        return x.reshape(B, H * W, C)


# ---------------------------------------------------------------------------
# FrozenDINO backbone
# ---------------------------------------------------------------------------
class FrozenDINONoDrop(nnx.Module):
    """Frozen DINO ViT-S/8 backbone for discriminator features."""

    RECIPES = {
        "S_8": dict(depth=12, key_depths=(2, 5, 8, 11), norm_eps=1e-6, patch_size=8,
                    in_chans=3, embed_dim=384, num_heads=6, mlp_ratio=4.0),
        "S_16": dict(depth=12, key_depths=(2, 5, 8, 11), norm_eps=1e-6, patch_size=16,
                     in_chans=3, embed_dim=384, num_heads=6, mlp_ratio=4.0),
        "B_16": dict(depth=12, key_depths=(2, 5, 8, 11), norm_eps=1e-6, patch_size=16,
                     in_chans=3, embed_dim=768, num_heads=12, mlp_ratio=4.0),
    }

    def __init__(
        self,
        recipe: str = "S_8",
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        cfg = dict(self.RECIPES[recipe])
        self.embed_dim = cfg["embed_dim"]
        self.img_size = 224
        self.patch_size = cfg["patch_size"]
        self.key_depths = frozenset(d for d in cfg["key_depths"] if d < cfg["depth"])

        self.patch_embed = PatchEmbed(
            self.img_size, cfg["patch_size"], cfg["in_chans"], cfg["embed_dim"],
            rngs=rngs, dtype=dtype
        )
        num_patches = (self.img_size // cfg["patch_size"]) ** 2

        self.cls_token = nnx.Param(jnp.zeros((1, 1, cfg["embed_dim"]), dtype=dtype))
        self.pos_embed = nnx.Param(jnp.zeros((1, num_patches + 1, cfg["embed_dim"]), dtype=dtype))

        self.blocks = [
            SABlockNoDrop(cfg["embed_dim"], cfg["num_heads"], cfg["mlp_ratio"], cfg["norm_eps"],
                          rngs=rngs, dtype=dtype)
            for _ in range(max(cfg["depth"], 1 + max(self.key_depths, default=0)))
        ]
        self.norm = nnx.LayerNorm(cfg["embed_dim"], epsilon=cfg["norm_eps"], dtype=dtype, rngs=rngs)

        # Normalization: x ∈ [-1, 1] → ImageNet normalized
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.x_scale = (0.5 / std).reshape(1, 1, 1, 3)  # NHWC
        self.x_shift = ((0.5 - mean) / std).reshape(1, 1, 1, 3)

    def __call__(self, x: jnp.ndarray) -> List[jnp.ndarray]:
        """Forward pass returning intermediate activations.

        Args:
            x: Images in [-1, 1], shape (B, C, H, W) — NCHW.

        Returns:
            List of activations (B, C, L) at key depths + final.
        """
        # NCHW → NHWC for JAX conv
        x = x.transpose(0, 2, 3, 1)

        # Resize to 224
        B = x.shape[0]
        if x.shape[1] != self.img_size or x.shape[2] != self.img_size:
            x = jax.image.resize(x, (B, self.img_size, self.img_size, x.shape[3]), method="bilinear")

        # Normalize
        x = x * self.x_scale + self.x_shift

        # Patch embed
        x = self.patch_embed(x)

        # Prepend CLS token + add pos embed
        cls_tokens = jnp.broadcast_to(self.cls_token.value, (B, 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x = x + self.pos_embed.value

        # Forward through blocks, collect activations at key depths
        activations = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.key_depths:
                # (B, L, C) → (B, C, L), skip CLS token
                activations.append(x[:, 1:, :].transpose(0, 2, 1))

        # Final activation (always included, inserted at front)
        activations.insert(0, x[:, 1:, :].transpose(0, 2, 1))
        return activations

    def load_pretrained_torch(self, path: str):
        """Load PyTorch DINO ViT-S checkpoint weights."""
        import torch

        state = torch.load(path, map_location="cpu")

        # Fix QKV bias (zero out K bias like PyTorch code)
        for key in sorted(state.keys()):
            if ".attn.qkv.bias" in key:
                bias = state[key]
                C = bias.numel() // 3
                bias[C:2 * C].zero_()

        def _t(name):
            return state[name].detach().cpu().numpy()

        # Patch embed
        # PyTorch Conv2d: (out, in, kH, kW) → JAX Conv: (kH, kW, in, out)
        self.patch_embed.proj.kernel.value = jnp.array(_t("patch_embed.proj.weight").transpose(2, 3, 1, 0))
        self.patch_embed.proj.bias.value = jnp.array(_t("patch_embed.proj.bias"))

        # CLS token, pos_embed
        self.cls_token.value = jnp.array(_t("cls_token"))
        self.pos_embed.value = jnp.array(_t("pos_embed"))

        # Blocks
        for i, block in enumerate(self.blocks):
            prefix = f"blocks.{i}."
            # norm1
            block.norm1.scale.value = jnp.array(_t(f"{prefix}norm1.weight"))
            block.norm1.bias.value = jnp.array(_t(f"{prefix}norm1.bias"))
            # attn
            block.attn.qkv.kernel.value = jnp.array(_t(f"{prefix}attn.qkv.weight").T)
            block.attn.qkv.bias.value = jnp.array(_t(f"{prefix}attn.qkv.bias"))
            block.attn.proj.kernel.value = jnp.array(_t(f"{prefix}attn.proj.weight").T)
            block.attn.proj.bias.value = jnp.array(_t(f"{prefix}attn.proj.bias"))
            # norm2
            block.norm2.scale.value = jnp.array(_t(f"{prefix}norm2.weight"))
            block.norm2.bias.value = jnp.array(_t(f"{prefix}norm2.bias"))
            # mlp
            block.mlp.fc1.kernel.value = jnp.array(_t(f"{prefix}mlp.fc1.weight").T)
            block.mlp.fc1.bias.value = jnp.array(_t(f"{prefix}mlp.fc1.bias"))
            block.mlp.fc2.kernel.value = jnp.array(_t(f"{prefix}mlp.fc2.weight").T)
            block.mlp.fc2.bias.value = jnp.array(_t(f"{prefix}mlp.fc2.bias"))

        # Final norm
        self.norm.scale.value = jnp.array(_t("norm.weight"))
        self.norm.bias.value = jnp.array(_t("norm.bias"))

        print(f"[FrozenDINONoDrop] Loaded pretrained weights from {path}")


# ---------------------------------------------------------------------------
# DinoDisc
# ---------------------------------------------------------------------------
class DinoDisc(nnx.Module):
    """DINO-based discriminator with multi-head feature discrimination."""

    def __init__(
        self,
        dino_ckpt_path: str,
        ks: int = 3,
        key_depths: Tuple[int, ...] = (2, 5, 8, 11),
        norm_type: str = "bn",
        using_spec_norm: bool = True,
        norm_eps: float = 1e-6,
        recipe: str = "S_8",
        *,
        rng: jax.random.PRNGKey,
    ):
        rngs = nnx.Rngs(int(rng[0]))
        self.dino = FrozenDINONoDrop(recipe=recipe, rngs=rngs)
        self.dino.load_pretrained_torch(dino_ckpt_path)

        dino_C = self.dino.embed_dim
        num_heads = len(self.dino.key_depths) + 1

        self.heads = [
            DiscHead(dino_C, kernel_size=ks, norm_eps=norm_eps, rngs=rngs)
            for _ in range(num_heads)
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Images (B, C, H, W) in [-1, 1].

        Returns:
            Logits (B, total_spatial_dims).
        """
        # Frozen DINO — stop gradients
        activations = jax.lax.stop_gradient(self.dino(x))

        outputs = []
        for head, act in zip(self.heads, activations):
            out = head(act)  # (B, 1, L)
            outputs.append(out.reshape(x.shape[0], -1))

        return jnp.concatenate(outputs, axis=1)
