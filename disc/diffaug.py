"""DiffAug — differentiable augmentation for GAN training. JAX port."""

from __future__ import annotations

import jax
import jax.numpy as jnp


class DiffAug:
    """Differentiable augmentation: translation + color jitter + cutout.

    Port of PyTorch DiffAug. All operations are JAX-compatible and
    differentiable. Requires explicit PRNG key.
    """

    def __init__(self, prob: float = 1.0, cutout: float = 0.2):
        self.prob = abs(prob)
        self.using_cutout = prob > 0
        self.cutout = cutout

    def __call__(self, x: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
        """Apply augmentation.

        Args:
            x: Images (B, C, H, W) in float32.
            rng: PRNG key.
        """
        if self.prob < 1e-6:
            return x

        B, C, H, W = x.shape
        rng, rng_choice, rng_aug = jax.random.split(rng, 3)

        # Decide which augmentations to apply
        choices = jax.random.uniform(rng_choice, (3,))
        do_trans = choices[0] <= self.prob
        do_color = choices[1] <= self.prob
        do_cut = choices[2] <= self.prob

        rng_t, rng_c, rng_k = jax.random.split(rng_aug, 3)

        # --- Translation ---
        x = jax.lax.cond(do_trans, lambda: self._translate(x, rng_t), lambda: x)

        # --- Color jitter ---
        x = jax.lax.cond(do_color, lambda: self._color(x, rng_c), lambda: x)

        # --- Cutout ---
        if self.using_cutout:
            x = jax.lax.cond(do_cut, lambda: self._cutout(x, rng_k), lambda: x)

        return x

    def _translate(self, x: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
        B, C, H, W = x.shape
        ratio = 0.125
        delta_h = round(H * ratio)
        delta_w = round(W * ratio)

        rng1, rng2 = jax.random.split(rng)
        th = jax.random.randint(rng1, (B, 1, 1, 1), -delta_h, delta_h + 1)
        tw = jax.random.randint(rng2, (B, 1, 1, 1), -delta_w, delta_w + 1)

        # Create shifted coordinates
        grid_h = jnp.arange(H).reshape(1, 1, H, 1)
        grid_w = jnp.arange(W).reshape(1, 1, 1, W)

        src_h = jnp.clip(grid_h - th, 0, H - 1)
        src_w = jnp.clip(grid_w - tw, 0, W - 1)

        # Gather — (B, C, H, W)
        batch_idx = jnp.arange(B).reshape(B, 1, 1, 1)
        batch_idx = jnp.broadcast_to(batch_idx, (B, C, H, W))
        chan_idx = jnp.arange(C).reshape(1, C, 1, 1)
        chan_idx = jnp.broadcast_to(chan_idx, (B, C, H, W))
        src_h = jnp.broadcast_to(src_h, (B, C, H, W))
        src_w = jnp.broadcast_to(src_w, (B, C, H, W))

        return x[batch_idx, chan_idx, src_h, src_w]

    def _color(self, x: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
        B = x.shape[0]
        rng1, rng2, rng3 = jax.random.split(rng, 3)

        # Brightness
        brightness = jax.random.uniform(rng1, (B, 1, 1, 1)) - 0.5
        x = x + brightness

        # Saturation
        mean_ch = jnp.mean(x, axis=1, keepdims=True)
        sat_factor = jax.random.uniform(rng2, (B, 1, 1, 1)) * 2.0
        x = (x - mean_ch) * sat_factor + mean_ch

        # Contrast
        mean_all = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
        con_factor = jax.random.uniform(rng3, (B, 1, 1, 1)) + 0.5
        x = (x - mean_all) * con_factor + mean_all

        return x

    def _cutout(self, x: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
        B, C, H, W = x.shape
        cutout_h = round(H * self.cutout)
        cutout_w = round(W * self.cutout)

        rng1, rng2 = jax.random.split(rng)
        offset_h = jax.random.randint(rng1, (B, 1, 1, 1), 0, H)
        offset_w = jax.random.randint(rng2, (B, 1, 1, 1), 0, W)

        grid_h = jnp.arange(H).reshape(1, 1, H, 1)
        grid_w = jnp.arange(W).reshape(1, 1, 1, W)

        # Mask: 1 where outside cutout, 0 inside
        mask_h = (grid_h >= offset_h - cutout_h // 2) & (grid_h < offset_h + cutout_h // 2)
        mask_w = (grid_w >= offset_w - cutout_w // 2) & (grid_w < offset_w + cutout_w // 2)
        mask = ~(mask_h & mask_w)
        mask = mask.astype(x.dtype)

        return x * mask
