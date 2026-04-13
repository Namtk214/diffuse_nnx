"""DinoDiscriminator — thin wrapper. JAX port."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from .dinodisc import DinoDisc


class DinoDiscriminator(DinoDisc):
    """Wrapper aligning with legacy API: forward(fake, real) -> (logits_fake, logits_real)."""

    def classify(self, img: jnp.ndarray) -> jnp.ndarray:
        return super().__call__(img)

    def __call__(self, fake: jnp.ndarray, real: jnp.ndarray | None = None):
        logits_fake = self.classify(fake)
        logits_real = self.classify(real) if real is not None else None
        return logits_fake, logits_real
