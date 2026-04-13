"""GAN loss functions — JAX port of PyTorch gan_loss.py."""

import jax.numpy as jnp
import jax.nn


def hinge_d_loss(logits_real: jnp.ndarray, logits_fake: jnp.ndarray) -> jnp.ndarray:
    """Hinge discriminator loss (VQGAN style)."""
    loss_real = jnp.mean(jax.nn.relu(1.0 - logits_real))
    loss_fake = jnp.mean(jax.nn.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real: jnp.ndarray, logits_fake: jnp.ndarray) -> jnp.ndarray:
    """Original GAN discriminator loss."""
    return 0.5 * (jnp.mean(jax.nn.softplus(-logits_real)) + jnp.mean(jax.nn.softplus(logits_fake)))


def vanilla_g_loss(logits_fake: jnp.ndarray) -> jnp.ndarray:
    """Original GAN generator loss."""
    return -jnp.mean(logits_fake)
