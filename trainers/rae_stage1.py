"""RAE Stage 1 Training.

Uses diffuse_nnx infrastructure (data loader, checkpoint, wandb, optax).
Training loop + losses ported from rae_jax/train_stage1.py.

Optimizations for TPUv4-8:
  - jax.lax.stop_gradient on frozen encoder output
  - Proper data sharding across all TPU cores
  - donate_argnums for zero-copy buffer reuse
"""

from __future__ import annotations

import functools
import math
import time
from collections import defaultdict, deque

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np
import optax
from flax import nnx
import ml_collections
from absl import logging

# ── diffuse_nnx utils ───────────────────────────────────────────────────
from data import local_imagenet_dataset
from networks.encoders.rae import RAE
from utils import checkpoint as ckpt_utils

# ── disc module (copied from rae_jax) ───────────────────────────────────
from disc import build_discriminator, LPIPS, hinge_d_loss, vanilla_g_loss
from disc.gan_loss import vanilla_d_loss


def _extract_last_layer_kernel(tree):
    """Return the decoder prediction kernel from an nnx state/grad pytree."""
    return tree.decoder_pred.kernel


def _fake_quantize_unit_range(x: jnp.ndarray) -> jnp.ndarray:
    """Match the public PyTorch discriminator fake quantization in [-1,1]."""
    x = jnp.clip(x, -1.0, 1.0)
    return jnp.round((x + 1.0) * 127.5) / 127.5 - 1.0


# ─────────────────────────────────────────────────────────────────────────
# Helpers: differentiable encode/decode for training
# (Stock RAE.encode / RAE.decode are decorated with @nnx.jit and can't
#  be called inside our outer jax.jit. We inline the logic here.)
# ─────────────────────────────────────────────────────────────────────────
def encode_for_training(rae_model, x: jnp.ndarray, rng: jnp.ndarray = None, training: bool = False) -> jnp.ndarray:
    """Encode images (B,H,W,3) in [-1,1] → latent (B,h,w,C). No @nnx.jit.

    The encoder is frozen so we wrap with stop_gradient for TPU efficiency.
    """
    _, h, w, _ = x.shape

    encoder_mean = np.asarray(rae_model.encoder_mean).reshape(1, 1, 1, 3)
    encoder_std = np.asarray(rae_model.encoder_std).reshape(1, 1, 1, 3)

    latent_mean = (0 if rae_model.latent_mean is None
                   else np.asarray(rae_model.latent_mean).transpose(1, 2, 0)[None, ...])
    latent_var = (1 if rae_model.latent_var is None
                  else np.asarray(rae_model.latent_var).transpose(1, 2, 0)[None, ...])

    if h != rae_model.encoder_input_size or w != rae_model.encoder_input_size:
        x = jax.image.resize(
            x, (x.shape[0], rae_model.encoder_input_size, rae_model.encoder_input_size, x.shape[-1]),
            method='bicubic',
        )

    # input range [-1,1] → [0,1] → ImageNet normalised
    x = (x + 1.0) / 2.0
    x = (x - encoder_mean) / encoder_std

    # ★ stop_gradient: encoder is frozen, no grad needed → saves TPU memory
    z = jax.lax.stop_gradient(rae_model.encoder(x, deterministic=True))

    if training and getattr(rae_model, 'noise_tau', 0.0) > 0 and rng is not None:
        rng1, rng2 = jax.random.split(rng)
        rand_factor = jax.random.uniform(rng1, (z.shape[0], 1, 1))
        noise = jax.random.normal(rng2, z.shape, dtype=z.dtype)
        z = z + rae_model.noise_tau * rand_factor * noise

    b, n, c = z.shape
    hw = int(math.sqrt(n))
    z = z.reshape(b, hw, hw, c)

    z = (z - latent_mean) / jnp.sqrt(latent_var + rae_model.eps)
    return z


def decode_for_training(rae_model, z: jnp.ndarray) -> jnp.ndarray:
    """Differentiable decoder: latent (B,H,W,C) → pixels (B,3,H',W') float32 [0,1]."""
    encoder_mean = np.asarray(rae_model.encoder_mean).reshape(1, 1, 1, 3)
    encoder_std = np.asarray(rae_model.encoder_std).reshape(1, 1, 1, 3)

    latent_mean = (0 if rae_model.latent_mean is None
                   else np.asarray(rae_model.latent_mean).transpose(1, 2, 0)[None, ...])
    latent_var = (1 if rae_model.latent_var is None
                  else np.asarray(rae_model.latent_var).transpose(1, 2, 0)[None, ...])

    # Un-normalise latent
    z = z * jnp.sqrt(latent_var + rae_model.eps) + latent_mean

    b, h, w, c = z.shape
    z = z.reshape(b, h * w, c)
    dec_out = rae_model.decoder(z, drop_cls_token=False).logits
    x_rec = rae_model.decoder.unpatchify(dec_out)         # (B, 3, pH, pW) NCHW
    x_rec_nhwc = x_rec.transpose(0, 2, 3, 1)              # → NHWC
    x_rec_nhwc = x_rec_nhwc * encoder_std + encoder_mean   # de-normalise to pixel [0,1]
    x_rec_nhwc = jnp.clip(x_rec_nhwc, 0.0, 1.0)

    return x_rec_nhwc.transpose(0, 3, 1, 2)               # → NCHW float32 [0,1]


# ─────────────────────────────────────────────────────────────────────────
# Combined train step
# ─────────────────────────────────────────────────────────────────────────
@functools.partial(
    jax.jit,
    donate_argnums=(0, 1, 2, 3, 4),
    static_argnames=(
        'rae_model', 'disc_graphdef', 'lpips_model', 'diffaug',
        'gen_optimizer', 'disc_optimizer',
        'ema_decay', 'disc_start', 'disc_upd_start', 'lpips_start',
        'perceptual_weight', 'disc_weight_val', 'max_d_weight', 'disc_loss_type', 'disc_updates',
    ),
)
def train_step(
    decoder_params, decoder_opt_state, ema_params,
    disc_params, disc_opt_state,
    images,
    rae_model, disc_graphdef, lpips_model, diffaug,
    gen_optimizer, disc_optimizer,
    rng, epoch,
    ema_decay, disc_start, disc_upd_start, lpips_start,
    perceptual_weight, disc_weight_val, max_d_weight, disc_loss_type, disc_updates,
):
    rng_noise, rng_gen_aug, rng_disc = jax.random.split(rng, 3)
    rng_disc_real, rng_disc_fake = jax.random.split(rng_disc)

    # ── Generator loss ───────────────────────────────────────────────
    def _reconstruct(dec_params, image_batch, noise_rng, training: bool):
        nnx.update(rae_model.decoder, dec_params)
        z = encode_for_training(rae_model, image_batch, rng=noise_rng, training=training)
        x_rec = decode_for_training(rae_model, z)
        target = ((image_batch + 1.0) / 2.0).transpose(0, 3, 1, 2)
        return x_rec, target

    def _nll_only(dec_params):
        x_rec, target = _reconstruct(dec_params, images, rng_noise, True)
        real_normed = images.transpose(0, 3, 1, 2)
        recon_normed = x_rec * 2.0 - 1.0
        rec_loss = jnp.mean(jnp.abs(x_rec - target))
        lpips_val = jax.lax.cond(
            epoch >= lpips_start,
            lambda: lpips_model(real_normed, recon_normed),
            lambda: jnp.zeros(()),
        )
        return rec_loss + perceptual_weight * lpips_val

    def _gan_only(dec_params):
        x_rec, _ = _reconstruct(dec_params, images, rng_noise, True)
        normed_x_rec = x_rec * 2.0 - 1.0
        return jax.lax.cond(
            epoch >= disc_start,
            lambda: vanilla_g_loss(nnx.merge(disc_graphdef, disc_params).classify(diffaug(normed_x_rec, rng_gen_aug))),
            lambda: jnp.zeros(()),
        )

    def _adp_w():
        nll_grads = jax.grad(_nll_only)(decoder_params)
        gan_grads = jax.grad(_gan_only)(decoder_params)
        nll_last = _extract_last_layer_kernel(nll_grads)
        gan_last = _extract_last_layer_kernel(gan_grads)
        return jnp.clip(
            jnp.linalg.norm(nll_last) / (jnp.linalg.norm(gan_last) + 1e-6),
            0.0,
            max_d_weight,
        )

    adp_w = jax.lax.cond(epoch >= disc_start, _adp_w, lambda: jnp.zeros(()))

    def gen_loss_fn(dec_params):
        x_rec, target = _reconstruct(dec_params, images, rng_noise, True)
        real_normed = images.transpose(0, 3, 1, 2)
        normed_x_rec = x_rec * 2.0 - 1.0
        rec_loss = jnp.mean(jnp.abs(x_rec - target))
        lpips_val = jax.lax.cond(
            epoch >= lpips_start,
            lambda: lpips_model(real_normed, normed_x_rec),
            lambda: jnp.zeros(()),
        )
        nll_loss = rec_loss + perceptual_weight * lpips_val
        def _gan_g():
            x_aug = diffaug(normed_x_rec, rng_gen_aug)
            return vanilla_g_loss(nnx.merge(disc_graphdef, disc_params).classify(x_aug))
        g_loss = jax.lax.cond(epoch >= disc_start, _gan_g, lambda: jnp.zeros(()))
        total = nll_loss + disc_weight_val * adp_w * g_loss
        return total, (rec_loss, lpips_val, nll_loss, g_loss, x_rec)

    (total_loss, (rec_loss, lpips_val, nll_loss, g_loss, x_rec)), grads = \
        jax.value_and_grad(gen_loss_fn, has_aux=True)(decoder_params)

    gen_upd, new_dec_opt = gen_optimizer.update(grads, decoder_opt_state, decoder_params)
    new_dec_params = optax.apply_updates(decoder_params, gen_upd)

    # EMA update
    new_ema = jax.tree.map(
        lambda e, p: ema_decay * e + (1 - ema_decay) * p,
        ema_params, new_dec_params,
    )

    # ── Discriminator loss ───────────────────────────────────────────
    def _disc_step(dp, dos):
        fresh_recon, _ = _reconstruct(new_dec_params, images, rng_noise, False)
        def disc_loss_fn(d_params):
            disc = nnx.merge(disc_graphdef, d_params)
            # Disc expects NCHW [-1,1]
            real_nchw = images.transpose(0, 3, 1, 2)   # NHWC [-1,1] → NCHW [-1,1]
            normed_fake = _fake_quantize_unit_range(jax.lax.stop_gradient(fresh_recon) * 2.0 - 1.0)  # [0,1] → [-1,1]
            real_aug = diffaug(real_nchw, rng_disc_real)
            fake_aug = diffaug(normed_fake, rng_disc_fake)
            lr, lf = disc.classify(real_aug), disc.classify(fake_aug)
            disc_loss = hinge_d_loss(lr, lf) if disc_loss_type == "hinge" else vanilla_d_loss(lr, lf)
            return disc_loss * disc_weight_val
        def one_disc_update(i, carry):
            cur_dp, cur_dos, loss_sum = carry
            d_loss, d_grads = jax.value_and_grad(disc_loss_fn)(cur_dp)
            d_upd, next_dos = disc_optimizer.update(d_grads, cur_dos, cur_dp)
            next_dp = optax.apply_updates(cur_dp, d_upd)
            return next_dp, next_dos, loss_sum + d_loss
        next_dp, next_dos, loss_sum = jax.lax.fori_loop(
            0,
            disc_updates,
            one_disc_update,
            (dp, dos, jnp.zeros(())),
        )
        return next_dp, next_dos, loss_sum / disc_updates

    new_disc_params, new_disc_opt, disc_loss = jax.lax.cond(
        epoch >= disc_upd_start,
        _disc_step,
        lambda dp, dos: (dp, dos, jnp.zeros(())),
        disc_params, disc_opt_state,
    )

    metrics = {
        "total_loss": total_loss, "rec_loss": rec_loss,
        "lpips_loss": lpips_val,  "g_loss":   g_loss,
        "d_loss":     disc_loss,  "d_weight":  adp_w,
    }
    return new_dec_params, new_dec_opt, new_ema, new_disc_params, new_disc_opt, metrics


# ─────────────────────────────────────────────────────────────────────────
# Scan-based gradient accumulation (single JIT dispatch per optimizer step)
# ─────────────────────────────────────────────────────────────────────────
@functools.partial(
    jax.jit,
    donate_argnums=(0, 1, 2, 3, 4),
    static_argnames=(
        'rae_model', 'disc_graphdef', 'lpips_model', 'diffaug',
        'gen_optimizer', 'disc_optimizer',
        'ema_decay', 'disc_start', 'disc_upd_start', 'lpips_start',
        'perceptual_weight', 'disc_weight_val', 'max_d_weight', 'disc_loss_type', 'disc_updates',
    ),
)
def train_step_accumulated(
    decoder_params, decoder_opt_state, ema_params,
    disc_params, disc_opt_state,
    images_stacked,            # [accum_steps, micro_bs, H, W, C]
    rae_model, disc_graphdef, lpips_model, diffaug,
    gen_optimizer, disc_optimizer,
    rng, epoch,
    ema_decay, disc_start, disc_upd_start, lpips_start,
    perceptual_weight, disc_weight_val, max_d_weight, disc_loss_type, disc_updates,
):
    """Full optimizer step with gradient accumulation using jax.lax.scan.

    Scans over accum_steps micro-batches inside XLA — only 1 JIT dispatch.
    """
    accum_steps = images_stacked.shape[0]

    # ── Generator micro-batch grad computation ───────────────────────
    def gen_micro_step(carry, xs):
        """carry = (acc_gen_grads, acc_metrics, rng)"""
        acc_gen, acc_m, rng_carry = carry
        images_mb = xs  # [micro_bs, H, W, C]

        rng_carry, rng_step = jax.random.split(rng_carry)
        rng_noise, rng_gen_aug = jax.random.split(rng_step, 2)

        def _reconstruct(dec_params, training: bool):
            nnx.update(rae_model.decoder, dec_params)
            z = encode_for_training(rae_model, images_mb, rng=rng_noise, training=True)
            x_rec = decode_for_training(rae_model, z)    # NCHW [0,1]
            target = ((images_mb + 1.0) / 2.0).transpose(0, 3, 1, 2)  # NCHW [0,1]
            return x_rec, target

        def _nll_only(dec_params):
            x_rec, target = _reconstruct(dec_params, True)
            real_normed = images_mb.transpose(0, 3, 1, 2)
            recon_normed = x_rec * 2.0 - 1.0
            rec_loss = jnp.mean(jnp.abs(x_rec - target))
            lpips_val = jax.lax.cond(
                epoch >= lpips_start,
                lambda: lpips_model(real_normed, recon_normed),
                lambda: jnp.zeros(()),
            )
            return rec_loss + perceptual_weight * lpips_val

        def _gan_only(dec_params):
            x_rec, _ = _reconstruct(dec_params, True)
            normed_x_rec = x_rec * 2.0 - 1.0
            return jax.lax.cond(
                epoch >= disc_start,
                lambda: vanilla_g_loss(nnx.merge(disc_graphdef, disc_params).classify(diffaug(normed_x_rec, rng_gen_aug))),
                lambda: jnp.zeros(()),
            )

        def _adp_w():
            nll_grads = jax.grad(_nll_only)(decoder_params)
            gan_grads = jax.grad(_gan_only)(decoder_params)
            nll_last = _extract_last_layer_kernel(nll_grads)
            gan_last = _extract_last_layer_kernel(gan_grads)
            return jnp.clip(
                jnp.linalg.norm(nll_last) / (jnp.linalg.norm(gan_last) + 1e-6),
                0.0,
                max_d_weight,
            )

        adp_w = jax.lax.cond(epoch >= disc_start, _adp_w, lambda: jnp.zeros(()))

        def gen_loss_fn(dec_params):
            x_rec, target = _reconstruct(dec_params, True)
            real_normed = images_mb.transpose(0, 3, 1, 2)
            normed_x_rec = x_rec * 2.0 - 1.0
            rec_loss = jnp.mean(jnp.abs(x_rec - target))
            lpips_val = jax.lax.cond(
                epoch >= lpips_start,
                lambda: lpips_model(real_normed, normed_x_rec),
                lambda: jnp.zeros(()),
            )
            nll_loss = rec_loss + perceptual_weight * lpips_val
            def _gan_g():
                x_aug = diffaug(normed_x_rec, rng_gen_aug)
                return vanilla_g_loss(nnx.merge(disc_graphdef, disc_params).classify(x_aug))
            g_loss = jax.lax.cond(epoch >= disc_start, _gan_g, lambda: jnp.zeros(()))
            total = nll_loss + disc_weight_val * adp_w * g_loss
            return total, (rec_loss, lpips_val, nll_loss, g_loss, x_rec)

        (total_loss, (rec_loss, lpips_val, nll_loss, g_loss, x_rec)), gen_grads = \
            jax.value_and_grad(gen_loss_fn, has_aux=True)(decoder_params)

        # Accumulate
        new_acc_gen  = jax.tree.map(lambda a, b: a + b, acc_gen,  gen_grads)
        new_acc_m    = {
            "total_loss": acc_m["total_loss"] + total_loss,
            "rec_loss":   acc_m["rec_loss"]   + rec_loss,
            "lpips_loss": acc_m["lpips_loss"] + lpips_val,
            "g_loss":     acc_m["g_loss"]      + g_loss,
            "d_weight":   acc_m["d_weight"]    + adp_w,
        }
        return (new_acc_gen, new_acc_m, rng_carry), None

    # Initialise accumulators
    zero_gen  = jax.tree.map(jnp.zeros_like, decoder_params)
    zero_metrics = {k: jnp.zeros(()) for k in
                    ["total_loss", "rec_loss", "lpips_loss", "g_loss", "d_weight"]}

    (acc_gen, acc_m, disc_rng), _ = jax.lax.scan(
        gen_micro_step,
        (zero_gen, zero_metrics, rng),
        images_stacked,           # xs: [accum_steps, micro_bs, H, W, C]
    )

    # Average
    avg_gen  = jax.tree.map(lambda g: g / accum_steps, acc_gen)
    metrics  = {k: v / accum_steps for k, v in acc_m.items()}

    # ── Apply gradients ──────────────────────────────────────────────
    gen_upd, new_dec_opt = gen_optimizer.update(avg_gen, decoder_opt_state, decoder_params)
    new_dec_params = optax.apply_updates(decoder_params, gen_upd)

    new_ema = jax.tree.map(
        lambda e, p: ema_decay * e + (1 - ema_decay) * p,
        ema_params, new_dec_params,
    )

    # ── Discriminator gradients after generator update ───────────────
    def disc_micro_step(active_disc_params, carry, xs):
        acc_disc, acc_dloss, rng_carry = carry
        images_mb = xs
        rng_carry, rng_step = jax.random.split(rng_carry)
        rng_disc_real, rng_disc_fake = jax.random.split(rng_step)

        def _disc_grads_fn(dp):
            nnx.update(rae_model.decoder, new_dec_params)
            fresh_z = encode_for_training(rae_model, images_mb, rng=None, training=False)
            fresh_rec = decode_for_training(rae_model, fresh_z)
            def disc_loss_fn(d_params):
                disc = nnx.merge(disc_graphdef, d_params)
                real_nchw = images_mb.transpose(0, 3, 1, 2)
                normed_fake = _fake_quantize_unit_range(jax.lax.stop_gradient(fresh_rec) * 2.0 - 1.0)
                real_aug = diffaug(real_nchw, rng_disc_real)
                fake_aug = diffaug(normed_fake, rng_disc_fake)
                lr, lf = disc.classify(real_aug), disc.classify(fake_aug)
                disc_loss = hinge_d_loss(lr, lf) if disc_loss_type == "hinge" else vanilla_d_loss(lr, lf)
                return disc_loss * disc_weight_val
            d_loss, d_grads = jax.value_and_grad(disc_loss_fn)(dp)
            return d_grads, d_loss

        disc_grads, disc_loss = jax.lax.cond(
            epoch >= disc_upd_start,
            _disc_grads_fn,
            lambda dp: (jax.tree.map(jnp.zeros_like, dp), jnp.zeros(())),
            active_disc_params,
        )
        new_acc_disc = jax.tree.map(lambda a, b: a + b, acc_disc, disc_grads)
        return (new_acc_disc, acc_dloss + disc_loss, rng_carry), None

    zero_disc = jax.tree.map(jnp.zeros_like, disc_params)

    def one_disc_update(i, carry):
        cur_disc_params, cur_disc_opt, loss_sum = carry
        (acc_disc, acc_dloss, _), _ = jax.lax.scan(
            lambda inner_carry, xs: disc_micro_step(cur_disc_params, inner_carry, xs),
            (zero_disc, jnp.zeros(()), disc_rng),
            images_stacked,
        )
        avg_disc = jax.tree.map(lambda g: g / accum_steps, acc_disc)
        disc_upd, next_disc_opt = disc_optimizer.update(avg_disc, cur_disc_opt, cur_disc_params)
        next_disc_params = optax.apply_updates(cur_disc_params, disc_upd)
        return next_disc_params, next_disc_opt, loss_sum + (acc_dloss / accum_steps)

    def run_disc_updates(dp, dos):
        return jax.lax.fori_loop(
            0,
            disc_updates,
            one_disc_update,
            (dp, dos, jnp.zeros(())),
        )

    new_disc_params, new_disc_opt, disc_loss_sum = jax.lax.cond(
        epoch >= disc_upd_start,
        run_disc_updates,
        lambda dp, dos: (dp, dos, jnp.zeros(())),
        disc_params,
        disc_opt_state,
    )
    metrics["d_loss"] = disc_loss_sum / jnp.maximum(disc_updates, 1)
    return new_dec_params, new_dec_opt, new_ema, new_disc_params, new_disc_opt, metrics


# ─────────────────────────────────────────────────────────────────────────
# Validation step (uses EMA decoder)
# ─────────────────────────────────────────────────────────────────────────
@functools.partial(jax.jit, static_argnames=('rae_model', 'lpips_model'))
def valid_step_fn(decoder_params, images, rae_model, lpips_model):
    """Compute val rec_loss & LPIPS with EMA decoder params."""
    nnx.update(rae_model.decoder, decoder_params)
    z = encode_for_training(rae_model, images, training=False)
    x_rec = decode_for_training(rae_model, z)
    target = ((images + 1.0) / 2.0).transpose(0, 3, 1, 2)
    rec_loss = jnp.mean(jnp.abs(x_rec - target))
    real_normed = images.transpose(0, 3, 1, 2)
    recon_normed = x_rec * 2.0 - 1.0
    lpips_val = lpips_model(real_normed, recon_normed)
    return rec_loss, lpips_val, x_rec, target


# ─────────────────────────────────────────────────────────────────────────
# WandB image grid helper
# ─────────────────────────────────────────────────────────────────────────
def _make_comparison_grid(originals, reconstructions, n=8):
    """Top row = original, bottom row = reconstruction.

    Args:
        originals: (B, 3, H, W) float32 [0,1]
        reconstructions: (B, 3, H, W) float32 [0,1]
    Returns:
        (H*2, W*n, 3) uint8 numpy array
    """
    n = min(n, originals.shape[0])
    orig = np.array(originals[:n]).transpose(0, 2, 3, 1)   # NCHW → NHWC
    recs = np.array(reconstructions[:n]).transpose(0, 2, 3, 1)
    orig = np.clip(orig * 255, 0, 255).astype(np.uint8)
    recs = np.clip(recs * 255, 0, 255).astype(np.uint8)

    top_row  = np.concatenate(list(orig), axis=1)
    bot_row  = np.concatenate(list(recs), axis=1)
    return np.concatenate([top_row, bot_row], axis=0)


# ─────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────
def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    import sys, os
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    # ── Mesh ────────────────────────────────────────────────────────
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, ('data',))
    data_sharding = NamedSharding(mesh, P('data'))
    repl_sharding = NamedSharding(mesh, P())

    # ── Dataset ─────────────────────────────────────────────────────
    dataset = local_imagenet_dataset.build_imagenet_dataset(
        is_train=True,
        data_dir=config.data.data_dir,
        image_size=config.data.image_size,
        latent_dataset=False,
    )
    # Loader uses micro_batch_size so each forward pass fits in TPU memory
    _loader_config = config.copy_and_resolve_references()
    _loader_config.data.batch_size = config.data.get('micro_batch_size', config.data.batch_size)
    loader = local_imagenet_dataset.build_imagenet_loader(_loader_config, dataset)

    # ── Models ──────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(config.seed)
    rng, rng_rae, rng_disc, rng_lpips = jax.random.split(rng, 4)

    rae_model = RAE(
        config=config.encoder,
        pretrained_path=config.encoder.pretrained_path,
        resolution=config.encoder.resolution,
        encoded_pixels=config.encoder.get('encoded_pixels', True),
        rngs=nnx.Rngs(int(rng_rae[0])),
    )
    # Attach noise_tau so encode_for_training can access it
    rae_model.noise_tau = float(config.encoder.get('noise_tau', 0.0))
    logging.info(f'[RAE Stage1] noise_tau={rae_model.noise_tau}')

    disc_model, diffaug = build_discriminator(config.gan.disc, rng=rng_disc)
    lpips_model = LPIPS(rngs=nnx.Rngs(int(rng_lpips[0])))

    # ── Optimizers ──────────────────────────────────────────────────
    steps_per_epoch = config.data.num_train_samples // config.data.batch_size
    total_steps     = config.stage1.epochs * steps_per_epoch
    warmup_steps    = config.stage1.warmup_epochs * steps_per_epoch

    logging.info(f'[RAE Stage1] steps_per_epoch={steps_per_epoch} total_steps={total_steps} warmup_steps={warmup_steps}')

    def make_optimizer(lr):
        schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(0.0, lr, warmup_steps),
                optax.cosine_decay_schedule(lr, total_steps - warmup_steps, alpha=2e-5 / lr),
            ],
            boundaries=[warmup_steps],
        )
        return optax.chain(
            optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, weight_decay=0.0),
        )

    gen_optimizer  = make_optimizer(config.stage1.lr)
    disc_optimizer = make_optimizer(config.stage1.lr)

    decoder_params    = nnx.state(rae_model.decoder)
    decoder_opt_state = gen_optimizer.init(decoder_params)
    ema_params        = jax.tree.map(jnp.copy, decoder_params)

    disc_params    = nnx.state(disc_model)
    disc_opt_state = disc_optimizer.init(disc_params)
    disc_graphdef, _ = nnx.split(disc_model)

    # ── Resume from latest checkpoint if available ───────────────────
    import orbax.checkpoint as ocp
    from etils import epath
    import shutil

    resume_epoch = 0
    _workdir_path = epath.Path(workdir)
    _existing = sorted(
        [d for d in _workdir_path.iterdir()
         if d.name.startswith('checkpoint_epoch_') and not d.name.endswith('-tmp')],
        key=lambda p: int(p.name.split('_')[-1])
    ) if _workdir_path.exists() else []

    if _existing:
        _latest = _existing[-1]
        logging.info(f'[RAE Stage1] Resuming from {_latest}')
        _ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
        _restored = _ckptr.restore(
            _latest,
            args=ocp.args.Composite(
                decoder_params=ocp.args.StandardRestore(decoder_params),
                ema_params=ocp.args.StandardRestore(ema_params),
                disc_params=ocp.args.StandardRestore(disc_params),
                decoder_opt_state=ocp.args.StandardRestore(decoder_opt_state),
                disc_opt_state=ocp.args.StandardRestore(disc_opt_state),
                metadata=ocp.args.JsonRestore({}),
            )
        )
        decoder_params    = _restored.decoder_params
        ema_params        = _restored.ema_params
        disc_params       = _restored.disc_params
        decoder_opt_state = _restored.decoder_opt_state
        disc_opt_state    = _restored.disc_opt_state
        _meta             = _restored.metadata or {}
        resume_epoch      = int(_meta.get('epoch', 0)) + 1
        logging.info(f'[RAE Stage1] Resumed: will train epoch {resume_epoch}→{config.stage1.epochs - 1}')
        del _ckptr, _restored, _meta

        # Reshard restored params to replicated sharding across all TPU devices
        repl_sharding = NamedSharding(Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ('data',)), P())
        decoder_params    = jax.device_put(decoder_params, repl_sharding)
        ema_params        = jax.device_put(ema_params, repl_sharding)
        disc_params       = jax.device_put(disc_params, repl_sharding)
        decoder_opt_state = jax.device_put(decoder_opt_state, repl_sharding)
        disc_opt_state    = jax.device_put(disc_opt_state, repl_sharding)
        logging.info('[RAE Stage1] Params resharded to all TPU devices')
    else:
        logging.info('[RAE Stage1] No checkpoint found, starting from scratch')

    # ── WandB ───────────────────────────────────────────────────────
    if jax.process_index() == 0:
        import wandb
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        entity = os.environ.get("WANDB_ENTITY", None)
        wandb.init(
            entity=entity,
            project='RAE_Stage1',
            name=config.exp_name,
            config=config.to_dict(),
            resume='allow',
            reinit=True,
        )

    # ── Config shortcuts ────────────────────────────────────────────
    loss_cfg       = config.stage1.loss
    log_every      = config.log_every_steps
    sample_every   = config.get('sample_every_steps', 500)
    metrics_buf    = defaultdict(lambda: deque(maxlen=log_every * 2))
    loader_iter    = iter(loader)
    step           = resume_epoch * steps_per_epoch
    t0             = time.time()

    micro_bs   = config.data.get('micro_batch_size', config.data.batch_size)
    accum_steps = config.data.batch_size // micro_bs
    logging.info(f'[RAE Stage1] batch_size={config.data.batch_size} micro_batch_size={micro_bs} accum_steps={accum_steps}')
    logging.info(f'[RAE Stage1] Training from epoch={resume_epoch} step={step} ...')

    # ── rFID evaluation setup ────────────────────────────────────────
    rfid_every = config.stage1.get('rfid_every_steps', 100)
    rfid_num_images = 2048  # fast eval
    rfid_eval_bs = 64

    # Pre-load InceptionV3 detector (one-time)
    from eval import utils as eval_utils, fid as fid_module
    import flax
    from eval import inception as inception_module
    _inception = inception_module.InceptionV3(pretrained=True)
    _inception_params = _inception.init(jax.random.PRNGKey(0), jnp.ones((1, 299, 299, 3)))
    _inception_params_repl = flax.jax_utils.replicate(_inception_params)

    def _inception_fwd(params, x):
        x = x.astype(jnp.float32) / 127.5 - 1
        x = jax.image.resize(x, (x.shape[0], 299, 299, x.shape[-1]), method='bilinear')
        features = _inception.apply(params, x, train=False).squeeze(axis=(1, 2))
        features = jax.lax.all_gather(features, axis_name='data', tiled=True)
        return features
    _inception_forward = jax.pmap(_inception_fwd, axis_name='data')

    # Pre-load rFID eval images (random subset from train set, in [0,255] NHWC)
    import torch as _torch
    _rfid_loader = _torch.utils.data.DataLoader(dataset, batch_size=rfid_eval_bs, num_workers=0, shuffle=True)
    _rfid_real_imgs = []
    _rfid_count = 0
    for _rb, _ in _rfid_loader:
        # _rb is NCHW [-1,1] → NHWC [0,255]
        _rb_nhwc = _rb.permute(0, 2, 3, 1).numpy()
        _rb_255 = ((_rb_nhwc + 1.0) / 2.0 * 255.0)
        _rfid_real_imgs.append(_rb_255)
        _rfid_count += _rb_255.shape[0]
        if _rfid_count >= rfid_num_images:
            break
    _rfid_real_imgs = np.concatenate(_rfid_real_imgs, axis=0)[:rfid_num_images]
    logging.info(f'[RAE Stage1] Pre-loaded {_rfid_real_imgs.shape[0]} images for rFID eval')

    # Pre-compute real stats (one-time)
    _rfid_real_stats = fid_module.calculate_stats_for_iterable(
        _rfid_real_imgs, _inception_forward, _inception_params_repl,
        batch_size=rfid_eval_bs, verbose=False
    )
    logging.info('[RAE Stage1] Pre-computed real Inception stats for rFID')

    # Keep raw images in NHWC [-1,1] for reconstruction (back-convert from [0,255])
    _rfid_raw_nhwc = (_rfid_real_imgs / 255.0) * 2.0 - 1.0  # [0,255] → [-1,1]

    def compute_rfid(dec_params_for_eval):
        """Compute rFID using current decoder params."""
        nnx.update(rae_model.decoder, dec_params_for_eval)

        recon_imgs_255 = []
        for i in range(0, rfid_num_images, rfid_eval_bs):
            batch_nhwc = _rfid_raw_nhwc[i:i+rfid_eval_bs]
            batch_jax = jax.device_put(batch_nhwc, data_sharding)
            z = encode_for_training(rae_model, batch_jax, training=False)
            x_rec = decode_for_training(rae_model, z)  # NCHW [0,1]
            # → NHWC [0,255]
            x_rec_nhwc = jax.device_get(x_rec).transpose(0, 2, 3, 1)
            recon_imgs_255.append(x_rec_nhwc * 255.0)

        recon_imgs_255 = np.concatenate(recon_imgs_255, axis=0)
        recon_stats = fid_module.calculate_stats_for_iterable(
            recon_imgs_255, _inception_forward, _inception_params_repl,
            batch_size=rfid_eval_bs, verbose=False
        )
        return eval_utils.calculate_fid(recon_stats, _rfid_real_stats)

    with mesh:
        for epoch in range(resume_epoch, config.stage1.epochs):
            for _ in range(steps_per_epoch):

                # ── Collect accum_steps micro-batches, stack into one tensor ─
                micro_list = []
                for _ in range(accum_steps):
                    try:
                        images_pt, _ = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(loader)
                        images_pt, _ = next(loader_iter)
                    # NCHW [-1,1] → NHWC [-1,1], keep as numpy
                    micro_list.append(images_pt.permute(0, 2, 3, 1).numpy())

                # [accum_steps, micro_bs, H, W, C] — put on devices with data sharding on batch dim
                images_stacked = jnp.stack([
                    jax.device_put(m, data_sharding) for m in micro_list
                ], axis=0)

                rng, rng_step = jax.random.split(rng)

                # Single JIT dispatch — scan runs inside XLA
                (decoder_params, decoder_opt_state, ema_params,
                 disc_params, disc_opt_state, step_metrics) = train_step_accumulated(
                    decoder_params, decoder_opt_state, ema_params,
                    disc_params, disc_opt_state,
                    images_stacked,
                    rae_model, disc_graphdef, lpips_model, diffaug,
                    gen_optimizer, disc_optimizer,
                    rng_step, epoch,
                 ema_decay        = config.stage1.ema_decay,
                    disc_start       = loss_cfg.disc_start,
                    disc_upd_start   = loss_cfg.disc_upd_start,
                    lpips_start      = loss_cfg.lpips_start,
                    perceptual_weight= loss_cfg.perceptual_weight,
                    disc_weight_val  = loss_cfg.disc_weight,
                    max_d_weight     = loss_cfg.get('max_d_weight', 10000.0),
                    disc_loss_type   = loss_cfg.disc_loss,
                    disc_updates     = loss_cfg.get('disc_updates', 1),
                )

                for k, v in step_metrics.items():
                    metrics_buf[k].append(float(v))

                if step == 0:
                    logging.info('[RAE Stage1] Initial compilation completed.')

                # ── Logging ──────────────────────────────────────
                if (step + 1) % log_every == 0 and jax.process_index() == 0:
                    sps = log_every / (time.time() - t0)
                    summary = {
                        f"train/{k}": sum(list(metrics_buf[k])[-log_every:]) / log_every
                        for k in metrics_buf
                    }

                    # ── Val loss (EMA decoder on next batch) ─────
                    try:
                        val_images_pt, _ = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(loader)
                        val_images_pt, _ = next(loader_iter)
                    val_images_np = val_images_pt.permute(0, 2, 3, 1).numpy()
                    val_images = jax.device_put(val_images_np, data_sharding)

                    val_rec, val_lpips, _, _ = valid_step_fn(
                        ema_params, val_images, rae_model, lpips_model
                    )
                    summary["val/rec_loss"]   = float(val_rec)
                    summary["val/lpips_loss"] = float(val_lpips)

                    summary["train/epoch"]         = epoch
                    summary["train/step"]          = step + 1
                    summary["train/steps_per_sec"] = sps

                    import wandb
                    wandb.log({k: v for k, v in summary.items()}, step=step + 1)

                    print(f"[Step {step+1}/{total_steps}] "
                          f"epoch={epoch} "
                          f"loss={summary.get('train/total_loss', 0):.4f} "
                          f"rec={summary.get('train/rec_loss', 0):.4f} "
                          f"lpips={summary.get('train/lpips_loss', 0):.4f} "
                          f"g={summary.get('train/g_loss', 0):.4f} "
                          f"d={summary.get('train/d_loss', 0):.4f} "
                          f"dw={summary.get('train/d_weight', 0):.4f} "
                          f"val_rec={summary.get('val/rec_loss', 0):.4f} "
                          f"val_lpips={summary.get('val/lpips_loss', 0):.4f} "
                          f"({sps:.1f} steps/s)",
                          flush=True)

                    metrics_buf = defaultdict(lambda: deque(maxlen=log_every * 2))
                    t0 = time.time()

                # ── Log reconstruction images ────────────────────
                if (step + 1) % sample_every == 0 and jax.process_index() == 0:
                    try:
                        viz_pt, _ = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(loader)
                        viz_pt, _ = next(loader_iter)
                    viz_np = viz_pt.permute(0, 2, 3, 1).numpy()
                    viz_img = jax.device_put(viz_np, data_sharding)

                    _, _, viz_rec, viz_tgt = valid_step_fn(
                        ema_params, viz_img, rae_model, lpips_model
                    )

                    grid = _make_comparison_grid(
                        jax.device_get(viz_tgt),
                        jax.device_get(viz_rec),
                        n=8,
                    )

                    import wandb
                    wandb.log({
                        "reconstructions": wandb.Image(
                            grid, caption=f"Top: Original, Bottom: Reconstruction (step {step+1})"
                        ),
                    }, step=step + 1)
                    logging.info(f'[RAE Stage1] Logged reconstruction images at step {step+1}')

                # ── rFID evaluation ────────────────────────────
                if (step + 1) % rfid_every == 0 and jax.process_index() == 0:
                    rfid_score = compute_rfid(ema_params)
                    import wandb
                    wandb.log({"eval/rFID": rfid_score, "train/step": step + 1}, step=step + 1)
                    print(f"[Step {step+1}] rFID = {rfid_score:.2f}", flush=True)

                step += 1

            # ── Checkpoint periodically ──────────────────────
            ckpt_interval = getattr(config.stage1, 'checkpoint_interval', 1)
            is_final_epoch = (epoch == config.stage1.epochs - 1)
            if jax.process_index() == 0 and ((epoch + 1) % ckpt_interval == 0 or is_final_epoch):
                logging.info(f'[RAE Stage1] Saving checkpoint at epoch {epoch}, step {step}')
                import orbax.checkpoint as ocp
                from etils import epath

                ckpt_path = epath.Path(workdir) / f'checkpoint_epoch_{epoch}'
                ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
                saved_dec = jax.device_get(decoder_params)
                saved_ema = jax.device_get(ema_params)
                saved_disc = jax.device_get(disc_params)
                saved_dec_opt = jax.device_get(decoder_opt_state)
                saved_disc_opt = jax.device_get(disc_opt_state)
                ckptr.save(
                    ckpt_path,
                    args=ocp.args.Composite(
                        decoder_params=ocp.args.StandardSave(saved_dec),
                        ema_params=ocp.args.StandardSave(saved_ema),
                        disc_params=ocp.args.StandardSave(saved_disc),
                        decoder_opt_state=ocp.args.StandardSave(saved_dec_opt),
                        disc_opt_state=ocp.args.StandardSave(saved_disc_opt),
                        metadata=ocp.args.JsonSave({'epoch': epoch, 'step': step}),
                    )
                )
                del saved_dec, saved_ema, saved_disc, saved_dec_opt, saved_disc_opt
                logging.info(f'[RAE Stage1] Checkpoint saved at {ckpt_path}')

                # Delete older epoch checkpoints to free disk space (keep only latest)
                _all_ckpts = sorted(
                    [d for d in epath.Path(workdir).iterdir()
                     if d.name.startswith('checkpoint_epoch_') and not d.name.endswith('-tmp')],
                    key=lambda p: int(p.name.split('_')[-1])
                )
                for _old in _all_ckpts[:-1]:  # keep only the last one
                    shutil.rmtree(str(_old))
                    logging.info(f'[RAE Stage1] Deleted old checkpoint {_old.name}')

    logging.info(f'[RAE Stage1] Training complete. Final step: {step}')
    return {}

