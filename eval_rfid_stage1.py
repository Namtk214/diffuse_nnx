"""Evaluate rFID of RAE Stage 1 decoder using diffuse_nnx InceptionV3 pipeline.

Loads EMA decoder weights from the latest checkpoint, reconstructs 4096 images,
and computes rFID using the same InceptionV3 FID pipeline used during training.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import jax
import jax.numpy as jnp
from flax import nnx
import flax
import torch
import numpy as np
from absl import logging, app
import orbax.checkpoint as ocp
from etils import epath

from eval import utils as eval_utils
from eval import fid
from eval import inception as inception_module
from data import local_imagenet_dataset
from configs.rae_stage1_celebahq import get_config
from trainers.rae_stage1 import encode_for_training, decode_for_training
import ml_collections
from tqdm import tqdm


def main(argv):
    config = get_config()
    logging.set_verbosity(logging.INFO)

    num_eval_images = 4096
    eval_batch_size = 64

    # 1. Setup dataloader
    print('=== Step 1: Loading CelebAHQ dataset ===', flush=True)
    dataset = local_imagenet_dataset.build_imagenet_dataset(
        is_train=True,
        data_dir='/home/ngothanhnam508/data/celebahq',
        image_size=config.data.image_size  # 256, matching training loop
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=eval_batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=False
    )

    # 2. Setup Model
    print('=== Step 2: Setting up RAE model ===', flush=True)
    from networks.encoders.rae import RAE

    rng = jax.random.PRNGKey(config.seed)
    rng_rae = jax.random.split(rng, 2)[1]

    rae_model = RAE(
        config=config.encoder,
        pretrained_path=config.encoder.pretrained_path,
        resolution=config.encoder.resolution,
        encoded_pixels=config.encoder.get('encoded_pixels', True),
        rngs=nnx.Rngs(int(rng_rae[0])),
    )

    # 3. Load EMA weights from checkpoint
    workdir = '/home/ngothanhnam508/diffuse_nnx/RAE-Stage1-CelebAHQ'
    import glob
    checkpoints = glob.glob(os.path.join(workdir, 'checkpoint_epoch_*'))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {workdir}")

    latest_ckpt = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f'  Loading EMA weights from {latest_ckpt}...', flush=True)

    ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    decoder_params = nnx.state(rae_model.decoder)
    state = ckptr.restore(
        epath.Path(latest_ckpt),
        args=ocp.args.Composite(
            ema_params=ocp.args.StandardRestore(decoder_params)
        )
    )
    nnx.update(rae_model.decoder, state['ema_params'])
    print('  Loaded EMA decoder params.', flush=True)

    # 4. Extract images into memory
    print(f'=== Step 3: Extracting {num_eval_images} images ===', flush=True)
    all_images_nhwc = []  # NHWC [-1,1] float32
    total = 0
    for batch, _ in loader:
        # batch is NCHW [-1,1] from the dataloader -> convert to NHWC
        nhwc = batch.permute(0, 2, 3, 1).numpy()
        all_images_nhwc.append(nhwc)
        total += nhwc.shape[0]
        if total >= num_eval_images:
            break

    all_images_nhwc = np.concatenate(all_images_nhwc, axis=0)[:num_eval_images]
    print(f"  Loaded {all_images_nhwc.shape[0]} images, range: [{all_images_nhwc.min():.2f}, {all_images_nhwc.max():.2f}]", flush=True)

    # 5. Reconstruct images
    print(f'=== Step 4: Reconstructing {num_eval_images} images ===', flush=True)

    mesh = jax.sharding.Mesh(jax.local_devices(), ('data',))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data'))

    recon_nhwc_01 = []  # NHWC [0,1]
    for i in tqdm(range(0, num_eval_images, eval_batch_size), desc="Reconstructing"):
        batch = all_images_nhwc[i:i+eval_batch_size]
        batch_jax = jnp.array(batch)

        z = encode_for_training(rae_model, batch_jax, training=False)
        x_rec = decode_for_training(rae_model, z)  # NCHW [0,1]

        # NCHW -> NHWC
        x_rec_np = jax.device_get(x_rec)
        x_rec_nhwc = x_rec_np.transpose(0, 2, 3, 1)  # NHWC [0,1]
        recon_nhwc_01.append(x_rec_nhwc)

        if i == 0:
            print(f"  x_rec shape: {x_rec_np.shape}, range: [{x_rec_np.min():.3f}, {x_rec_np.max():.3f}]", flush=True)

    recon_nhwc_01 = np.concatenate(recon_nhwc_01, axis=0)
    print(f"  Reconstructed shape: {recon_nhwc_01.shape}, range: [{recon_nhwc_01.min():.3f}, {recon_nhwc_01.max():.3f}]", flush=True)

    # Pixel-level comparison
    real_01 = (all_images_nhwc + 1.0) / 2.0
    mse = np.mean((real_01 - recon_nhwc_01) ** 2)
    print(f"  Pixel MSE (float [0,1]): {mse:.6f}, RMSE: {np.sqrt(mse):.6f}", flush=True)

    # 6. Compute rFID using diffuse_nnx InceptionV3 pipeline
    # The detector expects [0,255] NHWC float, reshaped for pmap
    print('=== Step 5: Computing rFID via diffuse_nnx InceptionV3 ===', flush=True)

    # Convert to [0,255] for the detector
    real_255 = real_01 * 255.0
    recon_255 = recon_nhwc_01 * 255.0

    # Set up InceptionV3 detector (same as training loop)
    inception = inception_module.InceptionV3(pretrained=True)
    inception_params = inception.init(jax.random.PRNGKey(0), jnp.ones((1, 299, 299, 3)))
    inception_params_repl = flax.jax_utils.replicate(inception_params)

    def _fwd(params, x):
        x = x.astype(jnp.float32) / 127.5 - 1
        x = jax.image.resize(x, (x.shape[0], 299, 299, x.shape[-1]), method='bilinear')
        features = inception.apply(params, x, train=False).squeeze(axis=(1, 2))
        features = jax.lax.all_gather(features, axis_name='data', tiled=True)
        return features
    detector = jax.pmap(_fwd, axis_name='data')

    print('  Computing Real Stats...', flush=True)
    stats_real = fid.calculate_stats_for_iterable(
        real_255, detector, inception_params_repl,
        batch_size=eval_batch_size, verbose=True
    )

    print('  Computing Recon Stats...', flush=True)
    stats_recon = fid.calculate_stats_for_iterable(
        recon_255, detector, inception_params_repl,
        batch_size=eval_batch_size, verbose=True
    )

    print('  Computing rFID...', flush=True)
    rfid_score = eval_utils.calculate_fid(stats_recon, stats_real)

    print("=" * 50)
    print(f"rFID Score (diffuse_nnx pipeline) on {num_eval_images} CelebAHQ images: {rfid_score:.4f}")
    print("=" * 50)

    # Also compute with RAE's regularized formula for comparison
    from scipy import linalg as scipy_linalg
    mu1 = np.asarray(stats_recon['mu'], dtype=np.float64)
    mu2 = np.asarray(stats_real['mu'], dtype=np.float64)
    s1 = np.asarray(stats_recon['sigma'], dtype=np.float64)
    s2 = np.asarray(stats_real['sigma'], dtype=np.float64)
    diff = mu1 - mu2
    offset = np.eye(s1.shape[0]) * 1e-6
    covmean, _ = scipy_linalg.sqrtm((s1 + offset) @ (s2 + offset), disp=False)
    covmean = np.real(covmean)
    rfid_rae_style = max(float(diff @ diff + np.trace(s1 + s2 - 2.0 * covmean)), 0.0)

    print(f"rFID Score (RAE-style regularized): {rfid_rae_style:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    app.run(main)
