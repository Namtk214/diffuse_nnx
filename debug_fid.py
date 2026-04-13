"""Debug FID pipeline: compare diffuse_nnx InceptionV3 vs RAE reference InceptionV3.
Also check torch_fidelity rFID against our JAX rFID.
"""
import sys
sys.path.insert(0, '/home/ngothanhnam508/diffuse_nnx')
sys.stdout.reconfigure(line_buffering=True)

import os
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import torch
from absl import logging, app
import orbax.checkpoint as ocp
from etils import epath

def main(argv):
    logging.set_verbosity(logging.INFO)

    from configs.rae_stage1_celebahq import get_config
    from data import local_imagenet_dataset
    from trainers.rae_stage1 import encode_for_training, decode_for_training

    config = get_config()

    # 1. Load 2048 images at 256x256
    print("=== Step 1: Load 2048 images ===", flush=True)
    dataset = local_imagenet_dataset.build_imagenet_dataset(
        is_train=True,
        data_dir='/home/ngothanhnam508/data/celebahq',
        image_size=256,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0, shuffle=False)

    real_nhwc_m1p1 = []  # NHWC [-1,1]
    count = 0
    for batch, _ in loader:
        nhwc = batch.permute(0, 2, 3, 1).numpy()
        real_nhwc_m1p1.append(nhwc)
        count += nhwc.shape[0]
        if count >= 2048:
            break
    real_nhwc_m1p1 = np.concatenate(real_nhwc_m1p1, axis=0)[:2048]
    print(f"  real images shape: {real_nhwc_m1p1.shape}, range: [{real_nhwc_m1p1.min():.2f}, {real_nhwc_m1p1.max():.2f}]", flush=True)

    # Convert to uint8 [0,255] for torch_fidelity
    real_uint8 = ((real_nhwc_m1p1 + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    # Convert to float [0,1] NHWC for JAX InceptionV3
    real_nhwc_01 = (real_nhwc_m1p1 + 1.0) / 2.0

    # 2. Load model and reconstruct
    print("\n=== Step 2: Load model & reconstruct ===", flush=True)
    from networks.encoders.rae import RAE
    rng = jax.random.PRNGKey(0)
    rng_rae = jax.random.split(rng, 2)[1]
    rae_model = RAE(
        config=config.encoder,
        pretrained_path=config.encoder.pretrained_path,
        resolution=config.encoder.resolution,
        encoded_pixels=config.encoder.get('encoded_pixels', True),
        rngs=nnx.Rngs(int(rng_rae[0])),
    )

    workdir = '/home/ngothanhnam508/diffuse_nnx/RAE-Stage1-CelebAHQ'
    import glob
    checkpoints = glob.glob(os.path.join(workdir, 'checkpoint_epoch_*'))
    if checkpoints:
        latest_ckpt = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"  Loading: {latest_ckpt}", flush=True)
        ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
        decoder_params = nnx.state(rae_model.decoder)
        state = ckptr.restore(
            epath.Path(latest_ckpt),
            args=ocp.args.Composite(
                ema_params=ocp.args.StandardRestore(decoder_params),
            )
        )
        nnx.update(rae_model.decoder, state['ema_params'])
        print("  Loaded EMA params", flush=True)
    else:
        print("  WARNING: No checkpoint!", flush=True)

    from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
    mesh = Mesh(jax.devices(), ('data',))
    data_sharding = NamedSharding(mesh, P('data'))

    recon_nhwc_01 = []
    for i in range(0, 2048, 64):
        batch = real_nhwc_m1p1[i:i+64]
        batch_jax = jnp.array(batch)
        z = encode_for_training(rae_model, batch_jax, training=False)
        x_rec = decode_for_training(rae_model, z)  # NCHW [0,1]
        x_rec_np = jax.device_get(x_rec)
        x_rec_nhwc = x_rec_np.transpose(0, 2, 3, 1)  # NHWC [0,1]
        recon_nhwc_01.append(x_rec_nhwc)
        if i == 0:
            print(f"  x_rec shape: {x_rec_np.shape}, range: [{x_rec_np.min():.3f}, {x_rec_np.max():.3f}]", flush=True)

    recon_nhwc_01 = np.concatenate(recon_nhwc_01, axis=0)
    recon_uint8 = (recon_nhwc_01 * 255).clip(0, 255).astype(np.uint8)
    print(f"  recon uint8 shape: {recon_uint8.shape}, range: [{recon_uint8.min()}, {recon_uint8.max()}]", flush=True)

    # Pixel-level comparison
    mse = np.mean((real_uint8.astype(float) - recon_uint8.astype(float)) ** 2)
    print(f"  Pixel MSE (uint8): {mse:.2f}, RMSE: {np.sqrt(mse):.2f}", flush=True)

    # Save for cross-testing
    np.save('/tmp/rfid_real.npy', real_uint8)
    np.save('/tmp/rfid_recon.npy', recon_uint8)
    print("  Saved arrays to /tmp/rfid_real.npy and /tmp/rfid_recon.npy", flush=True)

    # 3. Method A: RAE reference — torch_fidelity (gold standard)
    print("\n=== Step 3: rFID via torch_fidelity (RAE reference method) ===", flush=True)
    try:
        # Import RAE's FID module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("rae_fid", "/home/ngothanhnam508/RAE/eval/fid.py")
        rae_fid = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rae_fid)
        from torch_fidelity import calculate_metrics
        import torch as _tf_torch
        class ImgArrDataset:
            def __init__(self, arr):
                self.arr = arr
            def __len__(self):
                return len(self.arr)
            def __getitem__(self, idx):
                return _tf_torch.from_numpy(self.arr[idx]).permute(2, 0, 1)
        arr1_ds = ImgArrDataset(real_uint8)
        arr2_ds = ImgArrDataset(recon_uint8)
        metrics = calculate_metrics(input1=arr1_ds, input2=arr2_ds, batch_size=64, fid=True, cuda=False)
        rfid_torch = metrics['frechet_inception_distance']
        print(f"  rFID (torch_fidelity): {rfid_torch:.4f}", flush=True)
    except Exception as e:
        print(f"  torch_fidelity failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        rfid_torch = None

    # 4. Method B: RAE reference — JAX InceptionV3 (from RAE/eval/fid.py)
    print("\n=== Step 4: rFID via RAE JAX InceptionV3 ===", flush=True)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("rae_fid", "/home/ngothanhnam508/RAE/eval/fid.py")
        rae_fid = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rae_fid)
        fid_fn = rae_fid.get_fid_network()

        # Real
        real_acts = rae_fid.compute_fid_activations(real_uint8, fid_fn, batch_size=64)
        mu_real, sigma_real = rae_fid.moments_from_activations(real_acts)

        # Recon
        recon_acts = rae_fid.compute_fid_activations(recon_uint8, fid_fn, batch_size=64)
        mu_recon, sigma_recon = rae_fid.moments_from_activations(recon_acts)

        rfid_rae_jax = rae_fid.fid_from_stats(mu_recon, sigma_recon, mu_real, sigma_real)
        print(f"  rFID (RAE JAX InceptionV3): {rfid_rae_jax:.4f}", flush=True)

        # Sanity: FID(real, real) should be ~0
        rfid_self = rae_fid.fid_from_stats(mu_real, sigma_real, mu_real, sigma_real)
        print(f"  FID(real, real) sanity: {rfid_self:.4f} (should be ~0)", flush=True)
    except Exception as e:
        print(f"  RAE JAX InceptionV3 failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        rfid_rae_jax = None

    # 5. Method C: diffuse_nnx InceptionV3 (what our training loop uses)
    print("\n=== Step 5: rFID via diffuse_nnx InceptionV3 ===", flush=True)
    try:
        from eval import inception as inception_module
        from eval import utils as eval_utils, fid as fid_module
        import flax

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

        # Real stats — input [0,255] NHWC
        real_255 = real_uint8.astype(np.float32)
        stats_real = fid_module.calculate_stats_for_iterable(
            real_255, detector, inception_params_repl, batch_size=64, verbose=False
        )

        # Recon stats
        recon_255 = recon_uint8.astype(np.float32)
        stats_recon = fid_module.calculate_stats_for_iterable(
            recon_255, detector, inception_params_repl, batch_size=64, verbose=False
        )

        rfid_nnx = eval_utils.calculate_fid(stats_recon, stats_real)
        print(f"  rFID (diffuse_nnx InceptionV3): {rfid_nnx:.4f}", flush=True)
    except Exception as e:
        print(f"  diffuse_nnx InceptionV3 failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        rfid_nnx = None

    # 6. Summary
    print("\n" + "=" * 50, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 50, flush=True)
    if rfid_torch is not None:
        print(f"  torch_fidelity rFID:     {rfid_torch:.4f}")
    if rfid_rae_jax is not None:
        print(f"  RAE JAX InceptionV3 rFID: {rfid_rae_jax:.4f}")
    if rfid_nnx is not None:
        print(f"  diffuse_nnx rFID:        {rfid_nnx:.4f}")
    print(f"  Pixel RMSE (uint8):      {np.sqrt(mse):.2f}")
    print("=" * 50, flush=True)

if __name__ == '__main__':
    app.run(main)
