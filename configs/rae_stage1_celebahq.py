"""RAE Stage 1 config for CelebA-HQ 256."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.trainer           = 'RAE_Stage1'
    config.exp_name          = 'RAE_Stage1_CelebAHQ'
    config.seed              = 0
    config.log_every_steps   = 50
    config.sample_every_steps = 500

    # ── Data (dùng local_imagenet_dataset có sẵn) ────────────────────
    config.data = ml_collections.ConfigDict()
    config.data.data_dir          = '/home/ngothanhnam508/data/celebahq'
    config.data.val_data_dir      = '/home/ngothanhnam508/data/celebahq'
    config.data.image_size        = 256
    config.data.batch_size        = 512
    config.data.micro_batch_size  = 64    # actual per-step batch; grad accum over batch_size/micro_batch_size steps
    config.data.num_train_samples = 28000
    config.data.num_workers       = 8
    config.data.seed              = 0
    config.data.seed_pt           = 0

    # ── Encoder (dùng nguyên RAE của diffuse_nnx) ────────────────────
    config.encoder = ml_collections.ConfigDict()
    config.encoder.pretrained_path = 'facebook/dinov2-with-registers-base'
    config.encoder.resolution      = 224
    config.encoder.encoded_pixels  = True
    config.encoder.noise_tau       = 0.8  # Inject noise during training

    # ── Stage 1 hyperparams ──────────────────────────────────────────
    config.stage1 = ml_collections.ConfigDict()
    config.stage1.epochs        = 16
    config.stage1.lr            = 2e-4
    config.stage1.ema_decay     = 0.9978
    config.stage1.warmup_epochs = 1
    config.stage1.checkpoint_interval = 8

    config.stage1.loss = ml_collections.ConfigDict()
    config.stage1.loss.disc_loss         = 'hinge'
    config.stage1.loss.disc_weight       = 0.75
    config.stage1.loss.perceptual_weight = 1.0
    config.stage1.loss.disc_start        = 8   # epoch bắt đầu GAN generator loss
    config.stage1.loss.disc_upd_start    = 6   # epoch bắt đầu update discriminator
    config.stage1.loss.lpips_start       = 0
    config.stage1.loss.max_d_weight      = 10000.0

    # ── Discriminator ────────────────────────────────────────────────
    config.gan = ml_collections.ConfigDict()
    config.gan.disc = {
        'arch': {
            'dino_ckpt_path': 'models/discs/dino_vit_small_patch8_224.pth',
            'ks': 9,
            'norm_type': 'bn',
            'using_spec_norm': True,
            'recipe': 'S_8',
        },
        'augment': {'prob': 1.0, 'cutout': 0.0},
    }

    # ── Checkpoint (dùng format của diffuse_nnx) ─────────────────────
    config.checkpoint = ml_collections.ConfigDict()
    config.checkpoint.options = ml_collections.ConfigDict()
    config.checkpoint.options.save_interval_steps        = 1000
    config.checkpoint.options.max_to_keep                = 5
    config.checkpoint.options.keep_period                = 5000
    config.checkpoint.options.enable_async_checkpointing = False

    return config
