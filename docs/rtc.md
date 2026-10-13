# RTC training on clean RoboTwin data

RTC code lives directly in this repository on `lmx_rtc`. Training uses the
local `robbyant/lingbot-vla-v2-6b` checkpoint, not a checkpoint from the wrapper
repository. No patch is applied or reversed at process exit.

## Train on this host

```bash
cd /2025233210/lmx/lingbot-vla-v2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train_rtc.sh
```

The launcher activates `lingbotvla-train`, prepares a config from the official
RoboTwin recipe, and launches torchrun from this checkout. It uses all 50
datasets under `/2025233210/lmx/training-data/lerobot` and the clean statistics
at `/2025233210/lmx/training-data/robotwin_clean_norm_stats.json`. Defaults:

- Per-GPU micro batch 16, 8 GPUs, gradient accumulation 1, global batch 128.
- Gradient checkpointing enabled; optimizer and MoE settings follow
  `configs/vla/robotwin/robotwin.yaml`.
- RTC enabled with exclusive delay limit 10 (prefix lengths 0 through 9).
- Auxiliary depth/video teacher losses disabled. The policy trains from the RTC
  action-flow objective without loading MoGe, MoRGBD, or DINO-video teachers.
- Maximum training length: 50,000 optimizer steps; checkpoint every 1,000 steps.
- Checkpoints: `/2025233210/lmx/checkpoints/lingbot-vla-v2-6b-robotwin-rtc-no-moge`.
- Runtime YAML and manifest: `.runtime/rtc/robotwin_clean/` in this checkout.
- Training log: `.runtime/rtc/train.log` (or `RTC_LOG_PATH`).

Set `LMX_ROOT` to change the parent directory containing datasets/checkpoints.
Use `RTC_MICRO_BATCH_SIZE`, `RTC_GLOBAL_BATCH_SIZE`, `RTC_GRADIENT_CHECKPOINTING`,
`RTC_MAX_DELAY`, `RTC_SAVE_STEPS`, and `RTC_OUTPUT_DIR` to override defaults.
Accumulation is computed from the GPU/process count and `NNODES`. An explicit
YAML may be passed instead of automatic preparation; additional normal
`--model.*`, `--data.*`, and `--train.*` flags follow that YAML or `--auto`.
When overriding the batch on the CLI, also set its accumulation accordingly.
Set `RTC_USE_TEACHERS=true` to re-enable the official MoGe, MoRGBD, and DINO-video
auxiliary losses; this mode requires their teacher weights, including MoGe.

Preparation does not start training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train_rtc.sh --prepare-only
```

## Weights and environment

The already prepared `lingbotvla-train` environment uses Python 3.12,
PyTorch 2.8.0+cu128, Transformers 4.57.3, LeRobot 0.4.2, and FlashAttention 2.8.3.
`bash tools/create_rtc_env.sh` verifies it, or clones `lingbotvla` when the
training environment is absent. Its runtime checks do not require the MoGe or
MoRGBD Python packages. The upstream `tools/create_train_env.sh` provides
installation from scratch.

MoGe-2 ViT-B normal can be downloaded for the optional teacher-enabled recipe
with `bash tools/download_moge.sh`. In that recipe, MoGe converts RGB images
into estimated depth, and MoRGBD consumes RGB plus that depth to produce feature
targets for the VLA's depth alignment loss. The training launcher skips this
checkpoint by default. `tools/download_moge.sh` verifies/downloads
`Ruicheng/moge-2-vitb-normal` at revision
`ca5f0e07ff01d3e5a364c1d954ed12ee1814b368`, checks size and SHA-256, and installs
`$LMX_ROOT/checkpoints/moge2-vitb-normal.pt`. The configured mirror endpoint
can be changed with `RTC_MOGE_ENDPOINT`.

Normalization is wired through a generated `robot_configs/robotwin.yaml`,
because the current dataset loader reads that file's `norm_stats` value.
The `data.norm_stats_file` field is kept consistent as well.

## Model behavior and tests

With RTC enabled, each training example samples a clean action prefix and
tokenwise Beta(1.5, 1.0) flow times. Prefix time is zero. The action loss only
includes suffix tokens, valid joints and non-padded timesteps. Existing time
projections and adaptive-normalization weights are reused; no trainable
parameter keys are added. RTC defaults to off for other recipes.

The optional sampling API accepts normalized `previous_actions` and
`rtc_delay` and clamps the prefix throughout denoising. The helper
`align_previous_action_chunk` shifts an earlier plan by executed steps. The
RoboTwin server does not automatically use this API; asynchronous control
must supply the aligned previous plan explicitly.

```bash
bash tools/test_rtc.sh
```

Tests check RTC masks, tokenwise conditioning, default-off behavior, prefix
clamping, config parsing, batch arithmetic, no-teacher preparation, optional
teacher preparation, checkpoint interval, and actual normalizer wiring.
These checks do not run a full training job or measure task success.
