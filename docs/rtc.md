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
- MoGe, MoRGBD, and DINO-video auxiliary teacher losses enabled by default,
  alongside the RTC action-flow objective.
- Maximum training length: 50,000 optimizer steps; checkpoint every 1,000 steps.
- Checkpoint run directory: `/2025233210/lmx/checkpoints/lingbot-vla-v2-6b-robotwin-rtc`.
- Runtime YAML and manifest: `.runtime/rtc/robotwin_clean/` in this checkout.
- Training log: `.runtime/rtc/train.log` (or `RTC_LOG_PATH`).

Set `LMX_ROOT` to change the parent directory containing datasets/checkpoints.
Use `RTC_MICRO_BATCH_SIZE`, `RTC_GLOBAL_BATCH_SIZE`, `RTC_GRADIENT_CHECKPOINTING`,
`RTC_MAX_DELAY`, `RTC_SAVE_STEPS`, and `RTC_OUTPUT_DIR` to override defaults.
Accumulation is computed from the GPU/process count and `NNODES`. An explicit
YAML may be passed instead of automatic preparation; additional normal
`--model.*`, `--data.*`, and `--train.*` flags follow that YAML or `--auto`.
When overriding the batch on the CLI, also set its accumulation accordingly.
Teacher losses require the MoGe, MoRGBD, and DINO-video weights. Set
`RTC_USE_TEACHERS=false` to disable them and train with RTC action loss only.

## Progress and checkpoint lifecycle

Rank 0 prints a live progress bar such as `RTC train: 1200/50000`, including
elapsed time, estimated remaining time, step rate, current loss, and learning
rate. The other ranks keep the bar disabled so their logs remain readable.

`--name` selects the run directory below `$LMX_ROOT/checkpoints`. The native
distributed checkpointer then writes step snapshots below that directory:

```text
$LMX_ROOT/checkpoints/<name>/
└── checkpoints/
    ├── global_step_1000/
    │   └── hf_ckpt/       # best-effort Hugging Face export for inference
    └── global_step_2000/
```

These are directory snapshots because FSDP distributed state and optimizer
state are sharded across ranks; they are not a single `.pt` file. The explicit
run name is therefore the safe way to name a training checkpoint series.

Start a new named run:

```bash
./run_rtc_moge_run01.sh
```

This convenience script selects GPUs 0–7 by default and uses the run name
`rtc_moge_run01`. It can be launched from any working directory; the training
launcher activates the `lingbotvla-train` Conda environment. To choose a
different visible GPU set, set `CUDA_VISIBLE_DEVICES` before running it.

Resume its newest `global_step_N` snapshot:

```bash
bash train_rtc.sh --name rtc_moge_run01 --resume
```

To intentionally discard only that named run and start at step 0, use
`--overwrite`:

```bash
bash train_rtc.sh --name rtc_moge_run01 --overwrite
```

If a run directory already contains files, the launcher refuses to start
without either `--resume` or `--overwrite`. This prevents an accidental new
job from mixing with an earlier run. `--prepare-only` only generates the YAML
and prints the selected checkpoint directory; it never deletes or launches a
run.

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
one Uniform[0, 1) flow time shared by all action tokens, following the
paper's training-time schedule. Prefix time is set to zero in this model's
noise-to-action flow convention. The action loss only includes suffix tokens,
valid joints and non-padded timesteps. Existing time projections and
adaptive-normalization weights are reused; no trainable parameter keys are
added. RTC defaults to off for other recipes.

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

## Asynchronous RoboTwin evaluation

The `lmx_rtc` branch now includes an opt-in, single-flight RTC evaluator.
Inference runs in a background client thread while the main thread executes
old targets. SAPIEN observation, rendering, and `take_action` stay on the main
thread. The server can run locally or remotely and handles one request at a time.
Ordinary requests without `_rtc` retain the synchronous chunk behavior.

The initial request returns the full raw `(50, 14)` execution plan and normalized
`(50, 55)` model plan. With defaults `replan_steps=8`, `delay=6`, execution is:

```text
Execute A[0:8]. Observe at target step 8 and request B.
Fix normalized B[0:6] to normalized A[8:14].
While B is computed, execute A[8:14].
Install B and continue at B[6], without replaying B[0:6].
After 8 further targets, repeat using the current plan's actual cursor.
```

An early response is installed only after all committed targets execute.
A late response stalls further target execution until it arrives. It never
silently executes beyond the committed prefix. Delay counts RoboTwin joint
targets, not milliseconds or 25 Hz ticks: each target uses the unchanged TOPP
trajectory execution and a variable number of physics steps. This overlaps
work but does not implement hard real-time control or guarantee a speedup.

The server requires an RTC-trained config, finite full-dimensional normalized
plans, and `0 <= delay < training_time_rtc_max_delay` (this checkpoint: 0..9).
The scheduler checks request IDs and prefix continuity. It drains pending
requests before episode reset. A timeout or RPC error interrupts the episode;
it is restarted on resume rather than counted as a task failure.

### Local command on the evaluation machine

```bash
bash /home/asus/lmx/lingbot-vla-v2/eval_rtc.sh \
  /home/asus/lmx/checkpoints/global_step_5000/hf_ckpt \
  /home/asus/lmx/checkpoints/Qwen3-VL-4B-Instruct \
  demo_clean \
  --task_name stack_bowls_three \
  --num_episodes 1 \
  --rtc_replan_steps 8 --rtc_delay 6 \
  --run_dir /home/asus/lmx/RoboTwin/eval_results/step5000_rtc_stack_bowls_three
```

The standalone wrapper detects the Conda base directory. Defaults are
`lingbotvla` for inference and `robotwin-eval` for simulation, BF16, compilation
off, and CPU offload for the embedding and first 12 language MLP layers.
For a larger GPU, set `LINGBOT_CPU_OFFLOAD_MLP_LAYERS=0`; for other environments
or directories use `INFERENCE_ENV`, `SIM_ENV`, `CONDA_SH`, and `EVAL_WORKDIR`.
Video is enabled by default, with success/fail filenames. This machine's
RoboTwin already records continuous physics-step video; an unmodified upstream
RoboTwin uses its own recording implementation. Headless rendering still needs
GPU memory. Avoid running unrelated GPU workloads during evaluation.

Without `LINGBOT_INFERENCE_CONFIG`, `tools/render_robotwin_inference_config.py`
reads the exported checkpoint architecture and the RoboTwin data recipe into
`.runtime/`. An explicit training YAML takes priority. Specify the actual
training normalization with `LINGBOT_NORM_STATS_FILE=/path/to/stats.json`
when generating a config, and `LINGBOT_DATA_CONFIG=/path/to/training.yaml` if
the data representation differs. With an explicit inference YAML, edit its
`data.norm_stats_file` directly. Default `assets/norm_stats/robotwin.json` is
only a fallback; step5000's actual training stats have not been verified on
this machine, so a run using that fallback validates deployment but cannot
establish a reliable checkpoint success rate.

Repeat the identical command to resume completed episodes (both successes
and failures). `--num_episodes 25` can extend the same run. The run manifest
includes RTC mode, delay and replan interval; use a new directory if these,
the model, inference YAML or other evaluation settings change. Logs contain
`[rtc] request/install/wait/summary` entries, including latency, waiting time,
request count, and stalls. `stats.txt` reports success / completed episodes.
RoboTwin expert-seed checks, task success conditions, and trajectory planning
remain in use. This RTC scheduler is a custom inference mode, not the official
synchronous baseline; compare them using separate run directories.

For the usual seven tasks, replace `--task_name` with:

```bash
--tasks stack_blocks_three,blocks_ranking_rgb,pick_diverse_bottles,place_shoe,rotate_qrcode,open_laptop,hanging_mug \
--num_episodes 25
```

Run once with `demo_clean` and once with `demo_randomized`, using separate run
directories. To run synchronously use the existing deployment `eval.sh` or
call `experiment/robotwin/start_robotwin_infer_and_eval.sh` without `--rtc_async`.

### Protocol and checks

`_rtc` is a mapping with `request_id`, integer `delay`, and optional
`previous_actions`. The bootstrap uses delay 0 and no previous plan. RTC responses
contain `_rtc_request_id` and NumPy `_normalized_actions` in addition to raw action
keys and `server_timing`. RTC always samples a full fresh plan regardless of
ordinary cache and `use_length`. Only one connection/request should control a
server slot because episode state is shared by the policy.

```bash
conda activate lingbotvla
python -m unittest discover -s tests -p 'test_async_rtc.py' -v
python -m unittest discover -s tests -p 'test_rtc*.py' -v
```

Checks cover multi-plan action alignment, real thread overlap, detached camera
buffers, reset isolation, stale/corrupt responses, timeout cleanup, RPC validation,
server-to-sampler parameter forwarding, and incompatible resume rejection.
