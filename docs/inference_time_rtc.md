# Inference-Time RTC (arXiv:2506.07339)

This document covers the training-free, asynchronous inpainting method in
[Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339).
It uses a soft overlap mask and VJP guidance during each flow step; launch with
`--rtc_mode inference_time --use_compile False`. The evaluation-time cost is
higher than the training-time-conditioned sampler.

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
Execute B[6:8], then request C at target step 16.
Fix normalized C[0:6] to normalized B[8:14], execute B[8:14] while C is computed,
then install C at C[6]. Repeat every 8 target actions between requests.
```

An early response is installed only after all committed targets execute.
A late response stalls further target execution until it arrives. It never
silently executes beyond the committed prefix. `replan_steps` counts all target
actions between inference requests, including the committed prefix executed
while inference runs. With `replan_steps=8` and `delay=6`, the first request
starts after 8 actions; after switching plans, the next request starts after 2
more new-plan actions. Subsequent requests are 8 target actions apart. Delay
counts RoboTwin joint targets, not milliseconds or 25 Hz ticks: each target uses
the unchanged TOPP trajectory execution and a variable number of physics steps.
This overlaps work but does not implement hard real-time control or guarantee a
speedup.

The `training_time` mode requires an RTC-trained config and
`0 <= delay < training_time_rtc_max_delay`. The `inference_time` mode works
with the baseline config and uses the soft-mask/VJP procedure described above.
Both modes check request IDs; training-time RTC also checks exact prefix
continuity. Pending requests drain before episode reset. A timeout or RPC
error interrupts the episode; it is restarted on resume rather than counted as
a task failure.

### Local command on the evaluation machine

```bash
bash /home/asus/lmx/lingbot-vla-v2/eval_rtc.sh \
  /home/asus/lmx/checkpoints/global_step_5000/hf_ckpt \
  /home/asus/lmx/checkpoints/Qwen3-VL-4B-Instruct \
  demo_clean \
  --task_name stack_bowls_three \
  --num_episodes 1 \
  --rtc_mode inference_time --use_compile False \
  --rtc_replan_steps 8 --rtc_delay 6 \
  --run_dir /home/asus/lmx/RoboTwin/eval_results/step5000_rtc_stack_bowls_three
```

The standalone wrapper detects the Conda base directory. Defaults are
`lingbotvla` for inference and `robotwin-eval` for simulation, BF16, compilation
off, and GPU-dependent CPU offload for the embedding and language MLP layers
(24 layers up to 18 GiB, 12 layers up to 24 GiB, none above 24 GiB).
For a larger GPU, set `LINGBOT_CPU_OFFLOAD_MLP_LAYERS=0`; for other environments
or directories use `INFERENCE_ENV`, `SIM_ENV`, `CONDA_SH`, and `EVAL_WORKDIR`.
Video is enabled by default, with success/fail filenames. This machine's
RoboTwin already records continuous physics-step video; an unmodified upstream
RoboTwin uses its own recording implementation. Headless rendering still needs
GPU memory. Avoid running unrelated GPU workloads during evaluation.

Without `LINGBOT_INFERENCE_CONFIG`, `tools/render_robotwin_inference_config.py`
reads the exported checkpoint architecture and the RoboTwin data recipe into
`.runtime/`. The `xsy_utils/eval_rtc_checkpoint.sh` wrapper defaults
`LINGBOT_NORM_STATS_FILE` to
`/2025233210/lmx/training-data/robotwin_clean_norm_stats.json`, matching this
RTC training run; set the variable explicitly if the checkpoint was trained
with another stats file. Set `LINGBOT_DATA_CONFIG=/path/to/training.yaml` if the
data representation differs. With an explicit inference YAML, edit its
`data.norm_stats_file` directly because it takes precedence over generated
config settings.

Repeat the identical command to resume completed episodes (both successes
and failures). `--num_episodes 25` can extend the same run. The run manifest
includes RTC mode, delay and replan interval; use a new directory if these,
the model, inference YAML or other evaluation settings change. Logs contain
`[rtc] request/install/wait/summary` entries, including latency, waiting time,
request count, and stalls. `stats.txt` reports success / completed episodes.
Each run directory also contains `success_rate.md`, a Markdown table that VS
Code can preview. It is refreshed atomically after every committed episode and
shows each task's `clean` or `random` setting, successes, completed/target
episodes, and success rate. To generate or keep refreshing the table for a run
that was started before this feature, use:

```bash
python3 experiment/robotwin/eval_checkpoint.py report \
  --run-dir /path/to/RoboTwin/eval_results/<run> \
  --episodes 25 --watch --until-complete
```
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

## Remote inference over the SSH tunnel

Server startup and simulation-only evaluation are separate:
`serve_rtc.sh` activates the remote inference environment and loads weights;
`eval_rtc_remote.sh` activates only the local simulator environment. The latter
never starts, stops, or needs a local copy of the inference model or Qwen weights.
It defaults to the seven-task subset above, 25 episodes per task, RTC `s=8,d=6`,
and video enabled. A single server slot evaluates tasks sequentially.

### 1. On the remote server

The current SSH login is `ssh root@10.15.171.204 -p 30183`. The example paths
below use the training workspace `/2025233210/lmx`; replace the checkpoint
path with the actual step5000 training export if it is stored elsewhere.
Use the statistics generated for that training run, not a guessed replacement.
The remote Python environment may be `lingbotvla-train` from RTC training;
set `INFERENCE_ENV` to the environment actually installed there.

```bash
cd /2025233210/lmx/lingbot-vla-v2
git switch lmx_rtc
git pull --ff-only origin lmx_rtc

CUDA_VISIBLE_DEVICES=0 \
INFERENCE_ENV=lingbotvla-train \
LINGBOT_NORM_STATS_FILE=/2025233210/lmx/training-data/robotwin_clean_norm_stats.json \
bash serve_rtc.sh \
  /2025233210/lmx/checkpoints/rtc_moge_run01/checkpoints/global_step_5000/hf_ckpt \
  /2025233210/lmx/checkpoints/Qwen3-VL-4B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --rtc_mode inference_time --use_compile False
```

This uses BF16, compilation off, and selects 12 MLP offload layers only for
GPUs up to 18 GiB. Set `LINGBOT_CPU_OFFLOAD_MLP_LAYERS` explicitly for another
memory/performance tradeoff. No RoboTwin simulator or teacher weights are
loaded on the server for inference. A complete exported checkpoint config and
Qwen config/tokenizer are required. If using an explicit
`LINGBOT_INFERENCE_CONFIG`, ensure its `data.norm_stats_file` is the actual
training statistics; generated-config environment variables do not override
an existing explicit YAML.

### 2. Keep a tunnel open on this machine

```bash
ssh -p 30183 -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18000:127.0.0.1:8000 \
  root@10.15.171.204
```

The server listens on remote localhost; the simulator connects to local
localhost:18000 through SSH. The server does not need to expose a public
WebSocket port. Tunnel address/port can change between resumes because the
checkpoint signature, rather than the network route, identifies the policy.

### 3. Test one episode locally, then run the seven-task subset

```bash
bash /home/asus/lmx/lingbot-vla-v2/eval_rtc_remote.sh \
  127.0.0.1 18000 demo_clean \
  --checkpoint_id global_step_5000_rtc \
  --task_name stack_bowls_three --num_episodes 1 \
  --rtc_mode inference_time \
  --run_dir /home/asus/lmx/RoboTwin/eval_results/step5000_rtc_remote_stack_bowls_three
```

```bash
bash /home/asus/lmx/lingbot-vla-v2/eval_rtc_remote.sh \
  127.0.0.1 18000 demo_clean \
  --checkpoint_id global_step_5000_rtc --rtc_mode inference_time --seven --num_episodes 25 \
  --run_dir /home/asus/lmx/RoboTwin/eval_results/step5000_rtc_remote_seven_clean

bash /home/asus/lmx/lingbot-vla-v2/eval_rtc_remote.sh \
  127.0.0.1 18000 demo_randomized \
  --checkpoint_id global_step_5000_rtc --rtc_mode inference_time --seven --num_episodes 25 \
  --run_dir /home/asus/lmx/RoboTwin/eval_results/step5000_rtc_remote_seven_randomized
```

Rerun the identical evaluation command to resume. Fully completed tasks are
skipped after the server compatibility check. Interrupted episodes repeat
from their saved seed and instruction. RPC failures stop the local run and
leave the remote server running. Extend episode counts in place, but use a
new run directory for changed checkpoints/normalization/RTC parameters.
`--tasks` accepts CSV or quoted space-separated names. Set `--rtc_timeout`
for server startup and response deadlines, `--rtc_delay` and
`--rtc_replan_steps` to tune the scheduler, and `--no_video` if needed.
`SIM_ENV`, `CONDA_SH`, and `EVAL_WORKDIR` configure the local environment.

The server publishes protocol version, trained delay limit, full horizon, and
a checkpoint signature derived from model config, shard path/size/mtime,
inference YAML and normalization content. The local preflight verifies this
before launching simulation, and durable resume identity includes the server
signature and RTC settings. Shards are not fully rehashed at every startup;
replace/export changed weights normally so their metadata changes. Use a
single evaluator per server slot; the wrapper also locks an endpoint on the
local machine to prevent accidental concurrent runs.

Validation so far: all unit/transport checks pass, and the actual step5000
checkpoint loaded all 1,708 tensors and completed multiple full-plan RTC
switches locally with six-action prefix checks. The local episode was stopped
when deployment moved to the remote-server workflow; no full success-rate
result or remote-GPU benchmark is claimed. Remote evaluation can start once
the user launches the server and tunnel above.
