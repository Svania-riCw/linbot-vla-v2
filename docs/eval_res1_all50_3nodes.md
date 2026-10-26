# Res1 RoboTwin 50-task evaluation on three nodes

After cloning the repository, use the self-contained branch created for this
run:

```bash
git fetch origin eval
git switch --track origin/eval
```

This run evaluates the res1 checkpoint on all 50 RoboTwin tasks. The tasks are
split into 17, 17, and 16 tasks across three independent nodes. Each task is
run with both `demo_clean` and `demo_randomized`, with 10 episodes per setting.
The resulting total is 1,000 episodes:

```text
50 tasks x 2 settings x 10 episodes = 1,000 episodes
```

The launcher in this branch is
`xsy_utils/eval_rtc_res1_all50_3nodes.sh`. It selects the task list from
`EVAL_NODE_ID`, explicitly uses the res1 input size (`--image_size 256`), and
keeps the requested paper fixed-delay RTC settings:

```text
rtc_mode=paper_fixed_delay
rtc_replan_steps=32
rtc_delay=2
rtc_guidance_beta=5
use_compile=False
use_bf16=False
use_fp32=True
video=false
```

## Prerequisites

Use the same checkout and environments on all three nodes. The launcher
expects:

- `lingbotvla` inference environment and `robotwin-eval` simulation environment;
- `/2025233210/lmx/RoboTwin` with the 50 task environments and task configs;
- the checkpoint at
  `/2025233210/xiesy/lingbot_ckpts/d4rt_q96_ctxvs_res1_augon/checkpoints/global_step_15000/hf_ckpt`;
- Qwen3-VL under `/2025233210/lmx/checkpoints/Qwen3-VL-4B-Instruct`;
- normalization statistics at
  `/2025233210/lmx/training-data/robotwin_clean_norm_stats.json`.

Override any site-specific path with `MODEL_PATH`, `QWEN3VL_PATH`,
`EVAL_WORKDIR`, or `LINGBOT_NORM_STATS_FILE`. The launcher clears
`LINGBOT_INFERENCE_CONFIG` so the runtime config is regenerated from the
checkpoint and the explicit image-size override.

## Start the three nodes

Run exactly one command on each separate node. `START_PORT=9680` may be the
same on all three nodes because the nodes have separate network namespaces. If
multiple runs share one host, use different `START_PORT` values.

Node 1:

```bash
cd /2025233210/lmx/lingbot-vla-v2
EVAL_NODE_ID=1 bash xsy_utils/eval_rtc_res1_all50_3nodes.sh
```

Node 2:

```bash
cd /2025233210/lmx/lingbot-vla-v2
EVAL_NODE_ID=2 bash xsy_utils/eval_rtc_res1_all50_3nodes.sh
```

Node 3:

```bash
cd /2025233210/lmx/lingbot-vla-v2
EVAL_NODE_ID=3 bash xsy_utils/eval_rtc_res1_all50_3nodes.sh
```

The default run directories are:

```text
/2025233210/lmx/eval_rtc_res1_all50_10_2_32_node1
/2025233210/lmx/eval_rtc_res1_all50_10_2_32_node2
/2025233210/lmx/eval_rtc_res1_all50_10_2_32_node3
```

Because two task configurations are used, each directory contains separate
`demo_clean` and `demo_randomized` result trees. Re-running the same command
resumes incomplete episodes from the existing run directory.

## Dry-run and overrides

Before allocating GPUs, inspect the generated commands:

```bash
cd /2025233210/lmx/lingbot-vla-v2
EVAL_NODE_ID=1 DRY_RUN=true bash xsy_utils/eval_rtc_res1_all50_3nodes.sh
```

The main overrides are:

```bash
EVAL_NODE_ID=1 \
MODEL_PATH=/path/to/res1/hf_ckpt \
RUN_DIR=/path/to/results/node1 \
QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct \
EVAL_WORKDIR=/path/to/RoboTwin \
IMAGE_SIZE=256 \
bash xsy_utils/eval_rtc_res1_all50_3nodes.sh
```

`IMAGE_SIZE=256` is the res1 setting. The runtime log must contain
`Evaluation image input resolution: 256x256` before model initialization.

## Verify coverage and results

The launcher validates that node 1 and node 2 each contain 17 unique tasks and
node 3 contains 16. The canonical task list and task validation are provided
by `experiment/robotwin/start_robotwin_infer_and_eval.sh`.

Each task result is written under its setting directory as `_result.txt`; the
per-run progress and success summary are stored alongside the inference and
evaluation logs. Combine the two settings and three node directories only
after all six runs have completed.
