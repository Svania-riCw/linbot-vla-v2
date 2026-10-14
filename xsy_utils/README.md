# xsy_utils

xiesy's LingBot-VLA v2 workflow on this repo (branch `xiesy`, off lmx's `lmx_rtc`; RTC stays off
because `training_time_rtc` defaults to False). `WS=/2025233210/xiesy/lingbot_ws`.

## Setup (untracked pieces)

```bash
bash xsy_utils/env/setup.sh            # deps model_env sim_env assets gl
```

- `third_party/utils3d` @ `3fab839`, `external/RoboTwin/envs/curobo` @ `d64c4b0` (git-ignored clones).
- Conda envs on shared storage `$WS/conda_envs` (`CONDA_ENVS_PATH`): `lingbotvla` (py3.12, model)
  and `robotwin-eval` (py3.10, sim). Pins in `env/*.lock.txt`.
- RoboTwin assets: `objects` + `background_texture` symlinked from lmx (read-only), `embodiments` copied.
- `gl`: this container mounts only the NVIDIA compute libs, so sapien cannot find a Vulkan device.
  The step extracts the matching driver's userspace GL/Vulkan libs into `$WS/nvidia-gl-<ver>`;
  `eval_parallel.sh` picks them up automatically. Nothing under /usr is touched.

## Training

Baseline (no RTC) on RoboTwin clean data: 50 tasks x 50 episodes.

- Data (lmx, read-only): `/2025233210/lmx/training-data/lerobot/robotwin_clean_*`, listed in
  `configs/robotwin_clean.txt`; clean norm stats via `/2025233210/lmx/training-data/robot_config_clean`.
- Weights (lmx, read-only): `/2025233210/lmx/checkpoints/{lingbot-vla-v2-6b,Qwen3-VL-4B-Instruct}`;
  MoGe-2 teacher `Ruicheng/moge-2-vitb-normal` downloaded to `$WS/checkpoints/moge-2-vitb-normal/model.pt`.
- Config: `configs/robotwin_clean_baseline.yaml` = upstream `robotwin.yaml` with paths filled in
  (depth + DINO-video alignment on, fp32, compile on).

```bash
RUN_NAME=baseline_gbs128_5k MICRO_BS=16 bash xsy_utils/train_baseline.sh \
  --train.max_steps 5000 --train.save_steps 1000
```

micro 32/GPU OOMs on H20 (96GB); micro 16 x 8 GPUs = global 128 fits (~71GB/GPU, ~4.9 s/step).
Outputs: `$WS/outputs/$RUN_NAME/` (train.log, run_info.txt, code.diff, checkpoints).

## RoboTwin eval

`external/RoboTwin/policy/lingbot-vla-v2/` -- see the header of `eval_parallel.sh`.

```bash
cd external/RoboTwin
# smoke: 1 task, 1 episode, GPU 0
bash policy/lingbot-vla-v2/eval.sh $WS/outputs/baseline_gbs128_5k/checkpoints/global_step_5000/hf_ckpt \
    stack_bowls_three demo_clean 1 0
# full: 7 tasks x {clean, randomized} x 25 on 8 GPUs (= bash xsy_utils/eval_baseline.sh [hf_ckpt])
MODEL_PATH=.../hf_ckpt TEST_NUM=25 GPUS="0 1 2 3 4 5 6 7" \
TASKS="stack_blocks_three blocks_ranking_rgb hanging_mug pick_diverse_bottles open_laptop rotate_qrcode place_shoe" \
    bash policy/lingbot-vla-v2/eval_parallel.sh
```

Output: `external/RoboTwin/eval_result/<model>/run_<ts>/` with `_result.txt` (per unit + averages),
`<task>_<case>/episodeN_{success,fail}.mp4`, per-unit sim logs and `_server/gpu<id>.log`.
