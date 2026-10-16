# spatial-exp — handoff TODO (written 2026-10-16 ~17:30 UTC)

Hand-off note for the agent that resumes this work on another cluster. Read `README.md` first.

## The task (from the user)

1. Implement a **lightweight d4rt module** for LingBot-VLA v2 modelled on
   `/2025233210/xiesy/pi05-experiments/xsy_utils/d4rt_full_optimize/train/train_q8_10000.sh`,
   then **train 10k steps and run the eval**.
2. Then queue: a **baseline with MoGe/DINO alignment off**, also 10k steps of training plus the eval.
3. Every other variable must match the 10k baseline (`outputs/baseline_gbs128_10k`) exactly. Don't change anything else.
4. All code goes in `xsy_utils/spatial-exp`.
5. Kick yourself every hour (a cron self-check) and report progress.
6. Decisions already confirmed with the user:
   - Small hooks in core, off by default.
   - d4rt loss weight 0.1, as in pi05.
   - Run d4rt first, then no-align.

## Status

| item | state |
|---|---|
| d4rt probe + data (`d4rt_lingbot/`), configs, `run_10k.sh`, README | **done**, commit `b26f45b` on branch `xiesy` (unpushed at time of writing) |
| `validate_targets.py` geometric checks | **passed** (round trip median 0.000 px, p99 ≤ 0.004 px; see README) |
| depth-edge vs 224-image alignment overlay | **passed** (visually, all 3 cams) |
| d4rt smoke (`SMOKE=1 bash run_10k.sh d4rt`, 6 steps, 8 GPUs) | **passed**: D4rt_Loss 0.845→0.786, Mask ~0.97, no errors |
| eval path with a probe checkpoint (1 episode, place_shoe) | **passed** (server loads `d4rt_params` + probe weights) |
| no-align smoke (`SMOKE=1 bash run_10k.sh noalign`) | **NOT done**: OOM'd because lmx's job took the GPUs. Loader did print "Skipped 84 checkpoint weights for disabled depth/video alignment modules", so loading works. **Re-run it before the real run.** |
| d4rt 10k train + eval | **TODO** |
| no-align 10k train + eval | **TODO** |

Why it stopped: at 2026-10-16 17:05 UTC, lmx started an RTC training on all 8 GPUs of the
original cluster.
- The job runs from `/2025233210/lmx/lingbot-vla-v2` and writes to `/2025233210/lmx/checkpoints/lingbot-vla-v2-6b-robotwin-rtc`.
- It resumed at step 13000 of 50000, so it needs about 44 h.
- Don't touch it.

## Next steps

1. Check that the new cluster has everything:
   - The repo at commit `b26f45b` or later, on the `xiesy` branch.
   - The conda env `lingbotvla` (`/2025233210/xiesy/lingbot_ws/conda_envs/lingbotvla`).
   - The d4rt data `/2025233210/xiesy/data/lingbot_robotwin_clean_d4rt` (242 GB; 2500 episodes × 3 cams).
   - lmx's lerobot data `/2025233210/lmx/training-data/lerobot/robotwin_clean_*`.
   - The weights `/2025233210/lmx/checkpoints/lingbot-vla-v2-6b` and `Qwen3-VL-4B-Instruct`.
   - `/2025233210/xiesy/lingbot_ws/checkpoints/moge-2-vitb-normal`.
   - All paths are hard-coded under `/2025233210/...`. If storage is not shared, copy the data or edit the paths in `configs/*.yaml`, `d4rt_lingbot/hooks.py` (`root`) and `xsy_utils/train_baseline.sh`.
2. Make sure all 8 GPUs are free: `nvidia-smi`.
3. Run `SMOKE=1 bash run_10k.sh noalign`. The step log must show `Depth_Loss 0.0000`, no `D4rt_*` fields and no errors. You can delete the smoke checkpoints afterwards (73 GB each), but ask the user first.
4. Launch both runs in one detached chain, so they run back to back:
   ```bash
   cd /2025233210/xiesy/lingbot_ws
   cat > logs/chain_spatial.sh <<'EOF'
   bash /2025233210/xiesy/lingbot_ws/linbot-vla-v2/xsy_utils/spatial-exp/run_10k.sh d4rt
   bash /2025233210/xiesy/lingbot_ws/linbot-vla-v2/xsy_utils/spatial-exp/run_10k.sh noalign
   EOF
   setsid nohup bash logs/chain_spatial.sh > logs/chain_spatial.log 2>&1 < /dev/null & disown
   ```
   - Training logs: `logs/train_spatial_{d4rt,noalign}_10k.log`.
   - Eval logs: `logs/eval_spatial_*_10k.log`.
   - Each run is about 12.3 h of training plus about 1.2 h of eval, so roughly 27 h in total.
   - The d4rt run may be a bit slower per step, because d4rt targets are built in the data workers.
5. Set up an hourly self-check with CronCreate. Each check reports:
   - Stage, step and ETA.
   - `D4rt_Loss` against `D4rt_Base`. The probe has learned something once `D4rt_Loss < D4rt_Base`. At step 1 it is about 0.85 against a baseline of 0.167.
   - `D4rt_mm` per task.
   - Any errors.
6. After each eval, report the `_result.txt` averages against the earlier checkpoints, plus the train and eval times. Then delete the cron job once both runs are reported.

   | checkpoint | clean | random | all |
   |---|---|---|---|
   | baseline 10k | 0.623 | 0.303 | 0.463 |
   | baseline 5k | 0.440 | 0.269 | 0.354 |

## Things learned the hard way (don't re-debug)

- `LingbotVLAV2Config` drops unknown kwargs: its base `__init__` calls `super().__init__()` without `**kwargs`. So any new config key must be an explicit argument, as `d4rt_params` now is.
- With `model.post_training: true`, the weight loader asserts that no parameter is missing. The probe is exempted and initializes itself in `load_model_weights` through `d4rt().init_missing`.
- The d4rt dataset hook needs `item['pil_images']`, which only exists when `align_params` is on. That's fine for the d4rt line. A d4rt run without alignment would need `prepare_images` to also return the augmented 224 views.
- CoTracker track coordinates put pixel i's centre at i; everything else uses i + 0.5. `data.py` adds the 0.5.
- The GPUs are shared with lmx. Check `nvidia-smi` before launching anything.
- Git push: `GIT_SSH_COMMAND="ssh -i ~/.ssh/id_rsa_xiesy -o IdentitiesOnly=yes -p 443" git push ssh://git@ssh.github.com:443/Svania-riCw/linbot-vla-v2.git xiesy`. Port 22 is blocked here. Only push when the user asks.
