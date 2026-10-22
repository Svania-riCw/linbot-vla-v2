# spatial-exp — status / TODO (updated 2026-10-22)

## Status

| run | state |
|---|---|
| d4rt (q8), noalign, d4rt_q128, d4rt_ctxvs, d4rt_cotrain | trained 10k + evaluated, results in `README.md` |
| d4rt_l1 (probe depth 1) | training (started 2026-10-22 09:13 UTC, ~830 steps/h), eval follows in the same chain |

Logs: `/2025233210/xiesy/lingbot_ws/logs/{train,eval}_spatial_<run>_10k.log`; eval output
`external/RoboTwin/eval_result/spatial_<run>_10k_global_step_10000/`.

## Open

- Report d4rt_l1 vs q8 (paired McNemar on the shared eval seeds).
- All runs so far are within noise (per-split SE ≈ 3.7 points). To separate them, re-evaluate the
  10k checkpoints with more episodes (e.g. 100 per task) before training more variants.
- `rt_perturb.json`: the "bright" (blown-pixel share) statistic dominates the remaining distance;
  consider down-weighting it if the perturbation is refit.
- Smoke checkpoints `outputs/smoke_spatial_*` (~73 GB each) can be deleted — ask the user first.

## Things learned the hard way (don't re-debug)

- `LingbotVLAV2Config` drops unknown kwargs: its base `__init__` calls `super().__init__()` without `**kwargs`. So any new config key must be an explicit argument, as `d4rt_params` now is.
- With `model.post_training: true`, the weight loader asserts that no parameter is missing. The probe is exempted and initializes itself in `load_model_weights` through `d4rt().init_missing`.
- The d4rt dataset hook needs `item['pil_images']`, which only exists when `align_params` is on. That's fine for the d4rt line. A d4rt run without alignment would need `prepare_images` to also return the augmented 224 views.
- CoTracker track coordinates put pixel i's centre at i; everything else uses i + 0.5. `data.py` adds the 0.5.
- The GPUs are shared with lmx. Check `nvidia-smi` before launching anything.
- Git push: `GIT_SSH_COMMAND="ssh -i ~/.ssh/id_rsa_xiesy -o IdentitiesOnly=yes -p 443" git push ssh://git@ssh.github.com:443/Svania-riCw/linbot-vla-v2.git xiesy`. Port 22 is blocked here. Only push when the user asks.
