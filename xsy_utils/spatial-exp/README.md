# spatial-exp — spatial auxiliary objectives on LingBot-VLA v2

10k-step lines, each compared against `baseline_gbs128_10k` (`xsy_utils/train_baseline.sh`,
clean 0.623 / random 0.303 / all 0.463 on the 7-task eval). Everything except the one named
change is identical to that run: same 50-task RoboTwin clean data, micro 16 × 8 GPUs = global
128, lr 1e-4 → 5e-5 cosine over 10000 steps (no warmup), Muon, checkpoints every 2000.
The `d4rt_*` variants each change one thing relative to `d4rt` (q8).

| line | config | change |
|---|---|---|
| `d4rt` (q8) | `configs/robotwin_clean_d4rt.yaml` | baseline + lightweight d4rt probe (`train.d4rt_params`) |
| `noalign` | `configs/robotwin_clean_noalign.yaml` | baseline − MoGe/MoRGBD/DINO-video alignment (`train.align_params` removed) |
| `d4rt_q128` | `configs/robotwin_clean_d4rt_q128.yaml` | `num_queries: 128` |
| `d4rt_ctxvs` | `configs/robotwin_clean_d4rt_ctxvs.yaml` | `context: visual_state` — probe sees image patch tokens + raw state only |
| `d4rt_cotrain` | `configs/robotwin_clean_d4rt_cotrain.yaml` | `cotrain_every: 2` — + 5000 d4rt-only steps on perturbed images (below), 15000 steps total |
| `d4rt_l1` | `configs/robotwin_clean_d4rt_l1.yaml` | `depth: 1` — one cross-attention block |

```bash
bash run_10k.sh d4rt        # train -> outputs/spatial_d4rt_10k, then 7-task eval of global_step_10000
bash run_10k.sh noalign     # likewise d4rt_q128 / d4rt_ctxvs / d4rt_cotrain / d4rt_l1
SMOKE=1 bash run_10k.sh d4rt   # 6 steps
python validate_targets.py     # geometric checks of the d4rt targets (CPU)
python calibrate_perturb.py    # fit configs/rt_perturb.json (CPU)
```

### Results (7 tasks × 25 episodes per split, same eval seeds for every run)

| run | clean | random | all |
|---|---|---|---|
| baseline 10k | 0.623 | 0.303 | 0.463 |
| noalign | 0.611 | 0.326 | 0.469 |
| d4rt (q8) | 0.680 | 0.303 | 0.491 |
| d4rt_q128 | 0.669 | 0.286 | 0.477 |
| d4rt_ctxvs | 0.669 | 0.331 | 0.500 |
| d4rt_cotrain | 0.646 | 0.286 | 0.466 |

None of the differences is significant: per-split standard error ≈ 3.7 points; paired McNemar on
the shared seeds, ctxvs vs q8 p = 0.84 (all), cotrain vs q8 p = 0.36. Co-training lowers the probe's
3D error by ~15 % (34.6 vs 40.6 mm at 10k) without moving success.

### d4rt-only co-training (`cotrain_every`, `perturb`)

After every N-th action step, one d4rt-only step: prefix (VLM) forward only, d4rt loss only, the
action expert and MoE load-balance update untouched, LR schedule not advanced (it reuses the
current LR and shares optimizer state). It draws from a separate data stream whose images get a
photometric "random-domain" perturbation (`d4rt_lingbot/perturb.py`) instead of colour
augmentation; action steps keep clean, unaugmented images. Purely photometric, so d4rt targets
stay exact. `calibrate_perturb.py` fits the ranges (bounded Powell) so that perturbed clean eval
frames match the RoboTwin *random* eval frames in the 1-D Wasserstein distance of 13 global
image statistics: distance on train frames 37.6 → 9.3. `configs/rt_perturb_unbounded.*` is the
rejected unbounded fit (degenerate, ~12 % crushed-black pixels).

## The d4rt probe (`d4rt_lingbot/`)

PyTorch port of pi05-experiments' `DepthProbe` + `D4rtQuerySample` in the configuration of
`xsy_utils/d4rt_full_optimize/train/train_q8_10000.sh` (`pi05_robotwin_new_3view_d4rt_full_unified`):

```
(u, v) @ c_src, dt, F   ->   (x, y, z) of the 14x14 patch, in frame F, / 1.2 m
```

* **Query**: `LN(patch_proj(RGB 14×14) + fourier(u,v) + cam_embed(c_src) + dt_embed(dt) + tgt_embed(F))`;
  the RGB patch is cut from the model's own 224×224 view after colour augmentation.
* **Decoder**: 3 cross-attention blocks (`depth`), width 1024, 8 heads × 128, SwiGLU 4096, no query
  self-attention; keys/values = `context_proj` of the VLM **prefix hidden states** of the same
  forward pass (all prefix tokens, padding masked; `context: visual_state` keeps only the image
  patch tokens and appends the raw state as one token). Head `to_pred` → 14×14×3. ~54 M params.
* **Loss**: masked L1 on the normalized xyz, **× 0.1** added to the total loss. Unified
  decoding (no separate offset head).
* **Tasks**, K = 8 queries per frame, ¼ each, shuffled per frame; slots 0/1 pinned to the
  projected grippers in cam_high:

  | task | dt | c_src | F |
  |---|---|---|---|
  | depth | 0 | any | c_src @ t |
  | time | 1..49 (CoTracker) | any | c_src @ t (answer read at t+dt) |
  | hand-eye | 0 | cam_high | left/right gripper @ t |
  | registration | 0 | any | another camera @ t |

  F is an opaque id (0/1/2 cameras, 3/4 grippers) — no extrinsics, intrinsics or poses are fed in.
* **Inference**: unused (`sample_actions` never calls it); its weights ride along in the checkpoint.

### Data

Targets come from `xsy_utils/d4rt_data` (`/2025233210/xiesy/data/lingbot_robotwin_clean_d4rt`):
MoGe-3 depth scale-corrected per frame against the GT table/wall planes, CoTracker3 tracks
(window 50, stride 25), endpose, GT intrinsics/extrinsics. Differences from pi05's data side:

* lingbot's image is `Resize((224,224))` of the 640×480 lerobot frame (itself 2× the native
  320×240 render): an anisotropic squash with **no padding**. Query pixels live in that canvas;
  all geometry is done in native continuous coordinates (pixel i ↔ i + 0.5).
* The importance map (pi05's `w56`) and the canvas depth are computed on the fly.
* A query whose anchor frame has `depth_meta.reliable == False` is masked.

Checks (`validate_targets.py`, 4 tasks × 2 episodes × 40 frames, each task mapped back through
an independent path): depth / time / registration / hand-eye round trip median **0.000 px**,
p99 ≤ 0.004 px; gripper origin 0.000 mm; supervised 97.7–98.8 % per task; 0.058 % of target
values clipped at ±1.2 m (far wrist views) — the only round-trip outliers. A depth-edge overlay on
the model's 224 image lines up with silhouettes in all three views.

### Core hooks (all off unless `train.d4rt_params.enable`)

`lingbotvla/spatial_exp.py` bridges to this directory. Hooks: `d4rt_params` config field
(`configuration_lingbot_vla.py`, `train_lingbotvla.py`); probe built in `FlowMatchingV2.__init__`,
loss after the VLM forward and added in `LingbotVlaV2Policy.forward`; probe params exempt from the
post-training "no missing weights" check and self-initialized (`module_utils.load_model_weights`);
per-sample targets in `VLADataset.getitem`; `D4rt_*` fields in the step log / TensorBoard.

Log fields: `D4rt_Loss` (unweighted L1), `D4rt_Base` (best constant 3-vector — the
"learned nothing" bar), `D4rt_mm` (mean 3D error, mm) split by task, `D4rt_Mask` (supervised share).
