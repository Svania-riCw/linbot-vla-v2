# final_runs — d4rt-ctxvs × {resolution, colour augmentation}, DLC 8 × PPU

Four runs, all **global batch 1024, 50k steps** (the official RoboTwin recipe: lr 1e-4 → 5e-5 cosine,
Muon, fp32), **torch.compile off**, initialised from `lingbot-vla-v2-6b`, RoboTwin clean-50 (50 tasks ×
50 episodes, 549,787 frames).

| script | res | aug | d4rt queries | role |
|---|---|---|---|---|
| `train_d4rt_q08_ctxvs_res2_augon.sh` | 384² (432 tok) | on | 8 | hero |
| `train_d4rt_q96_ctxvs_res2_augon.sh` | 384² (432 tok) | on | 96 | hero |
| `train_d4rt_q08_ctxvs_res2_augoff.sh` | 384² (432 tok) | off | 8 | fallback |
| `train_d4rt_q08_ctxvs_res1_augoff.sh` | 256² (192 tok) | off | 8 | fallback |

- **d4rt-ctxvs** = `xsy_utils/spatial-exp/configs/robotwin_clean_d4rt_ctxvs.yaml` (alignment teachers +
  d4rt probe, `context: visual_state`), unchanged except paths. res1 keeps its torchvision-bilinear 256².
- **res2** = qianlei's full432 input: 384², OpenCV Lanczos4, fixed processor grid, 12×12 student grid.
- **aug** = qianlei's full432 observation augmentation (`configs/vla/robotwin/full432_augmentation_recipe.yaml`;
  colour/temperature, 2-D scale+shift, noise, blur, JPEG; occlusion is skipped — no foreground masks).
- Configs are generated: `python xsy_utils/final_runs/gen_configs.py` → `configs/*.yaml` (edit the generator).

## 1. Data to copy (you)

Only the d4rt targets are missing on this side. Everything else is already on CPFS.

| what | from (H20) | to | size |
|---|---|---|---|
| d4rt targets (all of it, incl. `*_INFO.json`) | `/2025233210/xiesy/data/lingbot_robotwin_clean_d4rt/` | `/mnt/cpfs/data/lingbot_robotwin_clean_d4rt/` | ≈ 262 GB (98 % `depth/`) |

Layout must stay `<task>/{ee,cam,depth/<cam>,depth_meta/<cam>,tracks/<cam>,tvis/<cam>,tmeta/<cam>}/episode_NNNNNN.*`.
Another location works too: change `D4RT_ROOT` in `gen_configs.py` and regenerate. Keep it on CPFS: every
sample opens ~21 files of a random episode, and OSS (ossfs) pays 0.2–0.5 s per first open.

Check after copying (also runs automatically at every launch):

```bash
source xsy_utils/env/ppu_env.sh
python xsy_utils/final_runs/check_data.py xsy_utils/final_runs/configs/d4rt_q08_ctxvs_res1_augoff.yaml
# -> [check_data] ok: 50 datasets, d4rt 2500 episodes
```

It checks that every episode has all 17 files and that `ee` has exactly lerobot-length + 1 rows. Run it once
with `--compare-ee` after the copy: the copied `ee/` and `cam/` must equal, value for value, what
`xsy_utils/d4rt_data/prep_ee.py` regenerated here from the PPU-side raw HDF5 (`.runtime/d4rt_ee_reference/`),
which proves the H20 targets were built from the same raw episodes as the LeRobot copy trained on here.
That LeRobot copy (`/mnt/cpfs/legacy_lingbot/data/robotwin_clean50_lerobot`, converted from the official raw
HDF5 on OSS) passed `prep_ee.py`'s own check for all 2,500 episodes (state == raw joint vector, length ==
raw − 1); the frame total matches the H20 set (549,787).

Already in place: weights `/mnt/cpfs/legacy_lingbot/checkpoints/{lingbot-vla-v2-6b,Qwen3-VL-4B-Instruct,moge-2-vitb-normal}`,
LeRobot data (list `xsy_utils/configs/ppu/robotwin_clean50.txt`), clean-50 norm stats (`xsy_utils/configs/ppu/`),
venv `/mnt/cpfs/venvs/lingbotvla`.

**CPFS quota.** On 2026-10-23 `/mnt/cpfs` held ≈ 1.89 TB (`data` 1.4 T, `legacy_lingbot` 367 G, ...). The first
training job also builds the HF-datasets Arrow cache of the LeRobot parquet in `.runtime/huggingface`
(≈ 141 GB, once, shared by all four runs). The old cache in `legacy_lingbot/.cache/huggingface` (141 GB) is
keyed on the pre-move path `/mnt/cpfs/lingbot-va2/...` and is not used by these runs.

## 2. Launch (DLC, 8 × PPU, same image as the dev box)

```bash
bash /mnt/cpfs/linbot-vla-v2/xsy_utils/final_runs/train_d4rt_q96_ctxvs_res2_augon.sh        # one job per run
RESUME=1 bash /mnt/cpfs/linbot-vla-v2/xsy_utils/final_runs/train_d4rt_q96_ctxvs_res2_augon.sh   # after a restart
DRYRUN=1 bash ...   # preflight + resolved settings only
```

- Run dir (CPFS): `.runtime/final_runs/<run>/` — `logs/train_node0_*.log`, TensorBoard `runs/`, `run_info.txt`,
  `code_*.diff`. `checkpoints/` there is a link to OSS `/mnt/oss/xiesy/lingbot-va-checkpoints/final_runs/<run>/checkpoints`
  (ossfs2 cannot append, so logs/TensorBoard must not live on OSS).
- Saves every 2,500 steps (≈ 75 GB each). A pruner keeps the newest 2 complete saves (resume) and the
  `hf_ckpt` of every 10k step (eval); everything else is deleted.
- Multi-node works the same way (DLC sets `WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT`); global batch stays 1024.
- A watchdog writes a node snapshot (py-spy stacks, memory, I/O) to `logs/` if the train log is idle for 15 min.

## 3. Batch, memory, speed (measured on 2 × PPU-ZW810E, 97.9 GB, compile off)

Global batch = MBS × world × grad-accum, so **memory depends on MBS only — bs 1024 and bs 512 need the
same memory**. Defaults (train.sh): **res1 MBS 32, res2 MBS 16, gradient checkpointing on** → bs 1024 on 8
GPUs = GA 4 (res1) / GA 8 (res2). Measured with `train.sh` itself on the smoke d4rt data (2 tasks):

| config | MBS | grad ckpt | micro-step | optimizer step | samples/s/GPU | max alloc / reserved (GB), alloc retries |
|---|---|---|---|---|---|---|
| res1 | **32** | on | 6.2 s | 2.7 s | **5.1** | 86.9 / 93.3, 0 |
| res1 | 16 | off | 3.6 s | 2.7 s | 4.5 | 93.1 / 94.7, 21–37 (too tight) |
| res1 | 64 | on | OOM | | | |
| res2 | **16** | on | 6.0 s | 2.7 s | **2.7** | 84.6 / 91.0, 0 (q96: same speed) |
| res2 | 8 | off | 3.4 s | 2.7 s | 2.4 | 90.8 / 94.7, 12 (too tight) |
| res2 | 32 | on | OOM | | | |

- micro-step / optimizer step from GA 1 vs GA 4 runs (the Muon update runs once per optimizer step);
  the two no-checkpointing rows had only a GA 1 run: micro-step = GA-1 step − 2.7 s.
- **torch.compile is slower on PPU** with these settings (res1 10.3 vs 8.9 s, res2 9.9 vs 8.75 s per
  GA-1 step, + ~150 s cold compile), so off is the right default for speed too, not only for memory.
- On 8 ranks the FSDP-sharded parameters / gradients / optimizer state per GPU shrink 4× vs 2 ranks
  (estimated ~25 GB less per GPU), so there is extra headroom; `MBS=64` (res1) / `MBS=32` (res2) may fit
  there but are untested (2-rank runs OOM).

**bs 1024 × 50k on one 8-PPU node** (extrapolated: GA × micro-step + optimizer step; 8-rank FSDP
communication not measured here, check `StepTime` in the first log lines):

| run | MBS × 8 × GA | s / step | 50k steps | samples (epochs) |
|---|---|---|---|---|
| res1 | 32 × 8 × 4 | ~28 s | **~16 days** | 51.2 M (93) |
| res2 (×3 runs) | 16 × 8 × 8 | ~51 s | **~29.5 days** | 51.2 M (93) |
| res1, bs 512 (`GBS=512`) | 32 × 8 × 2 | ~15 s | ~9 days | 25.6 M (47) |
| res2, bs 512 (`GBS=512`) | 16 × 8 × 4 | ~27 s | ~15.5 days | 25.6 M (47) |

For scale: the H20 spatial-exp lines were bs 128 × 10k = 1.28 M samples (2.3 epochs). More nodes cut
wall clock almost linearly (same bs 1024, smaller GA; e.g. 4 nodes → res2 ~8–9 days, cross-node
communication unmeasured).

## 4. Code changes on `xiesy` for these runs

- **Merge** `origin/feat/full432-augmentation` (qianlei, 19 commits: resolution baselines, full432 input,
  ROI-432, observation augmentation, W&B, loader resume) into `xiesy`. Conflicts: `prepare_images` gained a
  keyword on both sides (`visual_perturb` for spatial-exp cotrain, `fixed_image_size` for full432); both kept.
  Not merged: `lmx_rtc` (inference-time RTC), `lmx_sf`, `ysj/action-parameterization`.
- **d4rt × input resolution** (`spatial-exp/d4rt_lingbot/{data,hooks}.py`): the sampler assumes a 224×224
  canvas, but the model's view is `data.img_size` — 256 at res1 (the default; verified on a real sample:
  `pil_images` = 3×3×256×256), 384 at res2. The query RGB patches were therefore cut from the wrong place
  (top-left 224 region of the 256 view) while targets used the full-view mapping — this also applies to the
  H20 spatial-exp runs. The views are now area-resized to the 224 canvas first, so the d4rt task is identical
  at every resolution. With 224 inputs the new sampler is bit-identical to the old one (192 real samples,
  K = 8 and 96).
- **d4rt × augmentation**: the full432 augmentation's 2-D scale/shift moves the image under the probe.
  Queries are now sampled where the augmented image shows the scene, `d4rt_coords`/`d4rt_patches` come from
  the augmented canvas, and each target cell is the scene point under the augmented pixel
  (`CanvasAffine`; photometric parts need nothing). Tested against the augmenter's real `cv2.warpAffine`
  (error < 0.01 grey levels; half-pixel convention errors are detected) and by projecting depth targets back
  through the camera (`tests/test_d4rt_augment.py`).
- d4rt task names also accept the PPU-side LeRobot dirs (`<task>-aloha-agilex_clean_50-50`);
  `validate_targets.py --root`.
- **Speed, pure** (`base_dataset._query_hf_dataset`): select the column before the rows. Row-first decoded
  every column of the 50 queried action rows, i.e. 150 camera PNGs per sample: 1.02 s → 0.10 s of loader CPU
  per training sample, identical tensors. On 2 PPU the loader was just keeping up (9.1 → 8.9 s per step);
  it removes the data-bound risk on DLC and frees ~10× loader CPU.
- PPU env (`xsy_utils/env/{setup_venv_ppu.sh,ppu_env.sh}`, `configs/ppu/`, `train_baseline.sh` fallback).

Tests: `pytest tests/` → 132 passed (126 existing incl. qianlei's + 6 new).
