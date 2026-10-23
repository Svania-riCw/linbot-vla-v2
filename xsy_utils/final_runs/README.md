# final_runs — d4rt-ctxvs × {resolution, colour augmentation}, DLC 8 × PPU

Five runs, all **global batch 256, 20k steps** (5.1 M samples, 9.3 epochs; changed from 1024 × 50k on 2026-10-23,
which would have taken 16–30 days), lr as in the official RoboTwin recipe (1e-4 → 5e-5 cosine over the run,
Muon, fp32), **torch.compile off**, initialised from `lingbot-vla-v2-6b`, RoboTwin clean-50 (50 tasks ×
50 episodes, 549,787 frames).

| script | res | aug | d4rt queries | role |
|---|---|---|---|---|
| `train_d4rt_q08_ctxvs_res2_augon.sh` | 384² (432 tok) | on | 8 | hero |
| `train_d4rt_q96_ctxvs_res2_augon.sh` | 384² (432 tok) | on | 96 | hero |
| `train_d4rt_q08_ctxvs_res2_augoff.sh` | 384² (432 tok) | off | 8 | fallback |
| `train_d4rt_q96_ctxvs_res2_augoff.sh` | 384² (432 tok) | off | 96 | q96 twin of the res2 aug-off fallback |
| `train_d4rt_q08_ctxvs_res1_augoff.sh` | 256² (192 tok) | off | 8 | fallback |

- **d4rt-ctxvs** = `xsy_utils/spatial-exp/configs/robotwin_clean_d4rt_ctxvs.yaml` (alignment teachers +
  d4rt probe, `context: visual_state`), unchanged except paths. res1 keeps its torchvision-bilinear 256².
- **res2** = qianlei's full432 input: 384², OpenCV Lanczos4, fixed processor grid, 12×12 student grid.
- **aug** = qianlei's full432 observation augmentation (`configs/vla/robotwin/full432_augmentation_recipe.yaml`;
  colour/temperature, 2-D scale+shift, noise, blur, JPEG; occlusion is skipped — no foreground masks).
- Configs are generated: `python xsy_utils/final_runs/gen_configs.py` → `configs/*.yaml` (edit the generator).

## 1. Data

The d4rt targets (copy of H20 `/2025233210/xiesy/data/lingbot_robotwin_clean_d4rt/`, ≈ 262 GB, 98 % `depth/`) are
at **`/mnt/cpfs_shared/lingbot-data/lingbot_robotwin_clean_d4rt/`** (team CPFS, outside the personal 2 TB
quota of `/mnt/cpfs`); `D4RT_ROOT` in `gen_configs.py`. Also on OSS at `/mnt/oss/xiesy/lingbot-data/` — **do not
train from there**: in the 2 × 8 DLC test (2026-10-23) every loader worker failed at step 0 with `OSError(107,
'Transport endpoint is not connected')` (after a few `ECONNABORTED`) on both nodes, i.e. the node's ossfs mount
died under 8 ranks × 12 workers of d4rt reads. Layout `<task>/{ee,cam,depth/<cam>,depth_meta/<cam>,tracks/<cam>,tvis/<cam>,tmeta/<cam>}/episode_NNNNNN.*`.
Verified 2026-10-23: all 2,500 episodes complete, `ee` rows = LeRobot length + 1, and `--compare-ee` bit-identical.

**Reading it (measured on OSS, before the move).** Every sample opens the ~20 files of a random episode and reads a few depth frames; on
ossfs each open / page fault is a round trip (~2.5 ms per 4 KB page). `d4rt_lingbot/data.py` therefore opens an
episode's files concurrently and reads whole frames / track windows with one `pread` each (outputs bit-identical
to the memmap version: 480 samples over all 50 tasks, K = 8/96, with and without the augmentation affine).
Cold (never-read episode) d4rt cost per sample on the dev box, K = 96:

| workers | old (memmap) | new |
|---|---|---|
| 8  | 0.93 s, 8.6 samples/s | 0.29 s, 27.7 samples/s |
| 16 | 1.18 s, 13.6 samples/s | 0.28 s, 56.7 samples/s |
| 32–64 | 1.1–1.5 s, 28–43 samples/s | 0.5–1.1 s, ~57–60 samples/s (dev box saturated: ossfs kernel CPU on 16 cores) |

Needed per 8-GPU node: ~41 samples/s (res1), ~22 (res2); CPU in the worker itself is 0.04 s per sample.
CPFS (same task, copied): 0.07 s per sample.

Check (also runs automatically at every launch):

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
(≈ 150 GB = the parquet size, once, shared by all runs; a few minutes with `warm_hf_cache.py`'s 8 processes, estimated:
the first 4 datasets, 14 GB, took 26 s). The old cache in `legacy_lingbot/.cache/huggingface` (141 GB) is
keyed on the pre-move path `/mnt/cpfs/lingbot-va2/...` and is not used by these runs.

## 2. Launch (DLC, 8 × PPU, same image as the dev box)

```bash
bash /mnt/cpfs/linbot-vla-v2/xsy_utils/final_runs/train_d4rt_q96_ctxvs_res2_augon.sh        # one job per run
RESUME=1 bash /mnt/cpfs/linbot-vla-v2/xsy_utils/final_runs/train_d4rt_q96_ctxvs_res2_augon.sh   # after a restart
DRYRUN=1 bash ...   # preflight + resolved settings only
```

- Run dir (CPFS): `.runtime/final_runs/<run>/` — `logs/train_node0_*.log`, TensorBoard `runs/`, `run_info.txt`,
  `code_*.diff`. `checkpoints/` there is a link to OSS **`/mnt/oss/xiesy/lingbot-ckpts/<run>/checkpoints`**
  (`CKPT_BASE`; ossfs2 cannot append, so logs/TensorBoard must not live on OSS).
- Saves every 2,500 steps (≈ 75 GB each). A pruner keeps the newest 2 complete saves (resume) and the
  `hf_ckpt` of every 5k step (eval: 5k, 10k, 15k, 20k); everything else is deleted.
- The HF export of a save (rank 0, ~24 GB of safetensors written single-threaded) takes **~12 min on OSS**
  (40–50 s on local disk). Synchronous, it kept the other ranks in a collective past torch's 10-min NCCL
  timeout and killed the (simulated 2-node) test run at its first save — the full runs would have died at
  step 2,500. Now `--train.async_save_hf_weights true` (training continues while rank 0 exports) and
  `LINGBOT_PG_TIMEOUT_MIN=60` (end-of-run wait for the last export). `RESUME=1` only picks saves whose
  `hf_ckpt/config.json` exists (the export lands atomically; resume reads it for the augmentation check).
- Multi-node: section 2b.
- **Several jobs at once is fine.** The HF-datasets cache of all 50 datasets is built (2026-10-23, in
  `.runtime/huggingface`), so every job only opens it (~1 min in `warm_hf_cache.py`). If it ever has to be
  rebuilt (new data / parquet mtimes), jobs started together take turns through a lock dir next to the cache
  (`.warm_lock`, atomic mkdir on CPFS; HF's own file lock is per machine): the first builds, the rest wait
  and then open it. A lock left by a killed job is broken after 3 h.
- A watchdog writes a node snapshot (py-spy stacks, memory, I/O) to `logs/` if the train log is idle for 15 min.

## 2b. Multi-node (e.g. 2 × 8 = 16 PPU)

Same command on every node; DLC exports `WORLD_SIZE` (= number of nodes), `RANK` (= node rank), `MASTER_ADDR/PORT`
(`PET_NNODES/PET_NODE_RANK` take precedence if present). Global batch stays 256 (res1: 32 × 16 already covers it at GA 1 → `MBS=16`).

```bash
bash .../train_d4rt_q96_ctxvs_res2_augon.sh                 # full shard over all 16 ranks
DP_SHARD=8 bash .../train_d4rt_q96_ctxvs_res2_augon.sh      # HSDP: shard inside a node, replicate across nodes
```

- The trainer is multi-node clean: global rank for logging / TensorBoard / HF export, the sampler uses the
  global dp rank, DCP + per-rank `extra_state` go to the shared OSS dir; resume needs the same world size.
- `train.sh` additions for > 1 node: node 0 alone builds/opens the HF-datasets cache (`warm_hf_cache.py`; the
  cache's file lock does not work across nodes) while the others wait; node 0 alone creates the checkpoint link,
  `run_info.txt`, the code diff, and runs the pruner; every node logs to `logs/train_node<rank>_*.log`.
- **Full shard vs HSDP.** Full shard over 16 ranks puts the per-micro-step parameter all-gathers and gradient
  reduce-scatters on the inter-node links; HSDP (`DP_SHARD=8`) keeps them inside the node and only all-reduces
  1/8 of the gradients across nodes (~15× less inter-node traffic), at the per-GPU memory of the 8-GPU run.
  Which is faster depends on the RDMA fabric — `test_multinode.sh` measures both.
- Check the comm env in the first log lines: the dev box sets `NCCL_IB_DISABLE=1` (no RDMA); a DLC RDMA job must
  not, or inter-node traffic goes over TCP.

**Test first** (≈ 40 min on the dev box, ~20 of them the HF export to OSS; same command on every node; the test checkpoints are deleted afterwards):

```bash
bash /mnt/cpfs/linbot-vla-v2/xsy_utils/final_runs/test_multinode.sh
DP_SHARD=8 bash /mnt/cpfs/linbot-vla-v2/xsy_utils/final_runs/test_multinode.sh   # second job: HSDP
```

1. `comm`: rank → host placement, all_reduce correctness, all_gather / reduce_scatter / all_reduce bandwidth for
   all ranks, inside a node and between nodes (inter-node busbw of only a few GB/s = TCP fallback);
2. `train`: the q96 hero config at the real bs, 4 steps on the first 4 datasets (`TEST_DATASETS`), save;
3. `resume`: `RESUME=1` from that save, 2 more steps.
Node 0 prints a summary (StepTime per stage); the log is `.runtime/final_runs/<run>_mntest_2x8[_hsdp8]/logs/test_node*.log`.
Compare StepTime with the single-node numbers below (section 3).

## 3. Batch, memory, speed (measured on 2 × PPU-ZW810E, 97.9 GB, compile off)

Global batch = MBS × world × grad-accum, so **memory depends on MBS only**. Defaults (train.sh): **res1 MBS 32,
res2 MBS 16, gradient checkpointing on** → bs 256 on 8 GPUs = GA 1 (res1) / GA 2 (res2). Measured with `train.sh` itself on the smoke d4rt data (2 tasks):

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

**bs 256 × 20k on one 8-PPU node** (GA × micro-step + optimizer step; the DLC res1 run at bs 1024 / GA 4 measured
27.9 s/step = 4 × 6.2 + 3.1, i.e. 8-rank communication adds little):

| run | MBS × 8 × GA | s / step | 20k steps |
|---|---|---|---|
| res1 | 32 × 8 × 1 | ~9 s | **~2.1 days** |
| res2 (×4 runs) | 16 × 8 × 2 | ~15 s | **~3.5 days** |

Plus ~1 h of startup and saves (DCP every 2,500 steps, ~1 min; HF export async). For scale: the H20 spatial-exp
lines were bs 128 × 10k = 1.28 M samples (2.3 epochs).

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
- **Speed, pure** (`d4rt_lingbot/data.py`): concurrent episode open + one `pread` per depth frame / track window
  instead of memmap page faults (section 1), bit-identical outputs.
- Multi-node launcher (`train.sh`, `warm_hf_cache.py`, `comm_test.py`, `test_multinode.sh`; section 2b).
- d4rt task names also accept the PPU-side LeRobot dirs (`<task>-aloha-agilex_clean_50-50`);
  `validate_targets.py --root`.
- **Speed, pure** (`base_dataset._query_hf_dataset`): select the column before the rows. Row-first decoded
  every column of the 50 queried action rows, i.e. 150 camera PNGs per sample: 1.02 s → 0.10 s of loader CPU
  per training sample, identical tensors. On 2 PPU the loader was just keeping up (9.1 → 8.9 s per step);
  it removes the data-bound risk on DLC and frees ~10× loader CPU.
- PPU env (`xsy_utils/env/{setup_venv_ppu.sh,ppu_env.sh}`, `configs/ppu/`, `train_baseline.sh` fallback).

Tests: `pytest tests/` → 132 passed (126 existing incl. qianlei's + 6 new).
