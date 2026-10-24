# final_runs — d4rt-ctxvs × {resolution, colour augmentation}, DLC 8 × PPU

Seven runs, all **20k steps**; five on one 8-PPU node at **global batch 256** (5.1 M samples, 9.3 epochs; changed
from 1024 × 50k on 2026-10-23, which would have taken 16–30 days), and two at the **official global batch 1024**
(20.5 M samples, 37 epochs): `d4rt_q96_ctxvs_res1_augon` on 4 nodes × 8 PPU (MBS 32, no grad accumulation) and the
final run `final_d4rt_q96_ctxvs_res2_augon_bs1024` on 8 nodes × 8 PPU (MBS 16, no grad accumulation, ~1.7 days). lr as in the official RoboTwin recipe (1e-4 → 5e-5 cosine over the run,
Muon, fp32), **torch.compile off**, initialised from `lingbot-vla-v2-6b`, RoboTwin clean-50 (50 tasks ×
50 episodes, 549,787 frames).

| script | res | aug | d4rt queries | role |
|---|---|---|---|---|
| `train_d4rt_q08_ctxvs_res2_augon.sh` | 384² (432 tok) | on | 8 | hero |
| `train_d4rt_q96_ctxvs_res2_augon.sh` | 384² (432 tok) | on | 96 | hero |
| `train_d4rt_q08_ctxvs_res2_augoff.sh` | 384² (432 tok) | off | 8 | fallback |
| `train_d4rt_q96_ctxvs_res2_augoff.sh` | 384² (432 tok) | off | 96 | q96 twin of the res2 aug-off fallback |
| `train_d4rt_q08_ctxvs_res1_augoff.sh` | 256² (192 tok) | off | 8 | fallback |
| `train_d4rt_q96_ctxvs_res1_augon.sh` | 256² (192 tok) | on | 96 | **4 nodes, bs 1024** |
| `train_final_d4rt_q96_ctxvs_res2_augon_bs1024.sh` | 384² (432 tok) | on | 96 | **final run: 8 nodes, bs 1024**, save every 1000 |

- **d4rt-ctxvs** = `xsy_utils/spatial-exp/configs/robotwin_clean_d4rt_ctxvs.yaml` (alignment teachers +
  d4rt probe, `context: visual_state`), unchanged except paths. res1 keeps its torchvision-bilinear 256².
- **res2** = qianlei's full432 input: 384², OpenCV Lanczos4, fixed processor grid, 12×12 student grid.
- **aug** = qianlei's full432 observation augmentation (`configs/vla/robotwin/full432_augmentation_recipe.yaml`;
  colour/temperature, 2-D scale+shift, noise, blur, JPEG; occlusion is skipped — no foreground masks).
  At res1 the same recipe runs on the 256² frames (its input guard accepted only full432 before, section 4):
  everything is relative to the image except blur σ / motion-blur kernel, which are in pixels and so ~1.5×
  stronger relative to the frame than at 384².
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
- Saves every 1,000 steps (≈ 75 GB each; was 2,500 until 2026-10-25). A pruner keeps the newest 2 complete saves (resume) and the
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
(`PET_NNODES/PET_NODE_RANK` take precedence if present). The global batch is the config's (`GBS` overrides): the
256 runs on 16 GPUs need `MBS=16` at res1 (32 × 16 > 256). `train_d4rt_q96_ctxvs_res1_augon.sh` is the 4-node run
(1024 = 32 GPUs × 32) and stops on any other node count unless `ALLOW_NNODES=1` (it would accumulate gradients:
same batch, 2–4× slower).

Measured on DLC 2 × 8 (2026-10-24, `test_multinode.sh`, full shard over 16, `NCCL_IB_DISABLE=1` → inter-node
busbw ~10 GB/s): q96 res2 bs 256 **7.0 s/step** (MBS 16, GA 1) vs 13.6 s on one node (GA 2) = 1.94×. The 4-node
res1 run has the same per-GPU work as the 1-node res1 runs (MBS 32, GA 1; 7.1 s/step on DLC), so expect ~7–7.5
s/step (20k ≈ 1.7 days) if the inter-node traffic stays hidden as it did on 2 nodes — untested on 4 nodes.

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

Measured on DLC after the Muon change (2026-10-25, steps 4k–6k): res2 bs 256 StepTime median 13.5–13.9 s, wall
clock 14.2 s/step including saves (q96 aug-on); res1 7.1 s. Hence **`final_d4rt_q96_ctxvs_res2_augon_bs1024`**
on 8 nodes (16 × 64, GA 1): one micro-step ≈ (13.9 − ~0.7 s optimizer) / 2 ≈ 6.6 s + optimizer + communication;
the 2 × 8 test ran the same per-GPU work (q96 res2, MBS 16, GA 1) at 7.0 s/step with the inter-node traffic hidden
→ expect **~7–7.5 s/step, 20k steps ≈ 1.7 days** (+ ~1 h startup / saves). Untested at 8 nodes: per node the
full-shard traffic per step is the same as on 2 nodes, but if the first steps are much slower (> 9 s), relaunch
with `DP_SHARD=8` (HSDP: all-gathers inside the node, only gradient all-reduce across nodes). On one node the
same batch would take ~53 s/step (GA 8), ~12.5 days.

Plus ~1 h of startup and saves (DCP every 1,000 steps, ~1 min; HF export async). For scale: the H20 spatial-exp
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
- **Loader worker seeds per rank** (`lingbotvla/data/data_loader.py`, 2026-10-24): worker RNG derived from the
  global torch RNG, identical on every rank after `set_seed`, so worker k of every rank drew the same
  observation-augmentation seeds: the j-th sample of each rank's micro-batch got identical augmentation (1/8
  of the intended diversity per global batch on 8 GPUs; aug-off runs unaffected). The DataLoader now gets a
  generator seeded per data-parallel rank; the sampler (data partition) keeps the shared seed.
  `tests/test_dataloader_worker_seed.py` (fails on the old code).
- **Muon optimizer** (`lingbotvla/optim/muon.py`, 2026-10-24):
  - *Speed, same math*: every rank used to all-gather and Newton-Schulz-orthogonalise every 2D matrix of the
    model (831 matrices, 2.31 s per optimizer step per rank, independent of the number of GPUs). Now matrix i
    of a same-shape batch is orthogonalised on rank i % world only (all-to-all in / out): 0.31 s on 8 ranks.
    2 PPU res2 step 8.78 → 7.63 s; 8 PPU res2 (bs 256, GA 2) expected ~15.4 → ~13.4 s.
  - *Correctness*: the shard layout assumed "the first rows % world ranks get one extra row"; DTensor/FSDP2
    use torch.chunk (ceil-sized chunks). They differ for the d4rt head's 588-row `to_pred` on 8 ranks
    (7 × 74 + 70 vs 4 × 74 + 4 × 73): its Muon update was assembled from the wrong rows and applied at the
    wrong offsets, every step, silently (exact on 2 ranks, so the 2-PPU tests never saw it). Both paths
    now use the real layout and assert it. `tests/test_muon_distributed_ns.py` checks 8 and 3 CPU ranks
    against Muon on the full tensors (the old code fails it on the 588-row matrices by ~1 lr).
- d4rt task names also accept the PPU-side LeRobot dirs (`<task>-aloha-agilex_clean_50-50`);
  `validate_targets.py --root`.
- **Speed, pure** (`base_dataset._query_hf_dataset`): select the column before the rows. Row-first decoded
  every column of the 50 queried action rows, i.e. 150 camera PNGs per sample: 1.02 s → 0.10 s of loader CPU
  per training sample, identical tensors. On 2 PPU the loader was just keeping up (9.1 → 8.9 s per step);
  it removes the data-bound risk on DLC and frees ~10× loader CPU.
- PPU env (`xsy_utils/env/{setup_venv_ppu.sh,ppu_env.sh}`, `configs/ppu/`, `train_baseline.sh` fallback).
- **full432 augmentation at res1** (`lingbotvla/data/vla_data/utils.py`, 2026-10-24): the augmenter's input guard
  accepted only 384 Lanczos4 / fixed grid (432 tokens). It runs on the resized frames before the processor and
  keeps their size, so it is resolution-independent; the guard now also accepts the default 256 bilinear input
  with its fixed budget of 192 tokens (3 × 8 × 8), and the post-processor budget check uses the layout's value.
  Checked on real samples (`d4rt_q96_ctxvs_res1_augon`: augmented vs un-augmented pixels, budget check,
  d4rt fields), `tests/test_observation_augmentation.py` passes; full432 behaviour unchanged.

Tests: `pytest tests/` → 132 passed (126 existing incl. qianlei's + 6 new).
