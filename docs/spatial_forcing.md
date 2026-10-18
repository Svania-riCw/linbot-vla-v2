# Spatial Forcing for LingBot-VLA 2.0

This branch adds Spatial Forcing (SF) from [arXiv:2510.12276](https://arxiv.org/abs/2510.12276) to the V2 training path. SF runs frozen VGGT on the same augmented camera views used by the policy, strips VGGT's camera/register tokens, resizes its patch features onto the Qwen3-VL visual-token grid, adds the paper's 2-D sinusoidal position signal, and aligns an intermediate VLA layer with a BatchNorm + two-layer MLP and cosine loss. The weighted alignment loss is added to the existing flow-matching/action objective. VGGT and the projection head are training-only; inference uses the unchanged policy path.

## Install the optional teacher package

Use the same Python environment as LingBot-VLA:

```bash
pip install --no-deps -r requirements-spatial-forcing.txt
```

## Download the VGGT checkpoint

The official `facebook/VGGT-1B` checkpoint is 5,026,874,952 bytes. From the repository root, download it to the path used in `robotwin_sf.yaml`:

```bash
mkdir -p checkpoints/VGGT-1B
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="facebook/VGGT-1B",
    filename="model.pt",
    local_dir="checkpoints/VGGT-1B",
)
PY
```

If your machine cannot reach Hugging Face directly, configure `HF_ENDPOINT` to a reachable Hugging Face mirror before running the same command. The official repository distinguishes the original VGGT-1B checkpoint from its commercial-use checkpoint; check the upstream license for your intended use.

The downloaded file's SHA-256 should be `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe`. Keep the weight local; it is excluded from Git because of its size.

## Train

Set the model/tokenizer paths, dataset manifest, robot config, normalization statistics, and output directory in `configs/vla/robotwin/robotwin_sf.yaml`, then launch:

```bash
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/robotwin/robotwin_sf.yaml
```

The SF configuration enables alignment with Qwen layer 23 and the final cached VGGT feature, uses VGGT's 2048-dimensional concatenated frame/global representation, and sets the alignment coefficient to 0.5 as in the paper's default experiments. The original `robotwin.yaml` keeps SF disabled so existing training recipes are unaffected; set `train.align_params.spatial_forcing.enabled: true` there only if you explicitly want to combine SF with its depth/video losses.

The teacher can also load directly from the Hugging Face repo when the configured local checkpoint is absent and network access is available. `train.sh` runs in offline mode, so download the weight before using that launcher. `train.align_params.spatial_forcing.checkpoint_path` points at the local download above; `repo_id` is the fallback. VGGT targets are built only for training batches and are not written to checkpoints or the repository.

## Precision and no-RTC recipe

`configs/vla/robotwin/robotwin_sf_no_rtc.yaml` is the SF-only recipe with the precision and RTC choices written explicitly. It sets `training_time_rtc: false`, leaves `data.use_future_image: false`, and enables only the SF teacher. Launch it with:

```bash
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/robotwin/robotwin_sf_no_rtc.yaml
```

The repository's FSDP2 baseline is mixed precision at the model level: `enable_mixed_precision` defaults to `true`, while `enable_fp32: true` selects FP32 for the action-expert FSDP groups. The Qwen-VL groups remain BF16 under the FSDP2 policy, so this is not an all-parameter FP32 run. The depth/video/VGGT teacher paths also use BF16 CUDA autocast; SF targets are stored as BF16 by default. The new recipe makes these baseline choices explicit and does not enable RTC.

To combine SF with the baseline's future-image, depth, and video losses while still keeping RTC off, use `configs/vla/robotwin/robotwin_sf_with_teachers_no_rtc.yaml`:

```bash
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/robotwin/robotwin_sf_with_teachers_no_rtc.yaml
```

This combined recipe uses the local MoGe, MoRGBD, DINO-video, and VGGT teachers. It references the baseline run's task-list file and uses its seed (42) for an aligned sample order. It keeps gradient checkpointing disabled to match the baseline run; the VGGT teacher therefore needs enough free GPU memory. The SF-only recipe is better for an apples-to-apples SF ablation; after that run is validated, the combined recipe is the better choice when the goal is to maximize the full auxiliary-training signal. Its result mixes the SF gain with the depth/video losses.
