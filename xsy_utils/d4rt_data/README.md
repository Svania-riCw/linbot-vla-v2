# d4rt auxiliary data for lingbot's RoboTwin clean set

Depth / point tracks / end-effector data in the style of pi05-experiments' d4rt line
(`xsy_utils/d4rt_inject/prep_{depth3d_3cam,tracks,ee}.py` there), for lingbot's training set
`/2025233210/lmx/training-data/lerobot/robotwin_clean_*` (50 tasks x 50 episodes, 549,787 frames,
3 cameras), in case we train d4rt + lingbot. Not consumed by any training code yet.

Env: conda `d4rt-prep` (`bash xsy_utils/env/setup.sh d4rt_env`), weights in `$WS/checkpoints/`.

```bash
cd xsy_utils/d4rt_data
python prep_ee.py                      # CPU, ~1 min
bash run_all.sh depth                  # MoGe-3, all GPUs
bash run_all.sh tracks                 # CoTracker3, all GPUs
python validate.py --tasks place_shoe --episodes 0,1 --out /tmp/val   # numbers + overlay PNGs
```

## Source and conventions (verified, not assumed)

- Built from the **raw RoboTwin HDF5** (`/2025233210/lmx/RoboTwin/data/<task>/aloha-agilex_clean_50/data/episode{i}.hdf5`),
  which the lerobot set was converted from. lerobot episode `i` = `episode{i}.hdf5`; `prep_ee.py`
  re-checks every episode (`T_lerobot == T_raw - 1`, `state == joint_action/vector[:-1]`).
- `state[t] = raw[t]`, `action[t] = raw[t+1]`. **Every per-frame array has `T_raw` rows**; row `t` =
  lerobot frame `t`, the last row only exists as an action target.
- Raw RGB is **320x240** JPEG. The lerobot images are RoboTwin `process_data.py`'s
  `cv2.resize(.., (640, 480))` of it (re-encoded; mean |diff| 3-7 / 255 = JPEG noise). All pixel
  coordinates and maps here are **native 320x240**; lerobot pixel = 2x.
- cam_high = `head_camera`, cam_left_wrist = `left_camera`, cam_right_wrist = `right_camera`
  (checked by image difference). GT K identical for all three (fx = 358.6 px, fov_x = 48.1 deg).

## Outputs: `/2025233210/xiesy/data/lingbot_robotwin_clean_d4rt/<task>/`

| path | dtype / shape | content |
|---|---|---|
| `ee/episode_N.npy` | f32 [T_raw, 16] | `[Lpos, Lquat_wxyz, Lgrip, Rpos, Rquat_wxyz, Rgrip]`, world frame (pi05 layout) |
| `cam/episode_N.npz` | | `intrinsic` [3,T,3,3], `extrinsic` [3,T,3,4] world->cam (OpenCV), `grip` [T,3 cams,2 arms,(u,v,z,valid)] |
| `depth/<cam>/episode_N.npy` | u16 [T_raw, 240, 320] | scale-corrected metric depth, **mm**, 0 = invalid |
| `depth_meta/<cam>/episode_N.npz` | | MoGe `intrinsic`, `valid_frac`, `scale`, `scale_raw`, `inlier_frac`, `reliable` |
| `tracks/<cam>/episode_N.npy` | f16 [W, 50, 400, 2] | CoTracker3 (u, v), window 50, re-seeded every 25 (= chunk_size/2), 20x20 grid |
| `tvis/<cam>/episode_N.npy` | bool [W, 50, 400] | visibility |
| `tmeta/<cam>/episode_N.npz` | | `starts` [W], window, stride, grid_size, num_frames |

Plus `EE_INFO.json`, `DEPTH_INFO.json`, `TRACKS_INFO.json` at the root.

## Depth: MoGe-3 + per-frame plane scale correction

The raw HDF5 has **no depth** (pi05's RoboTwin set had simulator `depth_mm`; here we estimate).
MoGe-3 vitl runs with the **GT fov imposed**, so the focal is exact and the shape is right, but its
metric scale on these renders is wrong *and unstable*: on ep0 the required correction is
0.43-0.58 frame to frame on the **static** cam_high, 0.15-0.54 on the wrists (raw depth up to
~6x too far; median MoGe/GT at the gripper 1.6-1.7).

`scale_align.py` fixes it per frame against the two planes whose pose is known exactly -- the
table top (z = 0.74, 1.2 x 0.7 m; per-task offsets for `dump_bin_bigbin`, `put_bottles_dustbin`,
`place_dual_shoes`) and the back wall (y = 0.40) -- using each camera's GT extrinsic of that frame:

- per pixel, the depth the plane *would* have (ray/plane intersection);
- `scale` = mode of `log(plane_depth / moge_depth)` over those pixels. Objects / arm in front of
  the plane only push the ratio up, so the mode ignores them without any colour mask (pi05's
  `DEPTH_ANCHORING.md`: colour masks break on randomized tables);
- pure scale, no shift: the plane residual after correction is 1.00 +- 0.01 from near to far;
- each frame keeps **its own** fit (MoGe's raw scale jitters +-30% between consecutive frames of a
  static view, so temporal smoothing of the correction would be wrong). Frames with < 1500 plane
  inlier pixels get an interpolated scale and `reliable = False` -- mask them in training.

Checks (`validate.py`, place_shoe ep0-1):

| check | result |
|---|---|
| focal vs GT | 0 px (imposed) |
| depth at projected gripper / GT endpose z (cam_high) | 0.80-0.89 (endpose point is inside the gripper; the visible surface is a few cm closer, so < 1 is expected) |
| cross-camera, independent of the planes: wrist non-plane pixels -> world -> cam_high | ratio mode 1.01-1.09, 33-64% of points within 5% (rest mostly occlusion) |
| reliable frames | 100% on these episodes |

Throughput: ~22 frames/s per H20 (1200 tokens, 3 refine steps, fp16) -> ~2.6 h for 1.66 M frames on 8 GPUs.

## Tracks

Same scheme and format as pi05's `prep_tracks.py`: a 50-frame window (= lingbot `chunk_size`) from
a fresh 20x20 grid every 25 frames (= chunk_size / 2) plus a tail window, so every frame is covered.
For frame `t` use the window with the largest start <= t (25..49 frames of horizon). **Track ids are
not stable across windows.** CoTracker3 `scaled_offline.pth`, ~2.7 windows/s per GPU -> ~50 min for
all 3 cameras on 8 GPUs. Local patch on upstream CoTracker (`setup.sh d4rt_env` reapplies it):
`coords.view(B*T, ..)` -> `.reshape` in `cotracker3_offline.py`, otherwise batch > 1 crashes.

## EE

`endpose/` from the HDF5, pi05 layout (wxyz quaternion, world frame, includes RoboTwin's
`gripper_bias` tool offset = what `take_action(action_type='ee')` consumes). `cam.grip` is the
endpose position projected into every camera. A wrist camera never sees its *own* arm's endpose
point (it is ~4 cm behind that camera, z < 0, stored as u = v = -1, valid = 0).
