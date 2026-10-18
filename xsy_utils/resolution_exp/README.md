# Exp1：增加输入分辨率的轻量基线

## 三节点候选排队（2026-10-19）

空闲检查同时读取真实 GPU compute 进程、显存/利用率和 `/proc` 中已启动的训练/推理进程，避免把 CUDA 尚未分配显存的启动窗口当作空闲；配置准备完成后、真正启动训练前再检查一次。它降低同时启动的冲突概率，但不构成所有用户共同遵守的预约锁，仍不能阻止其他任务在最后一次检查之后启动。不要通过缩小 micro batch 与他人的任务挤在同一张卡上。

如果正式训练在首个 step 之前因这种争用退出，先核实没有训练进程、正步数日志或 checkpoint，保留失败的 W&B run、输出和日志，再由明确的恢复操作归档失败现场并接续原 claim；不删除锁、不自动跨节点接管。新的 W&B run 记录 attempt 编号和先前失败信息。已经产生训练进度的任务不能沿用这一流程从头重跑。

已选中节点的等待器需要更新时，可显式使用 `--resume-idle-claim` 接续原 claim。必须先核实并停止本人处于 `waiting_for_idle_gpus` 且没有训练子进程的旧等待器；该选项拒绝存活的旧 PID、不同节点/主机、运行中的训练阶段或非空子进程。共享 claim 不删除，已完成阶段不重跑，原 owner 信息归档。它不会自动接管失败或失联的任务。

正式训练会把已有 `EnvironMeter` 指标写入 TensorBoard 并同步 W&B，包括峰值显存、CPU 内存、token 吞吐与分配重试次数，并额外记录 `samples_per_second = global_batch / step_time`。FLOPS/MFU 沿用已有估算，不能当作硬件实测；跨分辨率的速度比较优先使用稳定的 step 时间和 samples/s。此前 GPU smoke 已验证 loss 与保存，但其日志没有持久化这一组性能指标。

三个节点各运行一个自有 tmux 等待器，共享唯一实验目录、成功 smoke checkpoint 和原子 `mkdir` 任务锁。首个连续确认8卡空闲的节点取得锁并接续实验，其他两个等待器自动记录取消并退出；只有锁的持有者能够调用训练入口。锁在失败或完成后仍保留，避免异常退出后其他节点悄悄重训。共享锁发布中也视为已占用。每节点还有本地进程锁，防止同一节点重复启动等待器。

`priority_worker.py` 复用 `queue_engine.py`，根据已完成阶段跳过成功的2步 smoke，先恢复至4步，再从初始权重训练10k；原5k暂停、350回合评测及每1000步保存策略保持不变。已完成的 smoke 先由 `stage_priority_storage.py` 复制并校验到自有共享目录，复制完成前候选不会运行训练。

等待器在服务器本地及时检测空闲；对话仍每小时检查一次。它们只竞争空闲资源，不暂停、终止或修改其他人的训练。这三台机器没有 Slurm/PBS/Pueue 等统一调度器，tmux 本身也没有跨用户作业优先级，因此该机制保证本实验全局只执行一份，不能约束别人绕过此锁手动启动作业。

## 当前执行计划（2026-10-18 更新）

用户提供现成的 `/2025233210/xiesy/lingbot_ws/outputs/baseline_gbs128_10k` 后，取消重训 256，也取消旧的 256/384 各 1000 步方案。下文两组配置用于解释对照；当前只训练 **384²、三路全图、432 visual tokens** 的改动版。

- 训练用全部 **50 个 clean 任务 × 每任务 50 episodes = 2500 episodes**；数据根目录 `/2025233210/lmx/training-data/lerobot`。不加入 random 训练数据。
- global batch **128**，训练 **10000 steps**，沿用 baseline 的初始化、优化器、学习率及 `enable_fp32=true`。这是仓库的 FP32 模式：融合 MoE、attention 和教师内部仍有显式 BF16 运算，不代表所有算子都使用纯 FP32。
- `tools/prepare_resolution_finetune.py` 从 baseline 实际的 `lingbotvla_cli.yaml` 生成配置，核验其 10k/batch128/FP32 与最终导出权重，再检查 50 个 clean 数据集的 episode 数。记录配置差异、baseline 的 seed/micro batch 和文件哈希。改动版为显存预留使用 micro4，smoke OOM 时可降至 2/1，global batch 保持128。
- **训练读取 `robot_config_root/robotwin.yaml` 的 norm_stats**；仅设置 `data.norm_stats_file` 不会覆盖训练映射。准备工具将训练映射与推理配置同时固定到同一份 clean 统计文件副本，并以 SHA256 检查 baseline 的统计内容相同。
- 每 **1000** 步保留 DCP 模型、优化器、学习率、数据加载器状态和 HF 推理权重。5000 和10000步都必须保留，不清理中途 checkpoint。
- 先跑384的2步保存，再恢复2步的 smoke，验证DCP恢复后才开始正式训练。正式训练始终设置 `max_steps=10000`，另设 `pause_after_steps=5000` 在保存后退出；完成5k评测后，从 **DCP global_step_5000** 恢复到10k。续训设置 `pause_after_steps=0`，不可仅用 HF 权重重启优化器。显式指定的 checkpoint 加载失败会报错，避免悄悄从头训练。
- 5k、10k 各评测 **7 tasks × clean/random × 25 = 350 episodes**：`blocks_ranking_rgb hanging_mug open_laptop pick_diverse_bottles rotate_qrcode stack_blocks_three place_shoe`。用户所写 BLOCK_RANKING_RGB 对应代码里的 `blocks_ranking_rgb`。random 只用于评测，统计仍用训练时的 clean 文件。
- `tools/eval_resolution_checkpoint.py` 显式传入 clean norm、FP32、相同动作长度与 seed，沿用训练 YAML 的384处理。底层脚本可能在部分任务失败时返回0，因此额外要求14项结果全部存在且每项25次，否则队列停止并保留日志。训练指标和逐任务评测指标均记录到用户指定 W&B 团队项目。

5k评测用于检查动作幅度、抓取/对齐、任务失败视频和输入映射是否异常，不能把5k改动版与10k baseline 的差异解释为改进收益。10k在相同评测条件下与已有 baseline 对比。已有 baseline 的插值、微批次或历史续训过程若不同，应在报告中明确记录；该实验不能单独证明纯 token 数量的因果作用。

准备命令（只检查并生成配置，不启动GPU任务）：

```bash
python tools/prepare_resolution_finetune.py \
  --destination /root/ql/linbot-vla-v2/.runtime/node1_queue/replan384
```

队列及部署状态以 `.runtime/node1_training_plan.json` 为准。2026-10-18 此次本地24项CPU测试通过；SSH握手失败，**新计划尚未部署，远端旧队列是否仍在等待尚未确认**。恢复连接后必须先处理本人旧队列，再应用新计划，不可并行启动第二个队列。

## 要证明什么

先比较三路全图输入的 **256×256** 和 **384×384**，只使用非 AI 插值，不增加 ROI、相机或训练数据。问题是：**同一批源 RGB，给模型更大的输入栅格和更多视觉 token，闭环任务成功率会不会提高，代价是多少？**

这一步建立两个视觉 token 预算，供后续在各自预算内重新分配全图与 ROI。它不能单独证明“纯粹增加 token”的因果作用，因为图像采样、patch 覆盖范围和计算量也一起变化；也不能证明插值恢复了新的真实细节。

## 现有图片到底多大

| 环节 | 分辨率（宽×高） | 证据和范围 |
| --- | --- | --- |
| RoboTwin 原始 RGB | 320×240 | 官方 `adjust_bottle/aloha-agilex_clean_50` 的 episode0，三个相机各检查第 0、69、139 帧，9 张均相同；不是全数据审计 |
| 默认采集配置 | 320×240 | `external/RoboTwin/task_config/_camera_config.yml` 的 D435；`demo_clean.yml` 的主相机和腕部均选择 D435 |
| 仓库保留的 pi0 转换 | 640×480 | `external/RoboTwin/policy/pi0/scripts/process_data.py` 对三路图调用 `cv2.resize(..., (640, 480))`，随后再次 JPEG 编码 |
| 团队现有 LeRobot 数据 | 640×480 | 2026-10-18 在 node1/2 核验训练清单全部 50 个数据集的 metadata，三路均为该尺寸；实际解码检查了 adjust_bottle 第 69 帧 |
| 模型默认输入 | 256×256 | `lingbotvla/data/dataset.py` 和部署入口默认 `img_size=256` |

320×240 是 76,800 像素，256×256 是 65,536 像素，原图面积只高约 **17.2%**，但宽高比也不同。640×480 如果来自上述转换，就是插值放大，不能视为原生高清数据。384×384 的像素数是默认输入的 2.25 倍、原始图的 1.92 倍。

本实验沿用现有数据。在模型入口使用 `cv2.INTER_LANCZOS4` 重采样，**不会消除数据转换时已有的插值或 JPEG 误差**，也不会引入超分辨率模型或额外训练信息。原始 4:3 图片仍按现有方式转换成正方形；此轮不同时改变为等比填充，以免叠加另一项变化。

## 两组匹配配置

| 设置 | B0：256 | B1：384 |
| --- | --- | --- |
| 三路图 | 主相机、左腕、右腕 | 相同 |
| `data.img_size` | 256 | 384 |
| `data.image_resize_mode` | `opencv_lanczos4` | `opencv_lanczos4` |
| `data.preserve_image_resolution` | `true` | `true` |
| `train.align_params.llm.image_token_size` | 8 | 12 |
| 每张视觉 token | 64（8×8） | 144（12×12） |
| 三张视觉 token | 192 | 432 |
| 加上每图两个边界 token | 198 | 438 |

这里按当前 Qwen3-VL 的 patch size=16、spatial merge size=2 计算；实际运行应核对 processor 配置和 `image_grid_thw`。`preserve_image_resolution=true` 用于避免 processor 再次缩放，使设定分辨率确实进入视觉编码器；校验应在网格不符时直接失败。

**B0 也重新运行 Lanczos4。**旧代码默认的 `torchvision_bilinear` 行为保留不变；历史 bilinear 训练结果不能直接充当这次匹配实验的 B0，否则插值方式和分辨率同时变化。

教师权重、损失权重和教师内部输入尺寸保持相同。`llm.image_token_size` 表示学生第一路图的 token 网格边长，需要随 8→12 修改，并不是把教师输入尺寸也改成 384。不过，送入教师前的源图经过了不同的重采样，教师目标像素可能有细微差异，因此不能宣称两组教师目标完全逐像素一致。

## 准备与 CPU 检查

以下命令从仓库根目录运行。路径占位符需要替换成实际文件；无需复制或修改原始数据。

```bash
python tools/prepare_resolution_baseline.py \
  --base-config xsy_utils/configs/robotwin_clean_baseline.yaml \
  --output-dir .runtime/resolution_exp \
  --run-root /root/ql/outputs
```

生成 `.runtime/resolution_exp/b0_256.yaml` 和 `b1_384.yaml`。检查两份配置，除上述分辨率相关设置外，应使用相同的 checkpoint、数据清单、归一化统计、教师和训练设置。基础配置包含服务器路径，运行前应确认这些路径在目标机器有效。

用一张实际 RGB 图检查完整 processor 的 CPU 路径，不加载模型权重：

```bash
python tools/check_resolution_baseline.py \
  --config .runtime/resolution_exp/b0_256.yaml \
  --image /path/to/head_camera_frame_69.png \
  --processor /path/to/Qwen3-VL-4B-Instruct

python tools/check_resolution_baseline.py \
  --config .runtime/resolution_exp/b1_384.yaml \
  --image /path/to/head_camera_frame_69.png \
  --processor /path/to/Qwen3-VL-4B-Instruct
```

此检查需要完整的 CPU 图像处理依赖和可访问的 processor 文件。脚本语法通过、插值函数通过，不等于真实 processor 已通过，更不等于 GPU 训练或闭环评测已通过。

预期：256 组单图网格为 `[1,16,16]`、merge 后 64 个 token；384 组为 `[1,24,24]`、merge 后 144 个。应确认不是“先放大到 384，随后又缩回 256”。两组均维持三路图、同一排序和有效 mask。

## 1000 步试跑

先用较小的两组匹配试跑确认吞吐、显存、loss 和 checkpoint 可用，再决定是否扩大训练预算。下例为 Linux Bash 命令；`xsy_utils/resolution_exp/train.sh` 为 node1/2 选择已核验的共享训练环境，并调用通用训练脚本。

```bash
# 示例假设八张卡均已分配给本实验；按实际分配修改，不能据此认定当前卡空闲。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash xsy_utils/resolution_exp/train.sh 256 --dry-run

# 首先各跑 2 步，包含一次保存，确认前向、反向、优化器和 checkpoint。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash xsy_utils/resolution_exp/train.sh 256 --train.max_steps 2 --train.save_steps 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash xsy_utils/resolution_exp/train.sh 384 --train.max_steps 2 --train.save_steps 2

# 两步检查通过后，用全新 run 从相同权重开始，默认各 1000 步。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash xsy_utils/resolution_exp/train.sh 256
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash xsy_utils/resolution_exp/train.sh 384
```

`MICRO_BS=4` 只是起点，不是显存可容纳的保证。固定两组 `CUDA_VISIBLE_DEVICES`、`GLOBAL_BS`、seed、初始化 checkpoint、学习率、训练步数和评测预算。脚本按 GPU 数与 micro batch 计算梯度累积，要求 `GLOBAL_BS` 可被 `MICRO_BS × GPU数` 整除。

384 若 OOM，降低 micro batch、保持等效全局 batch；为减少执行差异，必要时两组一起改为相同的 micro batch。不要只缩小 384 的全局 batch。每次重跑使用新的 `RUN_NAME`，避免已有 checkpoint 的自动恢复影响初始化；若改变梯度检查点等训练开关，也要记录并尽量两组一致。

启动器默认 `TRAIN_ENV=/2025233210/xiesy/lingbot_ws/conda_envs/lingbotvla`、`MICRO_BS=4`、`GLOBAL_BS=128`，输出到当前代码父目录的 `outputs/`，默认 run 名带时间戳，拒绝复用已有目录。可用环境变量覆盖。原脚本在 `/root/ql` 下会寻找不存在的 `/root/ql/conda_envs`，因此请用这个入口。

两台机器的 `/dev/shm` 均为 4 GiB。启动器默认每进程 `NUM_WORKERS=1`、`PREFETCH_FACTOR=1`；原配置 8 workers、默认预取 2，在 384 下有较高的共享内存峰值风险。遇到 worker bus error 可先两组都设 `NUM_WORKERS=0` 排查。吞吐比较时须保持两组 loader 设置一致，并确认没有被 CPU 供数限制。

上述命令是待执行的验证与训练步骤，不能视为已经完成 GPU 验证。

## 评测要恢复同一份图像配置

正常保留 `run/checkpoints/global_step_N/hf_ckpt` 目录结构时，部署入口会读取对应 run 的 `lingbotvla_cli.yaml`。`tools/render_robotwin_inference_config.py` 也优先从该文件恢复 `data`，从而保留训练时的分辨率、插值方式和 processor 网格约束。

如果将 `hf_ckpt` 单独复制到其他位置，应显式指定匹配的配置和训练归一化统计，不能让它回退到默认 256 配置。例如，对 384 组：

```bash
python tools/render_robotwin_inference_config.py \
  --checkpoint /path/to/exported/hf_ckpt \
  --qwen /path/to/Qwen3-VL-4B-Instruct \
  --data-config .runtime/resolution_exp/b1_384.yaml \
  --norm-stats /path/to/training_robotwin_norm_stats.json \
  --output .runtime/resolution_exp/b1_384_inference.yaml
```

启动服务或评测脚本时，设置 `LINGBOT_INFERENCE_CONFIG` 为生成文件的**绝对路径**。256 组同理使用 `b0_256.yaml`，两组归一化统计应与训练一致。图像配置未恢复就运行评测，会让实验失去可比性；384 模型的网格检查也可能直接阻止错误配置运行。

导出工具按显式 `--norm-stats`、`data.norm_stats_file`、训练 `robot_config_root/robotwin.yaml` 的顺序解析统计文件，与 RoboTwin 服务入口一致。如果训练映射在当前机器不可访问，会要求明确提供统计路径，避免误用仓库附带的 clean+randomized 统计。

## 记录什么，如何解释

两组在同一训练步数保存 checkpoint，用相同任务、clean/randomized 条件和评测预算比较：

- 闭环成功率与成功次数/总次数；保留逐任务结果和失败视频，关注是否减少小物体定位、接触或对齐失败。
- 实际视觉 token 数、峰值显存、稳定训练 step 时间和推理延迟；432/192 是视觉 token 倍数，不是整模型耗时倍数。
- action loss 与教师辅助 loss，用于排查训练异常；离线 loss 下降不能代替闭环成功率提升。
- 代码版本、两份实际配置、processor 版本、初始化权重、数据清单、seed、GPU 数、全局 batch 和训练步数。

开发阶段可先沿用团队的每任务/条件 **25 次**评测来筛选，但小样本波动较大。比赛要求的每任务/条件 **100 次**评测应单独执行，不能将 25 次开发结果当作正式结果。相同 seed 也不保证所有场景、语言和环境随机源完全匹配，应保留实际场景与指令记录；看到收益后再增加独立 seed 复核。

可能看到的现象：

- **成功率提高且代价可接受：**支持“大栅格/更多视觉 token 在当前源数据上有用”，值得继续；不等于证明新增了真实细节。
- **只增加耗时或显存，成功率没有清晰变化：**当前轻量预算未看到收益；1000 步或少量评测不足以否定更充分训练或 ROI。
- **辅助 loss 异常、token 数没变、训练/部署不一致：**先修复处理流程，不能解释为分辨率方案无效。

## 下一步：固定预算重新分配视觉内容

本轮不加入 ROI。确认 B0/B1 可运行并有匹配结果后，再分别以 **192** 和 **432** 为视觉 token 预算，比较全图与局部区域的分配。不要把“三路全图后直接追加一张 ROI”叫作 token 不变，它会额外增加预算。

ROI 实验应继续保留相同训练/评测条件，必要时加入重复全图或降采样 ROI 对照，区分局部放大、上下文变化和额外 token 的影响。训练与推理必须使用相同的裁剪、插值、图像顺序和分辨率设置。

## 本地验证记录（2026-10-17）

隔离环境 `.runtime/resolution-env`：PyTorch 2.8.0+cpu、torchvision 0.23.0+cpu、transformers 4.57.3、OpenCV 4.11.0。`tests/test_resolution_baseline.py` 的 **15 项全部通过，无跳过**。

使用官方 clean 样本 episode0 的三个相机第 69 帧（均为 320×240），实际执行共享 Lanczos 缩放与 `prepare_images`，结果如下：

| 输入 | 各相机实际 `image_grid_thw` | `pixel_values` shape | 三路视觉 token |
| --- | --- | --- | --- |
| 256² | `[1,16,16]` | `[3,256,1536]` | 192（含边界 198） |
| 384² | `[1,24,24]` | `[3,576,1536]` | 432（含边界 438） |

两组使用官方 `Qwen/Qwen3-VL-4B-Instruct` 的 processor 元数据，固定 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`，实际类为 `Qwen2VLImageProcessorFast`。原图取自官方 `TianxingChen/RoboTwin2.0` revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186` 的 `adjust_bottle/aloha-agilex_clean_50`。服务器副本及服务器 checkpoint 的 processor 配置仍需连接后复核。

本地结果保存在 `.runtime/resolution_exp/check_b0_256.json`、`check_b1_384.json` 与 `test_resolution_baseline.log`；插值样例在同目录的 `previews/`。这些运行产物与 CPU 环境由 `.gitignore` 排除，可按上面的命令重建。

已确认 384 没有被 processor 压回 256。此记录只覆盖图像处理、网格与配置检查，**未加载模型权重，未执行完整模型前向、GPU 训练或闭环成功率评测**。

## 服务器复查记录（2026-10-18）

- node1、node2 使用共享环境 torch 2.8.0+cu128、torchvision 0.23.0+cu128、transformers 4.57.3、opencv-python-headless 4.11.0.86、LeRobot 0.4.2。权重、教师、50 个数据集 metadata、归一化统计路径均存在。
- 两台机器各用实际 `robotwin_clean_adjust_bottle` 第 69 帧经过 `VLADataset`、`FeatureTransform` 和真实 Qwen processor；256/384 当前与未来图像均通过，输出有限值。训练样本状态为 `[55]`、动作为 `[50,55]`。
- 同一源图分别调用实际训练流程与部署 `resize_image` 方法、推理 `FeatureTransform` 后，`images` 和 `image_grid_thw` 完全相同；三路 token 为 192/432。
- 全程设置 `CUDA_VISIBLE_DEVICES=''`。原训练入口导入 Triton 算子时就查询 CUDA，因此 CPU 检查只加载其实际参数 dataclass 定义与部署 resize 方法，不加载完整模型；这不等价于训练入口、GPU 前后向或编译检查。
- 修复配置生成器对空 `align_params` 误启用教师的问题，新增自有输出目录选项与 H20 启动器。本地 16 项测试全部通过，无跳过。
- 检查证据保存在本地 `.runtime/h20_dataset_smoke.json`、`h20_progress_compact.json`。真实数据检查只读取源数据，没有运行新的训练或评测作业。

下一次获得 GPU 后，先运行两组各 2 步加 checkpoint 保存；通过后再做 1000 步匹配实验。尚未确认 384 的完整模型显存峰值、Triton/compile、FSDP 梯度累积及 checkpoint 恢复，不能保证首次 GPU 运行绝无问题。

## W&B 记录

用户实验通过 node1 的 `ql_roi_train` tmux 队列运行。队列从实验专用的受限文件读取密钥，仅向自己的训练子进程传入 `WANDB_API_KEY`；普通配置和 Git 中不保存密钥。

队列为两组分别传入 `use_wandb=true`、同一 `wandb_project`、独立的 `wandb_name`，并按实验组区分 smoke/train。W&B 在 TensorBoard writer 创建前初始化并开启 `sync_tensorboard`，记录现有 loss、学习率、梯度、显存和 step 时间指标；结束时完成同步。

新格式密钥使用 `.runtime/wandb_sdk` 中单独安装的 W&B 0.22.3，通过 `QL_WANDB_SDK_PATH` 注入本实验；共享 Conda 环境的 W&B 版本保持原状。账号、组织和项目的非敏感配置位于 `.runtime/node1_queue/wandb_settings.json`。运行时以该文件和 CLI 覆盖为准，基础 YAML 的默认 W&B 开关不影响队列设置。
