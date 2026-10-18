# Baseline 上的 Weighted / Spline 两组 5k 实验

以用户 2026-10-18 最后确认的要求为准：**绝对关节动作、无额外 EE 输入、保留 padding mask、fp32，分别训练两组 5000 次更新**。不使用 Joint Delta 的配置或 checkpoint 续训。

| 项目 | 两组共同设置 |
| --- | --- |
| 基础代码 | 个人分支，基于仓库 main；准备时记录 origin/main 和实际代码提交 |
| 配置来源 | 团队实际完成的 baseline_gbs128_5k/lingbotvla_cli.yaml；避免使用会误导的默认 config.yaml |
| 初始权重 | 同一官方预训练 lingbot-vla-v2-6b；两组分别从头 post-train |
| 动作、输入 | baseline 的绝对 12 维关节 + 2 维夹爪；不追加 EE state |
| 数据、归一化 | 原 50 个 clean task 数据，复用 baseline 的绝对动作统计并校验 SHA256 |
| 训练 | global BS128，8 卡每卡 16，5000 updates，seed42 |
| 精度 | enable_fp32=True、enable_mixed_precision=True，与团队 baseline 相同。FSDP2 参数/梯度/输出为 fp32；冻结视觉教师保留 baseline 的 bf16 autocast。不是声称所有算子都为 fp32 |
| 其他设置 | L1_fm，原 Muon/学习率/5k cosine schedule、图像设置、深度和视频辅助目标、训练 compile 设置保持来源配置 |
| mask | mask_action_padding=True；padding 不参与动作监督 |
| 资源适配 | prefetch_factor 从 baseline 的 4 降为 1，worker 仍为 8；此容器 /dev/shm 只有 4 GiB。不是改采样策略或学习率 |
| 评测 | 各 350 次：7 task × clean/random ×25；seed0，fp32，compile=False |

## Weighted：采用论文权重，保留 baseline 的 L1

依据：[Weighted Conditional Flow Matching](https://arxiv.org/abs/2507.22270)，第 3 节及附录 D。它不是之前的“夹爪权重 ×2”或“后续时间步降权”。

在当前归一化动作 y 和**同一次 forward 使用的噪声 x**之间，计算有效动作维度上的距离：

```
c_i = sqrt(sum_valid((noise_i - action_i)^2) / valid_dimensions_i)
w_i = exp(-c_i / epsilon - log_Z)
loss = sum_valid(w_i * 原始逐元素 L1 error) / sum(valid_mask)
```

这里用 RMS 距离，相当于将论文欧氏距离除以 sqrt(d)，epsilon 对应论文中的 kappa；这样 padding 长度不同不会仅因维数多而天然受罚。未使用的 55 维槽位不参与距离计算。

训练前，在 50 个任务各抽 8 个真实 chunk（总 400 个），配合 128 组独立 Gaussian 噪声，按附录 D 扫描带宽；选择源、目标两个边际倾斜代理的相对方差都不超过 0.01 的最小候选值。固定记录 log_Z；训练时不按每卡 minibatch 重算归一化，以免引入不同的 rank 归一化目标。校准使用独立 RNG，不消耗训练随机状态。

日志记录权重均值、最大值、有效样本量 ESS、距离和未加权动作 loss。两组的训练 loss 不可直接当作成功率比较，因为监督目标发生变化。

**与原论文的差别：**原论文理论和主要实验使用平方误差，且研究无条件生成。这里为保持 baseline 一致，保留 L1_fm，使用其 Gibbs 权重思想；常数势函数 f=g=1、可变长度 mask、机器人条件生成也是适配。因此这是 paper-inspired ablation，不是完整复现。边际代理通过不等于证明每个观察条件的动作分布都不偏移，也不保证成功率提升。

验证的问题：相同预算下，适度强调较短噪声—动作配对，能否改善最终策略成功率，而不是仅降低加权后的 loss？

## Spline：BEAST 启发的连续控制点表示

主要依据：[BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning，NeurIPS 2025 主会](https://papers.nips.cc/paper_files/paper/2025/hash/fc75cfcec16170f2b54d00283c739928-Abstract-Conference.html)。[论文第4节及附录D](https://arxiv.org/html/2506.06072v3)分别说明控制点拟合，以及与 flow matching 的结合。

将 baseline 的 50 步归一化绝对动作拟合成 16 个三次 clamped B-spline 控制点。模型仍用原 action head 和 L1 flow loss，预测 16 个控制点；推理时先还原为 50 步，再按 baseline 的绝对动作统计反归一化，交给原 qpos 控制器。输入、夹爪和关节动作语义不变。没有增加 delta 或 EE 监督。

完整 chunk 使用伪逆拟合，尾部不足 50 步时仅对有效点做 ridge 拟合；padding 不参与拟合。训练日志记录重建 RMSE 和夹爪误差；准备阶段输出真实数据的关节弧度、夹爪误差分位数。

**与 BEAST 的差别：**本次是连续控制点版本，无离散化 token，无更换 VLM，且未加入 BEAST 使用上一段动作固定首控制点的跨段约束。这里只保证单个 spline 曲线内部平滑，不能宣称跨 chunk 连续。16/50 是本次实验超参，不冒充论文原设置。急剧夹爪开合、接触动作可能被平滑，从而降低成功率。

补充阅读：[B-spline Policy](https://arxiv.org/abs/2607.09648)与动作表示非常直接相关，但作者标注为 2026 预印本，**不标成 ICLR/ICRA/CVPR/NeurIPS 正式论文**。本次优先以正式 NeurIPS 的 BEAST 为依据。

背景阅读：[Neural Dynamic Policies for End-to-End Sensorimotor Learning，NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/354ac345fd8c6d7ef634d9a8e3d47b83-Abstract.html)。它支持“让网络预测紧凑的轨迹参数，再生成连续动作”的总体动机，但使用 DMP 微分方程与高斯基函数，**不是 B-spline，也不是本次直接复现的算法**。不为凑会议名而将无关的 spline 视觉论文当成机器人动作证据。

验证的问题：压缩动作表示能否在同样 5k 更新下获得更好的策略或更快训练/推理；同时检查平滑是否损失抓取、接触精度。

## 验证与运行

```
python -m unittest discover -s tests/action_parameterization -v
python ysj_utils/run_baseline_supervision.py --run-dir /path/to/new_run --prepare-only
python ysj_utils/run_baseline_supervision.py --run-dir /path/to/new_run --run-prepared
```

准备程序输出 plan.json、source_baseline.yaml、gibbs_calibration.json、spline_fit.json，以及 Control/Weighted/Spline 的独立 train.yaml 和统计校验收据。**Control 仅生成配置，不自动占用额外 5k 训练预算。**

运行顺序：Weighted 5k → 350 次评测 → Spline 5k → 350 次评测。训练或评测失败即写 status.json 并停止；不自动换配置或跳过失败。进入正式训练后检查最初几次更新的 loss、梯度、精度和数据形状。训练器每 1000 步保存 DCP/HF；脚本启动时拒绝复用已启动的输出目录，避免意外重复训练。需要断点恢复时先检查完整 DCP，再明确指定恢复，不能拿 HF 文件宣称优化器也恢复了。

实测起跑后约 4.2 秒/步，Weighted 单组训练初估约 6 小时，加上 Spline 和两次评测，整个队列暂估 12–15 小时；这是起跑阶段外推，缓存、导出和后续吞吐仍会影响耗时。每组完整评测参考之前约 1 小时。tmux 能在本机断网/关机后继续运行。

2026-10-18 21:15（北京时间）已在 30375 的 8 张 H20 启动，tmux 会话 `ysj-supervision`。输出根目录：`/2025233210/ysj/lingbot_ws/outputs/baseline_supervision_5k_20261018`。代码提交 `8363bf8`，本机和服务器均 103 项测试通过。采集启动证据时 Weighted 已完成 29 次优化器更新，第29步 loss=0.1888，梯度范数=0.4033，LR≈1e-4，显存约82.3 GiB/卡、利用率约100%；Spline 尚在队列中，不能说已完成其完整 GPU 训练验证。

校准的 RMS 带宽为 0.53452266，log_Z=-2.23265968；源/目标边际倾斜代理相对方差约0.00255/0.00893。Spline 400段真实动作拟合的关节 MAE≈0.000432弧度，夹爪 MAE≈0.00126，最大夹爪误差≈0.0332。这里都是算法/压缩诊断，尚无新成功率结果。

启动脚本的数字进度匹配曾把上游日志分母误当作5000；实际是每epoch的4295。训练本身正常，**5000预算不变**。本地已修复后续启动的匹配；当前进度以实际 `Weighted/training.log` 为准，不用旧 launcher 的 step=0 判断训练停了。此修复没有热更新正在运行的训练代码。

```bash
tmux attach -t ysj-supervision
tail -n 5 /2025233210/ysj/lingbot_ws/outputs/baseline_supervision_5k_20261018/Weighted/training.log
```

启动、实际配置、校准与起跑日志证据保存在 [reports/baseline_supervision_5k_launch_2026-10-18.json](reports/baseline_supervision_5k_launch_2026-10-18.json)。

## 比较边界

两组彼此的公共配置相同，各自只改变方法。可以与历史 baseline 5k 作描述性比较，**但历史 baseline 没有 padding mask**；因此不能把成功率差异全部归因于 Weighted/Spline。要确认独立贡献，最终仍需一个相同 mask、fp32 和预算的 Control 5k。已生成其配置，未擅自追加第三组正式训练。

单任务每种环境仅 25 次，1 次成功对应 4 个百分点；小幅变化应多 seed 复测，不能预先承诺提升。
