# Baseline BS256 × 20k

**当前执行安排（2026-10-21用户调整）：已跳过5k、10k中途评测，从完整5k DCP直接续训到20k，只在20k做350次完整评测。当前运行在30375。**

从团队实际baseline10k的`lingbotvla_cli.yaml`复制模型、数据和训练配置，以同一原始预训练权重重新开始训练。本实验只研究新的batch/更新预算，不叠加Joint Delta、EE输入、加权loss或Spline。

- 8张H20，每卡micro-batch16，梯度累积2次，有效BS256。
- 保留baseline的FP32精度配置、原动作维度mask、末尾重复动作监督、绝对关节与夹爪动作、seed42、chunk50、Muon及专家学习率倍率。
- 主学习率从1e-4 cosine下降到5e-5，全程调度长度20000。沿用baseline warmup=0、weight_decay=0。
- 先完成1—50步短跑并保存完整DCP，再恢复到5000。按最新要求，从5000直接续训到20000。短跑计入20k预算；优化器、学习率和数据加载状态通过严格DCP恢复，学习率不重启。
- 当前只在20k评测7任务×clean/random×25=350次，seed0、推理FP32、compile=False、执行长度50。最初计划的5k、10k评测已取消。
- 累计采样量分别128万、256万、512万；若以后跑BS512，对应2.5k、5k、10k检查点。它们与历史BS128×10k总预算不同，不能仅归因于batch大小。
- 本容器只有4GiB共享内存，使用num_workers=2、prefetch_factor=1。每个DataLoader batch包含两个micro-batch，实际短跑检查内存与吞吐。
- 每1000步保存DCP/HF；不同评测阶段使用独立训练目录，保留各检查点对应的原始配置。

```bash
bash ysj_utils/train_baseline_bs256.sh --prepare-only
bash ysj_utils/train_baseline_bs256.sh
# 显式启用最新安排；协议会记录到plan.json，随后重复执行也会沿用：
bash ysj_utils/train_baseline_bs256.sh --eval-final-only
```

默认输出：`/2025233210/ysj/lingbot_ws/outputs/baseline_bs256_20k_20261020`。

在tmux会话`ysj-bs256-20k`运行。查看：

```bash
tmux attach -t ysj-bs256-20k
cat /2025233210/ysj/lingbot_ws/outputs/baseline_bs256_20k_20261020/status.json
```

`results.json`汇总完成的评测；各`step_XXXXX/evaluation_*.json`保存完整评测证据。重复执行脚本时，从最新完整DCP恢复，复用已经校验完成的评测。失败时记录FAILED并停止，不能跳过失败阶段。

本机4项配置/恢复/日志检查通过，服务器完整CPU测试套件执行成功。历史baseline10k为162/350（46.29%）；本次是否提升由实际评测决定。

2026-10-20 23:27（北京时间）在端口30010的8张H20启动，执行代码为`89a5541`。GPU短跑1—50步已完成，末步loss0.1348、梯度范数0.4236；已导出完整DCP与HF。随后严格恢复step50，完成51、52、53次更新，loss分别0.1242、0.1226、0.1313，无回退到重新初始化。2026-10-21 01:06之后采集证据时已经到866步，loss0.0465、梯度范数0.5880，主学习率约9.98e-5，稳定每步约6.36秒。GPU显存约88840—91142MiB（约87—89GiB），共享内存占用正常。

源配置与新配置中的model、全部align_params、精度、loss类型、优化器、seed、chunk、主学习率范围、warmup与weight_decay均已核对。缺失的旧动作实验字段在新版配置中显式写为关闭值，不代表启用了额外方法。资源、路径、保存频率、batch、目标步数、20k调度长度与恢复配置的差异保存在证据中。

启动证据：[reports/baseline_bs256_20k_started_2026-10-21.json](reports/baseline_bs256_20k_started_2026-10-21.json)。按当前吞吐，整轮训练及三轮评测先估计约39—43小时；后续以实际吞吐和保存/评测时间更新。

5k训练于2026-10-21 08:35（北京时间）完成，末步loss0.0281、梯度范数0.6031、主LR约9.27e-5，DCP和HF导出完整。首次评测因30010继承的HTTP代理转发本机WebSocket连接而失败，未形成有效成功率。修复为本机RPC显式`proxy=None`；本机及服务器模型/仿真两个环境各3项网络测试通过，含无效代理下的真实RPC测试。依据：[websockets代理说明](https://websockets.readthedocs.io/en/stable/topics/proxies.html)。

由于30010被队友ROI训练占用，恢复任务转至空闲的30375。用户随后要求尽量节省时间、直接训20k，已停止自有中途评测并切换`--eval-final-only`。未修改模型、数据、BS256、20k学习率日程或动作监督。新的训练目录为`step_20000/train`，断点来源为`step_05000/train/checkpoints/global_step_5000`。剩余15k步按此前6.4秒/步约26.7小时，加最终评测和保存开销，先估计余下28—30小时。

失败与修复记录：[首次评测失败](reports/baseline_bs256_5k_failed_eval_2026-10-21.json)、[网络修复与迁移](reports/baseline_bs256_eval_recovery_2026-10-21.json)。
