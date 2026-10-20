# Baseline BS256 × 20k

从团队实际baseline10k的`lingbotvla_cli.yaml`复制模型、数据和训练配置，以同一原始预训练权重重新开始训练。本实验只研究新的batch/更新预算，不叠加Joint Delta、EE输入、加权loss或Spline。

- 8张H20，每卡micro-batch16，梯度累积2次，有效BS256。
- 保留baseline的FP32精度配置、原动作维度mask、末尾重复动作监督、绝对关节与夹爪动作、seed42、chunk50、Muon及专家学习率倍率。
- 主学习率从1e-4 cosine下降到5e-5，全程调度长度20000。沿用baseline warmup=0、weight_decay=0。
- 先完成1—50步短跑并保存完整DCP，再恢复到5000、10000、20000步。短跑计入20k预算；中途评测前后的优化器、学习率和数据加载状态通过严格DCP恢复，学习率不重启。
- 在5k、10k、20k分别评测7任务×clean/random×25=350次，seed0、推理FP32、compile=False、执行长度50。
- 累计采样量分别128万、256万、512万；若以后跑BS512，对应2.5k、5k、10k检查点。它们与历史BS128×10k总预算不同，不能仅归因于batch大小。
- 本容器只有4GiB共享内存，使用num_workers=2、prefetch_factor=1。每个DataLoader batch包含两个micro-batch，实际短跑检查内存与吞吐。
- 每1000步保存DCP/HF；不同评测阶段使用独立训练目录，保留各检查点对应的原始配置。

```bash
bash ysj_utils/train_baseline_bs256.sh --prepare-only
bash ysj_utils/train_baseline_bs256.sh
```

默认输出：`/2025233210/ysj/lingbot_ws/outputs/baseline_bs256_20k_20261020`。

在tmux会话`ysj-bs256-20k`运行。查看：

```bash
tmux attach -t ysj-bs256-20k
cat /2025233210/ysj/lingbot_ws/outputs/baseline_bs256_20k_20261020/status.json
```

`results.json`汇总完成的评测；各`step_XXXXX/evaluation_*.json`保存完整评测证据。重复执行脚本时，从最新完整DCP恢复，复用已经校验完成的评测。失败时记录FAILED并停止，不能跳过失败阶段。

本机4项配置/恢复/日志检查通过。正式GPU短跑和恢复证据将在启动后记录。历史baseline10k为162/350（46.29%）；本次是否提升由实际评测决定。
